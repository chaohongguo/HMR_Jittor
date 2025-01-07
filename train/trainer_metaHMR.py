import jittor
from utils import BaseTrainer
import config
import constants
from datasets import BaseDataset_cliff as BaseDataset, MixedDataset
from models import SMPL, build_model
from train.loss import SetCriterion
from collections import OrderedDict
from utils.geometry import perspective_projection, cam_crop2full
from models.metaHMR.maml import gradient_update_parameters
import time
from eval import run_evaluation_metaHMR
import numpy as np


class Trainer(BaseTrainer):
    def init_fn(self):
        """
        init dataset,model,
        :return:
        """
        if self.options.train_dataset is not None:
            self.train_ds = BaseDataset(self.options, self.options.train_dataset, is_train=True,
                                        bbox_type=self.options.bbox_type)
        else:
            self.train_ds = MixedDataset(self.options, is_train=True)

        self.eval_ds = BaseDataset(self.options, self.options.eval_dataset, is_train=False,
                                   bbox_type=self.options.bbox_type)

        self.model_main = build_model(config.SMPL_MEAN_PARAMS, pretrained=True,
                                      backbone=self.options.backbone,
                                      model_name="metaHMR_main",
                                      option=self.options)
        self.model_aux = build_model(config.SMPL_MEAN_PARAMS, pretrained=True,
                                     backbone=self.options.backbone,
                                     model_name="metaHMR_aux",
                                     option=self.options)

        model_aux_st = self.model_aux.state_dict()
        model_main_st = self.model_main.state_dict()

        self.params = OrderedDict(self.model_main.meta_named_parameters())

        self.outer_lr = self.options.outer_lr  # default 1e-4
        self.inner_lr = self.options.inner_lr  # default 1e-5
        self.inner_steps = self.options.inner_step  # default 1
        self.val_steps = self.options.val_step  # default
        self.first_order = self.options.first_order  # default True

        if not self.options.no_learn_loss:
            print('------------------ with aux net, learn rate %.8f-----------------------' % self.options.aux_lr)

            self.optimizer = jittor.optim.Adam(
                params=self.model_main.parameters(), lr=self.outer_lr,
                betas=[self.options.beta1, self.options.beta2], eps=self.options.eps_adam,)

            self.optimizer_aux = jittor.optim.Adam(
                params=self.model_aux.parameters(), lr=self.options.aux_lr)
        else:
            print('------------------ without aux net -----------------------')
            self.optimizer = jittor.optim.Adam(self.model_main.parameters(),
                                               lr=self.outer_lr,
                                               betas=[self.options.beta1, self.options.beta2],
                                               eps=self.options.eps_adam)

        self.smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=self.options.batch_size, create_transl=False)

        self.joints_idx = 25
        self.joints_num = 49

        weight_dict = {
            "loss_keypoint_2d": self.options.keypoint_loss_weight,
            "loss_keypoint_3d": self.options.keypoint_loss_weight,
            "loss_shape": self.options.shape_loss_weight,
            "loss_pose": self.options.pose_loss_weight,
            "loss_beta": self.options.beta_loss_weight,
            "loss_con": self.options.con_loss_weight,
            "loss_camera": self.options.cam_loss_weight
        }

        self.criterion = SetCriterion(weight_dict=weight_dict, model_name="metaHMR")
        self.criterion_aux = SetCriterion(weight_dict=weight_dict, model_name="metaHMR")
        self.criterion_eval = SetCriterion(weight_dict=weight_dict, model_name="metaHMR")

        self.models_dict = {
            'model': self.model_main,
            'model_aux': self.model_aux
        }
        self.optimizers_dict = {
            'optimizer': self.optimizer,
            'optimizer_aux': self.optimizer_aux
        }

        self.bbox_type = self.options.bbox_type
        self.focal_length = constants.FOCAL_LENGTH

        if self.bbox_type == 'square':
            print(">>>>>>>>>>training using square bbox")
            self.crop_w = constants.IMG_RES
            self.crop_h = constants.IMG_RES
            self.bbox_size = constants.IMG_RES
        elif self.bbox_type == 'rect':
            print(">>>>>>>>>>training rect bbox")
            self.crop_w = constants.IMG_W
            self.crop_h = constants.IMG_H
            self.bbox_size = constants.IMG_H
        # if self.options.use_extraviews:
        #     print(">>>>>>>>>>training using extra view")

        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

    def train_step(self, input_batch, cur_step):
        self.model_main.train()
        self.model_aux.train()

        # img base info
        img_names = input_batch['img_name']  # [B,]
        dataset = input_batch['dataset_name']  # [B,]
        batch_size = len(img_names)

        images = input_batch['img']  # [B,3,h,w]

        # crop bbox 2D gt keypoint
        gt_keypoints_2d = input_batch['keypoints']  # [B,49,3]
        # origin full 2d gt keypoint
        gt_keypoints_2d_full = input_batch['keypoints_full']  # [B,49,3]

        gt_pose = input_batch['pose']  # [B,24*3]
        gt_betas = input_batch['betas']  # [B,10]
        gt_joints = input_batch['pose_3d']  # [B,24,4]
        has_smpl = input_batch['has_smpl']  # [B,]
        has_pose_3d = input_batch['has_pose_3d']  # [B,]
        is_flipped = input_batch['is_flipped']  # [B,]
        crop_trans = input_batch['crop_trans']  # [B,2,3]
        full_trans = input_batch['full_trans']  # [B,2,3]
        # inv_trans = input_batch['inv_trans']  # [B,2,3]
        bbox_info = input_batch['bbox_info']  # [B,3]
        center = input_batch['center']  # [B,2]
        scale = input_batch['scale']  # [B,1]
        focal_length = input_batch['focal_length'].float()  # [B,1]
        img_h, img_w = input_batch['img_h'].view(-1, 1), input_batch['img_w'].view(-1, 1)  # [B,1]
        full_img_shape = jittor.concat((img_h, img_w), dim=1)  # [B,2]
        camera_center = jittor.concat((img_w, img_h), dim=1) / 2  # [B,2]

        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        loss_keypoint_2d_outer = jittor.array(0, dtype=jittor.float32)
        loss_keypoint_3d_outer = jittor.array(0, dtype=jittor.float32)
        loss_regr_pose_outer = jittor.array(0, dtype=jittor.float32)
        loss_regr_betas_outer = jittor.array(0, dtype=jittor.float32)
        loss_shape_outer = jittor.array(0, dtype=jittor.float32)

        # TODO group size
        group_size = batch_size
        group_num = int(batch_size / group_size)

        for i in range(group_num):
            task_id = range(i * group_size, (i + 1) * group_size)
            task_id = list(task_id)
            images_tr = images[task_id]
            bbox_info_tr = bbox_info[task_id]
            center_tr = center[task_id]
            camera_center_tr = camera_center[task_id]
            full_img_shape_tr = full_img_shape[task_id]
            scale_tr = scale[task_id].squeeze()
            focal_length_tr = focal_length[task_id].squeeze()
            params_cp = self.params

            # inner loop
            for in_step in range(self.inner_steps):
                pred_rotmat_inner, pred_betas_inner, pred_camera_inner = self.model_main(images_tr, bbox_info_tr, params=params_cp)
                # data = np.load('predict_meta.npz')
                #
                # pred_rotmat_inner = data['pred_rotmat_inner']
                # pred_betas_inner = data['pred_betas_inner']
                # pred_camera_inner = data['pred_camera_inner']
                #
                # pred_rotmat_inner = jittor.array(pred_rotmat_inner)
                # pred_betas_inner = jittor.array(pred_betas_inner)
                # pred_camera_inner = jittor.array(pred_camera_inner)
                pred_rotmat_aux, pred_betas_aux, pred_camera_aux = self.model_aux(images_tr, bbox_info_tr)

                pred_output_inner = self.smpl(betas=pred_betas_inner, body_pose=pred_rotmat_inner[:, 1:],
                                              global_orient=pred_rotmat_inner[:, 0].unsqueeze(1), pose2rot=False)
                pred_vertices_inner = pred_output_inner.vertices  # [B,N=6890,3]
                pred_joints_inner = pred_output_inner.joints  # [B,49,3]

                pred_output_aux = self.smpl(betas=pred_betas_aux, body_pose=pred_rotmat_aux[:, 1:],
                                            global_orient=pred_rotmat_aux[:, 0].unsqueeze(1), pose2rot=False)
                pred_vertices_aux = pred_output_aux.vertices  # [B,N=6890,3]
                pred_joints_aux = pred_output_aux.joints  # [B,49,3]

                pred_cam_full_inner = cam_crop2full(pred_camera_inner, center_tr, scale_tr,
                                                    full_img_shape_tr, focal_length_tr)  # [B,3]
                pred_keypoints2d_full_inner = perspective_projection(
                    pred_joints_inner,
                    rotation=jittor.float32(jittor.init.eye(3).unsqueeze(0).expand(batch_size, -1, -1)),
                    translation=pred_cam_full_inner,
                    focal_length=focal_length_tr,
                    camera_center=camera_center_tr)  # [B,49,2]
                pred_keypoints2d_with_conf_inner = jittor.concat(
                    (pred_keypoints2d_full_inner, jittor.ones(batch_size, 49, 1)), dim=2)  # [B,49,3]
                pred_keypoints2d_bbox_inner = jittor.linalg.einsum('bij,bkj->bki', crop_trans, pred_keypoints2d_with_conf_inner)

                predict_inner = {
                    "pred_rotmat": pred_rotmat_inner,
                    "pred_betas": pred_betas_inner,
                    # "pred_camera": pred_camera,
                    "pred_output": pred_output_inner,
                    "pred_keypoint_2d_bbox": pred_keypoints2d_bbox_inner,

                }

                gt_inner = {
                    "gt_keypoint_2d": gt_keypoints_2d,
                    "gt_pose": pred_rotmat_aux,
                    "gt_betas": pred_betas_aux,
                    "gt_joints": pred_joints_aux,
                    # "gt_camera": gt_camera,
                    "gt_output": pred_output_aux,
                }

                const_inner = {
                    "openpose_train_weight": self.options.openpose_train_weight,
                    "gt_train_weight": self.options.gt_train_weight,
                    "has_pose_3d": has_pose_3d[task_id],
                    "has_smpl": has_smpl[task_id],
                    "crop_w": self.crop_w,
                    "crop_h": self.crop_h,
                    "center": center[task_id],
                    "scale": scale[task_id],
                    "full_img_shape": full_img_shape[task_id]
                }

                loss_inner, _ = self.criterion_aux(predict=predict_inner, gt=gt_inner, const=const_inner, using_pseudo=True)

                params_cp = gradient_update_parameters(self.model_main, loss_inner, params=params_cp,
                                                       step_size=self.inner_lr, first_order=self.first_order)

            pred_rotmat, pred_betas, pred_camera = self.model_main(images_tr, bbox_info_tr, params=params_cp)

            pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                    global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints

            pred_cam_crop = jittor.stack([pred_camera[:, 1],
                                         pred_camera[:, 2],
                                         2 * self.focal_length / (self.bbox_size * pred_camera[:, 0] + 1e-9)],
                                        dim=-1)

            pred_cam_full = cam_crop2full(pred_camera, center_tr, scale_tr, full_img_shape_tr, focal_length_tr)

            pred_keypoints2d_full = perspective_projection(
                pred_joints,
                rotation=jittor.float32(jittor.init.eye(3).unsqueeze(0).expand(batch_size, -1, -1)),
                translation=pred_cam_full,
                focal_length=focal_length.squeeze(),
                camera_center=camera_center)
            pred_keypoints2d_with_conf = jittor.concat(
                (pred_keypoints2d_full, jittor.ones(batch_size, 49, 1)), dim=2)
            pred_keypoints2d_bbox = jittor.linalg.einsum('bij,bkj->bki', crop_trans, pred_keypoints2d_with_conf)

            predict = {
                "pred_rotmat": pred_rotmat,
                "pred_betas": pred_betas,
                # "pred_camera": pred_camera,
                "pred_output": pred_output,
                "pred_keypoint_2d_bbox": pred_keypoints2d_bbox,
            }

            gt = {
                "gt_keypoint_2d": gt_keypoints_2d[task_id],
                "gt_pose": gt_pose[task_id],
                "gt_betas": gt_betas[task_id],
                "gt_joints": gt_joints[task_id],
                # "gt_camera": gt_camera,
                "gt_output": gt_out,
            }

            const = {
                "openpose_train_weight": self.options.openpose_train_weight,
                "gt_train_weight": self.options.gt_train_weight,
                "has_pose_3d": has_pose_3d[task_id],
                "has_smpl": has_smpl[task_id],
                "crop_w": self.crop_w,
                "crop_h": self.crop_h,
                "center": center[task_id],
                "scale": scale[task_id],
                "full_img_shape": full_img_shape[task_id]
            }

            loss, losses = self.criterion(predict, gt, const, using_pseudo=False)
            loss_keypoint_2d_outer += losses['loss_keypoints_2d']
            loss_keypoint_3d_outer += losses['loss_keypoints_3d']
            loss_regr_pose_outer += losses['loss_regr_pose']
            loss_regr_betas_outer += losses['loss_regr_betas']
            loss_shape_outer += losses['loss_shape']

        group_nums = jittor.array(group_num, dtype=jittor.float32)
        loss_keypoint_2d_outer.divide(group_nums)
        loss_keypoint_3d_outer.divide(group_nums)
        loss_regr_pose_outer.divide(group_nums)
        loss_regr_betas_outer.divide(group_nums)
        loss_shape_outer.divide(group_nums)

        loss = self.options.shape_loss_weight * loss_shape_outer + \
               self.options.keypoint_loss_weight * loss_keypoint_2d_outer + \
               self.options.keypoint_loss_weight * loss_keypoint_3d_outer + \
               self.options.pose_loss_weight * loss_regr_pose_outer + \
               self.options.beta_loss_weight * loss_regr_betas_outer

        loss *= 60

        start = time.time()
        self.optimizer.zero_grad()
        self.optimizer.step(loss)
        loss_copy = loss.clone()
        self.optimizer_aux.zero_grad()
        self.optimizer_aux.step(loss_copy)
        end = time.time()

        losses = {'loss': loss.detach().item(),
                  'loss_keypoints': loss_keypoint_2d_outer.detach().item(),
                  'loss_keypoints_3d': loss_keypoint_3d_outer.detach().item(),
                  'loss_regr_pose': loss_regr_pose_outer.detach().item(),
                  'loss_regr_betas': loss_regr_betas_outer.detach().item(),
                  'loss_shape': loss_shape_outer.detach().item()}

        if cur_step % 10 == 0:
            print(losses)

        output = {
            # 'pred_vertices': pred_vertices.detach(),
            # 'pred_cam_t': pred_cam_crop.detach()
        }

        return output, losses

    def test(self):
        self.model_main.eval()
        self.model_aux.eval()

        mpjpe_3dpw, pa_mpjpe_3dpw, pve_3dpw = run_evaluation_metaHMR(self.model_main, self.model_aux,
                                                                     dataset_name="3dpw",
                                                                     dataset=self.eval_ds,
                                                                     result_file=None,
                                                                     criterion=self.criterion_eval,
                                                                     crop_h=self.crop_h,
                                                                     crop_w=self.crop_w,
                                                                     options=self.options,
                                                                     batch_size=self.options.batch_size)
        results = {
            'mpjpe': mpjpe_3dpw,
            'pa_mpjpe': pa_mpjpe_3dpw,
            'pve': pve_3dpw
        }
        return results

    def train_summaries(self, input_batch, output, losses):
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
