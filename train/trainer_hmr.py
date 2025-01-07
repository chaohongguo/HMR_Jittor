from models import hmr, SMPL, build_model
from utils import BaseTrainer
import config
import constants
from datasets import BaseDataset_hmr as BaseDataset, MixedDataset
import jittor
from jittor import nn
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation
from eval import run_evaluation
from jittor import Var
import numpy as np

jittor.flags.use_cuda = 1


class Trainer(BaseTrainer):
    def init_fn(self):
        """
        init dataset,model,
        :return:
        """
        if self.options.train_dataset is not None:
            self.train_ds = BaseDataset(self.options, self.options.train_dataset, is_train=True)
        else:
            self.train_ds = MixedDataset(self.options, is_train=True)

        self.model = build_model(config.SMPL_MEAN_PARAMS, pretrained=True,
                                 backbone=self.options.backbone, model_name=self.options.model_name)
        self.optimizer = jittor.optim.Adam(params=self.model.parameters(),
                                           lr=self.options.lr)
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False)
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_shape = nn.L1Loss()
        self.criterion_regr = nn.MSELoss()
        self.focal_length = constants.FOCAL_LENGTH
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}

        self.bbox_type = self.options.bbox_type
        if self.bbox_type == 'square':
            print("----------using square bbox---------")
            self.crop_w = constants.IMG_RES
            self.crop_h = constants.IMG_RES
            self.bbox_size = constants.IMG_RES
        elif self.bbox_type == 'rect':
            print("----------using rect bbox-----------")
            self.crop_w = constants.IMG_W
            self.crop_h = constants.IMG_H
            self.bbox_size = constants.IMG_H

        # self.loss = nn.L1Loss
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        _mse = (pred_keypoints_2d - gt_keypoints_2d[:, :, :-1]) ** 2
        loss = conf * _mse
        loss = loss.mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            _mse = ((pred_keypoints_3d - gt_keypoints_3d) ** 2)
            return (conf * _mse).mean()
        else:
            return jittor.float32(0)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """
        Compute per-vertex loss on the shape for the examples that SMPL annotations are available.
        """
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return jittor.float32(0)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = jittor.float32(0)
            loss_regr_betas = jittor.float32(0)
        return loss_regr_pose, loss_regr_betas

    def camera_losses(self, crop_cam, center, scale, full_img_shape, agora_occ=None):
        pass

    def train_step(self, input_batch):
        self.model.train()

        images = input_batch['img']
        gt_keypoints_2d = input_batch['keypoints']  # 2D keypoints
        gt_pose = input_batch['pose']  # SMPL pose parameters
        gt_betas = input_batch['betas']  # SMPL beta parameters
        gt_joints = input_batch['pose_3d']  # 3D pose
        dataset = input_batch['dataset_name']
        has_smpl = input_batch['has_smpl']  # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d']  # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped']  # flag that indicates whether image was flipped during data augmentation
        # rot_angle = input_batch['rot_angle']  # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name']  # name of the dataset the image comes from
        # indices = input_batch['sample_index']  # index of example inside its dataset
        batch_size = images.shape[0]
        # bbox_info = input_batch['bbox_info']
        # center, scale, focal_length = input_batch['center'], input_batch['scale'], input_batch['focal_length'].float()
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)
        gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length,
                                        img_size=self.options.img_res)

        pred_rotmat, pred_betas, pred_camera = self.model(images)

        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)

        # np.savez("predict.npz", pred_rotmat=pred_rotmat.numpy(),
        #          pred_betas=pred_betas.numpy(),
        #          pred_camera=pred_camera.numpy())
        # # 提取 pred_output 的所有属性值并存为字典
        # pred_data = {key: value.numpy() if value is not None else None
        #              for key, value in pred_output.items()}
        #
        # # 保存为 npz 文件
        # np.savez("smpl_output.npz", **pred_data)
        # print("SMPLOutput 数据已保存为 smpl_output.npz")

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = jittor.stack([pred_camera[:, 1],
                                   pred_camera[:, 2],
                                   2 * self.focal_length / (self.options.img_res * pred_camera[:, 0] + 1e-9)], dim=-1)

        camera_center = jittor.zeros(batch_size, 2)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=jittor.init.eye(3).unsqueeze(0).expand(batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length,
                                                   camera_center=camera_center)

        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

        loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d)

        loss_keypoints_2d = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                               self.options.openpose_train_weight,
                                               self.options.gt_train_weight)

        loss_shape = self.shape_loss(pred_vertices, gt_vertices, has_smpl)
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl)

        # loss = loss_keypoints_2d + loss_keypoints_3d + loss_shape + loss_regr_pose + loss_regr_betas
        loss = self.options.shape_loss_weight * loss_shape + \
               self.options.keypoint_loss_weight * loss_keypoints_3d + \
               self.options.keypoint_loss_weight * loss_keypoints_2d + \
               self.options.pose_loss_weight * loss_regr_pose + \
               self.options.beta_loss_weight * loss_regr_betas + \
               ((jittor.exp(-pred_camera[:, 0] * 10)) ** 2).mean()

        loss *= 60
        losses = {'loss': loss.detach().item(),
                  'loss_keypoints_2d': loss_keypoints_2d.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                  'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regr_betas': loss_regr_betas.detach().item(),
                  'loss_shape': loss_shape.detach().item()}

        print(losses)

        self.optimizer.zero_grad()
        # self.optimizer.backward(loss)
        self.optimizer.step(loss)

        output = {'pred_vertices': pred_vertices.detach(),
                  'pred_cam_t': pred_cam_t.detach()
                  }

        return output, losses

    def test(self):
        self.model.eval()
        self.eval_dataset_3dpw = BaseDataset(None, '3dpw', is_train=False, )
        mpjpe_3dpw, pa_mpjpe_3dpw, pve_3dpw = run_evaluation(self.model, '3dpw', self.eval_dataset_3dpw,
                                                             result_file=None,
                                                             batch_size=self.options.batch_size,
                                                             shuffle=False,
                                                             log_freq=8)
        results = {
            'mpjpe': mpjpe_3dpw,
            'pa_mpjpe': pa_mpjpe_3dpw,
            'pve': pve_3dpw
        }
        return results

    def train_summaries(self, input_batch, output, losses):

        # images = input_batch['img']
        # pred_vertices = output['pred_vertices']
        # opt_vertices = output['opt_vertices']
        # pred_cam_t = output['pred_cam_t']
        # opt_cam_t = output['opt_cam_t']
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
