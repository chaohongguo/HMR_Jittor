import jittor
from models import hmr, SMPL, build_model
from utils import BaseTrainer
import config
import constants
from datasets import BaseDataset_cliff as BaseDataset, MixedDataset
from utils.geometry import perspective_projection, estimate_translation, cam_crop2full
from eval import run_evaluation
from .loss import SetCriterion

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

        self.eval_ds = BaseDataset(self.options, self.options.eval_dataset, is_train=False,
                                   bbox_type=self.options.bbox_type)

        self.model = build_model(config.SMPL_MEAN_PARAMS, pretrained=True,
                                 backbone=self.options.backbone, model_name=self.options.model_name)
        self.optimizer = jittor.optim.Adam(params=self.model.parameters(),
                                           lr=self.options.lr)
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False)

        self.joints_idx = 25
        self.joints_num = 49

        weight_dict = {
            "loss_keypoint_2d": self.options.keypoint_loss_weight,
            "loss_keypoint_3d": self.options.keypoint_loss_weight,
            "loss_shape": self.options.shape_loss_weight,
            "loss_pose": self.options.pose_loss_weight,
            "loss_beta": self.options.beta_loss_weight
        }
        self.criterion = SetCriterion(weight_dict=weight_dict, model_name="cliff")
        # self.criterion_keypoints = nn.MSELoss(reduction='none')
        # self.criterion_shape = nn.L1Loss()
        # self.criterion_regr = nn.MSELoss()
        # self.focal_length = constants.FOCAL_LENGTH
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}

        self.bbox_type = self.options.bbox_type
        if self.bbox_type == 'square':
            print("==================training using square bbox==================")
            self.crop_w = constants.IMG_RES
            self.crop_h = constants.IMG_RES
            self.bbox_size = constants.IMG_RES
        elif self.bbox_type == 'rect':
            print("==================training rect bbox==================")
            self.crop_w = constants.IMG_W
            self.crop_h = constants.IMG_H
            self.bbox_size = constants.IMG_H

        # self.loss = nn.L1Loss
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

    def train_step(self, input_batch, cur_step):
        self.model.train()

        # img base info
        img_names = input_batch['img_name']
        dataset = input_batch['dataset_name']
        batch_size = len(img_names)
        img_h, img_w = input_batch['img_h'].view(-1, 1), input_batch['img_w'].view(-1, 1)  # [B,1]

        images = input_batch['img']
        # crop bbox 2D gt keypoint
        gt_keypoints_2d = input_batch['keypoints'][:, :self.joints_num]  # [B,49,3]
        # origin full 2d gt keypoint
        gt_keypoints_2d_full = input_batch['keypoints_full']  # [B,49,3]
        gt_joints = input_batch['pose_3d']  # 3D pose:[B,24,4]

        gt_pose = input_batch['pose']  # [B,24*3]
        gt_betas = input_batch['betas']  # SMPL beta parameters

        # 3d flag
        has_smpl = input_batch['has_smpl']  # flag that indicates whether SMPL parameters are valid [B,]
        has_pose_3d = input_batch['has_pose_3d']  # flag that indicates whether 3D pose is valid [B,]

        # data augmentation info
        is_flipped = input_batch['is_flipped']  # flag that indicates whether image was flipped during data augmentation
        crop_trans = input_batch['crop_trans']
        rot_angle = input_batch['rot_angle']  # rotation angle used for data augmentation

        # indices = input_batch['sample_index']  # index of example inside its dataset

        bbox_info = input_batch['bbox_info']  # [B,3]
        center, scale, focal_length = input_batch['center'], input_batch['scale'].squeeze(), input_batch['focal_length'].squeeze()
        # [B,10] [B,72-3] [B,3] pose is aa format, pose2rot is True
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # De-normalize 2D keypoints from [-1,1] to crop pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        # TODO  origin_1
        # gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)
        # TODO  target_1
        gt_keypoints_2d_orig[:, :, 0] = 0.5 * self.crop_w * (gt_keypoints_2d_orig[:, :, 0] + 1)
        gt_keypoints_2d_orig[:, :, 1] = 0.5 * self.crop_h * (gt_keypoints_2d_orig[:, :, 1] + 1)

        # gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length,
        #                                 img_size=self.options.img_res)
        #
        pred_rotmat, pred_betas, pred_camera = self.model(images, bbox_info)

        # [B,10] [B,24,3,3]  [B,1,3] pose is rot matrix
        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        # print(temp.shape)
        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_crop = jittor.stack([pred_camera[:, 1],
                                      pred_camera[:, 2],
                                      2 * focal_length / (self.bbox_size * pred_camera[:, 0] + 1e-9)], dim=-1)  # [B,3]

        full_img_shape = jittor.concat((img_h, img_w), dim=1)  # [B,2]

        pred_cam_full = cam_crop2full(pred_camera, center, scale,
                                      full_img_shape=full_img_shape, focal_length=focal_length)
        # print(pred_cam_full.shape)

        camera_center = jittor.concat((img_w, img_h), dim=1) / 2
        pred_keypoints_2d_crop = perspective_projection(pred_joints,
                                                        rotation=jittor.init.eye(3).unsqueeze(0).expand(batch_size, -1,
                                                                                                        -1),
                                                        translation=pred_cam_crop,
                                                        focal_length=focal_length,
                                                        camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d_crop = pred_keypoints_2d_crop / (self.bbox_size / 2.)
        pred_keypoint_2d_full = perspective_projection(pred_joints,
                                                       rotation=jittor.init.eye(3).unsqueeze(0).expand(batch_size, -1,
                                                                                                       -1),
                                                       translation=pred_cam_full,
                                                       focal_length=focal_length,
                                                       camera_center=camera_center)  # [B,49,2]
        # print(pred_keypoint_2d_full)
        pred_keypoint_2d_full_with_conf = jittor.concat((pred_keypoint_2d_full, jittor.ones(batch_size, 49, 1)), dim=2)
        # trans @ pred_keypoint_2d_full_homo
        pred_keypoint_2d_bbox = jittor.linalg.einsum('bij,bkj->bki', crop_trans, pred_keypoint_2d_full_with_conf)
        # # TODO visual predict_full,gt,predict_crop
        # if self.options.viz_debug:
        #     index = dataset.index('3dpw')
        #     print(f'visualizing 3dpw {index}')
        #     images = images * jittor.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        #     images = images + jittor.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        #     cropped_img = (images[index].permute((1, 2, 0)).numpy()[..., ::-1] * 255.0).astype('uint8').copy()
        #     full_img = cv2.imread(img_names[index])
        #     # visualizing gt keypoint in full image
        #     for kp in gt_keypoints_2d_full[index][25:]:
        #         cv2.circle(full_img, (int(kp[0]), int(kp[1])), radius=3, color=(0, 0, 255), thickness=5)
        #     MAR = cv2.getRotationMatrix2D((int(center[index][0]), int(center[index][1])), int(rot_angle[index]), 1.0)
        #     rotated_img = cv2.warpAffine(full_img.copy(), MAR, (int(img_w[index][0]), int(img_h[index][0])))
        #     # visualizing predict keypoint in full image
        #     for kp in pred_keypoint_2d_full[index][self.joints_idx:]:
        #         cv2.circle(full_img, (int(kp[0]), int(kp[1])), radius=3, color=(0, 255, 0), thickness=-1)
        #
        #     for kp in gt_keypoints_2d[index][self.joints_idx:]:
        #         cv2.circle(cropped_img, (int(kp[0]), int(kp[1])), radius=3, color=(0, 0, 255), thickness=-1)
        #     for kp in pred_keypoint_2d_bbox[index][self.joints_idx:]:
        #         cv2.circle(cropped_img, (int(kp[0]), int(kp[1])), radius=3, color=(0, 255, 0), thickness=-1)
        #     os.makedirs('eval_result/img_viz/', exist_ok=True)
        #     cv2.imwrite(f'eval_result/img_viz/train_img_crop_visualized_{img_names[index]}.jpg', np.ascontiguousarray(cropped_img))
        #     cv2.imwrite(f'eval_result/img_viz/train_img_full_visualized_{img_names[index]}.jpg', np.ascontiguousarray(full_img))
        #     print("=====================visual success================")

        predict = {
            "pred_keypoint_2d": pred_keypoints_2d_crop,
            "pred_rotmat": pred_rotmat,
            "pred_betas": pred_betas,
            "pred_camera": pred_camera,
            "pred_output": pred_output,
            "pred_keypoint_2d_bbox": pred_keypoint_2d_bbox
        }
        gt = {
            "gt_keypoint_2d": gt_keypoints_2d,
            "gt_pose": gt_pose,
            "gt_betas": gt_betas,
            "gt_joints": gt_joints,
            # "gt_camera": gt_camera,
            "gt_output": gt_out,
        }
        const = {
            "openpose_train_weight": self.options.openpose_train_weight,
            "gt_train_weight": self.options.gt_train_weight,
            "has_pose_3d": has_pose_3d,
            "has_smpl": has_smpl,
            "crop_w": self.crop_w,
            "crop_h": self.crop_h
        }
        loss, losses = self.criterion(predict, gt, const)
        loss *= 60
        losses = {'loss': losses['loss'].detach().item(),
                  'loss_keypoints_2d': losses['loss_keypoints_2d'].detach().item(),
                  'loss_keypoints_3d': losses['loss_keypoints_3d'].detach().item(),
                  'loss_regr_pose': losses['loss_regr_pose'].detach().item(),
                  'loss_regr_betas': losses['loss_regr_betas'].detach().item(),
                  'loss_shape': losses['loss_shape'].detach().item(),
                  }
        if cur_step % 10 == 0:
            print(losses)

        self.optimizer.zero_grad()
        # self.optimizer.backward(loss)
        self.optimizer.step(loss)

        output = {
            'pred_vertices': pred_vertices.detach(),
            'pred_cam_t': pred_cam_crop.detach()
        }

        return output, losses

    def test(self):
        self.model.eval()

        mpjpe_3dpw, pa_mpjpe_3dpw, pve_3dpw = run_evaluation(self.model, "cliff", '3dpw', self.eval_ds,
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
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
