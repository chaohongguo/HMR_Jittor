import jittor
import numpy as np
import cv2
import os
import argparse
import config
from models import SMPL, hmr, build_model
from models.metaHMR.maml import gradient_update_parameters
from models.cliff.cliff_resnet50 import CLIFF_resnet50
from utils import CheckpointDataLoader
from datasets import BaseDataset_MutilROI, BaseDataset_cliff, BaseDataset_hmr
from tqdm import tqdm
import constants
from utils.pose_utils import reconstruction_error
from copy import deepcopy
from collections import OrderedDict
from utils.geometry import rot6d_to_rotmat, perspective_projection, cam_crop2full
from train.loss import SetCriterion

jittor.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',
                    default="data/best_ckpt/metaHMR_cliff_resnet_60.9_37.8.pkl",
                    help='Path to network checkpoint')
parser.add_argument('--dataset', default='3dpw', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp'],
                    help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=16, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
parser.add_argument('--model_name', default="mutilROI", help='If set, save detections to a .npz file')
parser.add_argument('--n_views', type=int, default=5, help='Views to use')
parser.add_argument('--is_pos_enc', default=True, action="store_true", help='using relative encodings')
parser.add_argument('--is_fuse', default=True, action="store_true", help='using fusion module')
parser.add_argument('--use_extraviews', default=True, action='store_true',
                    help='Use parallel stages for regression')
parser.add_argument('--shift_center', default=True, action='store_true', help='Shuffle data')
parser.add_argument('--rescale_bbx', default=True, action='store_true', help='Shuffle data')
parser.add_argument('--noise_factor', type=float, default=0.4,
                    help='Randomly multiply pixel values with factor in the range [1-noise_factor, 1+noise_factor]')
parser.add_argument('--rot_factor', type=float, default=30,
                    help='Random rotation in the range [-rot_factor, rot_factor]')
parser.add_argument('--scale_factor', type=float, default=0.25,
                    help='Rescale bounding boxes by a factor of [1-scale_factor,1+scale_factor]')
parser.add_argument('--bbox_type', default='rect',
                    help='Use square bounding boxes in 224x224 or rectangle ones in 256x192')

parser.add_argument('--inner_lr', type=float, default=1e-5, help='the inner lr')
parser.add_argument('--no_use_adam', action='store_true', help='not use_adam after 1 inner step')
parser.add_argument('--use_aug_trans', default=False, action="store_true", help='flag for using augment')
parser.add_argument('--use_aug_img', default=False, action="store_true", help='flag for using augment')


def write_params(model, param_dict):
    for name, param in model.meta_named_parameters():
        param.data = param_dict[name].data.copy()


def write_params_(model, param_dict):
    for name, param in model.meta_named_parameters():
        if isinstance(param_dict[name], jittor.Var):
            param.data.copy_(param_dict[name].data)  # 使用 .data 来访问 Var 中的数据
        else:
            param.data.copy_(param_dict[name])  # 如果 param_dict[name] 不是 Var 类型，可以直接拷贝


def run_evaluation(model, model_name, dataset_name, dataset, result_file,
                   batch_size=32, img_res=224,
                   num_workers=32, shuffle=False, log_freq=8):
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False)
    smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False)
    smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False)
    save_results = result_file is not None

    J_regressor = jittor.array(np.load(config.JOINT_REGRESSOR_H36M), dtype=jittor.float32)
    data_loader = CheckpointDataLoader(dataset=dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers)

    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    pve = np.zeros(len(dataset))

    eval_pose = False
    eval_masks = False
    eval_parts = False

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp':
        eval_pose = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = config.DATASET_FOLDERS['upi-s1h']

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14

    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader) // batch_size)):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose']
        gt_betas = batch['betas']
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        images = batch['img']
        gender = batch['gender']
        bbox = batch['bbox_info']
        curr_batch_size = images.shape[0]

        with jittor.no_grad():
            if model_name == 'cliff':
                pred_rotmat, pred_betas, pred_camera = model(images, bbox)
            elif model_name == 'hmr':
                pred_rotmat, pred_betas, pred_camera = model(images)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                       global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        if eval_pose:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
                gt_keypoints_3d = batch['pose_3d']
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                gt_vertices = smpl_male(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], betas=gt_betas).vertices
                gt_vertices_female = smpl_female(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                                 betas=gt_betas).vertices
                gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]
                gt_keypoints_3d = jittor.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = jittor.matmul(J_regressor_batch, pred_vertices)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :,
                :] = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            # Absolute error (MPJPE)
            error = jittor.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).numpy()
            # 32 * step: 32 * step + 32
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.numpy(),
                                           reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            # pve
            per_vertex_error = jittor.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(
                dim=-1).numpy()
            pve[step * batch_size:step * batch_size + curr_batch_size] = per_vertex_error

            # Print intermediate results during evaluation
            if step % log_freq == log_freq - 1:
                if eval_pose:
                    print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                    print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                    print('PVE: ' + str(1000 * pve[:step * batch_size].mean()))
                    print()

    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
        # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_pose:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print('PVE: ' + str(1000 * pve.mean()))
        print()
    return 1000 * mpjpe.mean(), 1000 * recon_err.mean(), 1000 * pve.mean()


def run_evaluation_mutilROI(model, model_name, dataset_name, dataset, result_file,
                            batch_size=32, img_res=224,
                            num_workers=32, shuffle=False, log_freq=8,
                            use_extra=True, use_fuse=True, n_views=5, ):
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False)
    smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False)
    smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False)
    save_results = result_file is not None

    J_regressor = jittor.array(np.load(config.JOINT_REGRESSOR_H36M), dtype=jittor.float32)
    data_loader = CheckpointDataLoader(dataset=dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers)

    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    pve = np.zeros(len(dataset))

    eval_pose = False
    eval_masks = False
    eval_parts = False

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp':
        eval_pose = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = config.DATASET_FOLDERS['upi-s1h']

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14

    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader) // batch_size)):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose']
        gt_betas = batch['betas']
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        images = batch['img']  # [b,3,h,w]
        gender = batch['gender']
        bbox_info = batch['bbox_info']  # [b,3]
        curr_batch_size = images.shape[0]
        if use_extra:
            bboxs_extras = batch['bbox_extras']  # [b,(v-1)*3]
            img_extras = batch['img_extras']  # [b,(v-1)*3,h,w]
            images = jittor.concat([images, img_extras], 1)  # [b,v*3,h,w]
            bbox_info = jittor.concat([bbox_info, bboxs_extras], 1)  # [b,v*3]

        index = 0
        with jittor.no_grad():
            if model_name == 'cliff':
                pred_rotmat, pred_betas, pred_camera = model(images, bbox_info)
            elif model_name == 'hmr':
                pred_rotmat, pred_betas, pred_camera = model(images)
            elif model_name == 'mutilROI':
                pred_rotmat, pred_betas, pred_camera, _1 = model(images, bbox_info)
                if use_fuse:
                    pred_rotmat = pred_rotmat.view(curr_batch_size, n_views, 24, 3, 3)[:, index]
                    pred_betas = pred_betas.view(curr_batch_size, n_views, -1)[:, index]
                    pred_camera = pred_camera.view(curr_batch_size, n_views, -1)[:, index]
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                       global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        if eval_pose:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
                gt_keypoints_3d = batch['pose_3d']
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                gt_vertices = smpl_male(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], betas=gt_betas).vertices
                gt_vertices_female = smpl_female(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                                 betas=gt_betas).vertices
                gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]
                gt_keypoints_3d = jittor.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = jittor.matmul(J_regressor_batch, pred_vertices)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :,
                :] = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            # Absolute error (MPJPE)
            error = jittor.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).numpy()
            # 32 * step: 32 * step + 32
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.numpy(),
                                           reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            # pve
            per_vertex_error = jittor.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(
                dim=-1).numpy()
            pve[step * batch_size:step * batch_size + curr_batch_size] = per_vertex_error

            # Print intermediate results during evaluation
            if step % log_freq == log_freq - 1:
                if eval_pose:
                    print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                    print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                    print('PVE: ' + str(1000 * pve[:step * batch_size].mean()))
                    print()

    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
        # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_pose:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print('PVE: ' + str(1000 * pve.mean()))
        print()
    return 1000 * mpjpe.mean(), 1000 * recon_err.mean(), 1000 * pve.mean()


def run_evaluation_metaHMR(model_main, model_aux, dataset_name, dataset, result_file, criterion, crop_w, crop_h,
                           batch_size=50, img_res=224,
                           num_workers=8, shuffle=False, log_freq=50,
                           with_train=False, eval_epoch=None, summary_writer=None, viz=False, bbox_type='rect', params=None,
                           options=None):

    smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False)
    smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False)
    smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False)
    J_regressor = jittor.array(np.load(config.JOINT_REGRESSOR_H36M), dtype=jittor.float32)
    data_loader = CheckpointDataLoader(dataset=dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers)

    save_results = result_file is not None
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    pve = np.zeros(len(dataset))

    eval_pose = False
    eval_masks = False
    eval_parts = False

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp':
        eval_pose = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = config.DATASET_FOLDERS['upi-s1h']

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14

    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader) // batch_size)):
        model_copy = deepcopy(model_main)
        params_test = OrderedDict(model_copy.meta_named_parameters())
        # Get ground truth annotations from the batch
        gt_pose = batch['pose']
        gt_betas = batch['betas']
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        images = batch['img']  # [b,3,h,w]
        gender = batch['gender']
        bbox_info = batch['bbox_info']  # [b,3]
        curr_batch_size = images.shape[0]
        img_h = batch['img_h']
        img_w = batch['img_w']
        crop_trans = batch['crop_trans']
        has_smpl = batch['has_smpl']
        has_pose_3d = batch['has_pose_3d']
        center = batch['center']
        scale = batch['scale'].squeeze()
        focal_length = batch['focal_length'].squeeze()
        gt_keypoints_2d = batch['keypoints']
    #
    #     inner_lr = options.inner_lr
    #     if not options.no_use_adam:
    #         optimAdam = jittor.optim.Adam(params=model_copy.parameters(), lr=options.after_innerlr, betas=[0.9, 0.999])
    #     img_h, img_w = img_h.view(-1, 1), img_w.view(-1, 1)
    #     full_img_shape = jittor.concat((img_h, img_w), dim=1)  # [B,2]
    #     camera_center = jittor.concat((img_w, img_h), dim=1) / 2  # [B,2]
    #
    #     # mutil-step optim
    #     for in_step in range(options.test_val_step + options.inner_step):
    #         pred_rotmat_inner, pred_betas_inner, pred_camera_inner = model_copy(images, bbox_info, params=params_test)
    #         # data = np.load('eval_predict_meta.npz')
    #         #
    #         # pred_rotmat_inner = data['pred_rotmat_inner']
    #         # pred_betas_inner = data['pred_betas_inner']
    #         # pred_camera_inner = data['pred_camera_inner']
    #         #
    #         # pred_rotmat_inner = jittor.array(pred_rotmat_inner)
    #         # pred_betas_inner = jittor.array(pred_betas_inner)
    #         # pred_camera_inner = jittor.array(pred_camera_inner)
    #         pred_output_inner = smpl_neutral(betas=pred_betas_inner, body_pose=pred_rotmat_inner[:, 1:],
    #                                          global_orient=pred_rotmat_inner[:, 0].unsqueeze(1), pose2rot=False)
    #         pred_vertices_inner = pred_output_inner.vertices
    #         pred_joints_inner = pred_output_inner.joints
    #
    #         pred_cam_full_inner = cam_crop2full(pred_camera_inner, center, scale, full_img_shape, focal_length)
    #         # print(pred_cam_full_inner.shape)
    #         pred_keypoints2d_full_inner = perspective_projection(
    #             pred_joints_inner,
    #             rotation=jittor.float32(jittor.init.eye(3).unsqueeze(0).expand(batch_size, -1, -1)),
    #             translation=pred_cam_full_inner,
    #             focal_length=focal_length,
    #             camera_center=camera_center)
    #         pred_keypoints2d_with_conf_inner = jittor.concat(
    #             (pred_keypoints2d_full_inner, jittor.ones(batch_size, 49, 1)), dim=2)
    #         pred_keypoints2d_bbox_inner = jittor.linalg.einsum('bij,bkj->bki', crop_trans,
    #                                                            pred_keypoints2d_with_conf_inner)
    #
    #         const_inner = {
    #             "openpose_train_weight": options.openpose_train_weight,
    #             "gt_train_weight": options.gt_train_weight,
    #             "has_pose_3d": has_pose_3d,
    #             "has_smpl": has_smpl,
    #             "crop_w": crop_w,
    #             "crop_h": crop_h,
    #             "center": center,
    #             "scale": scale,
    #             "full_img_shape": full_img_shape
    #         }
    #
    #         if in_step < options.inner_step:
    #             pred_rotmat_aux, pred_betas_aux, pred_camera_aux = model_aux(images, bbox_info)
    #             pred_output_aux = smpl_neutral(betas=pred_betas_aux, body_pose=pred_rotmat_aux[:, 1:],
    #                                            global_orient=pred_rotmat_aux[:, 0].unsqueeze(1), pose2rot=False)
    #             pred_vertices_aux = pred_output_aux.vertices
    #             pred_joints_aux = pred_output_aux.joints
    #
    #             predict_inner = {
    #                 "pred_rotmat": pred_rotmat_inner,
    #                 "pred_betas": pred_betas_inner,
    #                 # "pred_camera": pred_camera,
    #                 "pred_output": pred_output_inner,
    #                 "pred_keypoint_2d_bbox": pred_keypoints2d_bbox_inner
    #             }
    #             gt_2d = deepcopy(gt_keypoints_2d)
    #             gt_inner = {
    #                 "gt_keypoint_2d": gt_2d,
    #                 "gt_pose": pred_rotmat_aux,
    #                 "gt_betas": pred_betas_aux,
    #                 "gt_joints": pred_joints_aux,
    #                 # "gt_camera": gt_camera,
    #                 "gt_output": pred_output_aux,
    #             }
    #
    #             loss, losses = criterion(predict=predict_inner, gt=gt_inner, const=const_inner,
    #                                      using_pseudo=True)
    #             print(f"test optimi inner  loss {losses}")
    #             loss_ = loss
    #             params_test = gradient_update_parameters(model_copy, loss_, params=params_test,
    #                                                      step_size=inner_lr, first_order=True)
    #             # update model param
    #             write_params(model_copy, params_test)  # for multi-step optim
    #
    #         if in_step > (options.inner_step - 1):
    #             inner_lr = options.after_innerlr
    #
    #             predict_inner = {
    #                 "pred_keypoint_2d_bbox": pred_keypoints2d_bbox_inner,
    #             }
    #             gt_2d_ = deepcopy(gt_keypoints_2d)
    #             gt_inner = {
    #                 "gt_keypoint_2d": gt_2d_,
    #             }
    #
    #             loss__, _ = criterion(predict_inner, gt_inner, const_inner, is_eval=True)
    #             print(f"test optimi outer  loss {loss__}")
    #             optimAdam.zero_grad()
    #             # loss.backward()
    #             optimAdam.step(loss__)
    #             params_test = OrderedDict(model_copy.meta_named_parameters())
    #
        with jittor.no_grad():
            pred_rotmat, pred_betas, pred_camera = model_copy(images, bbox_info, params=params_test)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                       global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        if eval_pose:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
                gt_keypoints_3d = batch['pose_3d']
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                gt_vertices = smpl_male(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], betas=gt_betas).vertices
                gt_vertices_female = smpl_female(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                                 betas=gt_betas).vertices
                gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]
                gt_keypoints_3d = jittor.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = jittor.matmul(J_regressor_batch, pred_vertices)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :] = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            # Absolute error (MPJPE)
            error = jittor.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).numpy()
            # 32 * step: 32 * step + 32
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.numpy(),
                                           reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            # pve
            per_vertex_error = jittor.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(
                dim=-1).numpy()
            pve[step * batch_size:step * batch_size + curr_batch_size] = per_vertex_error

            # Print intermediate results during evaluation
            if step % log_freq == log_freq - 1:
                if eval_pose:
                    print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                    print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                    print('PVE: ' + str(1000 * pve[:step * batch_size].mean()))
                    print()

    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
        # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_pose:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print('PVE: ' + str(1000 * pve.mean()))
        print()
    return 1000 * mpjpe.mean(), 1000 * recon_err.mean(), 1000 * pve.mean()


if __name__ == '__main__':
    args = parser.parse_args()
    model = build_model(config.SMPL_MEAN_PARAMS, model_name=args.model_name, option=args, pretrained=False)
    checkpoint = jittor.load(args.checkpoint)
    # resnet_coco_ = jittor.load('models/backbones/pretrained/res50-PA45.7_MJE72.0_MVE85.3_3dpw.pkl')['model']
    # new_dict_ = dict([(k.replace('module.', ''), v) for k, v in resnet_coco_.items()])
    # # new_dict_ = dict([(k.replace('encoder.', ''), v) for k, v in new_dict_.items()])
    # model.load_state_dict(new_dict_)

    model.load_state_dict(checkpoint['model'])
    model.eval()

    if args.model_name == 'mutilROI':
        dataset = BaseDataset_MutilROI(args, args.dataset, is_train=False)
        run_evaluation_mutilROI(model, args.model_name, args.dataset, dataset,
                                result_file=None,
                                batch_size=40,
                                shuffle=False,
                                log_freq=8)
    elif args.model_name == 'metaHMR_main':
        dataset = BaseDataset_cliff(args, args.dataset, is_train=False)
        run_evaluation_metaHMR(model, None, args.dataset, dataset, options=args, result_file=None,
                               crop_h=256, crop_w=192, criterion=None)
    else:
        # Run evaluation
        dataset = BaseDataset_cliff(args, args.dataset, is_train=False)
        run_evaluation(model, args.model_name, args.dataset, dataset, args.result_file,
                       batch_size=args.batch_size,
                       shuffle=args.shuffle,
                       log_freq=args.log_freq)
