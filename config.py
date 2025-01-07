from os.path import join

# some npz
DATASET_NPZ_PATH = '/home/lab345/mnt4T/__gcode_/jittorLearn/data/dataset_extras'
DATASET_NPZ_PATH_ = '/home/lab345/mnt4T/__gcode_/HMR/SPIN-cuda11_fix/data/dataset_extras'
DATASET_NPZ_PATH__ = '/home/lab345/mnt4T/fmx/transmesh/cliff_metalearn/data/lcz'

DATASET_FILES = [
    # test.npz
    {
        '3dpw': join(DATASET_NPZ_PATH__, '3dpw_test_w2d_smpl3d_gender_openpose.npz'),
    },

    # train.npz
    {
        # 'h36m': join(DATASET_NPZ_PATH, 'h36m_train.npz'),
        'h36m': join(DATASET_NPZ_PATH__, 'h36m_mosh_train_fixname.npz'),
        'lsp-orig': join(DATASET_NPZ_PATH_, 'lsp_dataset_original_train.npz'),
        'mpii': join(DATASET_NPZ_PATH_, 'mpii_train.npz'),
        'coco': join(DATASET_NPZ_PATH__, 'coco_2014_smpl_train.npz'),
        'lspet': join(DATASET_NPZ_PATH_, 'hr-lspet_train.npz'),
        'mpi-inf-3dhp': join(DATASET_NPZ_PATH__, 'mpi_inf_3dhp_train_name_revised.npz'),
        '3dpw': join(DATASET_NPZ_PATH__, '3dpw_train_w2d_smpl3d_gender.npz'),
    }
]

# the position of raw data
PW3D_ROOT = '/home/lab345/mnt4T/__gcode_/_all_data_/hmr/3DPW'
H36M_ROOT = '/home/lab345/mnt4T/fmx/transmesh/datasets/h36m'
LSP_ROOT = '/home/lab345/mnt4T/fmx/transmesh/datasets/lsp_dataset_original'
LSP_ORIGINAL_ROOT = '/home/lab345/mnt4T/fmx/transmesh/datasets/lsp_dataset'
LSPET_ROOT = '/home/lab345/mnt4T/fmx/transmesh/datasets/hr_lspet'
MPII_ROOT = '/home/lab345/mnt4T/fmx/transmesh/datasets/mpii_human_pose_v1'
COCO_ROOT = '/home/lab345/mnt4T/fmx/transmesh/datasets/coco'
MPI_INF_3DHP_ROOT = '/home/lab345/mnt4T/fmx/transmesh/datasets/mpii3d'
AGORA_ROOT = '/home/lab345/mnt4T/fmx/transmesh/datasets/agora'
H36M_ROOT = '/home/lab345/mnt4T/fmx/transmesh/datasets/h36m'

DATASET_FOLDERS = {
    '3dpw': PW3D_ROOT,
    'mpii': MPII_ROOT,
    'coco': COCO_ROOT,
    'lsp-orig': LSP_ORIGINAL_ROOT,
    'lspet': LSPET_ROOT,
    'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,

    'h36m': H36M_ROOT,
    'h36m-p1': H36M_ROOT,
    'h36m-p2': H36M_ROOT,
}

#
SMPL_MEAN_PARAMS = '/home/lab345/mnt4T/__gcode_/jittorLearn/data/smpl_mean_params.npz'
JOINT_REGRESSOR_TRAIN_EXTRA = '/home/lab345/mnt4T/__gcode_/jittorLearn/data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = '/home/lab345/mnt4T/__gcode_/jittorLearn/data/J_regressor_h36m.npy'
SMPL_MEAN_PARAMS = '/home/lab345/mnt4T/__gcode_/jittorLearn/data/smpl_mean_params.npz'
SMPL_MODEL_DIR = '/home/lab345/mnt4T/__gcode_/jittorLearn/data/smpl'
