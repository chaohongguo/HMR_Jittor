from models.multi_ROI.mutil_resnet50_all import HMR_sim
from models.hmr import build_model
import config
import jittor
import argparse
import numpy as np

jittor.flags.use_cuda = 1
parser = argparse.ArgumentParser()
parser.add_argument('--n_views', type=int, default=5, help='Views to use')

options = parser.parse_args()

# 1. init model
model = build_model(config.SMPL_MEAN_PARAMS, pretrained=False, option=options)
weights = jittor.load('/home/lab345/mnt4T/__gcode_/jittorLearn/logs/A_best_test/previous_62_300_80.4_52.6.pkl')
model.load_state_dict(weights['model'])
# 2. prepare data
data = np.load('/home/lab345/mnt4T/__gcode_/jittorLearn/data/dataLooK/jittor_model_input.npz')
images = data['images']
bbox_info = data['bbox_info']
images = jittor.array(images, dtype=jittor.float32)
bbox_info = jittor.array(bbox_info, dtype=jittor.float32)
# 3. model forward
pred_rotmat, pred_betas, pred_camera, xf_g = model(images, bbox_info)

print("1")