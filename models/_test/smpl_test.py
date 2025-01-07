import jittor

import config
from models.smpl import SMPL
import numpy as np

smpl = SMPL(config.SMPL_MODEL_DIR,
            batch_size=20,
            create_transl=False)

data = np.load("/home/lab345/mnt4T/__gcode_/HMR/mutil_roi/predict.npz")

pred_rotmat = data['pred_rotmat']
pred_betas = data['pred_betas']
pred_rotmat = jittor.array(pred_rotmat)
pred_betas = jittor.array(pred_betas)

pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                   global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)

print(pred_output)
