import jittor
import jittor.nn as nn
import numpy as np
import math

from utils.geometry import rot6d_to_rotmat
from models.backbones.resnet import ResNet


class CLIFF_resnet50(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone"""

    def __init__(self, smpl_mean_params, img_feat_num=2048):
        super(CLIFF_resnet50, self).__init__()

        self.encoder = ResNet(layers=[3, 4, 6, 3])

        npose = 24 * 6
        nshape = 10
        ncam = 3
        nbbox = 3

        fc2_feat_num = 1024
        final_feat_num = fc2_feat_num

        self.fc1 = nn.Linear(512 * 4 + nbbox + npose + nshape + ncam, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        # decoder head
        self.decpose = nn.Linear(final_feat_num, npose)
        self.decshape = nn.Linear(final_feat_num, nshape)
        self.deccam = nn.Linear(final_feat_num, ncam)

        self.init_weights()
        # load smpl
        mean_params = np.load(smpl_mean_params)
        self.init_pose = jittor.unsqueeze(jittor.float32(mean_params['pose'][:]), 0)  # [B,1,144]
        self.init_shape = jittor.unsqueeze(jittor.float32(mean_params['shape'])[:], 0)  # [B,10]
        self.init_cam = jittor.unsqueeze(jittor.float32(mean_params['cam']), 0)  # [B,3]

    def init_weights(self):
        # Xavier 初始化 (对应 nn.init.xavier_uniform_)
        gain = 0.01
        jittor.init.xavier_uniform_(self.decpose.weight, gain=gain)
        jittor.init.xavier_uniform_(self.decshape.weight, gain=gain)
        jittor.init.xavier_uniform_(self.deccam.weight, gain=gain)

        for m in self.modules():
            if isinstance(m, nn.Conv):
                n = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                std = math.sqrt(2. / n)
                m.weight = jittor.init.gauss_(m.weight, mean=0, std=std)

            elif isinstance(m, nn.BatchNorm):
                m.weight = jittor.ones_like(m.weight)
                m.bias = jittor.zeros_like(m.bias)

    def execute(self, x, bbox, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = jittor.misc.expand(self.init_pose, [batch_size, -1])  # [batch, 144]
        if init_shape is None:
            init_shape = jittor.misc.expand(self.init_shape, [batch_size, -1])  # [batch, 10]
        if init_cam is None:
            init_cam = jittor.misc.expand(self.init_cam, [batch_size, -1])  # [batch, 3]

        xf = self.encoder(x)

        pred_pose = init_pose
        pred_shape = init_shape

        pred_cam = init_cam
        bbox = jittor.float32(bbox)
        for i in range(n_iter):
            xc = jittor.concat([xf, bbox, pred_pose, pred_shape, pred_cam], 1)

            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)

            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        return pred_rotmat, pred_shape, pred_cam