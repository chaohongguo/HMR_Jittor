import jittor
import jittor.nn as nn
import numpy as np
import math

from utils.geometry import rot6d_to_rotmat
from models.backbones.resnet import ResNet
from .pos_enc import PositionalEncoding


class MutilROI_resnet50(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone"""

    def __init__(self, smpl_mean_params, img_feat_num=2048,
                 n_extra_views=4, is_fuse=True, is_pos_enc=True,
                 ):
        super(MutilROI_resnet50, self).__init__()

        npose = 24 * 6
        nshape = 10
        ncam = 3
        nbbox = 3

        self.encoder = ResNet(layers=[3, 4, 6, 3])
        self.input_dim = 3
        self.n_view = n_extra_views
        self.is_fuse = is_fuse
        self.is_pos_enc = is_pos_enc
        self.pos_enc = PositionalEncoding(input_dim=self.input_dim, is_pos_enc=self.is_pos_enc)
        # 512 * 4 + 64 * 3
        self.fuse_fc = nn.Linear(512 * 4 + self.pos_enc.output_dim, 256)
        self.attention = nn.Sequential(
            nn.Linear(256 * (self.n_view + 1), 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.n_view + 1),
            nn.Tanh()
        )
        self.proj_head = nn.Sequential(
            nn.Linear(512 * 4, 512 * 4),
            nn.BatchNorm1d(512 * 4),
            # nn.ReLU(True)
            nn.ReLU(),
            nn.Linear(512 * 4, 512 * 4, bias=False),
            nn.BatchNorm1d(512 * 4),
        )
        self.softmax = nn.Softmax(dim=-1)

        fc2_feat_num = 1024
        final_feat_num = fc2_feat_num
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(512 * 4 + nbbox * (self.n_view + 1) + npose + nshape + ncam, 2048)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout()

        # decoder head
        self.decpose = nn.Linear(final_feat_num, npose)
        self.decshape = nn.Linear(final_feat_num, nshape)
        self.deccam = nn.Linear(final_feat_num, ncam)

        self.init_weights()
        # load smpl
        #
        mean_params = np.load(smpl_mean_params)
        self.init_pose = jittor.unsqueeze(jittor.float32(mean_params['pose'][:]), 0)  # [1,144,]
        self.init_shape = jittor.unsqueeze(jittor.float32(mean_params['shape'])[:], 0)  # [10,]
        self.init_cam = jittor.unsqueeze(jittor.float32(mean_params['cam']), 0)  # [3,]

    def init_weights(self):
        # Xavier 初始化 (对应 nn.init.xavier_uniform_)
        gain = 0.01
        jittor.init.xavier_uniform_(self.decpose.weight, gain=gain)
        jittor.init.xavier_uniform_(self.decshape.weight, gain=gain)
        jittor.init.xavier_uniform_(self.deccam.weight, gain=gain)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.weight.shape[0] * m.weight.shape[2] * m.weight.shape[3]
                jittor.init.gauss_(m.weight, 0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm):
                m.weight = jittor.ones_like(m.weight)
                m.bias = jittor.zeros_like(m.bias)

    def execute(self, x_all, bbox_all, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        """
        Args:
            x_all:[B,n_views*3,h,w]
                images from different view
            bbox_all:[B,n_views*3]

            init_pose:
            init_shape:
            init_cam:
            n_iter:

        Returns:

        """
        batch_size, _, h, w = x_all.shape  # [B,5*c,h,w]
        n_views = self.n_view + 1

        if init_pose is None:
            init_pose = jittor.misc.expand(self.init_pose, [n_views * batch_size, -1])  # [144,] =>
        if init_shape is None:
            init_shape = jittor.misc.expand(self.init_shape, [n_views * batch_size, -1])  # [batch, 10]
        if init_cam is None:
            init_cam = jittor.misc.expand(self.init_cam, [n_views * batch_size, -1])  # [batch, 3]

        x = x_all.view(-1, 3, h, w)  # [B*n_views,3,h,w]
        xf = self.encoder(x)  # [B*n_views,2048]
        xf = xf.view(batch_size, n_views, -1)  # [B, 5, 2048]

        xf_ = self.relu(xf.view(n_views * batch_size, -1))  # [B*n_views, 2048]
        alpha = self.sigmoid(xf_)
        # [B*5,2048] * [B*5,2048]
        xf_hidden = jittor.multiply(xf.view(n_views * batch_size, -1), alpha)  # [B*n_views, 2048]
        # global features
        xf_g = self.proj_head(xf_hidden)  # [B*n_views, 2048]

        bbox_all = bbox_all.view(batch_size, n_views, 3)  # [B, 5, 3]
        if self.is_fuse:
            # [0,0,0,0,0,1,1,1,1......]
            extra_index = jittor.arange(n_views).unsqueeze(-1).repeat(1, n_views).view(-1, )
            # [0,1,2,3,4]
            main_index = jittor.arange(n_views).unsqueeze(0).repeat(n_views, 1).view(-1, )
            # print(extra_inds, main_inds)
            bbox_trans = bbox_all[:, extra_index, :3] - bbox_all[:, main_index, :3]  # [B, 25, 3]
            if self.is_pos_enc:
                # using pos_emb
                bbox_trans_emb = self.pos_enc(bbox_trans.view(-1, self.input_dim))  # [B*25,64+3]
                xf = xf.repeat(1, n_views, 1).view(n_views * batch_size, n_views, -1)  # (5*B, 5, 2048)
                xf_cat = jittor.concat([xf, bbox_trans_emb.view(n_views * batch_size, n_views, -1)],
                                       -1)  # (5*B, 5, 2048+[32*2*3+3])

            else:
                xf = xf.repeat(1, n_views, 1).view(n_views * batch_size, n_views, -1)  # (5*B, 5, 2048)
                xf_cat = xf  # (5*B, 5, 2048)
            xf_attention = self.fuse_fc(xf_cat).view(n_views * batch_size, -1)  # [B*5,256*5=]
            # print('xf_cat', xf_attention.shape)
            xf_attention = self.attention(xf_attention)  # [B*5,5]
            xf_attention = self.softmax(xf_attention)
            # print('attention', xf_attention.shape)
            # [B*5,5,2048] * [b*5,5,1]
            xf_out = jittor.mul(xf, xf_attention[:, :, None])  # [B*5,5,2048]
            # print(xf_out.shape)
            xf_out = jittor.sum(xf_out, dim=1)  # [B*5,2048]
        else:
            # print("Not using fusion module !!")
            xf_out = xf.view(n_views * batch_size, -1)  # [n_views*B,2048]
        pred_pose = init_pose
        pred_shape = init_shape

        pred_cam = init_cam
        bbox_all = jittor.float32(bbox_all)
        # print(xf_out.shape)
        # print(bbox_all.shape)
        # print(bbox_all.repeat(1, n_views, 1).view(n_views * batch_size, -1).shape)
        for i in range(n_iter):
            # [B*5,2048]+[B*5,5*3]+[B*5,72]+[B*5,10]+[B*5,3]
            xc = jittor.concat(
                [xf_out, bbox_all.repeat(1, n_views, 1).view(n_views * batch_size, -1), pred_pose, pred_shape,
                 pred_cam], 1)  # [B*5,2048+15+72+13]

            xc = self.fc1(xc)  # [2048+15+72+13,2048]
            xc = self.drop1(xc)
            xc = self.fc2(xc)  # [1024,1024]
            xc = self.drop2(xc)

            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(n_views * batch_size, 24, 3, 3)
        return pred_rotmat, pred_shape, pred_cam, xf_g



