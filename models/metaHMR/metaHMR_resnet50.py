import jittor

from models.metaHMR.maml import MetaModule, MetaLinear, MetaConv2d, MetaBatchNorm2d
from utils.geometry import rot6d_to_rotmat
import jittor.nn as nn
import math
import numpy as np


class MetaHMR_main(MetaModule):
    def __init__(self, block, layers, smpl_mean_params, bbox_type='rect'):
        super(MetaHMR_main, self).__init__()
        self.inplanes = 64
        npose = 24 * 6
        nbbox = 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if bbox_type == 'square':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        elif bbox_type == 'rect':
            self.avgpool = nn.AvgPool2d((8, 6), stride=1)
        self.fc1 = MetaLinear(512 * 4 + npose + nbbox + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = MetaLinear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = MetaLinear(1024, npose)
        self.decshape = MetaLinear(1024, 10)
        self.deccam = MetaLinear(1024, 3)

        self.init_weights()

        mean_params = np.load(smpl_mean_params)
        self.init_pose = jittor.float32(mean_params['pose'][:]).unsqueeze(0)
        self.init_shape = jittor.float32(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.init_cam = jittor.float32(mean_params['cam']).unsqueeze(0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def init_weights(self):
        # Xavier 初始化 (对应 nn.init.xavier_uniform_)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                jittor.init.gauss_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight = jittor.ones_like(m.weight)
                m.bias = jittor.zeros_like(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                print(m)
                m.weight = jittor.ones_like(m.weight)
                m.bias = jittor.zeros_like(m.bias)
            elif isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                jittor.init.gauss_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, MetaBatchNorm2d):
                m.weight = jittor.ones_like(m.weight)
                m.bias = jittor.zeros_like(m.bias)

    def execute(self, x, bbox_info, init_pose=None, init_shape=None, init_cam=None, n_iter=3, params=None):
        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        for i in range(n_iter):
            xc = jittor.concat([xf, bbox_info, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc, params=self.get_subdict(params, 'fc1'))
            xc = self.drop1(xc)
            xc = self.fc2(xc, params=self.get_subdict(params, 'fc2'))
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc, params=self.get_subdict(params, 'decpose')) + pred_pose
            pred_shape = self.decshape(xc, params=self.get_subdict(params, 'decshape')) + pred_shape
            pred_cam = self.deccam(xc, params=self.get_subdict(params, 'deccam')) + pred_cam
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam


class MetaHMR_aux(MetaModule):
    def __init__(self, block, layers, smpl_mean_params, bbox_type='rect'):
        super(MetaHMR_aux, self).__init__()
        self.inplanes = 64
        npose = 24 * 6
        nbbox = 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if bbox_type == 'square':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        elif bbox_type == 'rect':
            self.avgpool = nn.AvgPool2d((8, 6), stride=1)
        self.fc1 = nn.Linear(512 * 4 + npose + nbbox + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        self.init_weights()
        mean_params = np.load(smpl_mean_params)
        # shape = mean_params['shape']
        self.init_pose = jittor.float32(mean_params['pose'][:]).unsqueeze(0)
        self.init_shape = jittor.float32(mean_params['shape'][:]).unsqueeze(0)
        self.init_cam = jittor.float32(mean_params['cam']).unsqueeze(0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def init_weights(self):
        # Xavier 初始化 (对应 nn.init.xavier_uniform_)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                jittor.init.gauss_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight = jittor.ones_like(m.weight)
                m.bias = jittor.zeros_like(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                print(m)
                m.weight = jittor.ones_like(m.weight)
                m.bias = jittor.zeros_like(m.bias)

    def execute(self, x, bbox_info, init_pose=None, init_shape=None, init_cam=None, n_iter=3, params=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)
        # print(xf)
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        for i in range(n_iter):
            xc = jittor.concat([xf, bbox_info, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            # xc = self.drop1(xc)
            xc = self.fc2(xc)
            # xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam

