import jittor
import jittor as jt
from jittor.dataset import Dataset
import config
import numpy as np
from os.path import join
import cv2
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa, get_affine_transform
import constants
from jittor import Var


def get_bbox_info(img, center, scale, img_shape=None, focal_length=None):
    """

    Args:
        img:
        center:
        scale:
        img_shape:
        focal_length:

    Returns:

    """
    if img_shape is not None:
        img_h = img_shape[0]
        img_w = img_shape[1]
    else:
        img_h, img_w = img.shape[:2]
    if focal_length is None:
        focal_length = estimate_focal_length(img_h, img_w)

    cx, cy = center
    s = scale
    bbox_info = np.stack([cx - img_w / 2., cy - img_h / 2., s * 200.])
    bbox_info[:2] = bbox_info[:2] / focal_length * 2.8
    bbox_info[2] = (bbox_info[2] - 0.24 * focal_length) / (0.06 * focal_length)
    return bbox_info, focal_length


def estimate_focal_length(img_h, img_w):
    return (img_w * img_w + img_h * img_h) ** 0.5


def pose_processing(pose, r, f):
    """Process SMPL theta parameters  and apply all augmentation transforms."""
    # rotation or the pose parameters
    pose[:3] = rot_aa(pose[:3], r)
    # flip the pose parameters
    if f:
        pose = flip_pose(pose)
    # (72),float
    pose = pose.astype('float32')
    return pose


def rgb_processing(rgb_img, center, scale, rot, flip, pn):
    """Process rgb image and do augmentation."""
    rgb_img = crop(rgb_img, center, scale,
                   [constants.IMG_RES, constants.IMG_RES], rot=rot)
    # flip the image
    if flip:
        rgb_img = flip_img(rgb_img)
    # in the rgb image we add pixel noise in a channel-wise manner
    rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
    rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
    rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
    # (3,224,224),float,[0,1]
    rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
    return rgb_img


def j2d_processing(kp, center, scale, r, f):
    """
    Process gt 2D keypoints and apply all augmentation transforms.
    Args:
        kp:[49,3]
        center: [2,]
        scale:
        r:
        f:
    Returns:

    """
    nparts = kp.shape[0]
    for i in range(nparts):
        kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                               [constants.IMG_RES, constants.IMG_RES], rot=r)
    # convert to normalized coordinates [-1,1]
    kp[:, :-1] = 2. * kp[:, :-1] / constants.IMG_RES - 1.
    # flip the x coordinates
    if f:
        kp = flip_kp(kp)
    kp = kp.astype('float32')
    return kp


def j3d_processing(S, r, f):
    """Process gt 3D keypoints and apply all augmentation transforms."""
    # in-plane rotation
    rot_mat = np.eye(3)
    if not r == 0:
        rot_rad = -r * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
    S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])
    # flip the x coordinates
    if f:
        S = flip_kp(S)
    S = S.astype('float32')
    return S


def bbox_from_keypoint(keypoint_2d, rescale=1.2):
    """
    Get center and scale of bounding box from gt 2d keypoint.
    Args:
        keypoint_2d: [24,3]
        rescale:

    Returns:

    """
    keypoint_valid = keypoint_2d[np.where(keypoint_2d[:, 2] > 0)]  # conf >0
    if len(np.where(keypoint_2d[:, 2] > 0)[0]) == 0:
        print(keypoint_2d)
    # print(np.where(keypoints[:, 2]>1), keypoints_valid)

    bbox = [min(keypoint_valid[:, 0]), min(keypoint_valid[:, 1]),  # left top [x_min,y_min]
            max(keypoint_valid[:, 0]), max(keypoint_valid[:, 1])]  # right bottom [x_max,y_max]

    # center
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    center = np.array([center_x, center_y])

    # scale
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox_size = max(bbox_w * constants.CROP_ASPECT_RATIO, bbox_h)
    scale = bbox_size / 200.0

    # adjust bounding box tightness
    scale *= rescale
    # print(center, scale)
    return center, scale


class BaseDataset(Dataset):
    def __init__(self, options, dataset, ignore_3d=False, is_train=True, use_augmentation=True, bbox_type='square'):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = config.DATASET_FOLDERS[dataset]
        filename = config.DATASET_FILES[is_train][dataset]
        self.data = np.load(filename)
        if self.is_train:
            print(">>Train dataset ", end=' ')
        else:
            print(">>Eval dataset ", end=' ')
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.imgname = self.data['imgname']
        print('{}: containing {} samples ...'.format(self.dataset, len(self.imgname)))
        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        self.length = self.scale.shape[0]
        self.use_augmentation = use_augmentation

        # get GT smpl params,if available
        try:
            self.pose = self.data['pose']
            self.betas = self.data['shape']
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
            print('No smpl params available!')

        # get GT 3D pose,if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
            print('No gt 3D keypoints available!')
        # get 2D GT keypoints or openpose keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)
        # TODO cliff
        self.bbox_type = bbox_type
        if self.bbox_type == 'square':
            print('Using original bboxes!')
            self.crop_w = constants.IMG_RES
            self.crop_h = constants.IMG_RES
            self.bbox_size = constants.IMG_RES
        elif self.bbox_type == 'rect':
            print('Using regenerated bboxes from gt 2d keypoints!')
            self.crop_w = constants.IMG_W
            self.crop_h = constants.IMG_H
            self.bbox_size = constants.IMG_H
        try:
            self.focal_length = self.data['focal_length']
            self.has_focal = True
        except KeyError:
            self.has_focal = False

        # only train using 2d data
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))
            self.has_pose_3d = 0

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augment_params()

        # load image
        # process the image
        imgname = join(self.img_dir, self.imgname[index])
        try:
            # cv default channel is BGR,use::-1 to transform the BGR to RGB
            img = cv2.imread(imgname)[:, :, ::-1].copy().astype(np.float32)  # [H,W,3]
            orig_shape = np.array(img.shape)[:2]
        except TypeError:
            print(imgname)
        img = rgb_processing(img, center, sc * scale, rot, flip, pn)
        img = jt.array(img)

        # TODO start cliff
        # get focal
        if self.has_focal:
            img_focal_length = self.focal_length[index]
        else:
            img_focal_length = None

        bbox_info, focal_length = get_bbox_info(img, center, sc * scale, focal_length=img_focal_length)
        item['bbox_info'] = jittor.array(bbox_info)
        # from full img to crop img
        crop_trans = get_affine_transform(center, scale, 0, (self.crop_w, self.crop_h))
        item['crop_trans'] = jittor.float32(crop_trans)
        item['focal_length'] = jittor.float32(focal_length)
        item['img_h'] = orig_shape[0]
        item['img_w'] = orig_shape[1]
        # TODO end cliff

        # get smpl params,if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # crop and aug image info
        item['img'] = self.normalize_img(img)  #
        item['img_name'] = imgname
        item['orig_shape'] = jittor.float32(orig_shape)
        item['dataset_name'] = self.dataset
        item['scale'] = jittor.float32(sc * scale)
        item['center'] = jittor.float32(center)
        item['is_flipped'] = jittor.float32(flip)
        item['rot_angle'] = jittor.float32(rot)
        # process the pose
        item['has_smpl'] = self.has_smpl[index]
        item['pose'] = jt.float32(pose)
        item['betas'] = jt.float32(betas)
        item['gender'] = self.gender[index]

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        keypoints_full = self.keypoints[index].copy()
        item['keypoints_full'] = jittor.array(keypoints_full, dtype=jittor.float32)
        item['keypoints'] = jt.array(j2d_processing(keypoints, center, sc * scale, rot, flip), dtype=jittor.float32)

        # Get 3D pose
        item['has_pose_3d'] = self.has_pose_3d
        if self.has_pose_3d:
            pose_3d = self.pose_3d[index].copy()
            item['pose_3d'] = jt.array(j3d_processing(pose_3d, rot, flip))
        else:
            item['pose_3d'] = jt.zeros(24, 4, dtype=jt.float32)

        return item

    def __len__(self):
        return len(self.imgname)

    def augment_params(self):
        """
        Get augmentation parameters.
        """
        flip = 0  # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0  # rotation
        sc = 1  # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1

            # Each channel is multiplied with a number
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1 - self.options.noise_factor, 1 + self.options.noise_factor, 3)

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2 * self.options.rot_factor,
                      max(-2 * self.options.rot_factor, np.random.randn() * self.options.rot_factor))

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor] default=0.25
            sc = min(1 + self.options.scale_factor,
                     max(1 - self.options.scale_factor, np.random.randn() * self.options.scale_factor + 1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        return flip, pn, rot, sc


class Normalize:

    def __init__(self, mean, std, inplace=False):
        super().__init__()

        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Var) -> Var:
        return jt.transform.image_normalize(tensor, self.mean, self.std)

    def __call__(self, tensor: Var) -> Var:
        return self.forward(tensor)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
