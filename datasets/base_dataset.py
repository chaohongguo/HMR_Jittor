import jittor
from jittor.dataset import Dataset
import config
import copy
import numpy as np
from os.path import join
import cv2
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa, get_affine_transform, affine_transform
import constants
from jittor import Var
import albumentations as A
import jpeg4py as jpeg
from loguru import logger


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


def rgb_processing_(rgb_img, center, scale, rot, flip, pn):
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
    def __init__(self, options, dataset, ignore_3d=False, is_train=True,
                 use_augmentation=True, bbox_type='rect'):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = config.DATASET_FOLDERS[dataset]
        filename = config.DATASET_FILES[is_train][dataset]
        self.data = np.load(filename)
        if self.is_train:
            print(">>>Train dataset ", end=' ')
        else:
            print(">>>Eval dataset ", end=' ')
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.imgname = self.data['imgname']
        print('{}: containing {} samples ...'.format(self.dataset, len(self.imgname)))
        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        self.length = self.scale.shape[0]
        self.use_augmentation = self.options.use_aug_trans
        self.use_syn_occ = False
        # fix 3dpw
        if self.dataset == '3dpw':
            self.half_size = True
        else:
            self.half_size = False
        # get GT smpl params,if available
        try:
            self.pose = self.data['pose'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
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
            # 'male=0' 'female=1'
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)
        # TODO cliff
        self.bbox_type = self.options.bbox_type
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
            print('No focal lengths available! Using estimated focal length...')

        # only train using 2d data
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))
            self.has_pose_3d = 0

    def __getitem__(self, index):
        item = {}

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        keypoints_full = self.keypoints[index].copy()

        # TODO fix 3dpw
        if self.half_size:
            keypoints[:, :2] = keypoints[:, :2] / 2.
            keypoints_full[:, :2] = keypoints_full[:, :2] / 2.

        # get bounding box from keypoint
        center, scale = bbox_from_keypoint(keypoints[25:], rescale=1.2)

        # Get augmentation parameters
        if self.use_augmentation:
            flip, pn, rot, sc = self.augment_params()
        else:
            flip = 0
            pn = np.ones(3)
            rot = 0
            sc = 1

        scale = scale * sc

        # load image
        imgname = join(self.img_dir, self.imgname[index])
        try:
            # cv default channel is BGR,use::-1 to transform the BGR to RGB
            # img = cv2.imread(imgname)[:, :, ::-1].copy()  # [H,W,3]
            img = self.read_img(imgname)
        except TypeError:
            print(imgname)

        # TODO fix 3dpw
        if self.half_size:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

        orig_shape = np.array(img.shape)[:2]
        # get smpl params,if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Get 3D pose
        item['has_pose_3d'] = self.has_pose_3d
        if self.has_pose_3d:
            pose_3d = self.pose_3d[index].copy()
        else:
            pose_3d = np.zeros((24, 4), dtype=np.float32)

        if flip:
            center[0] = orig_shape[1] - 1 - center[0]
            img = flip_img(img)
            keypoints = flip_kp(keypoints, orig_shape[1])
            keypoints_full = flip_kp(keypoints_full, orig_shape[1])
            pose_3d = flip_kp(pose_3d)
            pose = self.flip_smpl_pose(pose)

        full_trans = get_affine_transform(center, scale, rot,
                                          (self.crop_w, self.crop_h))  # from full to crop (with rotate)
        inv_trans = get_affine_transform(center, scale, 0, (self.crop_w, self.crop_h),
                                         inv=True)  # from crop to full (no rotate)
        crop_trans = get_affine_transform(center, scale, 0, (self.crop_w, self.crop_h))  # from full to crop (no rotate)

        keypoints_2d = self.rotate_joints_2d(keypoints, full_trans).astype('float32')
        keypoints_3d = self.rotate_joints_3d(pose_3d, rot).astype('float32')
        pose = self.rotate_smpl_pose(pose, rot).astype('float32')

        item['img_name'] = imgname
        item['dataset_name'] = self.dataset
        item['img_h'] = orig_shape[0]
        item['img_w'] = orig_shape[1]
        item['gender'] = self.gender[index]
        item['orig_shape'] = orig_shape
        item['scale'] = jittor.float32(scale)
        item['center'] = jittor.float32(center)

        # data aug info
        item['is_flipped'] = jittor.float32(flip)
        item['rot_angle'] = jittor.float32(rot)

        # 2d keypoints
        item['keypoints'] = jittor.float32(keypoints_2d)
        item['keypoints_full'] = jittor.float32(keypoints_full)

        # smpl params
        item['has_smpl'] = self.has_smpl[index]
        item['pose'] = jittor.float32(pose)
        item['betas'] = jittor.float32(betas)

        # 3d keypoints
        item['has_pose_3d'] = self.has_pose_3d
        item['pose_3d'] = jittor.array(keypoints_3d)

        # TODO start cliff
        # get focal
        if self.has_focal:
            img_focal_length = self.focal_length[index]/2. if self.half_size else self.focal_length[index]
        else:
            img_focal_length = None

        bbox_info, focal_length = get_bbox_info(img, center, scale, focal_length=img_focal_length)
        item['bbox_info'] = jittor.array(bbox_info, dtype=jittor.float32)
        item['crop_trans'] = jittor.float32(crop_trans)
        item['focal_length'] = jittor.float32(focal_length)
        item['full_trans'] = jittor.float32(full_trans)
        # TODO end cliff
        # Process image
        # item['img_origin'] = img  # for viz

        img = self.rgb_processing(img, full_trans, pn)

        img = jittor.float32(img)
        item['img'] = self.normalize_img(img)
        # item['img_'] = img   # for viz
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

    def rotate_joints_2d(self, keypoints, trans):
        keypoints_rot = copy.copy(keypoints)
        for kp in keypoints_rot:
            kp[:2] = affine_transform(kp[:2], trans)
        return keypoints_rot

    def _construct_rotation_matrix(self, rot, size=3):
        """Construct the in-plane rotation matrix.
        Args:
            rot (float): Rotation angle (degree).
            size (int): The size of the rotation matrix.
                Candidate Values: 2, 3. Defaults to 3.
        Returns:
            rot_mat (np.ndarray([size, size]): Rotation matrix.
        """
        rot_mat = np.eye(size, dtype=np.float32)
        if rot != 0:
            rot_rad = np.deg2rad(rot)
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]

        return rot_mat

    def rotate_joints_3d(self, joints_3d, rot):
        """Rotate the 3D joints in the local coordinates.
        Notes:
            Joints number: K
        Args:
            joints_3d (np.ndarray([K, 3])): Coordinates of keypoints.
            rot (float): Rotation angle (degree).
        Returns:
            joints_3d_rotated
        """
        # in-plane rotation
        # 3D joints are rotated counterclockwise,
        # so the rot angle is inversed.
        rot_mat = self._construct_rotation_matrix(-rot, 3)
        # print(rot_mat.shape, joints_3d.shape)

        joints_3d[:, :-1] = np.einsum('ij,kj->ki', rot_mat, joints_3d[:, :-1])
        joints_3d = joints_3d.astype('float32')
        return joints_3d

    def rotate_smpl_pose(self, pose, rot):
        """Rotate SMPL pose parameters.
        SMPL (https://smpl.is.tue.mpg.de/) is a 3D
        human model.
        Args:
            pose (np.ndarray([72])): SMPL pose parameters
            rot (float): Rotation angle (degree).
        Returns:
            pose_rotated
        """
        pose_rotated = pose.copy()
        if rot != 0:
            rot_mat = self._construct_rotation_matrix(-rot)
            orient = pose[:3]
            # find the rotation of the body in camera frame
            per_rdg, _ = cv2.Rodrigues(orient.astype(np.float32))
            # apply the global rotation to the global orientation
            res_rot, _ = cv2.Rodrigues(np.dot(rot_mat, per_rdg))
            pose_rotated[:3] = (res_rot.T)[0]

        return pose_rotated

    def flip_smpl_pose(self, pose):
        """Flip SMPL pose parameters horizontally.
        Args:
            pose (np.ndarray([72])): SMPL pose parameters
        Returns:
            pose_flipped
        """

        flippedParts = [
            0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18, 19,
            20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32, 36, 37,
            38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58,
            59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68
        ]
        pose_flipped = pose[flippedParts]
        # Negate the second and the third dimension of the axis-angle
        pose_flipped[1::3] = -pose_flipped[1::3]
        pose_flipped[2::3] = -pose_flipped[2::3]
        return pose_flipped

    def rgb_processing(self, rgb_img, trans, pn, use_syn_occ=False, kp=None, dir=None):
        """Process rgb image and do augmentation."""

        # if self.is_train and self.options.ALB:
        if self.is_train and self.options.use_aug_img:
            rgb_img_full = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            aug_comp = [A.Downscale(0.5, 0.9, interpolation=0, p=0.1),
                        A.ImageCompression(20, 100, p=0.1),
                        A.RandomRain(blur_value=4, p=0.1),
                        A.MotionBlur(blur_limit=(3, 15), p=0.2),
                        A.Blur(blur_limit=(3, 9), p=0.1),
                        A.RandomSnow(brightness_coeff=1.5,
                                     snow_point_lower=0.2, snow_point_upper=0.4)]
            aug_mod = [A.CLAHE((1, 11), (10, 10), p=0.2), A.ToGray(p=0.2),
                       A.RandomBrightnessContrast(p=0.2),
                       A.MultiplicativeNoise(multiplier=[0.5, 1.5],
                                             elementwise=True, per_channel=True, p=0.2),
                       A.HueSaturationValue(hue_shift_limit=20,
                                            sat_shift_limit=30, val_shift_limit=20,
                                            always_apply=False, p=0.2),
                       A.Posterize(p=0.1),
                       A.RandomGamma(gamma_limit=(80, 200), p=0.1),
                       A.Equalize(mode='cv', p=0.1)]
            albumentation_aug = A.Compose([A.OneOf(aug_comp,
                                                   p=0.3),
                                           A.OneOf(aug_mod,
                                                   p=0.3)])
            rgb_img = albumentation_aug(image=rgb_img_full)['image']
        rgb_img = cv2.warpAffine(
            rgb_img,
            trans, (self.crop_w, self.crop_h), flags=cv2.INTER_LINEAR)
        if self.is_train and use_syn_occ:
            # if np.random.uniform() <= 0.5:
            rgb_img = self.syn_occlusion.make_occlusion(rgb_img)
            # print('Using syn-occ for DA ...')
        if kp is not None:
            print(kp.shape, kp)
            for keypoint in kp:
                cv2.circle(rgb_img, (int(keypoint[0]), int(keypoint[1])), color=255. * np.random.rand(3, ), radius=4,
                           thickness=-1)
                # print(keypoint)
            print(dir)
            cv2.imwrite(dir, rgb_img[:, :, ::-1])
        # cv2.circle(rgb_img, (rgb_img.shape[1]//2, rgb_img.shape[0]//2), color=(255, 0, 0), radius=4, thickness=-1)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))

        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
        return rgb_img

    def read_img(self, img_fn):
        #  return pil_img.fromarray(
        #  cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB))
        #  with open(img_fn, 'rb') as f:
        #  img = pil_img.open(f).convert('RGB')
        #  return img
        if img_fn.endswith('jpeg') or img_fn.endswith('jpg'):
            try:
                with open(img_fn, 'rb') as f:
                    img = np.array(jpeg.JPEG(f).decode())
            except jpeg.JPEGRuntimeError:
                logger.warning('{} produced a JPEGRuntimeError', img_fn)
                img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
        else:
            #  elif img_fn.endswith('png') or img_fn.endswith('JPG') or img_fn.endswith(''):
            img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
        return img


class Normalize:

    def __init__(self, mean, std, inplace=False):
        super().__init__()

        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Var) -> Var:
        return jittor.transform.image_normalize(tensor, self.mean, self.std)

    def __call__(self, tensor: Var) -> Var:
        return self.forward(tensor)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
