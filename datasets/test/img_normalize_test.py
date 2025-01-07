import numpy as np
import jpeg4py as jpeg
from loguru import logger
import cv2
import albumentations as A


def read_img(img_fn):
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


def rgb_processing(rgb_img, trans, pn, use_syn_occ=False, kp=None, dir=None):
    """Process rgb image and do augmentation."""

    # if self.is_train and self.options.ALB:
    if True:
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
        trans, (192, 256), flags=cv2.INTER_LINEAR)
    # if self.is_train and use_syn_occ:
    #     # if np.random.uniform() <= 0.5:
    #     rgb_img = self.syn_occlusion.make_occlusion(rgb_img)
    #     # print('Using syn-occ for DA ...')
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


img_fn = '/home/lab345/mnt4T/fmx/transmesh/datasets/coco/train2014/COCO_train2014_000000044474.jpg'