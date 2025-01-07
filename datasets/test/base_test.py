from utils import TrainOptions
from datasets.base_dataset import BaseDataset
import argparse
import config
import cv2
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--n_views', default=5, help='Batch size for testing')
parser.add_argument('--use_extraviews', default=True, action='store_true', help='Shuffle data')
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
parser.add_argument('--use_aug_trans', default=False, action="store_true", help='flag for using augment')
parser.add_argument('--use_aug_img', default=False, action="store_true", help='flag for using augment')

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = BaseDataset(options=args, dataset='3dpw', is_train=True)
    item = dataset[999]
    imgname = item['img_name']
    print(imgname)
    origin_h = item['img_h']
    origin_w = item['img_w']
    print(f'origin hight:{origin_h} origin width:{origin_w}')
    center = item['center'].numpy()
    scale = item['scale'].numpy()
    print(f'center:{center} scale:{scale}')
    bbox = item['bbox_info'].numpy()
    print(f"bounding box info{bbox}")

    # TODO visualiztion
    # 1. full image with origin kp2d
    img_origin = item['img_origin']  # rgb
    img_ = cv2.cvtColor(img_origin, cv2.COLOR_RGB2BGR)
    cv2.circle(img_, (int(center[0]), int(center[1])), color=255. * np.random.rand(3, ), radius=10,
               thickness=-1)
    cv2.imwrite("Image_full_pre.png", img_)

    kp = item['keypoints_full']
    kp = kp[25:]

    for keypoint in kp:
        cv2.circle(img_, (int(keypoint[0]), int(keypoint[1])), color=255. * np.random.rand(3, ), radius=4,
                   thickness=-1)
        # print(keypoint)
    cv2.imwrite("Image_full_post.png", img_)

    # 2. crop image
    img_crop = item['img_'].numpy() * 255.0
    img_crop = img_crop.transpose(1, 2, 0)
    img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
    cv2.imwrite("Image_crop_pre.png", img_crop)
    kp = item['keypoints'][25:]
    for keypoint in kp:
        cv2.circle(img_crop, (int(keypoint[0]), int(keypoint[1])), color=255. * np.random.rand(3, ), radius=4,
                   thickness=-1)
    cv2.imwrite("Image_crop_post.png", img_crop)


