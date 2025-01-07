# HMR_jittor

**Jittor version** of Code repository for the following papers about Human Mesh Recovery:

CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation [paper](https://arxiv.org/abs/2208.00571) [project](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)

Multi-RoI Human Mesh Recovery with Camera Consistency and Contrastive Losses(ECCV2024)  [paper](https://arxiv.org/abs/2402.02074) [project](https://github.com/CptDiaos/Multi-RoI?tab=readme-ov-file)

Incorporating Test-Time Optimization into Training with Dual Networks for Human Mesh Recovery  [paper](https://arxiv.org/abs/2401.14121) [project](https://github.com/fmx789/Meta-HMR)

## Installation instructions

```
conda create -n hmr_jittor pytyon=3.9
conda activate hmr_jittor
pip install jittor
python -m jittor.test.test_example
pip install numpy==1.23.5
pip install -r requirements.txt
```

## Fetch data

let your data path as following

```
data
├── best_ckpt
│   ├── cliff_resnet_72.0_45.7.pkl
│   ├── metaHMR_cliff_resnet_84.0_53,6.pkl
│   └── mutilROI_resnet_68.8_43.7.pkl
├── dataset_extras
│   ├── 3dpw_test_w2d_smpl3d_gender.npz
│   ├── 3dpw_train_w2d_smpl3d_gender.npz
│   └── coco_2014_smpl_train.npz
├── J_regressor_extra.npy
├── J_regressor_h36m.npy
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── smpl_kid_template.npy
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
└── smpl_mean_params.npz
```

download part of dataset_extras from [here]( https://pan.baidu.com/s/1_vXFwFtp8xbPoUxtJcQuAA?pwd=bq8v)

download smpl from [here](https://www.bing.com/search?q=smpl&cvid=57cea722454a4e58a2ec22a8fd748ba3&gs_lcrp=EgRlZGdlKgYIABBFGDsyBggAEEUYOzIGCAEQABhAMgYIAhBFGDsyBggDEAAYQDIGCAQQABhAMgYIBRBFGDwyBggGEEUYPDIGCAcQRRg8MgYICBBFGEHSAQgxMjg3ajBqOagCCLACAQ&FORM=ANAB01&adppc=EdgeStart&PC=LCTS)

## Eval

```python
python eval.py --model_name mutilROI 
               --checkpoint data/best_ckpt/mutilROI_resnet_68.8_43.7.pkl  # eval mutilROI
python eval.py --model_name metaHMR
               --checkpoint data/best_ckpt/metaHMR_cliff_resnet_84.0_53,6.pkl # eval metaHMR
python eval.py --model_name cliff
               --checkpoint data/best_ckpt/cliff_resnet_72.0_45.7.pkl # eval cliff    

```

## Train

```python
>>> train on single datset with mutilROI
python train.py --fixname --name example_train
                --lr 1e-4 --model_name mutilROI
                --viz_debug --train_dataset coco
                --bbox_type rect
                --batch_size 36
                --use_aug_trans
                
>>> train on single datset with metaHMR
pytyon train.py --model_name metaHMR
                --viz_debug
                --train_dataset coco
                --bbox_type rect
                --batch_size 64
                --first_order
                --adapt_val_ste
>>>  train on single datset with cliff
pytyon train.py --model_name cliff
                --viz_debug
                --train_dataset coco
                --bbox_type rect
                --batch_size 64
     
```

