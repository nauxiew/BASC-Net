import os
import torch
from monai.utils.enums import CommonKeys
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    ConcatItemsd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    SaveImaged,
    ScaleIntensityd,
    NormalizeIntensityd,
    Resized,
)
from PIL import Image
import numpy as np

        
class get_basic_info():
    def __init__(self, keys):
        self.gt_key= keys
    def __call__(self, data):
        d = dict(data)
        gt =d[self.gt_key]
        h,w, = gt.shape
        d['row'] = h
        d['col'] = w
        return d
        

def get_base_transforms(
    minv=0, 
    maxv=1,
    period=None
):
    
    tfms=[]
    tfms+=[LoadImaged(keys=['image','segmentation', 'edge'], image_only=True, reader='ITKReader')]
    tfms+=[get_basic_info(keys='segmentation')]
    tfms+=[EnsureChannelFirstd(keys=['image', 'segmentation','edge'])]
    return tfms


def get_train_transforms(cfg, p=0.175):
    tfms=get_base_transforms()
    if cfg.RandBiasFieldd>0:
        from monai.transforms import RandBiasFieldd
        tfms+=[
            RandBiasFieldd(
                keys=['image'],
                degree=10,
                coeff_range=[0.0, 0.01],
                prob=p
            )
        ]

    if cfg.RandGaussianSmoothd > 0:
        from monai.transforms import RandGaussianSmoothd
        tfms+=[
            RandGaussianSmoothd(
                keys=['image'],
                sigma_x= [0.25, 1.5],
                sigma_y= [0.25, 1.5],
                prob=p
            )
        ]

    if cfg.RandGibbsNoised >0:
        from monai.transforms import RandGibbsNoised
        tfms+=[
            RandGibbsNoised(
                keys=['image'],
                alpha=[0.5, 1],
                prob=p
            )
        ]

    if cfg.RandAffined >0:
        from monai.transforms import RandAffined
        tfms+=[
            RandAffined(
                keys=['image', 'segmentation', 'edge'],
                rotate_range=5,
                shear_range=0.5,
                translate_range=25,
                mode=["bilinear", 'nearest', 'nearest'],
                prob=p
            )
        ]

    if cfg.RandRotate90d >0:
        from monai.transforms import RandRotate90d
        tfms+=[
            RandRotate90d(
                keys=['image', 'segmentation', 'edge'],
                spatial_axes=[0,1],
                prob=p
            )
        ]

    if cfg.RandRotated >0:
        from monai.transforms import RandRotated
        tfms+=[
            RandRotated(
                keys=['image', 'segmentation', 'edge'],
                range_x=0.1,
                range_y=0.1,
                mode=['bilinear', 'nearest', 'nearest'],
                prob=p
            )
        ]

    tfms+=[
        Resized(
            keys=['image', 'segmentation', 'edge'],
            spatial_size=[384,384],
            mode = ['bilinear', 'nearest','nearest'],
        )
    ]


    if cfg.RandCropByPosNegLabeld >0:
        from monai.transforms import RandCropByPosNegLabeld
        tfms+=[
            RandCropByPosNegLabeld(
                keys=['image', 'segmentation', 'edge'],
                label_key='segmentation',
                spatial_size=[256, 256],
                pos=3,
                neg=1,
                num_samples=4,
            )
        ]
        

    if cfg.RandGaussianNoised>0:
        from monai.transforms import RandGaussianNoised
        tfms+=[
            RandGaussianNoised(
                keys=['image'],
                mean=0.1,
                std=0.25,
                prob=p
            )
        ]

    if cfg.RandShiftIntensityd>0:
        from monai.transforms import RandShiftIntensityd
        tfms+=[
            RandShiftIntensityd(
                keys=['image'],
                offsets=0.2,
                prob=p
            )
        ]

    if cfg.RandGaussianSharpend>0:
        from monai.transforms import RandGaussianSharpend
        tfms+=[
            RandGaussianSharpend(
                keys=['image'],
                sigma1_x=[0.5, 1.0],
                sigma1_y=[0.5, 1.0],
                sigma2_x=[0.5, 1.0],
                sigma2_y=[0.5, 1.0],
                alpha=[10.0, 30.0],
                prob=p
            )
        ]

    if cfg.RandAdjustContrastd>0:
        from monai.transforms import RandAdjustContrastd
        tfms+=[
            RandAdjustContrastd(
                keys=['image'],
                gamma=2.0,
                prob=p
            )
        ]
        
    return Compose(tfms)


def get_val_transforms():
    tfms=get_base_transforms()
    tfms+=[
        Resized(
            keys=['image', 'segmentation', 'edge'],
            spatial_size=[256,256],
            mode = ['bilinear', 'nearest','nearest'],
        )
    ] 
    return Compose(tfms)


def get_test_transforms():
    tfms=get_base_transforms(period='test')
    tfms+=[
        Resized(
            keys=['image', 'segmentation', 'edge'],
            spatial_size=[256,256],
            mode = ['bilinear', 'nearest','nearest'],
        )
    ]

    
    return Compose(tfms)

