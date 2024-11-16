# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

class Configuration():
    def __init__(self):
        self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__"),'..','..'))
        self.EXP_NAME = 'exp10'

        self.DATA_NAME = 'Prostate158'
        self.RandBiasFieldd = 1
        self.RandGaussianSmoothd = 1
        self.RandGibbsNoised = 1
        self.RandAffined =1
        self.RandRotate90d = 1
        self.RandRotated = 1
        self.RandElasticd = 1
        self.RandZoomd = 1
        self.RandCropByPosNegLabeld = 1
        self.RandGaussianNoised = 1
        self.RandShiftIntensityd = 1
        self.RandGaussianSharpend = 1
        self.RandAdjustContrastd = 1

        self.DATA_WORKERS = 8
        self.MODEL_NAME = 'BASC'
        self.MODEL_BACKBONE = 'res101_atrous'
        self.MODEL_OUTPUT_STRIDE = 16
        self.MODEL_ASPP_OUTDIM = 256
        self.MODEL_SHORTCUT_DIM = 128
        self.MODEL_SHORTCUT_KERNEL = 1
        self.MODEL_NUM_CLASSES = 3
        self.MODEL_SAVE_DIR = os.path.join(self.ROOT_DIR,'model',self.EXP_NAME)

        self.TRAIN_LR = 1e-4
        self.TRAIN_LR_GAMMA = 0.1
        self.TRAIN_MOMENTUM = 0.9
        self.TRAIN_WEIGHT_DECAY = 0.00001
        self.TRAIN_BN_MOM = 0.0003
        self.TRAIN_POWER = 0.9 
        self.TRAIN_GPUS = 1
        self.TRAIN_BATCHES = 36
        self.TRAIN_SHUFFLE = True
        self.TRAIN_MINEPOCH = 0	
        self.TRAIN_EPOCHS = 400
        self.TRAIN_LOSS_LAMBDA = 0
        self.TRAIN_CKPT = os.path.join(self.ROOT_DIR,'model/deeplabv3plus_res101_atrous_VOC2012_epoch46_all.pth')

        self.TEST_CKPT = None
        self.TEST_GPUS = 1
        self.TEST_BATCHES = 36

        self.__check()
        self.__add_path(os.path.join(self.ROOT_DIR, 'lib'))
        
    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('config.py: cuda is not avalable')
        if self.TRAIN_GPUS == 0:
            raise ValueError('config.py: the number of GPU is 0')
        #if self.TRAIN_GPUS != torch.cuda.device_count():
        #	raise ValueError('config.py: GPU number is not matched')

        if not os.path.isdir(self.MODEL_SAVE_DIR):
            os.makedirs(self.MODEL_SAVE_DIR)

    def __add_path(self, path):
        if path not in sys.path:
            sys.path.insert(0, path)

cfg = Configuration() 	
