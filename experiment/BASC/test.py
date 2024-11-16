# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
import surface_distance

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from net.sync_batchnorm.replicate import patch_replication_callback

from torch.utils.data import DataLoader
from scipy import ndimage
import time



def test_net():
    dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test')
    dataloader = DataLoader(dataset, 
                batch_size=cfg.TEST_BATCHES, 
                shuffle=False, 
                num_workers=cfg.DATA_WORKERS)
    net = generate_net(cfg)
    if cfg.TEST_CKPT is None:
        raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
    print('Use %d GPU'%cfg.TEST_GPUS)
    device = torch.device('cuda')
    if cfg.TEST_GPUS > 1:
        net = nn.DataParallel(net)
        patch_replication_callback(net)
    net.to(device)
    print('start loading model %s'%cfg.TEST_CKPT)
    model_dict = torch.load(cfg.TEST_CKPT,map_location=device)
    net.load_state_dict(model_dict)
    net.eval()
    # starttime = time.time()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            [batch, channel, height, width] = sample_batched['image'].size()
            inputs_batched = sample_batched['image']
            predicts_batched = net(inputs_batched.cuda())
            predicts_batched = predicts_batched.cuda()
            result_seg = torch.argmax(predicts_batched, dim=1, keepdim=True).cpu().astype(np.uint8)
            labels_batched_seg = sample_batched['segmentation'].cpu()
            mean_dice = eval(result_seg, labels_batched_seg)        

        # endtime = time.time()
def eval(prediction, ground_truth):
    B, _, H, W = ground_truth.size()
    n = prediction.size(1)
    ground_truth_one_hot = torch.zeros(B, n, H, W, device=prediction.device)
    ground_truth_one_hot.scatter_(1, ground_truth, 1)
    dice_scores = []
    for class_idx in [1, 2]:
        pred_mask = prediction[:, class_idx, :, :] 
        gt_mask = ground_truth_one_hot[:, class_idx, :, :] 
        intersection = (pred_mask * gt_mask).sum(dim=(1, 2)) 
        pred_area = pred_mask.sum(dim=(1, 2)) 
        gt_area = gt_mask.sum(dim=(1, 2))
        dice_score = (2 * intersection + 1e-8) / (pred_area + gt_area + 1e-8)
        dice_scores.append(dice_score)
    dice_scores = torch.stack(dice_scores, dim=1) 
    mean_dice_score = dice_scores.mean(dim=1).mean()
    return mean_dice_score

if __name__ == '__main__':
    test_net()




