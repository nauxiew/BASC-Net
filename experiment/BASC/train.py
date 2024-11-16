# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np
import torch.nn.functional as F
import cv2
import surface_distance
from utils import *
from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from net.sync_batchnorm.replicate import patch_replication_callback
from net.loss import BAC_loss


def train_net():
    # laod segmentation data
    dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train', cfg.DATA_AUG)
    dataloader = DataLoader(dataset, 
                batch_size=cfg.TRAIN_BATCHES, 
                shuffle=cfg.TRAIN_SHUFFLE, 
                num_workers=cfg.DATA_WORKERS,
                drop_last=True)

    val_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'val')
    val_dataloader = DataLoader(val_dataset, 
                batch_size=cfg.TEST_BATCHES, 
                shuffle=False, 
                num_workers=cfg.DATA_WORKERS)
    
    test_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test')
    test_dataloader = DataLoader(test_dataset, 
                batch_size=cfg.TEST_BATCHES, 
                shuffle=False, 
                num_workers=cfg.DATA_WORKERS)
    net = generate_net(cfg)
    print('Use %d GPU'%cfg.TRAIN_GPUS)
    device = torch.device(0)
    if cfg.TRAIN_GPUS > 1:
        net = nn.DataParallel(net)
        patch_replication_callback(net)
    net.to(device)
    if cfg.TRAIN_CKPT:
        pretrained_dict = torch.load(cfg.TRAIN_CKPT)
        net_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
        net_dict.update(pretrained_dict)        
        net.load_state_dict(net_dict)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    criterion_BAC = BAC_loss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.TRAIN_LR)
    best_jacc = 0.
    best_epoch = 0
    for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):
        total_loss = 0.0
        net.train()
        for i_batch, sample_batched in enumerate(dataloader):       
            labels_batched_edge = sample_batched['edge']
            inputs_batched, labels_batched = sample_batched['image'], sample_batched['segmentation']
            labels_batched_edge = labels_batched_edge.long().cuda()
            labels_batched = labels_batched.long().cuda()
            inputs_batched = inputs_batched.cuda()
            one_hot_edge_batched = make_one_hot_edge(labels_batched_edge[:,0,:,:])
            one_hot_edge_batched_train = one_hot_edge_batched.permute(0,3,1,2)
            optimizer.zero_grad()
            predicts_batched, aux_loss, final_feature = net(x=inputs_batched, edge=one_hot_edge_batched_train, period='train')   
            predicts_batched = predicts_batched.cuda()
            loss_bac = criterion_BAC(final_feature, one_hot_edge_batched_train).to(0)
            loss_ce = criterion(predicts_batched, labels_batched[:,0,:,:]).to(0)
            aux_loss = aux_loss.to(0)
            loss_function = loss_ce + loss_bac + aux_loss
            loss_function.backward()
            optimizer.step()
            total_loss += loss_function.item()
        print('epoch:%d/%d\t  loss:%g\t \n' %(epoch, cfg.TRAIN_EPOCHS, total_loss/i_batch))
        net.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(val_dataloader):
                row_batched = sample_batched['row']
                col_batched = sample_batched['col']
                spacing_batched = sample_batched['spacing']
                [batch, channel, height, width] = sample_batched['image'].size()
                inputs_batched = sample_batched['image']
                predicts_batched = net(inputs_batched.cuda())
                predicts_batched = predicts_batched.cuda()
                result_seg = torch.argmax(predicts_batched, dim=1, keepdim=True).cpu().astype(np.uint8)
                labels_batched_seg = sample_batched['segmentation'].cpu()
                mean_dice = eval(result_seg, labels_batched_seg)        
            if mean_dice > best_jacc:
                model_snapshot(net.state_dict(), new_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_dice%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch,mean_dice)),
                        old_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_dice%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,best_epoch,best_jacc)))
                best_jacc = mean_dice
                best_epoch = epoch

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

def get_params(model, key):
    for m in model.named_modules():
        if key == '1x':
            if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p
        elif key == '10x':
            if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p

def make_one_hot_edge(labels):
    
    target = torch.eye(6)[labels].float().to(0)
    return target

def model_snapshot(model, new_file=None, old_file=None):
    if os.path.exists(old_file) is True:
        os.remove(old_file) 
    torch.save(model, new_file)
    print('%s has been saved'%new_file)



if __name__ == '__main__':
    train_net()


