from datasets.transform import *
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import monai
from monai.data import CacheDataset

def Prostate158_dataset(data_name, cfg, period):
    if period=='train':
        train_root_dir = r'/home/wx/zonal_seg/data/Prostate158'
        img_dir = os.path.join(train_root_dir, 'image')
        edge_dir = os.path.join(train_root_dir, 'edge')
        gt_dir = os.path.join(train_root_dir, 'label')
        file_name = os.path.join(train_root_dir, 'train.txt')
        df = pd.read_csv(file_name, names=['filename'])
        name_list = df['filename'].values
        data_dict = []
        for i in range(len(name_list)):
            name = name_list[i]
            img_file = os.path.join(img_dir, '%s.jpg'%name_list[i])
            gt_file = os.path.join(gt_dir, '%s.png'%name_list[i])
            edge_file = os.path.join(edge_dir, '%s.png'%name_list[i])
            data_dict.append({'image':img_file, 'segmentation':gt_file, 'edge':edge_file})
            transforms = get_train_transforms(cfg)
            
    elif period=='val':
        train_root_dir = r'/home/wx/zonal_seg/data/Prostate158'
        img_dir = os.path.join(train_root_dir, 'image')
        edge_dir = os.path.join(train_root_dir, 'edge')
        gt_dir = os.path.join(train_root_dir, 'label')
        file_name = os.path.join(train_root_dir, 'val.txt')
        df = pd.read_csv(file_name, names=['filename'])
        name_list = df['filename'].values
        data_dict = []
        for i in range(len(name_list)):
            name = name_list[i]
            img_file = os.path.join(img_dir, '%s.jpg'%name_list[i])
            gt_file = os.path.join(gt_dir, '%s.png'%name_list[i])
            edge_file = os.path.join(edge_dir, '%s.png'%name_list[i])
            data_dict.append({'image':img_file, 'segmentation':gt_file, 'edge':edge_file})
            transforms = get_val_transforms()

    else:        
        test_root_dir = r'/home/wx/zonal_seg/data/Prostate158'
        img_dir = os.path.join(test_root_dir, 'image')
        edge_dir = os.path.join(test_root_dir, 'edge')
        gt_dir = os.path.join(test_root_dir, 'label')
        file_name = os.path.join(test_root_dir, 'test.txt')
        df = pd.read_csv(file_name, names=['filename'])
        name_list = df['filename'].values
        data_dict = []
        for i in range(len(name_list)):
            name = name_list[i]
            img_file = os.path.join(img_dir, '%s.jpg'%name_list[i])
            gt_file = os.path.join(gt_dir, '%s.png'%name_list[i])
            edge_file = os.path.join(edge_dir, '%s.png'%name_list[i])
            data_dict.append({'image':img_file, 'segmentation':gt_file, 'edge':edge_file, 'name':name})
            transforms = get_test_transforms()

    dataset = CacheDataset(data=data_dict, transform=transforms)
    return dataset




