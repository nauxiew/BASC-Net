# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from datasets.ISBI_Dataset import ISBI_dataset
from datasets.Prostate158_Dataset import Prostate158_dataset

def generate_dataset(dataset_name, cfg, period, aug=False):
	if dataset_name == 'Prostate158':
		return Prostate158_dataset(dataset_name, cfg, period)
	if dataset_name == 'ISBI':
		return ISBI_dataset(dataset_name, cfg, period)
	else:
		raise ValueError('generateData.py: dataset %s is not support yet'%dataset_name)
