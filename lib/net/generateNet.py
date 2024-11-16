# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from net.deeplabv3plus import deeplabv3plus
from net.BASC_v2 import BASC_v2


def generate_net(cfg):
	if cfg.MODEL_NAME == 'BASC':
		return BASC_v2(cfg)
	else:
		raise ValueError('generateNet.py: network %s is not support yet'%cfg.MODEL_NAME)
