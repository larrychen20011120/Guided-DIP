import torch.nn as nn
import os
import sys

# in order to import other files
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from models.backbone import UNet

class DIP(nn.Module):
    def __init__(self, **config):
        super(DIP, self).__init__()
        self.backbone = UNet(**config)
        
    def forward(self, x):
        return self.backbone(x)

class DDPM(nn.Module):
    def __init__(self, **config):
        super(DDPM, self).__init__()
        pass
    def forward(self, x):
        pass

class PreDDPM(nn.Module):
    def __init__(self, **config):
        super(PreDDPM, self).__init__()
        pass
    def forward(self, x):
        pass
