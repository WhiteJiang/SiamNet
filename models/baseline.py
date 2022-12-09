from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url

# from ..utils.serialization import load_checkpoint, copy_state_dict
""""
基础模型 resnet50
"""


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Baseline(nn.Module):
    def __init__(self, code_length=12, num_classes=200, att_size=4, feat_size=2048, device='cpu', pretrained=True):
        super(Baseline, self).__init__()
        self.backbone = Baseline(pretrained=pretrained)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, code_length),
            nn.Tanh(),
        )
        self.device = device

    def forward(self, x, targets):
        ret = self.backbone(x)
        return ret


def baseline(code_length, num_classes, att_size, feat_size, device, pretrained=False, progress=True, **kwargs):
    model = Baseline(code_length, num_classes, att_size, feat_size, device, pretrained, **kwargs)
    return model

