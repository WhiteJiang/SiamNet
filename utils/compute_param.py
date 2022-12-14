# -*- coding: utf-8 -*-
# @Time    : 2022/10/30
# @Author  : White Jiang
import numpy as np
from models.SiamNet import SiamNet


def param_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()


if __name__ == '__main__':
    model = SiamNet()
    print(param_count(model))
