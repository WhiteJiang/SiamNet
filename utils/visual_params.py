# -*- coding: utf-8 -*-
# @Time    : 2022/12/9
# @Author  : White Jiang
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    params = torch.load('/home/jx/code/SiamNet/checkpoints/SiamNet/12/model.pth')
    # params = torch.load('D:\Science\pth\model1.pth')
    w_g = params['W_G']
    w_g_abs = torch.abs(w_g)
    # values, index = torch.sort(w_g_abs[0])
    # scale = torch.clamp((w_g_abs[0] > values[100]).float(), min=0.0000).cuda()
    # scale = scale.float()
    # print(w_g[0])
    # for i in range(len(w_g_abs[0])):
    #     if w_g_abs[0][i] < values[100]:
    #         w_g[0][i] = 0.0
    # print('%f' % w_g[0])
    for i in range(12):
        # print(w_g[0].size())
        # temp_w = np.array(w_g_abs[i])
        values, index = torch.sort(w_g_abs[i])
        values = values.cuda()
        for i in range(len(w_g_abs[0])):
            if w_g_abs[0][i] < values[100]:
                w_g[0][i] = 0.0
        print(w_g[0])
