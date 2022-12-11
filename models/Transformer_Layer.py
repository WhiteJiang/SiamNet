# -*- coding: utf-8 -*-
# @Time    : 2022/12/11
# @Author  : White Jiang
import torch.nn as nn
import torch


class Enconder(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        # 将qkv做映射的卷积
        self.qkv = nn.Conv2d(dim, dim * 3, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        # 将输入映射为 q k v  在通道方向均等划分为num_head个特征向量
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W).transpose(0, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 第一阶段的注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 将注意力与原值相加
        attn = torch.sign(attn) * torch.sqrt(torch.abs(attn) + 1e-5)
        # 做softmax
        attn = attn.softmax(dim=-1)
        # 残差链接 同时转换成d个n块
        y = self.norm(x)
        x = self.relu(y)
        return x
