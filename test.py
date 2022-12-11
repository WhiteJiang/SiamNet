# -*- coding: utf-8 -*-
# @Time    : 2022/12/10
# @Author  : White Jiang
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    var1 = torch.randn(2, 4)

    print(var1[0])
    print(torch.topk(var1[0], 2))
    # soft = F.softmax(dim=-1)
    mask = F.softmax(var1, dim=-1)
    # var2 = torch.FloatTensor(1, 1024, 14, 14)
    # result = (var1 @ var2.transpose(-2, -1))
    print(var1)