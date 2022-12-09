from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import math
from models.CSA import SA_FBSM, CA_SE
from models.CRFC import CRA_DOT
from models.resnet import ResNet_Backbone, Bottleneck, BasicBlock, conv1x1


# 设计的backbone只有三层 第四层用于获取局部和全局特征
def Backbone(pretrained=True, progress=True, **kwargs):
    # model = ResNet_Backbone(Bottleneck, [3, 4, 6], **kwargs)
    model = ResNet_Backbone(BasicBlock, [2, 2, 2], **kwargs)
    if pretrained:
        state_dict = torch.load('/home/jx/code/SiamNet/cub_ft.t')['model_state_dict']
        # print(state_dict.keys())
        for name in list(state_dict.keys()):
            if 'fc' in name or 'layer4' in name:
                state_dict.pop(name)
        for name in model.state_dict().keys():
            pretrained_name = 'pretrained_model.' + name
            model.state_dict()[name].copy_(state_dict[pretrained_name])
    print(model.state_dict())
    return model


class Refine(nn.Module):

    def __init__(self, block, layer, is_local=True, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(Refine, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 256
        self.dilation = 1

        self.is_local = is_local
        self.groups = groups
        self.base_width = width_per_group
        # self.attention = PosAttention(in_channels=1024, SA=True, CA=True)
        self.layer4 = self._make_layer(block, 512, layer, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer4(x)

        pool_x = self.avgpool(x)
        pool_x = torch.flatten(pool_x, 1)
        if self.is_local:
            return x, pool_x
        else:
            return pool_x

    def forward(self, x):
        return self._forward_impl(x)


def refine(is_local=True, pretrained=True, progress=True, **kwargs):
    # model = Refine(Bottleneck, 3, is_local, **kwargs)
    model = Refine(BasicBlock, 2, is_local, **kwargs)
    if pretrained:
        state_dict = torch.load('/home/jx/code/SiamNet/cub_ft.t')['model_state_dict']
        for name in list(state_dict.keys()):
            if not 'layer4' in name:
                state_dict.pop(name)
        for name in model.state_dict().keys():
            if 'layer4' in name:
                pretrained_name = 'pretrained_model.' + name
                model.state_dict()[name].copy_(state_dict[pretrained_name])
        # model.load_state_dict(state_dict, strict=False)
    return model

"""
Visual
"""


class SiamNet(nn.Module):
    def __init__(self, code_length=12, num_classes=200, att_size=3, feat_size=2048, device='cpu', pretrained=True):
        super(SiamNet, self).__init__()
        # 骨干网
        self.backbone = Backbone(pretrained=pretrained)
        # 用于生成全局特征。 最后只返回一个bs c h*w的特征 全局级别的转换网络
        self.refine_global = refine(is_local=False, pretrained=pretrained)
        # 用于生成局部特征 局部特征会返回未经池化的特征bs c h w和bs c h*w
        self.cls = nn.Linear(512, 200)
        # 哈希激活代码
        self.hash_layer_active = nn.Sequential(
            nn.Tanh(),
        )
        self.code_length = code_length
        # 12 24 32 48
        # global
        self.W_G = nn.Parameter(torch.Tensor(code_length, 512))
        torch.nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))
        # 伯努利分布
        self.bernoulli = torch.distributions.Bernoulli(0.5)
        self.device = device

    def forward(self, x):
        # 提取特征
        out = self.backbone(x)  # .detach()
        # 生成全局特征,基于第三层特征生成
        global_f = self.refine_global(out)
        # local_f1, avg_local_f1 = self.refine_local(out)

        # 特征映射到哈希码上
        deep_S_G = F.linear(global_f, self.W_G)
        # deep_S_1 = F.linear(avg_local_f1, self.W_L1)
        deep_S = torch.cat([deep_S_G], dim=1)
        # 哈希激活层
        ret = self.hash_layer_active(deep_S)
        if self.training:
            cls = self.cls(global_f)
            # cls1 = self.cls_loc(avg_local_f1)
            return ret, global_f, cls

        return ret, global_f


def siamnet(code_length, num_classes, att_size, feat_size, device, pretrained=False, **kwargs):
    # 实际att_size = 1
    model = SiamNet(code_length, num_classes, att_size, feat_size, device, pretrained, **kwargs)
    return model


if __name__ == '__main__':
    var1 = torch.FloatTensor(1, 1024, 14, 14)
    var2 = torch.FloatTensor(1, 1024, 14, 14)
    result = (var1 @ var2.transpose(-2, -1))
    print(result.shape)
