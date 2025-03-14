#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
from .resnet import BasicBlock, ResNet
from .convolution import Swish


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class Conv3dResNet(torch.nn.Module):
    """Conv3dResNet module"""

    def __init__(self, backbone_type="resnet", relu_type="swish"):
        """__init__.
        :param backbone_type: str, the type of a visual front-end.
        :param relu_type: str, activation function used in an audio front-end.
        """
        super(Conv3dResNet, self).__init__()
        self.frontend_nout = 64
        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1, self.frontend_nout, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
        )
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        self.proj = nn.Linear(512, 1024)

    def forward(self, xs_pad, length):
        xs_pad = xs_pad.transpose(1, 2)  # [B, T, C, H, W] -> [B, C, T, H, W]
        B, C, T, H, W = xs_pad.size()
        xs_pad = self.frontend3D(xs_pad)
        Tnew = xs_pad.shape[2]
        xs_pad = threeD_to_2D_tensor(xs_pad)
        # print(xs_pad.shape) # [592, 64, 22, 22]
        xs_pad, length = self.trunk(xs_pad, length)
        xs_pad = self.proj(xs_pad)
        return xs_pad.view(B, Tnew, xs_pad.size(1)), length

    def get_num_params_and_final_out_channels(self):
        return self.trunk.get_num_params_and_final_out_channels()