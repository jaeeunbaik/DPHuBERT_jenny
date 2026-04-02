#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import torch
from .resnet1d import (
    BasicBlock1D,
    ResNet1D,
)


class Conv1dResNet(torch.nn.Module):
    def __init__(self, relu_type="swish", a_upsample_ratio=1):
        super().__init__()
        self.a_upsample_ratio = a_upsample_ratio
        self.trunk = ResNet1D(
            BasicBlock1D,
            [2, 2, 2, 2],
            relu_type=relu_type,
            a_upsample_ratio=a_upsample_ratio,
        )

    def forward(self, xs_pad, length):  
        """forward.
        :param xs_pad: torch.Tensor, batch of padded input sequences (B, Tmax, idim)
        """# batch_size, 
        xs_pad = xs_pad.unsqueeze(dim=2)
        B, T, C = xs_pad.size()
        xs_pad = xs_pad[:, : T // 640 * 640, :]
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad, length = self.trunk(xs_pad)


        return xs_pad.transpose(1, 2), length

    def get_num_params_and_final_out_channels(self):
        return self.trunk.get_num_params_and_final_out_channels()