#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:41:58 2024

@author: ArthurC

Adapting the U-Net pytorch code from "Tunable U-Net implementation in PyTorch"
# https://github.com/jvanvugt/pytorch-unet to implement the tuning parameter 
described in"Tunable U-Net: Controlling Image-ti-Image Outputs Using a 
Tunable Scalar Value" https://ieeexplore.ieee.org/document/9481244.
"""

import torch
from torch import nn
import torch.nn.functional as F

class tUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=4,
        kernel_sz=3,
        input_size=(1536,608), # This sample size was set based on the GPU memory of the GEFORCE GTX TITAN X
        up_mode='upsample'
    ):
        super(tUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        padding = kernel_sz//2
        self.depth = depth
        self.kernel_sz = kernel_sz
        prev_channels = in_channels
        self.mid_size = tuple(d//(2**(self.depth-1)) for d in input_size)
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, kernel_sz = self.kernel_sz)
            )
            prev_channels = 2 ** (wf + i)
        self.midconv = nn.Conv2d(prev_channels, prev_channels-1, kernel_size=self.kernel_sz, padding=padding, padding_mode='reflect') # kernel_size is originally 3 with padding = 1
        self.mlp = TuningNetBlock(self.depth,self.mid_size)
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, kernel_sz = self.kernel_sz)
            )
            prev_channels = 2 ** (wf + i)
            
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=self.kernel_sz,padding=padding, padding_mode='reflect') # kernel_size is originally 3 with padding = 1

    def forward(self, x, pos):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.midconv(x), negative_slope = 0.1)
        x_t = self.mlp(x,pos)
        x = torch.cat((x,x_t),1)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, kernel_sz = 3):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=kernel_sz, padding=padding, padding_mode='reflect')) # kernel_size is originally 3 with padding = 1
        # block.append(nn.InstanceNorm2d(out_size)) # Testing instance_norm
        block.append(nn.LeakyReLU(0.1))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=kernel_sz, padding=padding, padding_mode='reflect')) # kernel_size is originally 3 with padding = 1
        # block.append(nn.InstanceNorm2d(out_size))
        block.append(nn.LeakyReLU(0.1))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class TuningNetBlock(nn.Module):
    def __init__(self, depth, mid_size):
        super(TuningNetBlock, self).__init__()
        self.layers = []
        self.output_size = mid_size
        self.depth = depth 
        self.mlp = []
        
        for i in range(8-self.depth):
            self.layers.append(nn.Linear(4**(i), 4**(i+1)))
        # Add output layer    
        self.layers.append(nn.Linear(4**(i+1),self.output_size[0]*self.output_size[1]))
        self.mlp = nn.Sequential(*self.layers)
        
    def forward(self, x, pos):
        out = self.mlp(pos)
        out = out.view(*self.output_size)
        return out.unsqueeze(0).unsqueeze(0)
    

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, kernel_sz = 3):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_sz, stride=2) # kernel_size is originally 2
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2,align_corners=False),# Align corners was declared here
                    nn.Conv2d(in_size, out_size, kernel_size=kernel_sz, padding=padding, padding_mode='reflect'), # kernel_size is originally 3 with padding = 1
                )
        self.conv_block = UNetConvBlock(in_size, out_size, padding, kernel_sz = kernel_sz)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]
    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat((up, crop1), 1)
        out = self.conv_block(out)
        return out
