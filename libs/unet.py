import torch
from torch import nn
import torch.nn.functional as F

# Adapted from "Tunable U-Net implementation in PyTorch"
# https://github.com/jvanvugt/pytorch-unet

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=4,
        kernel_sz=3,
        up_mode='upsample'
    ):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        padding = kernel_sz//2
        self.depth = depth
        self.kernel_sz = kernel_sz
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, kernel_sz = self.kernel_sz)
            )
            prev_channels = 2 ** (wf + i)
        self.midconv = nn.Conv2d(prev_channels, prev_channels, kernel_size=self.kernel_sz, padding=padding, padding_mode='reflect') # kernel_size is originally 3 with padding = 1

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, kernel_sz = self.kernel_sz)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=self.kernel_sz,padding=padding, padding_mode='reflect') # kernel_size is originally 3 with padding = 1

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.midconv(x), negative_slope = 0.1)
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
