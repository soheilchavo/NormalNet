import torch
from torch import nn

from Scripts.Models.classes import DoubleConv, UpConv, DownConv

#Simplified U-Net structure (in order to save on training time)
class UNet(torch.nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.initial_conv = DoubleConv(n_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)

        self.up1 = UpConv(256, 128)
        self.up2 = UpConv(128, 64)
        self.final_conv = nn.Conv2d(64, n_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.initial_conv(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.up1(x3, x2)
        x6 = self.up2(x4, x1)

        x7 = self.final_conv(x6)
        return x7