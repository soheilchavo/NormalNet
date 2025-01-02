import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.conv(x)

class UpConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.transpose_conv = torch.nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)
    def forward(self, x, x2):
        x = self.transpose_conv(x)
        x = torch.cat([x, x2], dim=1)
        return self.double_conv(x)