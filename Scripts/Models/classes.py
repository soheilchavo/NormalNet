import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        x2 = self.conv(x)
        return x2

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2,2)
        )
    def forward(self, x):
        x2 = self.conv(x)
        return self.conv(x2)

class UpConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.transpose_conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(out_channels*2, out_channels)
    def forward(self, x, x2):
        x3 = self.transpose_conv(x)
        x4 = torch.cat([x3, x2], dim=1)
        return self.double_conv(x4)