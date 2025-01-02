import torch
from torch import nn

class UNet(torch.nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.predict = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Linear(64 * 4 * 4, 512),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.predict(x)