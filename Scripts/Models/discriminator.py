import torch
from torch import nn

class DiscriminatorCNN(torch.nn.Module):
    def __init__(self, n_channels):
        super(DiscriminatorCNN, self).__init__()
        self.predict = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),  # No BatchNorm for the first layer

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(169, 1)
        )

    def forward(self, x):
        x2 = self.predict(x).view(-1)
        return x2