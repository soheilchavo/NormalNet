import torch
from torch import nn

class DiscriminatorCNN(torch.nn.Module):
    def __init__(self, n_channels):
        super(DiscriminatorCNN, self).__init__()
        self.predict = nn.Sequential(

            nn.Conv2d(n_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128*64*64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x2 = self.predict(x).view(-1)
        return x2