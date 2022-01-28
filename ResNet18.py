import torch
from torch import nn
from Residual import *


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.b1 =  nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.net = nn.Sequential(self.b1,
                                 self.b2,
                                 self.b3,
                                 self.b4,
                                 self.b5,
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Linear(512, 1024),
                                 nn.Dropout(0.5),
                                 nn.ReLU(),
                                 nn.Linear(1024, 100)
                                 )

    def forward(self, x):
        return self.net(x)