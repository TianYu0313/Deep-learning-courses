import torch
from torch import nn
from Residual_CBAM_Pro import *


class ResNet18_CBAM_Pro(nn.Module):
    def __init__(self, classes=100):
        super(ResNet18_CBAM_Pro, self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
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
                                 nn.Linear(1024, classes)
                                 )

    def forward(self, x):
        return self.net(x)