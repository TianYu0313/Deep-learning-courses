import torch
from torch import nn
from Residual_CBAM import *


class ResNet101_CBAM(nn.Module):
    def __init__(self, blocks, expansion=4):
        super(ResNet101_CBAM, self).__init__()
        self.expansion = expansion
        self.b1 =  nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.b2 = resnet_block101(64, 64, blocks[0], 1)
        self.b3 = resnet_block101(256, 128, blocks[1], 2)
        self.b4 = resnet_block101(512, 256, blocks[2], 2)
        self.b5 = resnet_block101(1024, 512, blocks[3], 2)
        self.net = nn.Sequential(self.b1,
                                 self.b2,
                                 self.b3,
                                 self.b4,
                                 self.b5,
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Linear(2048, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096, 1024),
                                 nn.Dropout(0.5),
                                 nn.Linear(1024, 100)
                                 )

    def forward(self, x):
        return self.net(x)
