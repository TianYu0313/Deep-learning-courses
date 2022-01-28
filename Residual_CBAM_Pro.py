from torch import nn
from torch.nn import functional as F
from Pre_CA_SA import *
import matplotlib.pyplot as plt

class Residual_CBAM(nn.Module):
    def __init__(self, input_channels, num_channels,
                 down_sample=False, strides=1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if down_sample:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = nn.Sequential()
        self.ca = ChannelAttention(num_channels=num_channels)
        self.sa = SpatialAttention()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y21 = self.ca(Y)
        Y22 = self.sa(Y)
        Y = Y21 * Y
        Y = Y22 * Y

        X = self.conv3(X)
        Y += X
        # 查看特征图
        # plt.imshow(Y[0][0].cpu().detach().numpy())
        # plt.show()
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual_CBAM(input_channels, num_channels,
                                down_sample=True, strides=2))
        else:
            blk.append(Residual_CBAM(num_channels, num_channels))
    return blk


class Residual101_CBAM(nn.Module):
    def __init__(self, input_channels, num_channels,
                 strides=1, down_sample=False, expansion=4):
        super().__init__()
        self.expansion = expansion
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,out_channels=num_channels,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_channels*self.expansion),
        )
        if down_sample:
            self.ds = nn.Sequential(nn.Conv2d(input_channels, num_channels*self.expansion,
                                   kernel_size=1, stride=strides),
                                    nn.BatchNorm2d(num_channels*self.expansion))
        else:
            self.ds = nn.Sequential()

        self.ca = ChannelAttention(num_channels*self.expansion)
        self.sa = SpatialAttention()

    def forward(self, X):
        out = self.block(X)
        out21 = self.ca(out)
        out22 = self.sa(out)
        out = out21 * out
        out = out22 * out
        X = self.ds(X)
        out += X
        return F.relu(out)


def resnet_block101(input_channels, num_channels, block, stride=1, down_sample=1, expansion=4):
    blk = []
    blk.append(Residual101_CBAM(input_channels, num_channels, stride, down_sample=1))
    for i in range(1, block):
        blk.append(Residual101_CBAM(num_channels*4, num_channels))
    return nn.Sequential(*blk)
