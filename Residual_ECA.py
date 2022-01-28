from torch import nn
from torch.nn import functional as F
from math import log


class Residual_ECA(nn.Module):
    def __init__(self, input_channels, num_channels,
                 down_sample=False, strides=1, gamma=2, b=1):
        super().__init__()
        self.gamma = gamma
        self.b = b
        t = int(abs(log(num_channels, 2)+self.b)/self.gamma)
        k = t if t%2 else t+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

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

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y2 = self.avg_pool(Y)
        Y2 = Y2.squeeze(-1).transpose(-1, -2)
        Y2 = self.conv(Y2)
        Y2 = self.sigmoid(Y2)
        Y2 = Y2.transpose(-1, -2).unsqueeze(-1)
        Y = Y*Y2.expand_as(Y)

        X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual_ECA(input_channels, num_channels,
                                down_sample=True, strides=2))
        else:
            blk.append(Residual_ECA(num_channels, num_channels))
    return blk


class Residual101_ECA(nn.Module):
    def __init__(self, input_channels, num_channels,
                 strides=1, down_sample=False, expansion=4, gamma=2, b=1):
        super().__init__()
        self.expansion = expansion

        self.gamma = gamma
        self.b = b
        t = int(abs(log(num_channels, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

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

    def forward(self, X):
        out = self.block(X)
        out2 = self.avg_pool(out)
        out2 = out2.squeeze(-1).transpose(-1, -2)
        out2 = self.conv(out2)
        out2 = self.sigmoid(out2)
        out2 = out2.transpose(-1, -2).unsqueeze(-1)
        out = out * out2.expand_as(out)
        X = self.ds(X)
        out += X
        return F.relu(out)


def resnet_block101(input_channels, num_channels, block, stride=1, down_sample=1, expansion=4):
    blk = []
    blk.append(Residual101_ECA(input_channels, num_channels, stride, down_sample=1))
    for i in range(1, block):
        blk.append(Residual101_ECA(num_channels*4, num_channels))
    return nn.Sequential(*blk)
