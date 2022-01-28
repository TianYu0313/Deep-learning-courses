from torch import nn
from torch.nn import functional as F


class Residual_SE(nn.Module):
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

        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))

        self.se = nn.Sequential(
                            nn.Linear(in_features=num_channels, out_features=round(num_channels / 16)),
                            nn.ReLU(),
                            nn.Linear(in_features=round(num_channels / 16), out_features=num_channels),
                            nn.Sigmoid()
                                )

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y2 = self.globalAvgPool(Y)
        Y2 = Y2.view(Y2.size(0), -1)
        Y2 = self.se(Y2)
        Y2 = Y2.view(Y2.size(0), Y2.size(1), 1, 1)
        Y = Y2 * Y

        X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual_SE(input_channels, num_channels,
                                down_sample=True, strides=2))
        else:
            blk.append(Residual_SE(num_channels, num_channels))
    return blk


class Residual101_SE(nn.Module):
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

        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.se = nn.Sequential(
            nn.Linear(in_features=num_channels*self.expansion, out_features=round(num_channels*self.expansion / 16)),
            nn.ReLU(),
            nn.Linear(in_features=round(num_channels*self.expansion / 16), out_features=num_channels*self.expansion),
            nn.Sigmoid()
        )

    def forward(self, X):
        out = self.block(X)
        out2 = self.globalAvgPool(out)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.se(out2)
        out2 = out2.view(out2.size(0), out2.size(1), 1, 1)
        out = out2*out
        X = self.ds(X)
        out += X
        return F.relu(out)


def resnet_block101(input_channels, num_channels, block, stride=1, down_sample=1, expansion=4):
    blk = []
    blk.append(Residual101_SE(input_channels, num_channels, stride, down_sample=1))
    for i in range(1, block):
        blk.append(Residual101_SE(num_channels*expansion, num_channels))
    return nn.Sequential(*blk)
