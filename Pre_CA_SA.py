from torch import nn
import torch
from math import *


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, sratio=16, gamma=2, b=1):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.gamma = gamma
        self.b = b

        t = int(abs(log(num_channels, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(2, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.sigmoid(self.conv(self.avg_pool(x).squeeze(-1).transpose(-1, -2)))
        # max_out = self.sigmoid(self.conv(self.max_pool(x).squeeze(-1).transpose(-1, -2)))
        avg_out = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        max_out = self.max_pool(x).squeeze(-1).transpose(-1, -2)
        avg_max = torch.cat([avg_out, max_out], dim=1)
        avg_max_out = self.conv(avg_max)
        return self.sigmoid(avg_max_out.transpose(-1, -2).unsqueeze(-1))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)