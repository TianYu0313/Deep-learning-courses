import torch
from torch import nn


class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.net = nn.Sequential(

            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(16*5*5, 1024), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 100)
                        )

    def forward(self, x):
        return self.net(x)