import torch
from torch.nn import functional as F
import argparse
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
import time
from typing import Union, NamedTuple
import os
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
import dataLoader as myData
from torchvision.models.video import r3d_18

class VideoClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(VideoClassificationModel, self).__init__()
        
        # 3D Convolutional Layers
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input: (batch_size, channels, frames, height, width)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Reshape before fully connected layers
        x = x.view(x.size(0), -1)
        
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class C3DVideoClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(C3DVideoClassificationModel, self).__init__()

        self.c3d = r3d_18(pretrained=True)
        in_features = self.c3d.fc.in_features
        self.c3d.fc = nn.Linear(in_features, num_classes)
    

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.c3d(x)

        return x


        
# Example usage:
num_classes = 2  # Replace with the actual number of classes
model = VideoClassificationModel(num_classes)
