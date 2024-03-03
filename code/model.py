import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN3D(nn.Module):
    def __init__(self, t_dim=120, img_x=90, img_y=120, drop_p=0.2, fc1_hidden=256, fc2_hidden=256, num_classes=2):
        super(CNN3D, self).__init__()

        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        self.drop_p = drop_p
        self.fc1_hidden = fc1_hidden
        self.fc2_hidden = fc2_hidden
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.kernel1, self.kernel2 = (5,5,5), (3, 3, 3)
        self.stride1, self.stride2 = (2, 2, 2), (2, 2, 2)
        self.padd1, self.padd2 = (0, 0, 0), (0, 0, 0)

        self.conv1_out = self.conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.padd1, self.kernel1, self.stride1)
        self.conv2_out = self.conv3D_output_size(self.conv1_out, self.padd2, self.kernel2, self.stride2)

        self.conv1 = nn.Conv3d(in_channels=300, out_channels=self.ch1, kernel_size=self.kernel1, stride=self.stride1, padding=self.padd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)

        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.kernel2, stride=self.stride2, padding=self.padd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2 * self.conv2_out[0] * self.conv2_out[1] * self.conv2_out[2], self.fc1_hidden)
        self.fc2 = nn.Linear(self.fc1_hidden, self.fc2_hidden)
        self.fc3 = nn.Linear(self.fc2_hidden, 1)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = torch.sigmoid(self.fc3(x))

        return x

    
    @staticmethod
    def conv3D_output_size(input_size, padding, kernel_size, stride):
        output_size = [(input_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) // stride[0] + 1,
                       (input_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) // stride[1] + 1,
                       (input_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) // stride[2] + 1]
        return output_size