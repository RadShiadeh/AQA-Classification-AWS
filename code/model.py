import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
import numpy as np


class ClassifierCNN3D(nn.Module):
    def __init__(self, t_dim=16, img_x=256, img_y=256, fc_hidden=256, num_classes=2):
        super(ClassifierCNN3D, self).__init__()

        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        self.fc_hidden = fc_hidden
        self.num_classes = num_classes
        self.ch = 3
        self.kernel = (2, 2, 2)
        self.stride = (2, 2, 2)  # Adjusted stride for desired output shape
        self.padd = (0, 0, 0)

        self.conv1_out = self.conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.padd, (2, 2, 2), (2, 2, 2))
        self.conv2_out = self.conv3D_output_size(self.conv1_out, self.padd, self.kernel, self.stride)
        self.conv3_out = self.conv3D_output_size(self.conv2_out, self.padd, self.kernel, self.stride)
        self.conv4_out = self.conv3D_output_size(self.conv3_out, self.padd, self.kernel, self.stride)

        self.conv1 = nn.Conv3d(in_channels=self.ch, out_channels=self.ch, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        self.bn1 = nn.BatchNorm3d(self.ch)
        self.conv2 = nn.Conv3d(in_channels=self.ch, out_channels=self.ch, kernel_size=self.kernel, stride=self.stride, padding=self.padd)
        self.bn2 = nn.BatchNorm3d(self.ch)
        self.conv3 = nn.Conv3d(in_channels=self.ch, out_channels=self.ch, kernel_size=self.kernel, stride=self.stride, padding=self.padd)
        self.bn3 = nn.BatchNorm3d(self.ch)
        self.conv4 = nn.Conv3d(in_channels=self.ch, out_channels=self.ch, kernel_size=self.kernel, stride=self.stride, padding=self.padd)
        self.bn4 = nn.BatchNorm3d(self.ch)

        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(self.ch * self.conv4_out[0] * self.conv4_out[1] * self.conv4_out[2], self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, 1)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        #x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        #x = self.bn4(x)
        x = self.relu(x)

        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(x)
        
        return x
    
    @staticmethod
    def conv3D_output_size(input_size, padding, kernel_size, stride):
        output_size = [
            (input_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) // stride[0] + 1,
            (input_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) // stride[1] + 1,
            (input_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) // stride[2] + 1
        ]
        return output_size


class C3DC(nn.Module):
    def __init__(self):
        super(C3DC, self).__init__()
        self.pre_conv = nn.Conv3d(3, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pre_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))


        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))

        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.pre_conv(x))
        h = self.pre_pool(h)

        h = self.relu(self.conv1(h))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.reshape(h.size(0), -1)

        return h


class FullyConnected(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(8192, 4096)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return x


class ScoreRegressor(nn.Module):
    def __init__(self):
        super(ScoreRegressor, self).__init__()
        self.fc1 = nn.Linear(4096, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

class FeatureExtractionC3D(nn.Module):
    def __init__(self, num_classes=101):
        c3d_model = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(c3d_model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)

        return x

class FeatureExtractionRes3D(nn.Module):
    def __init__(self, num_classes = 400):
        super(FeatureExtractionRes3D, self).__init__()
        res3d_model = models.video.r3d_18()
        self.features = nn.Sequential(*list(res3d_model.children())[::-1])
        self.avgpool1 = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool1(x)
        x = x.reshape(x.size(0), -1)

        return x
    

#sample final look of endto end not sure if iys right
class EndToEndModel(nn.Module):
    def __init__(self, classifier, cnn, fully_connected, final_score_regressor):
        super(EndToEndModel, self).__init__()
        self.classifier = classifier
        self.final_score_regressor = final_score_regressor
        self.cnn = cnn
        self.fully_connected = fully_connected

    def forward(self, x):
        classification_output = self.classifier(x)

        features = []
        for frame in x:
            feature_out = self.cnn(frame)
            features.append(feature_out)

        features = torch.stack(features, dim=0)
        features = features.flatten(start_dim=1, end_dim=2)

        fc_out = self.fully_connected(features)
        final_score = self.final_score_regressor(fc_out)

        return {
            'classification': classification_output,
            'final_score': final_score
        }