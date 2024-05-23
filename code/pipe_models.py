import torch.nn as nn
import torch
import torchvision.models as models

class BinaryClassifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        r3d_model = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(r3d_model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return torch.sigmoid(x)


class AQAResNet18(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        r3d_model = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(r3d_model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


class C3DAQA(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc1 = nn.Linear(16384, 8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, 1)


        self.relu = nn.ReLU()
    
    def forward(self, x):
        features = []
        
        for frame in x:
            h = self.relu(self.conv1(frame))
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

            features.append(h)
        
        features = torch.stack(features, dim=0)
        features = features.flatten(start_dim=1, end_dim=2)

        h = self.relu(self.fc1(features))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))

        return h
