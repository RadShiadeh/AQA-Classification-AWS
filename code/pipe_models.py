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


class OverHeadPressAQA(nn.Module):
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


class BarbellSquatsAQA(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        r3d_model = models.video.r3d_18(pretrained=True)
        self.features = nn.Sequential(*list(r3d_model.children()))[:-1]
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x