import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class ClassifierCNN3D(nn.Module):
    def __init__(self, t_dim=16, img_x=256, img_y=256, fc_hidden=256, num_classes=2):
        super(ClassifierCNN3D, self).__init__()

        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        self.fc_hidden = fc_hidden
        self.num_classes = num_classes
        self.ch = 3
        self.kernel = (3, 3, 3)
        self.stride = (2, 2, 2)
        self.padd = (0, 0, 0)

        self.conv1_out = self.conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.padd, self.kernel, self.stride)
        self.conv2_out = self.conv3D_output_size(self.conv1_out, self.padd, self.kernel, self.stride)

        self.conv1 = nn.Conv3d(in_channels=self.ch, out_channels=self.ch, kernel_size=self.kernel, stride=self.stride, padding=self.padd)
        self.bn = nn.BatchNorm3d(self.ch)

        self.conv2 = nn.Conv3d(in_channels=self.ch, out_channels=self.ch, kernel_size=self.kernel, stride=self.stride, padding=self.padd)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch * self.conv2_out[0] * self.conv2_out[1] * self.conv2_out[2], self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, 1)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(F.relu(self.fc2(x)))

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

        # Adjust the spatial dimensions accordingly
        h = h.view(h.size(0), -1)
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
        x = x.view(x.size(0), -1)

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
        x = x.view(x.size(0), -1)

        return x
    

#sample final look of endto end not sure if iys right
class EndToEndModel(nn.Module):
    def __init__(self, classifier, cnnLayer, fully_connected, final_score_regressor):
        super(EndToEndModel, self).__init__()
        self.classifier = classifier
        self.cnnLayer = cnnLayer
        self.fully_connected = fully_connected
        self.final_score_regressor = final_score_regressor

    def forward(self, x):
        classification_output = self.classifier(x)

        cnnLayer_out = self.cnnLayer(x)
        fully_connected_output = self.fully_connected(cnnLayer_out)
        final_score = self.final_score_regressor(fully_connected_output)

        return {
            'classification': torch.sigmoid(classification_output),  # Apply sigmoid activation
            'final_score': final_score
        }


# Class Definition CNN3D:
# CNN3D is a subclass of nn.Module, which is the base class for all neural network modules in PyTorch.
# Initialization:

# The __init__ method initializes the parameters and layers of the model.
# Parameters include dimensions (t_dim, img_x, img_y), dropout probability (drop_p), hidden layer sizes (fc1_hidden, fc2_hidden), and the number of output classes (num_classes).
# Convolutional layer parameters such as number of channels (ch1, ch2), kernel sizes (kernel1, kernel2), strides (stride1, stride2), and paddings (padd1, padd2) are also set.
# Convolutional Layers:

# Two convolutional layers (conv1, conv2) are defined with batch normalization (bn1, bn2) and ReLU activation (relu).
# Max pooling (pool) is applied after each convolutional layer.
# Linear Layers (Fully Connected):

# Three fully connected layers (fc1, fc2, fc3) are defined with ReLU activation in the first two.
# The final layer (fc3) outputs a single value after applying the sigmoid activation function.
# Forward Method:

# The forward method defines the forward pass of the network.
# Applies the convolutional layers, batch normalization, ReLU activation, and dropout.
# Reshapes the output and applies fully connected layers with ReLU activation and dropout.
# The final output is obtained by applying the sigmoid activation.
# Static Method:

# conv3D_output_size is a static method that computes the output size of a 3D convolutional layer given input size, padding, kernel size, and stride