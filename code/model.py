import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class CNN3D(nn.Module):
    def __init__(self, t_dim=300, img_x=480, img_y=480, fc_hidden=256, num_classes=2):
        super(CNN3D, self).__init__()

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



class ScoreRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(ScoreRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, out_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

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
    def __init__(self, classifier, final_score_regressor):
        super(EndToEndModel, self).__init__()
        self.classifier = classifier
        self.final_score_regressor = final_score_regressor

    def forward(self, x):
        classification_output = self.classifier(x)

        # Final Score Regression
        final_score = self.final_score_regressor(x)

        return {
            'classification': classification_output,
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