import torch
import torch.nn as nn
import torch.optim as optim


class ToyCNN(nn.Module):
    """The inner model that gets trained on re-weighted data."""

    def __init__(self):
        super(ToyCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=32*5*5, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        return self.layers(x)


class DataRater(nn.Module):
    """The outer model (meta-learner) that learns to rate data."""

    def __init__(self, temperature=1.0):
        super(DataRater, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.head = nn.Linear(400, 1)
        self.temperature = temperature

    def forward(self, x):
        features = self.layers(x)
        return self.head(features).squeeze(-1)


class ToyMLP(nn.Module):
    """Simple 2-layer MLP for regression tasks."""
    
    def __init__(self, input_dim=10, hidden_dim=64):
        super(ToyMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single output for regression
        )
    
    def forward(self, x):
        return self.layers(x).squeeze(-1)  # Remove last dimension to get shape (batch_size,)

class RegressionDataRater(nn.Module):
    """The outer model (meta-learner) that learns to rate regression data."""
    
    def __init__(self, input_dim=10, hidden_dim=64, temperature=1.0):
        super(RegressionDataRater, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Using Tanh like in the CNN version
            nn.Linear(hidden_dim, 1)
        )
        self.temperature = temperature
    
    def forward(self, x):
        return self.layers(x).squeeze(-1)  # Remove last dimension

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SimpleSegNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(SimpleSegNet, self).__init__()

        resnet = models.resnet18(pretrained=pretrained)
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4


        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.up4(e4) + e3
        d3 = self.up3(d4) + e2
        d2 = self.up2(d3) + e1
        d1 = self.up1(d2) + e0

        out = self.out_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

def construct_model(model_class):
    if model_class == 'unet':
      return SimpleSegNet
    if model_class == 'ToyCNN':
        return ToyCNN()
    elif model_class == 'DataRater':
        return DataRater()
    elif model_class == 'ToyMLP':
        return ToyMLP()
    elif model_class == 'RegressionDataRater':
        return RegressionDataRater()
    else:
        raise ValueError(f"Model {model_class} not found")