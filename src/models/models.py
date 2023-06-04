import torch
import torch.nn as nn
from torchvision import models


class VGG16:
    def __init__(self, freeze_conv=True):
        model = models.vgg16(pretrained=True)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 10)  # Add new fully connected layer
        if freeze_conv:
            # Freeze pre-trained layers
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

    def save(self, path2save):
        torch.save(self.model.state_dict(), path2save)

    def load(self, path2model):
        self.model.load_state_dict(torch.load(path2model))
        self.model.eval()
        return self.model


class ResNet18:
    def __init__(self, freeze_conv=True):
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)  # Add new fully connected layer
        if freeze_conv:
            # Freeze pre-trained layers
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

    def save(self, path2save):
        torch.save(self.model.state_dict(), path2save)

    def load(self, path2model):
        self.model.load_state_dict(torch.load(path2model))
        self.model.eval()
        return self.model