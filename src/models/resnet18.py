import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # Базовая ResNet18
        self.model = models.resnet18(weights=None)

        # Адаптация под 40x40
        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        setattr(self.model, 'maxpool', nn.Identity())

        self.model.layer4[1].bn2 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.2)
        )
    
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
