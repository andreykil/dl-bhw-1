import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 1)  # Добавил max(..., 1) для безопасности, если channels малы
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.view(y.size(0), y.size(1))  # (B, C) — критично: size(1) = C, не -1 или size(2)!
        y = self.fc(y)
        return x * y.view(y.size(0), y.size(1), 1, 1)

class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, alpha=1.0):
        super().__init__()
        mid_channels = max(int(out_channels // self.expansion * alpha), 1)  # min 1 для избежания 0 channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        return F.relu(out + self.shortcut(x))

class CustomEfficientPyramid(nn.Module):
    def __init__(self, num_classes=200, depth=110, alpha=1.2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        layers_per_stage = (depth - 2) // 3
        channels = [64, 128, 256]  # Убрал лишний 512, т.к. только 3 слоя; это не влияло, но для чистоты
        self.layer1 = self._make_layer(64, channels[0], layers_per_stage, stride=1, alpha=alpha)
        self.layer2 = self._make_layer(channels[0] * BottleneckBlock.expansion, channels[1], layers_per_stage, stride=2, alpha=alpha)
        self.layer3 = self._make_layer(channels[1] * BottleneckBlock.expansion, channels[2], layers_per_stage, stride=2, alpha=alpha)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[2] * BottleneckBlock.expansion, num_classes)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, alpha):
        layers = [BottleneckBlock(in_channels, out_channels * BottleneckBlock.expansion, stride, alpha)]
        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(out_channels * BottleneckBlock.expansion, out_channels * BottleneckBlock.expansion, alpha=alpha))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        return self.fc(out)