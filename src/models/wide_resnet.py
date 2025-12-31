import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=4, num_classes=200, dropout_rate=0.3):
        super().__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, "Depth must be 6n+4"
        n_blocks = (depth - 4) // 6  # 4 blocks per group for WRN-28
        
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = self._make_layer(n_channels[0], n_channels[1], n_blocks, stride=1, dropout_rate=dropout_rate)
        self.block2 = self._make_layer(n_channels[1], n_channels[2], n_blocks, stride=2, dropout_rate=dropout_rate)
        self.block3 = self._make_layer(n_channels[2], n_channels[3], n_blocks, stride=2, dropout_rate=dropout_rate)
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels[3], num_classes)
    
    def _make_layer(self, in_channels, out_channels, n_blocks, stride, dropout_rate):
        layers = [BasicBlock(in_channels, out_channels, stride, dropout_rate)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, dropout_rate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out