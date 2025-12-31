import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Pre-activation WideResNet block:
    BN -> ReLU -> Conv -> Dropout -> BN -> ReLU -> Conv
    """
    def __init__(self, in_channels, out_channels, stride, dropout_rate):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        # shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=stride, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    """
    WideResNet.
    Глубина должна быть 6n + 4 (28, 40).
    """
    def __init__(self, depth=28, widen_factor=8, num_classes=200, dropout_rate=0.1):
        super().__init__()

        assert (depth - 4) % 6 == 0, "Depth must be 6n + 4"
        n_blocks = (depth - 4) // 6

        widths = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        ]

        # Initial conv
        self.conv1 = nn.Conv2d(
            3, widths[0],
            kernel_size=3, stride=1, padding=1, bias=False
        )

        # Stages
        self.stage1 = self._make_stage(
            widths[0], widths[1], n_blocks, stride=1, dropout_rate=dropout_rate
        )
        self.stage2 = self._make_stage(
            widths[1], widths[2], n_blocks, stride=2, dropout_rate=dropout_rate
        )
        self.stage3 = self._make_stage(
            widths[2], widths[3], n_blocks, stride=2, dropout_rate=dropout_rate
        )

        # Head
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)

        self._init_weights()

    def _make_stage(self, in_channels, out_channels, n_blocks, stride, dropout_rate):
        blocks = []
        blocks.append(BasicBlock(in_channels, out_channels, stride, dropout_rate))
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock(out_channels, out_channels, 1, dropout_rate))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)
