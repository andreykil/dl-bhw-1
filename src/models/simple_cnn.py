import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Простая CNN (Baseline).
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Сверточные слои
        self.features = nn.Sequential(
            # 3 x 40 x 40 -> 16 x 40 x 40
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 16 x 20 x 20

            # 16 x 20 x 20 -> 32 x 20 x 20
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 32 x 10 x 10

            # 32 x 10 x 10 -> 64 x 10 x 10
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 64 x 5 x 5
        )

        # Полносвязные слои
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
