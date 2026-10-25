import torch
import torch.nn as nn


class GhostModule(nn.Module):
    """Lightweight convolution module."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()

        if out_channels % 2 != 0:
            raise ValueError("out_channels must be an even number.")

        half_channels = out_channels // 2

        self.primary_conv = nn.Conv2d(
            in_channels,
            half_channels,
            kernel_size,
            stride,
            padding,
        )

        self.cheap_operation = nn.Conv2d(
            half_channels,
            half_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        return self.relu(torch.cat([x1, x2], dim=1))


class GhostFaceNetV2(nn.Module):
    """Custom Ghost-style CNN for face image classification."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            GhostModule(1, 16),
            nn.MaxPool2d(2, 2),

            GhostModule(16, 32),
            nn.MaxPool2d(2, 2),

            GhostModule(32, 64),
            nn.MaxPool2d(2, 2),
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(
            64 * 14 * 14,
            num_classes,
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.fc(x)