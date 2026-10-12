"""Vanilla CNN baseline for MNIST: two conv blocks (32 -> 64) + classifier head."""
import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Block 1: 1x28x28 (1 meaning greyscale, single brightness per image) -> 32x28x28 -> 32x14x14
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Block 2: 32x14x14 -> 64x14x14 -> 64x7x7
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Head: 64x7x7 -> num_classes logits
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(64 * 7 * 7), out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28) -> returns logits (batch, num_classes)
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)
        
if __name__ == "__main__":
    # Quick shape + parameter check: python -m models.cnn (from baselines/)
    model = CNN()
    x = torch.randn(8, 1, 28, 28)
    print(model(x).shape)  # expect torch.Size([8, 10])
    print(sum(p.numel() for p in model.parameters()))
