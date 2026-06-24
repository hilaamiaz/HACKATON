import torch
import torch.nn as nn


class ModelArchitecture(nn.Module):
    def __init__(self):
        super(ModelArchitecture, self).__init__()

        # Convolutional Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.AdaptiveAvgPool2d((28, 28))
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 20)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x