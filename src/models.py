"""CNN-only dengan jumlah fitur input yang dapat dipilih."""
import torch.nn as nn

class CNNOnly(nn.Module):
    def __init__(self, in_features=90, num_classes=317):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x.transpose(1, 2)))
