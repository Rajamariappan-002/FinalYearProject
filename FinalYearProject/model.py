import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Load pretrained ResNet18
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()  # Remove fully connected layer

        self.feature_extractor = resnet
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # Added BatchNorm
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Added BatchNorm
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        feat1 = self.feature_extractor(img1)
        feat2 = self.feature_extractor(img2)

        # Compute absolute difference
        diff = torch.abs(feat1 - feat2)
        output = self.fc(diff)
        return output
