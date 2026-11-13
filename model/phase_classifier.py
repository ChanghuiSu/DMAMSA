import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        # Extract latent arterial-phase branch features from the target image.
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, return_logits=False):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        if return_logits:
            return logits
        if self.training:
            return gumbel_softmax(logits)
        phase = F.one_hot(logits.argmax(dim=-1), self.num_classes).to(logits.dtype)
        return 1 - phase if self.num_classes == 2 else phase


def gumbel_softmax(logits):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
    y = logits + gumbel_noise
    y = F.softmax(y, dim=-1)

    y_hard = torch.zeros_like(y)
    y_hard.scatter_(1, torch.argmax(y, dim=1, keepdim=True), 1.0)
    y = (y_hard - y).detach() + y

    # Preserve the legacy binary branch ordering; return one-hot vectors directly for more branches.
    return 1 - y if logits.shape[-1] == 2 else y
