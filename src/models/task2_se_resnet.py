import torch.nn as nn
from torchvision import models

from src.models.se_resnet import SEBasicBlock


def build_inference_model(num_classes=61):
    model = models.resnet18(weights=None)

    model.layer3 = nn.Sequential(
        SEBasicBlock(128, 256, stride=2, downsample=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        ), reduction=32),
        *[SEBasicBlock(256, 256, reduction=32) for _ in range(4)]
    )

    model.layer4 = nn.Sequential(
        SEBasicBlock(256, 512, stride=2, downsample=nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        ), reduction=32),
        *[SEBasicBlock(512, 512, reduction=32) for _ in range(2)]
    )

    model.fc = nn.Linear(512, num_classes)
    return model
