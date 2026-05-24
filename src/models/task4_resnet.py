import torch.nn as nn
from torchvision import models


class MultiTaskResNet(nn.Module):
    def __init__(self, num_type_classes=61, num_severity_classes=3, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.resnet50(weights=weights)
        backbone_fc_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.type_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(backbone_fc_features, num_type_classes)
        )

        self.severity_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(backbone_fc_features, num_severity_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        type_output = self.type_head(features)
        severity_output = self.severity_head(features)
        return type_output, severity_output
