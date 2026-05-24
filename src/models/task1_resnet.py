import torch.nn as nn
from torchvision import models


def build_model(num_classes=61, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total params: {total_params / 1e6:.2f}M")
    return model
