from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool,
    freeze_stages: int = 0,
) -> nn.Module:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "convnext_tiny":
        if freeze_stages != 0:
            raise ValueError("convnext_tiny currently supports freeze_stages=0")
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")

    model._rgel_architecture = name

    if freeze_stages < 0 or freeze_stages > 4:
        raise ValueError("freeze_stages must be in [0, 4]")
    if name.startswith("resnet") and freeze_stages >= 1:
        for module in (model.conv1, model.bn1, model.layer1):
            for parameter in module.parameters():
                parameter.requires_grad = False
    if name.startswith("resnet") and freeze_stages >= 2:
        for module in model.layer2:
            for parameter in module.parameters():
                parameter.requires_grad = False
    if name.startswith("resnet") and freeze_stages >= 3:
        for module in model.layer3:
            for parameter in module.parameters():
                parameter.requires_grad = False
    if name.startswith("resnet") and freeze_stages >= 4:
        for module in model.layer4:
            for parameter in module.parameters():
                parameter.requires_grad = False

    return model


def forward_features(model: nn.Module, images):
    """Return penultimate features and classifier logits."""
    architecture = getattr(model, "_rgel_architecture", None)
    if architecture is None and hasattr(model, "conv1") and hasattr(model, "layer4"):
        architecture = "resnet"

    if architecture in {"resnet18", "resnet50", "resnet"}:
        x = model.conv1(images)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        features = torch.flatten(x, 1)
        return features, model.fc(features)

    if architecture == "convnext_tiny":
        x = model.features(images)
        x = model.avgpool(x)
        x = model.classifier[0](x)
        features = model.classifier[1](x)
        return features, model.classifier[2](features)

    raise ValueError(f"Unsupported model architecture: {architecture}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
