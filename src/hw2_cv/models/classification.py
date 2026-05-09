from __future__ import annotations

from torch import nn
from torchvision import models

from .attention import AttentionWrapper, CBAM, SEBlock


def _replace_resnet_head(model: nn.Module, num_classes: int) -> nn.Module:
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _replace_swin_head(model: nn.Module, num_classes: int) -> nn.Module:
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def _wrap_resnet_attention(model: nn.Module, attention: str) -> nn.Module:
    if attention in ("none", "", None):
        return model
    attention = attention.lower()
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for idx, block in enumerate(layer):
            channels = block.conv2.out_channels
            if attention == "se":
                attn = SEBlock(channels)
            elif attention == "cbam":
                attn = CBAM(channels)
            else:
                raise ValueError(f"Unsupported attention: {attention}")
            layer[idx] = AttentionWrapper(block, attn)
    return model


def create_classifier(
    name: str,
    num_classes: int = 37,
    pretrained: bool = True,
    attention: str | None = None,
) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model = _wrap_resnet_attention(model, attention or "none")
        return _replace_resnet_head(model, num_classes)
    if name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        model = _wrap_resnet_attention(model, attention or "none")
        return _replace_resnet_head(model, num_classes)
    if name == "swin_t":
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.swin_t(weights=weights)
        return _replace_swin_head(model, num_classes)
    if name in {"vit_tiny", "vit_tiny_patch16_224"}:
        try:
            import timm
        except ImportError as exc:
            raise ImportError("ViT-Tiny requires timm. Install requirements.txt first.") from exc
        return timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unsupported classifier: {name}")

