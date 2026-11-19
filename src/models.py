"""Neural architectures, BatchNorm freezing, and differential parameter-group builders."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torchvision.models as tv_models


class SimpleMLP(nn.Module):
    """Two-layer MLP baseline operating on flattened RGB images."""

    def __init__(
        self,
        num_classes: int = 7,
        image_size: int = 64,
        in_channels: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = in_channels * image_size * image_size
        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNN(nn.Module):
    """Four-block custom CNN with configurable BatchNorm2d and AdaptiveAvgPool2d."""

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256),
        hidden_dim: int = 512,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        use_adaptive_pool: bool = False,
        first_kernel_size: int = 3,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if len(channels) > 4:
            raise ValueError(f"SimpleCNN supports at most 4 convolutional layers, got {len(channels)}.")
        if first_kernel_size % 2 == 0 or first_kernel_size < 1:
            raise ValueError(f"first_kernel_size must be a positive odd integer, got {first_kernel_size}.")

        self.use_batchnorm = use_batchnorm
        self.use_adaptive_pool = use_adaptive_pool

        blocks: list[nn.Module] = []
        prev_c = in_channels
        for idx, out_c in enumerate(channels):
            k = first_kernel_size if idx == 0 else 3
            pad = k // 2
            blocks.append(
                nn.Conv2d(
                    prev_c,
                    out_c,
                    kernel_size=k,
                    padding=pad,
                    bias=not use_batchnorm,
                )
            )
            if use_batchnorm:
                blocks.append(nn.BatchNorm2d(out_c))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool2d(kernel_size=2))
            prev_c = out_c

        if use_adaptive_pool:
            blocks.append(nn.AdaptiveAvgPool2d((1, 1)))
            flat_dim = prev_c
        else:
            divisor = 2 ** len(channels)
            if image_size % divisor != 0 or image_size < divisor:
                raise ValueError(
                    f"image_size={image_size} is incompatible with {len(channels)} MaxPool2d(2) layers "
                    "when use_adaptive_pool=False."
                )
            spatial = image_size // divisor
            flat_dim = prev_c * spatial * spatial

        self.features = nn.Sequential(*blocks)

        classifier_layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            classifier_layers.append(nn.Dropout(p=dropout))
        classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)


def freeze_bn(model: nn.Module) -> None:
    """Freeze all BatchNorm layers in eval mode and disable gradient updates for affine parameters."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def build_resnet18(
    num_classes: int = 7,
    pretrained: bool = True,
    freeze_bn_stats: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a ResNet-18 classifier with optional ImageNet weights and frozen BatchNorm statistics."""
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    if freeze_bn_stats:
        freeze_bn(model)

    return model


def build_model(
    model_name: str,
    num_classes: int = 7,
    image_size: int = 64,
    dropout: float = 0.3,
    hidden_dim: int = 512,
    use_batchnorm: bool = True,
    use_adaptive_pool: bool = False,
    first_kernel_size: int = 3,
    pretrained: bool = True,
    freeze_bn_stats: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Factory function constructing `mlp`, `cnn`, or `resnet18` models."""
    key = model_name.lower().strip()
    if key in {"mlp", "simple_mlp"}:
        return SimpleMLP(
            num_classes=num_classes,
            image_size=image_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if key in {"cnn", "simple_cnn"}:
        return SimpleCNN(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            use_adaptive_pool=use_adaptive_pool,
            first_kernel_size=first_kernel_size,
            image_size=image_size,
        )
    if key in {"resnet18", "resnet-18"}:
        return build_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_bn_stats=freeze_bn_stats,
            freeze_backbone=freeze_backbone,
        )
    raise ValueError(f"Unsupported model {model_name!r}. Choose from 'mlp', 'cnn', or 'resnet18'.")


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_name: str = "adam",
    backbone_lr: float | None = None,
    head_lr: float | None = None,
) -> torch.optim.Optimizer:
    """Create an optimizer over trainable parameters only (`requires_grad=True`).

    When both `backbone_lr` and `head_lr` are provided (e.g., for ResNet-18 fine-tuning),
    parameters are partitioned into non-`fc.*` backbone parameters and `fc.*` head parameters,
    while cleanly excluding any frozen parameters (such as frozen BatchNorm affine weights).
    """
    opt_key = optimizer_name.lower().strip()

    if backbone_lr is not None and head_lr is not None:
        backbone_params = [
            p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc.")
        ]
        head_params = [
            p for n, p in model.named_parameters() if p.requires_grad and n.startswith("fc.")
        ]
        param_groups: list[dict[str, Any]] = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr, "name": "head"})
        if not param_groups:
            raise ValueError("No trainable parameters found when building optimizer.")
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found when building optimizer.")
        param_groups = [{"params": trainable_params, "lr": lr, "name": "all"}]

    if opt_key == "adam":
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    if opt_key == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    if opt_key == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer {optimizer_name!r}. Choose 'adam', 'adamw', or 'sgd'.")

