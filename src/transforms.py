"""Domain-specific image transformation pipelines and normalization utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torchvision import transforms

from src.datasets import resolve_norm_stats


IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

AUG_MODE_ALIASES: dict[str, str] = {
    "none": "none",
    "geometric": "geometric",
    "color": "color",
    "colour": "color",
    "combined": "combined",
    "all": "combined",
    "both": "combined",
}


def canonicalize_aug_mode(aug_mode: str) -> str:
    """Normalize augmentation mode aliases to one of {'none', 'geometric', 'color', 'combined'}."""
    key = aug_mode.lower().strip()
    if key not in AUG_MODE_ALIASES:
        valid = ", ".join(sorted({"none", "geometric", "color", "combined"}))
        raise ValueError(f"Unsupported aug_mode {aug_mode!r}. Choose from: {valid}")
    return AUG_MODE_ALIASES[key]


def resolve_normalization_stats(
    dataset_name: str,
    normalization: str = "dataset",
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Resolve RGB normalization mean and std for either dataset stats or ImageNet stats."""
    if mean is not None and std is not None:
        return (
            (float(mean[0]), float(mean[1]), float(mean[2])),
            (float(std[0]), float(std[1]), float(std[2])),
        )
    norm_kind = normalization.lower().strip()
    if norm_kind == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if norm_kind == "dataset":
        return resolve_norm_stats(dataset_name)
    raise ValueError(f"Unsupported normalization mode: {normalization!r}. Use 'dataset' or 'imagenet'.")


def _domain_augmentations(dataset_name: str, aug_mode: str) -> list[object]:
    """Return domain-appropriate PIL augmentations prior to tensor conversion."""
    mode = canonicalize_aug_mode(aug_mode)
    if mode == "none":
        return []

    dataset_key = dataset_name.lower().strip()
    ops: list[object] = []

    if dataset_key == "skin_lesions":
        if mode in {"geometric", "combined"}:
            ops.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
        if mode in {"color", "combined"}:
            ops.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
        return ops

    if dataset_key == "facial_expressions":
        if mode in {"geometric", "combined"}:
            ops.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        if mode in {"color", "combined"}:
            ops.extend([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ])
        return ops

    raise ValueError(f"No augmentation policy defined for dataset {dataset_name!r}.")


def build_transforms(
    dataset_name: str,
    image_size: int = 64,
    aug_mode: str = "none",
    normalization: str = "dataset",
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Build deterministic evaluation transforms and domain-aware training transforms."""
    norm_mean, norm_std = resolve_normalization_stats(
        dataset_name=dataset_name,
        normalization=normalization,
        mean=mean,
        std=std,
    )

    train_ops: list[object] = [transforms.Resize((image_size, image_size))]
    train_ops.extend(_domain_augmentations(dataset_name=dataset_name, aug_mode=aug_mode))
    train_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=list(norm_mean), std=list(norm_std)),
    ])

    eval_ops: list[object] = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(norm_mean), std=list(norm_std)),
    ]

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def build_visualization_transform(
    dataset_name: str,
    image_size: int = 64,
    aug_mode: str = "combined",
) -> transforms.Compose:
    """Build a transform returning an unnormalized `[0, 1]` tensor for augmentation inspection."""
    ops: list[object] = [transforms.Resize((image_size, image_size))]
    ops.extend(_domain_augmentations(dataset_name=dataset_name, aug_mode=aug_mode))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """Convert a normalized `(C, H, W)` tensor into a clipped `(H, W, C)` numpy image in `[0, 1]`."""
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    std_arr = np.asarray(std, dtype=np.float32)
    mean_arr = np.asarray(mean, dtype=np.float32)
    img = img * std_arr + mean_arr
    return np.clip(img, 0.0, 1.0)

