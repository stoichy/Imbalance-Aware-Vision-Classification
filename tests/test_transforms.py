"""Unit tests for domain-specific augmentation transforms and normalization utilities."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
    canonicalize_aug_mode,
    denormalize_tensor,
)


def test_domain_specific_augmentation_policies() -> None:
    skin_geo, skin_eval = build_transforms("skin_lesions", image_size=64, aug_mode="geometric")
    skin_color, _ = build_transforms("skin_lesions", image_size=64, aug_mode="color")
    skin_comb, _ = build_transforms("skin_lesions", image_size=64, aug_mode="combined")

    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in skin_geo.transforms)
    assert any(isinstance(t, transforms.RandomVerticalFlip) for t in skin_geo.transforms)
    assert any(isinstance(t, transforms.ColorJitter) for t in skin_color.transforms)
    assert any(isinstance(t, transforms.RandomVerticalFlip) for t in skin_comb.transforms)
    assert any(isinstance(t, transforms.ColorJitter) for t in skin_comb.transforms)
    assert len(skin_eval.transforms) == 3  # Resize, ToTensor, Normalize

    cam_geo, _ = build_transforms("facial_expressions", image_size=64, aug_mode="geometric")
    cam_color, _ = build_transforms("facial_expressions", image_size=64, aug_mode="color")
    cam_comb, _ = build_transforms("facial_expressions", image_size=64, aug_mode="combined")

    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in cam_geo.transforms)
    assert any(isinstance(t, transforms.RandomRotation) for t in cam_geo.transforms)
    assert not any(isinstance(t, transforms.RandomVerticalFlip) for t in cam_comb.transforms)
    assert any(isinstance(t, transforms.ColorJitter) for t in cam_color.transforms)
    assert any(isinstance(t, transforms.GaussianBlur) for t in cam_color.transforms)


def test_eval_transform_is_deterministic_and_output_shapes_match() -> None:
    img = Image.new("RGB", (130, 100), color=(200, 120, 90))
    train_tf, eval_tf = build_transforms(
        "skin_lesions",
        image_size=224,
        aug_mode="combined",
        normalization="imagenet",
    )
    out1 = eval_tf(img)
    out2 = eval_tf(img)
    assert out1.shape == (3, 224, 224)
    assert torch.equal(out1, out2)

    out_train = train_tf(img)
    assert out_train.shape == (3, 224, 224)

    denorm = denormalize_tensor(out1, IMAGENET_MEAN, IMAGENET_STD)
    assert denorm.shape == (224, 224, 3)
    assert float(denorm.min()) >= 0.0 and float(denorm.max()) <= 1.0
    assert np.allclose(denorm[0, 0], np.array([200 / 255.0, 120 / 255.0, 90 / 255.0]), atol=1e-2)


def test_aug_mode_aliases() -> None:
    assert canonicalize_aug_mode("all") == "combined"
    assert canonicalize_aug_mode("both") == "combined"
    assert canonicalize_aug_mode("colour") == "color"

