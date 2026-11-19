"""Unit tests for dataset parsing, stratified splitting, and DataLoader construction."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.datasets import (
    CSVImageDataset,
    InferenceImageDataset,
    build_dataloaders,
    create_stratified_split,
    load_dataset_splits,
    parse_csv_samples,
)
from src.transforms import build_transforms


def _make_dummy_image(path: Path, color: tuple[int, int, int] = (120, 80, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def test_parse_csv_samples_normalizes_paths_and_labels(tmp_path: Path) -> None:
    img_root = tmp_path / "imgs"
    _make_dummy_image(img_root / "sub" / "a.jpg")
    _make_dummy_image(img_root / "sub" / "b.jpg")
    csv_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "path": ["\\sub\\a.jpg", "sub/b.jpg"],
            "label": [1, 7],
        }
    ).to_csv(csv_path, index=False)

    samples, class_to_idx = parse_csv_samples(
        csv_path=csv_path,
        img_root=img_root,
        raw_labels=("1", "2", "3", "4", "5", "6", "7"),
        verify_exists=True,
    )
    assert len(samples) == 2
    assert samples[0] == (img_root / "sub" / "a.jpg", 0)
    assert samples[1] == (img_root / "sub" / "b.jpg", 6)
    assert class_to_idx["7"] == 6


def test_create_stratified_split_preserves_proportions_and_disjointness() -> None:
    # 5:1 imbalanced dataset (100 class-0, 20 class-1)
    labels = [0] * 100 + [1] * 20
    tr_idx, val_idx = create_stratified_split(labels, val_ratio=0.15, seed=42)

    assert len(set(tr_idx.tolist()) & set(val_idx.tolist())) == 0
    assert len(tr_idx) + len(val_idx) == len(labels)

    tr_counts = Counter(labels[i] for i in tr_idx)
    val_counts = Counter(labels[i] for i in val_idx)
    assert tr_counts[0] == 85 and tr_counts[1] == 17
    assert val_counts[0] == 15 and val_counts[1] == 3

    # Deterministic given same seed
    tr_idx2, val_idx2 = create_stratified_split(labels, val_ratio=0.15, seed=42)
    assert np.array_equal(tr_idx, tr_idx2)
    assert np.array_equal(val_idx, val_idx2)

    with pytest.raises(ValueError):
        create_stratified_split(labels, val_ratio=1.2)


def test_load_dataset_splits_no_test_leakage_and_independent_transforms(tmp_path: Path) -> None:
    root = tmp_path / "skin_lesions"
    classes = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    train_rows = []
    test_rows = []

    for c_idx, cls_name in enumerate(classes):
        for i in range(10):
            fname = f"{cls_name}_tr_{i}.jpg"
            _make_dummy_image(root / "train" / fname, color=(c_idx * 30, 50, 50))
            train_rows.append({"imagine": fname, "diagnostic": cls_name})
        for i in range(3):
            fname = f"{cls_name}_te_{i}.jpg"
            _make_dummy_image(root / "test" / fname, color=(c_idx * 30, 100, 100))
            test_rows.append({"imagine": fname, "diagnostic": cls_name})

    pd.DataFrame(train_rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(test_rows).to_csv(root / "test.csv", index=False)

    splits = load_dataset_splits(
        dataset_name="skin_lesions",
        data_root=root,
        val_ratio=0.2,
        seed=123,
        verify_exists=True,
    )
    assert len(splits.train_samples) == 56
    assert len(splits.val_samples) == 14
    assert len(splits.test_samples) == 21

    train_paths = {p for p, _ in splits.train_samples}
    val_paths = {p for p, _ in splits.val_samples}
    test_paths = {p for p, _ in splits.test_samples}
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    tr_tf, ev_tf = build_transforms("skin_lesions", image_size=64, aug_mode="combined")
    loaders = build_dataloaders(
        splits=splits,
        train_transform=tr_tf,
        eval_transform=ev_tf,
        batch_size=16,
        use_weighted_sampler=True,
        num_workers=0,
    )
    assert loaders["train_dataset"].transform is not loaders["val_dataset"].transform
    assert loaders["val_dataset"].transform is ev_tf
    x, y = next(iter(loaders["train_loader"]))
    assert x.shape == (16, 3, 64, 64)
    assert y.dtype == torch.int64


def test_inference_dataset_resolves_nested_images(tmp_path: Path) -> None:
    img_dir = tmp_path / "DATASET" / "test"
    _make_dummy_image(img_dir / "1" / "test_001.jpg")
    _make_dummy_image(img_dir / "4" / "test_002.jpg")
    csv_path = tmp_path / "remote_test.csv"
    pd.DataFrame({"id": ["test_001.jpg", "test_002.jpg"]}).to_csv(csv_path, index=False)

    _, ev_tf = build_transforms("facial_expressions", image_size=64, aug_mode="none")
    ds = InferenceImageDataset(csv_file=csv_path, img_dir=img_dir, transform=ev_tf)
    assert len(ds) == 2
    tensor0, sid0 = ds[0]
    assert sid0 == "test_001.jpg"
    assert tensor0.shape == (3, 64, 64)

