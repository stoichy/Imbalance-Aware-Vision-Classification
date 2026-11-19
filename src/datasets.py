"""Dataset loading, stratified train/val splitting, and DataLoader construction."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata describing a supported vision classification dataset."""

    name: str
    default_root: str
    train_csv: str
    train_img_subdir: str
    test_csv: str
    test_img_subdir: str
    inference_csv: str
    inference_img_subdir: str
    raw_labels: tuple[str, ...]
    class_names: tuple[str, ...]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    norm_stats_file: str | None = None
    output_prediction_as_idx: bool = False


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "skin_lesions": DatasetSpec(
        name="skin_lesions",
        default_root="data/skin_lesions",
        train_csv="train.csv",
        train_img_subdir="train",
        test_csv="test.csv",
        test_img_subdir="test",
        inference_csv="kaggle_test.csv",
        inference_img_subdir="kaggle_test",
        raw_labels=("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"),
        class_names=("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"),
        mean=(0.7646, 0.5464, 0.5712),
        std=(0.1410, 0.1532, 0.1706),
        norm_stats_file=None,
        output_prediction_as_idx=True,
    ),
    "facial_expressions": DatasetSpec(
        name="facial_expressions",
        default_root="data/facial_expressions",
        train_csv="splits/train.csv",
        train_img_subdir="",
        test_csv="splits/local_test.csv",
        test_img_subdir="",
        inference_csv="splits/remote_test.csv",
        inference_img_subdir="DATASET/test",
        raw_labels=("1", "2", "3", "4", "5", "6", "7"),
        class_names=("Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"),
        mean=(0.5752126091262171, 0.4497227663589234, 0.4012869135370431),
        std=(0.26532880449272844, 0.24229550144470252, 0.24068486736909714),
        norm_stats_file="splits/norm_stats.json",
        output_prediction_as_idx=False,
    ),
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    """Look up a dataset specification by canonical name."""
    key = dataset_name.lower().strip()
    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset {dataset_name!r}. Available: {available}")
    return DATASET_REGISTRY[key]


def resolve_norm_stats(
    dataset_name: str,
    data_root: str | Path | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Load RGB mean/std from `norm_stats.json` when available, otherwise use spec defaults."""
    spec = get_dataset_spec(dataset_name)
    if data_root is not None and spec.norm_stats_file:
        stats_path = Path(data_root) / spec.norm_stats_file
        if stats_path.exists():
            with stats_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            mean = tuple(float(x) for x in payload["mean"])
            std = tuple(float(x) for x in payload["std"])
            if len(mean) == 3 and len(std) == 3:
                return mean, std  # type: ignore[return-value]
    return spec.mean, spec.std


def _normalize_rel_path(raw_path: Any) -> str:
    cleaned = str(raw_path).strip().replace("\\", "/")
    while cleaned.startswith("/"):
        cleaned = cleaned[1:]
    return cleaned


def parse_csv_samples(
    csv_path: str | Path,
    img_root: str | Path,
    raw_labels: Sequence[str] | None = None,
    verify_exists: bool = False,
) -> tuple[list[tuple[Path, int]], dict[str, int]]:
    """Parse a two-column `(image_path, label)` CSV into `(Path, class_idx)` pairs."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    if df.empty or df.shape[1] < 2:
        raise ValueError(f"Expected non-empty CSV with >= 2 columns at {csv_file}")

    label_strings = [str(v).strip() for v in df.iloc[:, 1].tolist()]
    if raw_labels is None:
        ordered_labels = sorted(set(label_strings))
    else:
        ordered_labels = [str(lbl).strip() for lbl in raw_labels]

    class_to_idx = {label: idx for idx, label in enumerate(ordered_labels)}
    unknown = sorted(set(label_strings) - set(class_to_idx))
    if unknown:
        raise ValueError(f"Unexpected labels {unknown} in {csv_file}; expected {ordered_labels}")

    root = Path(img_root)
    samples: list[tuple[Path, int]] = []
    for raw_rel, lbl in zip(df.iloc[:, 0].tolist(), label_strings):
        rel = _normalize_rel_path(raw_rel)
        full_path = root / rel
        if verify_exists and not full_path.exists():
            raise FileNotFoundError(f"Image listed in {csv_file} does not exist: {full_path}")
        samples.append((full_path, class_to_idx[lbl]))

    return samples, class_to_idx


class CSVImageDataset(Dataset):
    """PyTorch Dataset backed by explicit `(image_path, label_idx)` pairs or a CSV manifest."""

    def __init__(
        self,
        samples: Sequence[tuple[Path | str, int]] | None = None,
        class_to_idx: dict[str, int] | None = None,
        class_names: Sequence[str] | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        csv_file: str | Path | None = None,
        root_dir: str | Path | None = None,
        raw_labels: Sequence[str] | None = None,
        verify_exists: bool = False,
    ) -> None:
        if samples is not None:
            self.samples: list[tuple[Path, int]] = [(Path(p), int(y)) for p, y in samples]
            if class_to_idx is None:
                unique_labels = sorted({y for _, y in self.samples})
                self.class_to_idx = {str(y): y for y in unique_labels}
            else:
                self.class_to_idx = dict(class_to_idx)
        elif csv_file is not None and root_dir is not None:
            self.samples, self.class_to_idx = parse_csv_samples(
                csv_path=csv_file,
                img_root=root_dir,
                raw_labels=raw_labels,
                verify_exists=verify_exists,
            )
        else:
            raise ValueError("Provide either `samples` or both `csv_file` and `root_dir`.")

        self.transform = transform
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        if class_names is not None:
            self.class_names = list(class_names)
        else:
            self.class_names = [self.idx_to_class[i] for i in sorted(self.idx_to_class)]
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
        if self.transform is not None:
            output = self.transform(rgb_img)
        else:
            output = rgb_img
        return output, label


class InferenceImageDataset(Dataset):
    """Dataset for unlabeled test manifests (`kaggle_test.csv`, `remote_test.csv`)."""

    def __init__(
        self,
        csv_file: str | Path,
        img_dir: str | Path,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"Inference CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f"Inference CSV is empty: {csv_path}")

        img_root = Path(img_dir)
        if not img_root.exists():
            raise FileNotFoundError(f"Inference image directory not found: {img_root}")

        indexed_by_name: dict[str, Path] = {}
        for root, _, files in os.walk(img_root):
            for fname in files:
                indexed_by_name[fname] = Path(root) / fname

        self.items: list[tuple[str, Path]] = []
        for raw_id in df.iloc[:, 0].tolist():
            sample_id = str(raw_id).strip()
            rel = _normalize_rel_path(sample_id)
            direct_path = img_root / rel
            if direct_path.exists():
                resolved = direct_path
            else:
                base_name = os.path.basename(rel)
                if base_name not in indexed_by_name:
                    raise FileNotFoundError(
                        f"Could not locate inference image {sample_id!r} under {img_root}"
                    )
                resolved = indexed_by_name[base_name]
            self.items.append((sample_id, resolved))

        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[Any, str]:
        sample_id, path = self.items[idx]
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
        output = self.transform(rgb_img) if self.transform is not None else rgb_img
        return output, sample_id


def create_stratified_split(
    labels: Sequence[int],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic, non-overlapping stratified train and validation index arrays."""
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must lie strictly in (0, 1), got {val_ratio}")

    labels_arr = np.asarray(labels, dtype=np.int64)
    if len(labels_arr) < 2:
        raise ValueError("Need at least 2 samples to create a train/val split.")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    indices = np.arange(len(labels_arr))
    train_idx, val_idx = next(splitter.split(indices, labels_arr))
    return train_idx, val_idx


def _subsample_stratified(
    samples: list[tuple[Path, int]],
    max_samples: int | None,
    seed: int,
) -> list[tuple[Path, int]]:
    """Deterministically subsample `(path, label)` pairs while preserving class stratification."""
    if max_samples is None or max_samples >= len(samples):
        return samples
    labels = np.asarray([y for _, y in samples], dtype=np.int64)
    num_classes = len(np.unique(labels))
    if max_samples < num_classes:
        raise ValueError(
            f"max_samples ({max_samples}) must be >= number of classes ({num_classes})."
        )
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
    subset_idx, _ = next(splitter.split(np.arange(len(samples)), labels))
    return [samples[int(i)] for i in subset_idx]


@dataclass
class DatasetSplits:
    """Container holding stratified train/val splits and held-out test samples."""

    dataset_name: str
    data_root: Path
    train_samples: list[tuple[Path, int]]
    val_samples: list[tuple[Path, int]]
    test_samples: list[tuple[Path, int]]
    raw_labels: tuple[str, ...]
    class_names: tuple[str, ...]
    class_to_idx: dict[str, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def train_class_counts(self) -> np.ndarray:
        labels = np.asarray([y for _, y in self.train_samples], dtype=np.int64)
        return np.bincount(labels, minlength=self.num_classes)

    @property
    def val_class_counts(self) -> np.ndarray:
        labels = np.asarray([y for _, y in self.val_samples], dtype=np.int64)
        return np.bincount(labels, minlength=self.num_classes)

    @property
    def test_class_counts(self) -> np.ndarray:
        labels = np.asarray([y for _, y in self.test_samples], dtype=np.int64)
        return np.bincount(labels, minlength=self.num_classes)


def load_dataset_splits(
    dataset_name: str,
    data_root: str | Path | None = None,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    verify_exists: bool = False,
) -> DatasetSplits:
    """Load training manifest, create a stratified train/val split, and load held-out test set.

    Validation samples are drawn strictly from the training manifest; test splits remain
    held out until final evaluation.
    """
    spec = get_dataset_spec(dataset_name)
    root = Path(data_root) if data_root is not None else Path(spec.default_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root for {dataset_name!r} does not exist: {root}. "
            "Pass --data-root or set dataset root in config."
        )

    train_img_root = root / spec.train_img_subdir if spec.train_img_subdir else root
    test_img_root = root / spec.test_img_subdir if spec.test_img_subdir else root

    all_train_samples, class_to_idx = parse_csv_samples(
        csv_path=root / spec.train_csv,
        img_root=train_img_root,
        raw_labels=spec.raw_labels,
        verify_exists=verify_exists,
    )
    test_samples, _ = parse_csv_samples(
        csv_path=root / spec.test_csv,
        img_root=test_img_root,
        raw_labels=spec.raw_labels,
        verify_exists=verify_exists,
    )

    if max_train_samples is not None:
        all_train_samples = _subsample_stratified(all_train_samples, max_train_samples, seed=seed)
    if max_test_samples is not None:
        test_samples = _subsample_stratified(test_samples, max_test_samples, seed=seed)

    train_labels = [y for _, y in all_train_samples]
    train_idx, val_idx = create_stratified_split(train_labels, val_ratio=val_ratio, seed=seed)

    train_samples = [all_train_samples[int(i)] for i in train_idx]
    val_samples = [all_train_samples[int(i)] for i in val_idx]
    mean, std = resolve_norm_stats(dataset_name, root)

    return DatasetSplits(
        dataset_name=spec.name,
        data_root=root,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        raw_labels=spec.raw_labels,
        class_names=spec.class_names,
        class_to_idx=class_to_idx,
        mean=mean,
        std=std,
    )


def build_weighted_sampler(labels: Sequence[int], num_classes: int) -> WeightedRandomSampler:
    """Create an inverse-frequency `WeightedRandomSampler` for class-balanced batches."""
    labels_arr = np.asarray(labels, dtype=np.int64)
    class_counts = np.bincount(labels_arr, minlength=num_classes)
    inv_freq = np.where(class_counts > 0, 1.0 / np.maximum(class_counts, 1), 0.0)
    sample_weights = inv_freq[labels_arr]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(
    splits: DatasetSplits,
    train_transform: Callable[[Image.Image], torch.Tensor],
    eval_transform: Callable[[Image.Image], torch.Tensor],
    batch_size: int = 64,
    use_weighted_sampler: bool = False,
    num_workers: int = 2,
    pin_memory: bool = False,
    drop_last_if_singleton: bool = True,
) -> dict[str, Any]:
    """Construct independent Dataset and DataLoader objects for train, val, and held-out test."""
    train_dataset = CSVImageDataset(
        samples=splits.train_samples,
        class_to_idx=splits.class_to_idx,
        class_names=splits.class_names,
        transform=train_transform,
    )
    train_eval_dataset = CSVImageDataset(
        samples=splits.train_samples,
        class_to_idx=splits.class_to_idx,
        class_names=splits.class_names,
        transform=eval_transform,
    )
    val_dataset = CSVImageDataset(
        samples=splits.val_samples,
        class_to_idx=splits.class_to_idx,
        class_names=splits.class_names,
        transform=eval_transform,
    )
    test_dataset = CSVImageDataset(
        samples=splits.test_samples,
        class_to_idx=splits.class_to_idx,
        class_names=splits.class_names,
        transform=eval_transform,
    )

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sampler = build_weighted_sampler(train_dataset.targets, num_classes=splits.num_classes)
        shuffle = False

    drop_last = False
    if drop_last_if_singleton and len(train_dataset) > batch_size and (len(train_dataset) % batch_size == 1):
        drop_last = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "train_eval_dataset": train_eval_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }

