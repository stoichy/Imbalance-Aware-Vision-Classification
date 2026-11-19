"""Evaluation metrics, reporting helpers, and matplotlib/seaborn plot generators."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torchvision import transforms


sns.set_theme(style="whitegrid", context="notebook")


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute Accuracy, Macro/Micro/Weighted F1, per-class P/R/F1, and confusion matrices."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    if len(y_true_arr) == 0:
        raise ValueError("Cannot compute metrics on an empty sample list.")

    if class_names is not None:
        num_classes = len(class_names)
        labels = list(range(num_classes))
        names = list(class_names)
    else:
        num_classes = int(max(y_true_arr.max(), y_pred_arr.max()) + 1)
        labels = list(range(num_classes))
        names = [str(i) for i in labels]

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    f1_macro = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0))
    f1_micro = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="micro", zero_division=0))
    f1_weighted = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0))

    prec, rec, f1_per_cls, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        zero_division=0,
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums > 0,
    )

    per_class: dict[str, dict[str, float | int]] = {}
    for idx, cls_name in enumerate(names):
        per_class[cls_name] = {
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "f1": float(f1_per_cls[idx]),
            "support": int(support[idx]),
        }

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "precision_macro": float(np.mean(prec)),
        "recall_macro": float(np.mean(rec)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }


def format_metrics_report(metrics: dict[str, Any], split_name: str = "Test") -> str:
    """Format global and per-class metrics into a readable multi-line table."""
    lines = [
        f"[{split_name}] Accuracy={metrics['accuracy']:.4f} | "
        f"Macro-F1={metrics['f1_macro']:.4f} | "
        f"Micro-F1={metrics['f1_micro']:.4f} | "
        f"Weighted-F1={metrics['f1_weighted']:.4f}",
        f"{'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
        "-" * 58,
    ]
    for cls_name, stats in metrics["per_class"].items():
        lines.append(
            f"{cls_name:<14} "
            f"{stats['precision']:>10.4f} "
            f"{stats['recall']:>10.4f} "
            f"{stats['f1']:>10.4f} "
            f"{stats['support']:>10d}"
        )
    return "\n".join(lines)


def figure_to_tensor(fig: plt.Figure) -> torch.Tensor:
    """Render a Matplotlib figure to a `(3, H, W)` float tensor for TensorBoard logging."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as img:
        rgb = img.convert("RGB")
        return transforms.ToTensor()(rgb)


def create_confusion_matrix_figure(
    cm: np.ndarray | Sequence[Sequence[int]],
    class_names: Sequence[str],
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Create a side-by-side figure with raw counts and row-normalized recall percentages."""
    cm_arr = np.asarray(cm, dtype=np.int64)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_arr.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm_arr, dtype=np.float64),
        where=row_sums > 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=axes[0],
        cbar_kws={"shrink": 0.85},
    )
    axes[0].set_title("Counts", fontsize=11)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        cm_norm * 100.0,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0.0,
        vmax=100.0,
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=axes[1],
        cbar_kws={"shrink": 0.85, "label": "Row Recall (%)"},
    )
    axes[1].set_title("Row-Normalized (%)", fontsize=11)
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].tick_params(axis="y", rotation=0)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray | Sequence[Sequence[int]],
    class_names: Sequence[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Save a raw + normalized confusion matrix PNG to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = create_confusion_matrix_figure(cm=cm, class_names=class_names, title=title)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: str | Path,
    title: str = "Training History",
) -> Path:
    """Plot Loss, Accuracy, and Macro-F1 / Micro-F1 learning curves across epochs."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Loss
    axes[0].plot(epochs, history["train_loss"], marker="o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", markersize=3, label="Val Loss")
    axes[0].set_title("Cross-Entropy / Focal Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], marker="o", markersize=3, label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], marker="s", markersize=3, label="Val Acc")
    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend()

    # Macro-F1 and Micro-F1
    if "train_f1_macro" in history and history["train_f1_macro"]:
        axes[2].plot(epochs, history["train_f1_macro"], marker="o", markersize=3, label="Train Macro-F1")
    axes[2].plot(epochs, history["val_f1_macro"], marker="s", markersize=3, label="Val Macro-F1")
    if "val_f1_micro" in history and history["val_f1_micro"]:
        axes[2].plot(
            epochs,
            history["val_f1_micro"],
            linestyle="--",
            marker="^",
            markersize=3,
            label="Val Micro-F1",
        )
    axes[2].set_title("Macro-F1 & Micro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].legend()

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_augmentation_grid(
    images_by_mode: dict[str, list[np.ndarray]],
    class_names: Sequence[str],
    output_path: str | Path,
    title: str = "Domain-Specific Augmentation Comparison",
) -> Path:
    """Save a grid showing one representative image per class across augmentation modes.

    `images_by_mode` maps mode label (e.g., `'Original'`, `'Geometric'`, `'Color'`, `'Combined'`)
    to a list of `(H, W, 3)` images in `[0, 1]`, aligned with `class_names`.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    modes = list(images_by_mode.keys())
    n_rows = len(modes)
    n_cols = len(class_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r_idx, mode in enumerate(modes):
        imgs = images_by_mode[mode]
        for c_idx, cls_name in enumerate(class_names):
            ax = axes[r_idx, c_idx]
            ax.imshow(np.clip(imgs[c_idx], 0.0, 1.0))
            if r_idx == 0:
                ax.set_title(cls_name, fontsize=10, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(mode, fontsize=10, fontweight="bold")
                ax.set_yticks([])
                ax.set_xticks([])
            else:
                ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


SUITE_DISPLAY_NAMES: dict[str, str] = {
    "mlp_dropout": "MLP Dropout",
    "cnn_batchnorm": "CNN BatchNorm & Global Pooling",
    "augmentation": "Domain Augmentation",
    "imbalance": "Class Imbalance Mitigation",
    "resnet_warmup": "ResNet-18 Warmup Schedule",
    "curriculum": "Self-Paced Curriculum Learning",
}


def plot_ablation_comparison(
    df: pd.DataFrame,
    suite_name: str,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Plot held-out test Macro-F1 and Accuracy across ablation variants for each dataset."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    subset = df[df["suite"] == suite_name].copy()
    if subset.empty:
        raise ValueError(f"No rows found in ablation summary for suite={suite_name!r}.")

    datasets = list(dict.fromkeys(subset["dataset"].tolist()))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.8 * len(datasets), 4.6), squeeze=False)

    for idx, ds_name in enumerate(datasets):
        ax = axes[0, idx]
        ds_df = subset[subset["dataset"] == ds_name]
        x = np.arange(len(ds_df))
        width = 0.36

        bars1 = ax.bar(x - width / 2, ds_df["test_f1_macro"], width, label="Test Macro-F1", color="#2563eb")
        bars2 = ax.bar(x + width / 2, ds_df["test_accuracy"], width, label="Test Accuracy", color="#10b981")

        ax.set_xticks(x)
        ax.set_xticklabels(ds_df["variant"], rotation=20, ha="right")
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Score")
        ax.set_title(ds_name.replace("_", " ").title())
        ax.legend(loc="upper right")

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    pretty_suite = SUITE_DISPLAY_NAMES.get(suite_name, suite_name.replace("_", " ").title())
    fig.suptitle(title or f"Ablation Comparison — {pretty_suite}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out

