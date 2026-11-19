"""Unit tests for FocalLoss, class weights, LR schedulers, and training pipeline."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.losses import FocalLoss, build_criterion, compute_class_weights
from src.models import SimpleMLP, build_optimizer
from src.trainer import build_scheduler, run_experiment


def test_compute_class_weights_balanced_formula() -> None:
    counts = [900, 100]
    w = compute_class_weights(counts)
    assert torch.allclose(w, torch.tensor([1000.0 / (2 * 900.0), 1000.0 / (2 * 100.0)]))


def test_focal_loss_matches_ce_at_gamma_zero_and_focuses_hard_samples() -> None:
    logits = torch.tensor([[3.0, 0.1, -1.0], [-1.0, 2.0, 2.1]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    ce = nn.CrossEntropyLoss(reduction="none")(logits, targets)
    fl_gamma0 = FocalLoss(alpha=None, gamma=0.0, reduction="none")(logits, targets)
    assert torch.allclose(fl_gamma0, ce, atol=1e-6)

    fl_gamma2 = FocalLoss(alpha=None, gamma=2.0, reduction="none")(logits, targets)
    # Sample 0 is high confidence (p_t >> 0.9), sample 1 is ambiguous (p_t < 0.5)
    easy_ratio = (fl_gamma2[0] / fl_gamma0[0]).item()
    hard_ratio = (fl_gamma2[1] / fl_gamma0[1]).item()
    assert easy_ratio < hard_ratio
    assert easy_ratio < 0.05

    # Alpha weighting + sum/mean reductions
    alpha = torch.tensor([1.0, 4.0, 2.0])
    fl_alpha = FocalLoss(alpha=alpha, gamma=2.0, reduction="none")(logits, targets)
    assert torch.allclose(fl_alpha, torch.tensor([1.0, 4.0]) * fl_gamma2, atol=1e-6)
    assert torch.isclose(
        FocalLoss(alpha=alpha, gamma=2.0, reduction="mean")(logits, targets),
        fl_alpha.mean(),
    )
    assert torch.isclose(
        FocalLoss(alpha=alpha, gamma=2.0, reduction="sum")(logits, targets),
        fl_alpha.sum(),
    )

    crit = build_criterion("focal", class_counts=[90, 10, 50], gamma=2.0)
    assert isinstance(crit, FocalLoss)


def test_warmup_cosine_scheduler_trajectory() -> None:
    model = SimpleMLP(num_classes=3, image_size=8, hidden_dim=16, dropout=0.0)
    opt = build_optimizer(model, lr=1.0)
    scheduler = build_scheduler(opt, scheduler_name="warmup_cosine", epochs=10, warmup_epochs=3)
    assert scheduler is not None

    lrs = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(10):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            scheduler.step()

    assert len(caught) == 0
    # Warmup epochs 0, 1, 2 ramp up 1/3 -> 2/3 -> 1.0, then cosine decays
    assert lrs[0] == pytest.approx(1.0 / 3.0, rel=1e-4)
    assert lrs[1] == pytest.approx(2.0 / 3.0, rel=1e-4)
    assert lrs[2] == pytest.approx(1.0, rel=1e-4)
    assert lrs[3] > lrs[-1]


def test_end_to_end_smoke_training_with_curriculum(tmp_path: Path) -> None:
    root = tmp_path / "skin_lesions"
    classes = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    tr_rows, te_rows = [], []
    for c_idx, cls in enumerate(classes):
        for i in range(6):
            fname = f"{cls}_tr_{i}.jpg"
            (root / "train").mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color=(c_idx * 35, 60, 120)).save(root / "train" / fname)
            tr_rows.append({"imagine": fname, "diagnostic": cls})
        for i in range(2):
            fname = f"{cls}_te_{i}.jpg"
            (root / "test").mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color=(c_idx * 35, 60, 120)).save(root / "test" / fname)
            te_rows.append({"imagine": fname, "diagnostic": cls})

    pd.DataFrame(tr_rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(te_rows).to_csv(root / "test.csv", index=False)

    result = run_experiment(
        {
            "dataset": "skin_lesions",
            "data_root": str(root),
            "model": "cnn",
            "image_size": 32,
            "hidden_dim": 32,
            "use_batchnorm": True,
            "aug_mode": "combined",
            "imbalance_mode": "focal",
            "curriculum": True,
            "curriculum_start_frac": 0.5,
            "curriculum_step": 0.25,
            "epochs": 2,
            "batch_size": 8,
            "val_ratio": 0.25,
            "num_workers": 0,
            "device": "cpu",
            "log_tensorboard": False,
            "export_plots": True,
            "run_name": "test_curriculum_cnn",
            "output_dir": str(tmp_path / "outputs"),
        }
    )

    summary = result["summary"]
    assert Path(summary["checkpoint_path"]).exists()
    assert (tmp_path / "outputs" / "test_curriculum_cnn" / "summary.json").exists()
    assert (tmp_path / "outputs" / "test_curriculum_cnn" / "plots" / "test_curriculum_cnn_confusion.png").exists()
    assert len(result["history"]["curriculum_frac"]) == 2
    assert 0.0 <= summary["test_metrics"]["f1_macro"] <= 1.0

