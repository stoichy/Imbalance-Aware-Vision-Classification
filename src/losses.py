"""Loss functions and class-weighting utilities for imbalanced classification."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    class_counts: Sequence[int] | np.ndarray,
    mode: str = "inverse_freq",
) -> torch.Tensor:
    """Compute per-class loss weights `w_c = N_total / (K * N_c)` from training class counts."""
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim != 1 or len(counts) == 0:
        raise ValueError("class_counts must be a non-empty 1D sequence.")
    if np.any(counts < 0):
        raise ValueError("class_counts cannot contain negative entries.")

    num_classes = len(counts)
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("Total sample count must be positive.")

    safe_counts = np.maximum(counts, 1.0)
    mode_key = mode.lower().strip()
    if mode_key in {"inverse_freq", "balanced"}:
        weights = total / (num_classes * safe_counts)
    elif mode_key == "sqrt_inv":
        raw = np.sqrt(total / (num_classes * safe_counts))
        weights = raw / raw.mean()
    else:
        raise ValueError(f"Unsupported class weighting mode: {mode!r}")

    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., 2017) with optional per-class alpha weights.

    Computes:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where `p_t = exp(-CE_unweighted)` is the true predicted probability of the ground-truth class.
    """

    def __init__(
        self,
        alpha: torch.Tensor | Sequence[float] | float | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction {reduction!r}. Use 'mean', 'sum', or 'none'.")

        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.detach().clone().to(dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(list(alpha), dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"Expected 2D logits (N, C), got shape {tuple(logits.shape)}")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Expected 1D targets of length {logits.shape[0]}, got shape {tuple(targets.shape)}"
            )

        ce_unweighted = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_unweighted)
        focal_weight = (1.0 - p_t).clamp(min=0.0) ** self.gamma
        loss = focal_weight * ce_unweighted

        if self.alpha is not None:
            alpha_tensor = self.alpha.to(device=logits.device, dtype=logits.dtype)
            if alpha_tensor.ndim == 0:
                alpha_t = alpha_tensor
            else:
                if alpha_tensor.numel() != logits.shape[1]:
                    raise ValueError(
                        f"alpha length ({alpha_tensor.numel()}) does not match num_classes ({logits.shape[1]})."
                    )
                alpha_t = alpha_tensor[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(
    loss_name: str = "ce",
    class_counts: Sequence[int] | np.ndarray | None = None,
    gamma: float = 2.0,
    focal_use_alpha: bool = True,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Build classification loss (`ce`, `weighted_ce`, `focal`, `focal_unweighted`)."""
    key = loss_name.lower().strip()
    if key == "ce":
        return nn.CrossEntropyLoss()

    if key == "weighted_ce":
        if class_counts is None:
            raise ValueError("class_counts is required for 'weighted_ce'.")
        weights = compute_class_weights(class_counts).to(device)
        return nn.CrossEntropyLoss(weight=weights)

    if key == "focal":
        alpha = None
        if focal_use_alpha and class_counts is not None:
            alpha = compute_class_weights(class_counts).to(device)
        return FocalLoss(alpha=alpha, gamma=gamma, reduction="mean").to(device)

    if key == "focal_unweighted":
        return FocalLoss(alpha=None, gamma=gamma, reduction="mean").to(device)

    raise ValueError(
        f"Unsupported loss {loss_name!r}. Choose from 'ce', 'weighted_ce', 'focal', 'focal_unweighted'."
    )

