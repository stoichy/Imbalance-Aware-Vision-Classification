"""Training, validation, self-paced curriculum, scheduler, and TensorBoard orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler, SequentialLR
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

from src.datasets import build_dataloaders, build_weighted_sampler, load_dataset_splits
from src.losses import build_criterion
from src.metrics import (
    compute_classification_metrics,
    create_confusion_matrix_figure,
    figure_to_tensor,
    format_metrics_report,
    plot_confusion_matrix,
    plot_training_curves,
)
from src.models import build_model, build_optimizer, freeze_bn
from src.transforms import build_transforms
from src.utils import count_parameters, get_device, save_json, seed_everything


logger = logging.getLogger("imbalanced_vision.trainer")


@dataclass
class TrainConfig:
    """Configuration governing optimization, scheduling, regularization, and logging."""

    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    backbone_lr: float | None = None
    head_lr: float | None = None
    weight_decay: float = 1e-4
    optimizer_name: str = "adam"
    scheduler_name: str = "none"
    warmup_epochs: int = 3
    freeze_bn_stats: bool = False
    grad_clip_norm: float | None = None
    curriculum: bool = False
    curriculum_start_frac: float = 0.3
    curriculum_step: float = 0.05
    checkpoint_metric: str = "f1_macro"
    use_weighted_sampler: bool = False
    log_tensorboard: bool = True
    seed: int = 42
    run_name: str = "experiment"
    output_dir: str = "outputs"
    tensorboard_dir: str = "runs"
    extra_metadata: dict[str, Any] = field(default_factory=dict)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int = 3,
) -> LRScheduler | None:
    """Build LR scheduler ('none', 'cosine', or 'warmup_cosine' via SequentialLR)."""
    key = scheduler_name.lower().strip()
    if key == "none":
        return None
    if key == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if key == "warmup_cosine":
        if epochs <= 1:
            return LambdaLR(optimizer, lr_lambda=lambda _e: 1.0)
        eff_warmup = min(max(1, warmup_epochs), epochs - 1)
        warmup = LambdaLR(
            optimizer,
            lr_lambda=lambda e: min(1.0, float(e + 1) / float(eff_warmup)),
        )
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - eff_warmup))
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[eff_warmup])
    raise ValueError(f"Unsupported scheduler {scheduler_name!r}. Choose 'none', 'cosine', or 'warmup_cosine'.")


@torch.no_grad()
def compute_per_sample_losses(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 0,
) -> torch.Tensor:
    """Evaluate unweighted per-sample cross-entropy losses for self-paced curriculum selection."""
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    losses: list[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        per_item = F.cross_entropy(logits, labels, reduction="none")
        losses.append(per_item.detach().cpu())
    if was_training:
        model.train()
    return torch.cat(losses, dim=0)


def _build_curriculum_loader(
    train_dataset: Dataset,
    losses: torch.Tensor,
    epoch: int,
    start_frac: float,
    step: float,
    batch_size: int,
    num_classes: int,
    use_weighted_sampler: bool,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, float, float, int]:
    """Construct a DataLoader over the easy subset whose per-sample loss <= quantile(q)."""
    q = min(1.0, max(0.05, start_frac + step * epoch))
    threshold = float(torch.quantile(losses, q=q).item())
    easy_idx = (losses <= threshold).nonzero(as_tuple=True)[0].tolist()

    min_required = min(len(train_dataset), max(2, min(batch_size, len(train_dataset))))
    if len(easy_idx) < min_required:
        sorted_idx = torch.argsort(losses).tolist()
        easy_idx = sorted_idx[:min_required]

    subset = Subset(train_dataset, easy_idx)
    sampler = None
    shuffle = True
    if use_weighted_sampler and hasattr(train_dataset, "targets"):
        subset_targets = [int(train_dataset.targets[i]) for i in easy_idx]
        sampler = build_weighted_sampler(subset_targets, num_classes=num_classes)
        shuffle = False

    drop_last = len(subset) > batch_size and (len(subset) % batch_size == 1)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader, q, threshold, len(easy_idx)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_names: Sequence[str] | None = None,
    freeze_bn_stats: bool = False,
    grad_clip_norm: float | None = None,
) -> dict[str, Any]:
    """Run one training epoch and return loss and classification metrics."""
    model.train()
    if freeze_bn_stats:
        freeze_bn(model)

    running_loss = 0.0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_n = labels.size(0)
        running_loss += float(loss.item()) * batch_n
        total += batch_n
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    metrics = compute_classification_metrics(all_labels, all_preds, class_names=class_names)
    metrics["loss"] = running_loss / max(1, total)
    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a model on a validation or test DataLoader without gradient tracking."""
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_n = labels.size(0)
        running_loss += float(loss.item()) * batch_n
        total += batch_n
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    metrics = compute_classification_metrics(all_labels, all_preds, class_names=class_names)
    metrics["loss"] = running_loss / max(1, total)
    return metrics


def fit_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    class_names: Sequence[str],
    config: TrainConfig,
    device: torch.device,
    train_dataset: Dataset | None = None,
    train_eval_dataset: Dataset | None = None,
    num_workers: int = 2,
) -> dict[str, Any]:
    """Train a model selecting the best checkpoint strictly on `val_loader`, then evaluate on `test_loader`."""
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=config.scheduler_name,
        epochs=config.epochs,
        warmup_epochs=config.warmup_epochs,
    )

    writer: SummaryWriter | None = None
    if config.log_tensorboard:
        tb_dir = Path(config.tensorboard_dir) / config.run_name
        writer = SummaryWriter(log_dir=str(tb_dir))

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1_macro": [],
        "val_f1_macro": [],
        "val_f1_micro": [],
        "lr": [],
    }
    if config.curriculum:
        history["curriculum_frac"] = []
        history["curriculum_samples"] = []

    best_score = -float("inf") if config.checkpoint_metric != "val_loss" else float("inf")
    best_epoch = 0
    best_val_metrics: dict[str, Any] = {}
    start_time = time.time()

    param_counts = count_parameters(model)
    logger.info(
        "Starting run=%s | epochs=%d | params total=%d trainable=%d frozen=%d | device=%s",
        config.run_name,
        config.epochs,
        param_counts["total"],
        param_counts["trainable"],
        param_counts["frozen"],
        device,
    )

    try:
        for epoch in range(config.epochs):
            active_train_loader = train_loader
            if config.curriculum:
                if train_dataset is None or train_eval_dataset is None:
                    raise ValueError("train_dataset and train_eval_dataset are required when curriculum=True.")
                sample_losses = compute_per_sample_losses(
                    model=model,
                    dataset=train_eval_dataset,
                    device=device,
                    batch_size=max(config.batch_size, 128),
                    num_workers=num_workers,
                )
                active_train_loader, q_frac, loss_thresh, n_easy = _build_curriculum_loader(
                    train_dataset=train_dataset,
                    losses=sample_losses,
                    epoch=epoch,
                    start_frac=config.curriculum_start_frac,
                    step=config.curriculum_step,
                    batch_size=config.batch_size,
                    num_classes=len(class_names),
                    use_weighted_sampler=config.use_weighted_sampler,
                    num_workers=num_workers,
                    pin_memory=(device.type == "cuda"),
                )
                history["curriculum_frac"].append(q_frac)
                history["curriculum_samples"].append(float(n_easy))
                if writer is not None:
                    writer.add_scalar("Curriculum/fraction", q_frac, epoch)
                    writer.add_scalar("Curriculum/loss_threshold", loss_thresh, epoch)
                    writer.add_scalar("Curriculum/num_samples", n_easy, epoch)

            train_metrics = train_one_epoch(
                model=model,
                loader=active_train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                class_names=class_names,
                freeze_bn_stats=config.freeze_bn_stats,
                grad_clip_norm=config.grad_clip_norm,
            )
            val_metrics = evaluate_model(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                class_names=class_names,
            )

            current_lr = float(optimizer.param_groups[-1]["lr"])
            history["train_loss"].append(float(train_metrics["loss"]))
            history["val_loss"].append(float(val_metrics["loss"]))
            history["train_acc"].append(float(train_metrics["accuracy"]))
            history["val_acc"].append(float(val_metrics["accuracy"]))
            history["train_f1_macro"].append(float(train_metrics["f1_macro"]))
            history["val_f1_macro"].append(float(val_metrics["f1_macro"]))
            history["val_f1_micro"].append(float(val_metrics["f1_micro"]))
            history["lr"].append(current_lr)

            if writer is not None:
                writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
                writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                writer.add_scalar("Acc/train", train_metrics["accuracy"], epoch)
                writer.add_scalar("Acc/val", val_metrics["accuracy"], epoch)
                writer.add_scalar("F1_macro/train", train_metrics["f1_macro"], epoch)
                writer.add_scalar("F1_macro/val", val_metrics["f1_macro"], epoch)
                writer.add_scalar("F1_micro/val", val_metrics["f1_micro"], epoch)
                for group in optimizer.param_groups:
                    tag = group.get("name", "lr")
                    writer.add_scalar(f"LR/{tag}", group["lr"], epoch)

            if scheduler is not None:
                scheduler.step()

            if config.checkpoint_metric == "val_loss":
                is_best = epoch == 0 or val_metrics["loss"] < best_score
                current_score = float(val_metrics["loss"])
            else:
                is_best = epoch == 0 or val_metrics["f1_macro"] > best_score
                current_score = float(val_metrics["f1_macro"])

            if is_best:
                best_score = current_score
                best_epoch = epoch + 1
                best_val_metrics = val_metrics
                torch.save(
                    {
                        "epoch": best_epoch,
                        "model_state_dict": model.state_dict(),
                        "val_metrics": val_metrics,
                        "class_names": list(class_names),
                        "train_config": asdict(config),
                        "metadata": config.extra_metadata,
                    },
                    checkpoint_path,
                )

            logger.info(
                "Epoch [%02d/%02d] | lr=%.2e | tr_loss=%.4f tr_acc=%.4f | "
                "val_loss=%.4f val_acc=%.4f val_f1_macro=%.4f",
                epoch + 1,
                config.epochs,
                current_lr,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["f1_macro"],
            )

        # Load best validation checkpoint and evaluate ONCE on strictly held-out test set
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate_model(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
        )

        if writer is not None:
            cm_fig = create_confusion_matrix_figure(
                cm=test_metrics["confusion_matrix"],
                class_names=class_names,
                title=f"{config.run_name} (Held-Out Test)",
            )
            writer.add_image("ConfusionMatrix/test", figure_to_tensor(cm_fig), global_step=best_epoch)
            writer.add_hparams(
                {
                    "lr": float(config.lr),
                    "batch_size": int(config.batch_size),
                    "epochs": int(config.epochs),
                    "scheduler": config.scheduler_name,
                    "curriculum": bool(config.curriculum),
                },
                {
                    "hparam/val_f1_macro": float(best_val_metrics["f1_macro"]),
                    "hparam/test_f1_macro": float(test_metrics["f1_macro"]),
                    "hparam/test_accuracy": float(test_metrics["accuracy"]),
                },
            )
    finally:
        if writer is not None:
            writer.close()

    elapsed_sec = round(time.time() - start_time, 2)
    logger.info("\n%s", format_metrics_report(test_metrics, split_name=f"{config.run_name} Held-Out Test"))

    summary = {
        "run_name": config.run_name,
        "best_epoch": best_epoch,
        "elapsed_seconds": elapsed_sec,
        "parameter_counts": param_counts,
        "checkpoint_path": str(checkpoint_path),
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "train_config": asdict(config),
    }
    save_json(history, output_dir / "history.json")
    save_json(summary, output_dir / "summary.json")
    return {
        "history": history,
        "summary": summary,
        "checkpoint_path": checkpoint_path,
    }


def resolve_imbalance_mode(imbalance_mode: str) -> tuple[bool, str]:
    """Map an imbalance strategy string to `(use_weighted_sampler, loss_name)`."""
    key = imbalance_mode.lower().strip()
    mapping = {
        "ce": (False, "ce"),
        "none": (False, "ce"),
        "weighted_sampler": (True, "ce"),
        "sampler": (True, "ce"),
        "weighted_ce": (False, "weighted_ce"),
        "focal": (False, "focal"),
        "focal_unweighted": (False, "focal_unweighted"),
    }
    if key not in mapping:
        valid = ", ".join(sorted({"ce", "weighted_sampler", "weighted_ce", "focal", "focal_unweighted"}))
        raise ValueError(f"Unsupported imbalance_mode {imbalance_mode!r}. Choose from: {valid}")
    return mapping[key]


def run_experiment(cfg: dict[str, Any]) -> dict[str, Any]:
    """End-to-end training, validation checkpointing, held-out testing, and optional plotting."""
    seed = int(cfg.get("seed", 42))
    seed_everything(seed=seed, deterministic=bool(cfg.get("deterministic", True)))
    device = get_device(str(cfg.get("device", "auto")))

    dataset_name = str(cfg["dataset"])
    model_name = str(cfg["model"])
    image_size = int(cfg.get("image_size", 224 if model_name == "resnet18" else 64))
    batch_size = int(cfg.get("batch_size", 32 if model_name == "resnet18" else 64))
    num_workers = int(cfg.get("num_workers", 2))
    normalization = str(cfg.get("normalization", "imagenet" if model_name == "resnet18" else "dataset"))
    aug_mode = str(cfg.get("aug_mode", "none"))
    imbalance_mode = str(cfg.get("imbalance_mode", "ce"))
    use_weighted_sampler, loss_name = resolve_imbalance_mode(imbalance_mode)

    splits = load_dataset_splits(
        dataset_name=dataset_name,
        data_root=cfg.get("data_root"),
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        seed=seed,
        max_train_samples=cfg.get("max_train_samples"),
        max_test_samples=cfg.get("max_test_samples"),
    )

    custom_mean = cfg.get("mean") if normalization == "dataset" else None
    custom_std = cfg.get("std") if normalization == "dataset" else None
    train_transform, eval_transform = build_transforms(
        dataset_name=dataset_name,
        image_size=image_size,
        aug_mode=aug_mode,
        normalization=normalization,
        mean=custom_mean,
        std=custom_std,
    )

    loaders = build_dataloaders(
        splits=splits,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=batch_size,
        use_weighted_sampler=use_weighted_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_kwargs = {
        "model_name": model_name,
        "num_classes": splits.num_classes,
        "image_size": image_size,
        "dropout": float(cfg.get("dropout", 0.3)),
        "hidden_dim": int(cfg.get("hidden_dim", 512)),
        "use_batchnorm": bool(cfg.get("use_batchnorm", True)),
        "use_adaptive_pool": bool(cfg.get("use_adaptive_pool", False)),
        "first_kernel_size": int(cfg.get("first_kernel_size", 3)),
        "pretrained": bool(cfg.get("pretrained", True)),
        "freeze_bn_stats": bool(cfg.get("freeze_bn", model_name == "resnet18")),
        "freeze_backbone": bool(cfg.get("freeze_backbone", False)),
    }
    model = build_model(**model_kwargs).to(device)

    criterion = build_criterion(
        loss_name=loss_name,
        class_counts=splits.train_class_counts,
        gamma=float(cfg.get("focal_gamma", 2.0)),
        focal_use_alpha=bool(cfg.get("focal_use_alpha", True)),
        device=device,
    )

    optimizer = build_optimizer(
        model=model,
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        optimizer_name=str(cfg.get("optimizer", "adam")),
        backbone_lr=float(cfg["backbone_lr"]) if cfg.get("backbone_lr") is not None else None,
        head_lr=float(cfg["head_lr"]) if cfg.get("head_lr") is not None else None,
    )

    run_name = str(cfg.get("run_name", f"{dataset_name}_{model_name}"))
    train_cfg = TrainConfig(
        epochs=int(cfg.get("epochs", 20)),
        batch_size=batch_size,
        lr=float(cfg.get("lr", 1e-3)),
        backbone_lr=float(cfg["backbone_lr"]) if cfg.get("backbone_lr") is not None else None,
        head_lr=float(cfg["head_lr"]) if cfg.get("head_lr") is not None else None,
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        optimizer_name=str(cfg.get("optimizer", "adam")),
        scheduler_name=str(cfg.get("scheduler", "none")),
        warmup_epochs=int(cfg.get("warmup_epochs", 3)),
        freeze_bn_stats=bool(cfg.get("freeze_bn", model_name == "resnet18")),
        grad_clip_norm=float(cfg["grad_clip_norm"]) if cfg.get("grad_clip_norm") is not None else None,
        curriculum=bool(cfg.get("curriculum", False)),
        curriculum_start_frac=float(cfg.get("curriculum_start_frac", 0.3)),
        curriculum_step=float(cfg.get("curriculum_step", 0.05)),
        checkpoint_metric=str(cfg.get("checkpoint_metric", "f1_macro")),
        use_weighted_sampler=use_weighted_sampler,
        log_tensorboard=bool(cfg.get("log_tensorboard", True)),
        seed=seed,
        run_name=run_name,
        output_dir=str(cfg.get("output_dir", "outputs")),
        tensorboard_dir=str(cfg.get("tensorboard_dir", "runs")),
        extra_metadata={
            "dataset": dataset_name,
            "data_root": str(splits.data_root),
            "val_ratio": float(cfg.get("val_ratio", 0.15)),
            "raw_labels": list(splits.raw_labels),
            "class_names": list(splits.class_names),
            "normalization": normalization,
            "mean": list(splits.mean),
            "std": list(splits.std),
            "model_kwargs": model_kwargs,
            "imbalance_mode": imbalance_mode,
            "aug_mode": aug_mode,
            "split_sizes": {
                "train": len(splits.train_samples),
                "val": len(splits.val_samples),
                "test": len(splits.test_samples),
            },
        },
    )

    result = fit_and_evaluate(
        model=model,
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        test_loader=loaders["test_loader"],
        criterion=criterion,
        optimizer=optimizer,
        class_names=splits.class_names,
        config=train_cfg,
        device=device,
        train_dataset=loaders["train_dataset"],
        train_eval_dataset=loaders["train_eval_dataset"],
        num_workers=num_workers,
    )

    result["summary"]["dataset_split_sizes"] = {
        "train": len(splits.train_samples),
        "val": len(splits.val_samples),
        "test": len(splits.test_samples),
    }
    result["summary"]["train_class_counts"] = splits.train_class_counts.tolist()
    save_json(result["summary"], Path(train_cfg.output_dir) / run_name / "summary.json")

    if cfg.get("export_plots", False):
        plot_dir = Path(cfg.get("plot_dir", Path(train_cfg.output_dir) / run_name / "plots"))
        plot_training_curves(
            history=result["history"],
            output_path=plot_dir / f"{run_name}_curves.png",
            title=f"{run_name} Learning Curves",
        )
        plot_confusion_matrix(
            cm=result["summary"]["test_metrics"]["confusion_matrix"],
            class_names=splits.class_names,
            output_path=plot_dir / f"{run_name}_confusion.png",
            title=f"{run_name} (Held-Out Test)",
        )

    return result
