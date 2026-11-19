"""CLI entrypoint for training a single classification experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer import run_experiment
from src.utils import deep_merge, load_yaml, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single image classification model.")
    parser.add_argument("--config", type=str, default=None, help="Experiment YAML config path.")
    parser.add_argument("--dataset-config", type=str, default=None, help="Dataset YAML config path.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["skin_lesions", "facial_expressions"],
        default=None,
        help="Target dataset.",
    )
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root path.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "cnn", "resnet18"],
        default=None,
        help="Model architecture.",
    )
    parser.add_argument("--image-size", type=int, default=None, help="Input spatial resolution.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Classifier hidden units.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout probability.")
    parser.add_argument(
        "--batchnorm",
        dest="use_batchnorm",
        action="store_true",
        default=None,
        help="Enable BatchNorm2d in SimpleCNN.",
    )
    parser.add_argument(
        "--no-batchnorm",
        dest="use_batchnorm",
        action="store_false",
        default=None,
        help="Disable BatchNorm2d in SimpleCNN.",
    )
    parser.add_argument(
        "--adaptive-pool",
        dest="use_adaptive_pool",
        action="store_true",
        default=None,
        help="Enable AdaptiveAvgPool2d(1) in SimpleCNN.",
    )
    parser.add_argument("--first-kernel-size", type=int, default=None, help="First CNN kernel size.")
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=None,
        help="Load ImageNet weights for ResNet-18.",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        default=None,
        help="Initialize ResNet-18 randomly (offline / test mode).",
    )
    parser.add_argument(
        "--freeze-bn",
        dest="freeze_bn",
        action="store_true",
        default=None,
        help="Freeze BatchNorm running stats and affine parameters.",
    )
    parser.add_argument(
        "--no-freeze-bn",
        dest="freeze_bn",
        action="store_false",
        default=None,
        help="Keep BatchNorm stats trainable.",
    )
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        default=None,
        help="Freeze ResNet-18 backbone weights (train head only).",
    )
    parser.add_argument(
        "--aug-mode",
        type=str,
        choices=["none", "geometric", "color", "combined"],
        default=None,
        help="Domain-specific augmentation preset.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["dataset", "imagenet"],
        default=None,
        help="Normalization statistics source.",
    )
    parser.add_argument(
        "--imbalance-mode",
        type=str,
        choices=["ce", "weighted_sampler", "weighted_ce", "focal", "focal_unweighted"],
        default=None,
        help="Class imbalance mitigation strategy.",
    )
    parser.add_argument("--focal-gamma", type=float, default=None, help="Focal loss focusing parameter.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size.")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], default=None)
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate.")
    parser.add_argument("--backbone-lr", type=float, default=None, help="Backbone LR for ResNet-18.")
    parser.add_argument("--head-lr", type=float, default=None, help="Head LR for ResNet-18.")
    parser.add_argument("--weight-decay", type=float, default=None, help="L2 weight decay.")
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "cosine", "warmup_cosine"],
        default=None,
        help="Learning rate schedule.",
    )
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Linear warmup epochs.")
    parser.add_argument(
        "--curriculum",
        action="store_true",
        default=None,
        help="Enable self-paced learning (per-sample loss thresholding).",
    )
    parser.add_argument("--curriculum-start-frac", type=float, default=None)
    parser.add_argument("--curriculum-step", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None, help="Stratified validation fraction.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes.")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('auto', 'cpu', 'cuda').")
    parser.add_argument("--run-name", type=str, default=None, help="Experiment identifier.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for checkpoints & JSON.")
    parser.add_argument("--tensorboard-dir", type=str, default=None, help="TensorBoard log root.")
    parser.add_argument("--export-plots", action="store_true", default=None, help="Save curves and CM PNGs.")
    parser.add_argument("--plot-dir", type=str, default=None, help="Output directory for exported plots.")
    parser.add_argument("--no-tensorboard", dest="log_tensorboard", action="store_false", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Stratified train cap (debug/smoke).")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Stratified test cap (debug/smoke).")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.dataset_config:
        cfg = deep_merge(cfg, load_yaml(args.dataset_config))
    if args.config:
        cfg = deep_merge(cfg, load_yaml(args.config))

    cli_fields = [
        "dataset",
        "data_root",
        "model",
        "image_size",
        "hidden_dim",
        "dropout",
        "use_batchnorm",
        "use_adaptive_pool",
        "first_kernel_size",
        "pretrained",
        "freeze_bn",
        "freeze_backbone",
        "aug_mode",
        "normalization",
        "imbalance_mode",
        "focal_gamma",
        "epochs",
        "batch_size",
        "optimizer",
        "lr",
        "backbone_lr",
        "head_lr",
        "weight_decay",
        "scheduler",
        "warmup_epochs",
        "curriculum",
        "curriculum_start_frac",
        "curriculum_step",
        "val_ratio",
        "seed",
        "num_workers",
        "device",
        "run_name",
        "output_dir",
        "tensorboard_dir",
        "export_plots",
        "plot_dir",
        "log_tensorboard",
        "max_train_samples",
        "max_test_samples",
    ]
    for field in cli_fields:
        val = getattr(args, field, None)
        if val is not None:
            cfg[field] = val

    if "dataset" not in cfg or "model" not in cfg:
        raise ValueError("Both `dataset` and `model` must be specified via configs or CLI flags.")
    return cfg


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = build_config_from_args(args)
    result = run_experiment(cfg)
    summary = result["summary"]
    test_m = summary["test_metrics"]
    print(
        f"\nCompleted {summary['run_name']} (best val epoch={summary['best_epoch']}) -> "
        f"Held-Out Test Acc={test_m['accuracy']:.4f}, Macro-F1={test_m['f1_macro']:.4f}, "
        f"Micro-F1={test_m['f1_micro']:.4f}"
    )


if __name__ == "__main__":
    main()

