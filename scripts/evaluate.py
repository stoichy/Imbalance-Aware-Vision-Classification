"""CLI entrypoint to evaluate a saved model checkpoint on validation or held-out test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import build_dataloaders, load_dataset_splits
from src.metrics import format_metrics_report, plot_confusion_matrix
from src.models import build_model
from src.trainer import evaluate_model
from src.transforms import build_transforms
from src.utils import get_device, save_json, seed_everything, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on val or held-out test.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved `best_model.pt`.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["skin_lesions", "facial_expressions"],
        default=None,
        help="Dataset name (inferred from checkpoint metadata if omitted).",
    )
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root path.")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test", help="Split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON metrics export path.")
    parser.add_argument("--output-cm", type=str, default=None, help="Optional confusion matrix PNG path.")
    return parser.parse_args()


def main() -> None:
    logger = setup_logging()
    args = parse_args()
    device = get_device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt.get("metadata", {})
    train_cfg = ckpt.get("train_config", {})

    dataset_name = args.dataset or meta.get("dataset")
    if not dataset_name:
        raise ValueError("Could not infer dataset from checkpoint; please pass --dataset.")

    data_root = args.data_root or meta.get("data_root")
    seed = int(train_cfg.get("seed", 42))
    seed_everything(seed)

    model_kwargs = meta.get("model_kwargs")
    if not model_kwargs:
        raise ValueError("Checkpoint metadata is missing `model_kwargs` needed to reconstruct model.")

    # Never download weights again when loading a saved checkpoint
    load_kwargs = dict(model_kwargs)
    if load_kwargs.get("model_name") in {"resnet18", "resnet-18"}:
        load_kwargs["pretrained"] = False

    model = build_model(**load_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    splits = load_dataset_splits(
        dataset_name=dataset_name,
        data_root=data_root,
        val_ratio=float(meta.get("val_ratio", 0.15)),
        seed=seed,
    )
    _, eval_transform = build_transforms(
        dataset_name=dataset_name,
        image_size=int(load_kwargs.get("image_size", 64)),
        aug_mode="none",
        normalization=str(meta.get("normalization", "dataset")),
    )
    loaders = build_dataloaders(
        splits=splits,
        train_transform=eval_transform,
        eval_transform=eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    loader = loaders["val_loader"] if args.split == "val" else loaders["test_loader"]
    metrics = evaluate_model(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        class_names=splits.class_names,
    )

    logger.info("\n%s", format_metrics_report(metrics, split_name=args.split.upper()))

    if args.output_json:
        save_json(metrics, args.output_json)
        logger.info("Saved evaluation JSON: %s", args.output_json)

    if args.output_cm:
        plot_confusion_matrix(
            cm=metrics["confusion_matrix"],
            class_names=splits.class_names,
            output_path=args.output_cm,
            title=f"{ckpt_path.parent.name} ({args.split})",
        )
        logger.info("Saved confusion matrix PNG: %s", args.output_cm)


if __name__ == "__main__":
    main()
