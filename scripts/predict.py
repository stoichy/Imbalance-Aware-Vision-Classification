"""CLI entrypoint for generating predictions on unlabeled test manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import InferenceImageDataset, get_dataset_spec
from src.models import build_model
from src.transforms import build_transforms
from src.utils import get_device, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission CSV predictions from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved `best_model.pt`.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["skin_lesions", "facial_expressions"],
        default=None,
        help="Target dataset (inferred from checkpoint if omitted).",
    )
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root path.")
    parser.add_argument("--output-csv", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device.")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["auto", "idx", "label"],
        default="auto",
        help="Write zero-based integer class index ('idx') or raw dataset label ('label').",
    )
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
    dataset_name = args.dataset or meta.get("dataset")
    if not dataset_name:
        raise ValueError("Could not infer dataset from checkpoint; please pass --dataset.")

    spec = get_dataset_spec(dataset_name)
    data_root = Path(args.data_root or meta.get("data_root") or spec.default_root)

    model_kwargs = meta.get("model_kwargs")
    if not model_kwargs:
        raise ValueError("Checkpoint metadata is missing `model_kwargs`.")
    load_kwargs = dict(model_kwargs)
    if load_kwargs.get("model_name") in {"resnet18", "resnet-18"}:
        load_kwargs["pretrained"] = False

    model = build_model(**load_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, eval_transform = build_transforms(
        dataset_name=dataset_name,
        image_size=int(load_kwargs.get("image_size", 64)),
        aug_mode="none",
        normalization=str(meta.get("normalization", "dataset")),
    )

    inf_dataset = InferenceImageDataset(
        csv_file=data_root / spec.inference_csv,
        img_dir=data_root / spec.inference_img_subdir,
        transform=eval_transform,
    )
    loader = DataLoader(inf_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    raw_labels = list(meta.get("raw_labels", spec.raw_labels))
    as_idx = spec.output_prediction_as_idx if args.output_format == "auto" else (args.output_format == "idx")

    ids: list[str] = []
    preds: list[int | str] = []
    with torch.no_grad():
        for images, batch_ids in loader:
            images = images.to(device)
            pred_indices = model(images).argmax(dim=1).cpu().tolist()
            ids.extend(batch_ids)
            for idx in pred_indices:
                preds.append(int(idx) if as_idx else raw_labels[int(idx)])

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Id": ids, "Prediction": preds}).to_csv(out_path, index=False)
    logger.info("Saved %d predictions to %s", len(preds), out_path)


if __name__ == "__main__":
    main()

