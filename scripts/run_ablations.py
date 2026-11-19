"""CLI runner for controlled ablation suites across datasets, architectures, and imbalance methods."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import CSVImageDataset, load_dataset_splits
from src.metrics import (
    SUITE_DISPLAY_NAMES,
    plot_ablation_comparison,
    plot_augmentation_grid,
    plot_confusion_matrix,
    plot_training_curves,
)
from src.transforms import build_visualization_transform
from src.trainer import run_experiment
from src.utils import deep_merge, load_yaml, save_json, seed_everything, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute controlled ablation suites.")
    parser.add_argument("--config", type=str, default="configs/ablations.yaml", help="Ablation suite YAML.")
    parser.add_argument(
        "--suites",
        nargs="+",
        default=None,
        help="Subset of suites to execute (default: all suites in config).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["skin_lesions", "facial_expressions"],
        default=None,
        help="Subset of datasets to run (default: all datasets in config).",
    )
    parser.add_argument("--skin-root", type=str, default=None, help="Override skin_lesions root directory.")
    parser.add_argument(
        "--expressions-root",
        type=str,
        default=None,
        help="Override facial_expressions root directory.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count across all runs.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap train+val size (smoke test).")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Cap held-out test size (smoke test).")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'cuda').")
    parser.add_argument("--output-dir", type=str, default=None, help="Checkpoint and run log output dir.")
    parser.add_argument("--results-dir", type=str, default=None, help="Aggregated CSV/JSON output dir.")
    parser.add_argument("--assets-dir", type=str, default=None, help="Generated plot export dir.")
    parser.add_argument("--tensorboard-dir", type=str, default=None, help="TensorBoard log root.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download/load ImageNet weights.")
    parser.add_argument("--skip-aug-grids", action="store_true", help="Skip generating augmentation grid PNGs.")
    return parser.parse_args()


def export_domain_augmentation_grids(
    dataset_name: str,
    data_root: str | Path,
    output_path: Path,
    image_size: int = 64,
    seed: int = 42,
) -> Path:
    """Generate a deterministic 4x7 grid (Original / Geometric / Color / Combined across all classes)."""
    seed_everything(seed)
    splits = load_dataset_splits(dataset_name=dataset_name, data_root=data_root, seed=seed)
    raw_ds = CSVImageDataset(
        samples=splits.train_samples,
        class_to_idx=splits.class_to_idx,
        class_names=splits.class_names,
        transform=None,
    )

    first_per_class: dict[int, int] = {}
    for idx, (_, label) in enumerate(raw_ds.samples):
        if label not in first_per_class:
            first_per_class[label] = idx
        if len(first_per_class) == splits.num_classes:
            break

    mode_labels = [
        ("Original", "none"),
        ("Geometric", "geometric"),
        ("Color", "color"),
        ("Combined", "combined"),
    ]
    images_by_mode: dict[str, list[Any]] = {}
    for label_text, aug_mode in mode_labels:
        tf = build_visualization_transform(dataset_name=dataset_name, image_size=image_size, aug_mode=aug_mode)
        row_imgs = []
        for cls_idx in range(splits.num_classes):
            pil_img, _ = raw_ds[first_per_class[cls_idx]]
            torch.manual_seed(seed + cls_idx)
            tensor_img = tf(pil_img)
            row_imgs.append(tensor_img.permute(1, 2, 0).numpy())
        images_by_mode[label_text] = row_imgs

    pretty_name = dataset_name.replace("_", " ").title()
    return plot_augmentation_grid(
        images_by_mode=images_by_mode,
        class_names=splits.class_names,
        output_path=output_path,
        title=f"{pretty_name} — Domain-Specific Augmentation Matrix",
    )


def main() -> None:
    logger = setup_logging()
    args = parse_args()
    suite_cfg = load_yaml(args.config)

    seed = int(args.seed if args.seed is not None else suite_cfg.get("seed", 42))
    val_ratio = float(suite_cfg.get("val_ratio", 0.15))
    num_workers = int(args.num_workers if args.num_workers is not None else suite_cfg.get("num_workers", 2))
    output_dir = Path(args.output_dir or suite_cfg.get("output_dir", "outputs"))
    results_dir = Path(args.results_dir or suite_cfg.get("results_dir", "results"))
    assets_dir = Path(args.assets_dir or suite_cfg.get("assets_dir", "assets"))
    tb_dir = Path(args.tensorboard_dir or suite_cfg.get("tensorboard_dir", "runs"))

    results_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "runs").mkdir(parents=True, exist_ok=True)

    datasets_map = suite_cfg.get("datasets", {})
    selected_datasets = args.datasets or list(datasets_map.keys())

    # Resolve dataset configurations & data roots
    resolved_datasets: dict[str, dict[str, Any]] = {}
    for ds_name in selected_datasets:
        ds_entry = datasets_map[ds_name]
        ds_base = load_yaml(ds_entry["config"]) if "config" in ds_entry else {"dataset": ds_name}
        ds_merged = deep_merge(ds_base, ds_entry)
        if ds_name == "skin_lesions" and args.skin_root:
            ds_merged["data_root"] = args.skin_root
        if ds_name == "facial_expressions" and args.expressions_root:
            ds_merged["data_root"] = args.expressions_root
        resolved_datasets[ds_name] = ds_merged

        if not args.skip_aug_grids:
            grid_path = assets_dir / f"aug_grid_{ds_name}.png"
            export_domain_augmentation_grids(
                dataset_name=ds_name,
                data_root=ds_merged["data_root"],
                output_path=grid_path,
                seed=seed,
            )
            logger.info("Exported augmentation grid: %s", grid_path)

    all_suites = suite_cfg.get("suites", {})
    selected_suites = args.suites or list(all_suites.keys())

    records: list[dict[str, Any]] = []
    full_summaries: list[dict[str, Any]] = []

    is_subset_or_short = bool(args.max_train_samples or args.max_test_samples or args.epochs)
    title_suffix = ""
    if is_subset_or_short:
        parts = []
        if args.epochs is not None:
            parts.append(f"{args.epochs} epochs")
        if args.max_train_samples is not None:
            parts.append(f"subset N={args.max_train_samples}")
        title_suffix = f" [Smoke Test: {', '.join(parts)}]"

    for suite_name in selected_suites:
        if suite_name not in all_suites:
            raise ValueError(f"Unknown suite {suite_name!r}. Available: {list(all_suites.keys())}")
        suite_spec = all_suites[suite_name]
        logger.info("=== Executing ablation suite: %s (%s) ===", suite_name, suite_spec.get("description", ""))

        for ds_name in selected_datasets:
            ds_cfg = resolved_datasets[ds_name]
            pretty_ds = ds_name.replace("_", " ").title()
            pretty_suite = SUITE_DISPLAY_NAMES.get(suite_name, suite_name.replace("_", " ").title())
            for run_spec in suite_spec["runs"]:
                variant = str(run_spec["variant"])
                run_name = f"{suite_name}_{ds_name}_{variant}"
                pretty_var = variant.replace("_", " ")

                exp_cfg = deep_merge(ds_cfg, run_spec)
                exp_cfg["dataset"] = ds_name
                exp_cfg["run_name"] = run_name
                exp_cfg["seed"] = seed
                exp_cfg["val_ratio"] = val_ratio
                exp_cfg["num_workers"] = num_workers
                exp_cfg["device"] = args.device
                exp_cfg["output_dir"] = str(output_dir)
                exp_cfg["tensorboard_dir"] = str(tb_dir)
                exp_cfg["log_tensorboard"] = not args.no_tensorboard

                if args.epochs is not None:
                    exp_cfg["epochs"] = int(args.epochs)
                if args.max_train_samples is not None:
                    exp_cfg["max_train_samples"] = int(args.max_train_samples)
                if args.max_test_samples is not None:
                    exp_cfg["max_test_samples"] = int(args.max_test_samples)
                if args.no_pretrained and exp_cfg.get("model") == "resnet18":
                    exp_cfg["pretrained"] = False

                result = run_experiment(exp_cfg)
                summary = result["summary"]
                history = result["history"]

                curves_path = assets_dir / "runs" / f"{run_name}_curves.png"
                cm_path = assets_dir / "runs" / f"{run_name}_cm.png"
                plot_training_curves(
                    history=history,
                    output_path=curves_path,
                    title=f"{pretty_suite} — {pretty_ds} ({pretty_var}){title_suffix}",
                )
                plot_confusion_matrix(
                    cm=summary["test_metrics"]["confusion_matrix"],
                    class_names=ds_cfg["class_names"],
                    output_path=cm_path,
                    title=f"{pretty_suite} — {pretty_ds} ({pretty_var}){title_suffix}",
                )

                val_m = summary["val_metrics"]
                test_m = summary["test_metrics"]
                split_sizes = summary["dataset_split_sizes"]

                records.append(
                    {
                        "suite": suite_name,
                        "dataset": ds_name,
                        "variant": variant,
                        "model": exp_cfg["model"],
                        "epochs": int(exp_cfg["epochs"]),
                        "train_samples": int(split_sizes["train"]),
                        "val_samples": int(split_sizes["val"]),
                        "test_samples": int(split_sizes["test"]),
                        "best_epoch": int(summary["best_epoch"]),
                        "val_accuracy": round(float(val_m["accuracy"]), 4),
                        "val_f1_macro": round(float(val_m["f1_macro"]), 4),
                        "val_f1_micro": round(float(val_m["f1_micro"]), 4),
                        "test_accuracy": round(float(test_m["accuracy"]), 4),
                        "test_f1_macro": round(float(test_m["f1_macro"]), 4),
                        "test_f1_micro": round(float(test_m["f1_micro"]), 4),
                        "test_f1_weighted": round(float(test_m["f1_weighted"]), 4),
                        "elapsed_seconds": float(summary["elapsed_seconds"]),
                        "trainable_params": int(summary["parameter_counts"]["trainable"]),
                        "checkpoint_path": summary["checkpoint_path"],
                    }
                )
                full_summaries.append(
                    {
                        "suite": suite_name,
                        "dataset": ds_name,
                        "variant": variant,
                        **summary,
                    }
                )

        # Export suite-level comparison plot and best variant confusion matrix
        suite_df = pd.DataFrame([r for r in records if r["suite"] == suite_name])
        if not suite_df.empty:
            pretty_suite = SUITE_DISPLAY_NAMES.get(suite_name, suite_name.replace("_", " ").title())
            plot_ablation_comparison(
                df=suite_df,
                suite_name=suite_name,
                output_path=assets_dir / f"ablation_{suite_name}.png",
                title=f"{pretty_suite}{title_suffix}",
            )
            for ds_name in selected_datasets:
                ds_sub = suite_df[suite_df["dataset"] == ds_name]
                if not ds_sub.empty:
                    best_row = ds_sub.sort_values("val_f1_macro", ascending=False).iloc[0]
                    best_run = f"{suite_name}_{ds_name}_{best_row['variant']}"
                    src_cm = assets_dir / "runs" / f"{best_run}_cm.png"
                    dst_cm = assets_dir / f"cm_{suite_name}_{ds_name}.png"
                    if src_cm.exists():
                        dst_cm.write_bytes(src_cm.read_bytes())

    summary_df = pd.DataFrame(records)
    csv_out = results_dir / "ablation_summary.csv"
    json_out = results_dir / "ablation_summary.json"
    summary_df.to_csv(csv_out, index=False)
    save_json(
        {
            "run_mode": "smoke_test" if is_subset_or_short else "full",
            "runs": records,
            "detailed": full_summaries,
        },
        json_out,
    )
    logger.info("Saved aggregated ablation CSV: %s", csv_out)
    logger.info("Saved aggregated ablation JSON: %s", json_out)


if __name__ == "__main__":
    main()
