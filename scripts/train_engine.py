# -*- coding: utf-8 -*-
"""
Train anomaly detection models using Anomalib's high-level Engine API.

Supports:
- Folder dataset (default): data/datasets/<name> with train/normal, test/normal, test/abnormal
- MVTecAD dataset: requires --dataset_type mvtec and --mvtec_category

Examples (PowerShell):
  # Folder dataset (recommended for your own data)
  .\.venv\Scripts\python.exe scripts\train_engine.py --dataset_type folder --dataset_root data\datasets\your_dataset --name your_dataset --max_epochs 1

  # MVTec AD (download externally or via anomalib tools)
  .\.venv\Scripts\python.exe scripts\train_engine.py --dataset_type mvtec --dataset_root data\datasets\mvtec --mvtec_category bottle --max_epochs 1
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List


def setup_logger(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with Anomalib Engine API")

    # Dataset options
    parser.add_argument(
        "--dataset_type",
        choices=["folder", "mvtec"],
        default="folder",
        help="Dataset type to use.",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("data/datasets/your_dataset"),
        help="Root path to dataset.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="your_dataset",
        help="Dataset name (Folder datamodule only).",
    )
    parser.add_argument(
        "--mvtec_category",
        type=str,
        default=None,
        help="MVTec category (e.g., bottle, screw, transistor). Required if dataset_type=mvtec.",
    )

    # Model options (PatchCore)
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        default=["layer2", "layer3"],
        help="Backbone layers to use.",
    )
    parser.add_argument("--coreset_sampling_ratio", type=float, default=0.1)
    parser.add_argument("--num_neighbors", type=int, default=9)

    # Trainer options
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Lightning accelerator: auto|cpu|gpu.",
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of devices (gpus/cpus)")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)

    return parser.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()

    # Lazy imports to avoid overhead when checking environment
    logging.info("Importing Anomalib components...")
    try:
        from anomalib.engine import Engine  # type: ignore
        from anomalib.models import Patchcore  # type: ignore
        if args.dataset_type == "folder":
            from anomalib.data.datamodules.image.folder import Folder  # type: ignore
        else:
            from anomalib.data import MVTecAD  # type: ignore
    except Exception as e:
        logging.error("Failed to import anomalib components: %s", e)
        raise SystemExit(2)

    # Build datamodule
    if args.dataset_type == "folder":
        root = args.dataset_root
        if not root.exists():
            logging.error("Folder dataset root not found: %s", root)
            logging.error(
                "Expected structure: train/normal, test/normal, test/abnormal under %s",
                root,
            )
            raise SystemExit(2)
        datamodule = Folder(
            name=args.name,
            root=str(root),
            normal_dir="train/normal",
            abnormal_dir="test/abnormal",
            normal_test_dir="test/normal",
            test_split_mode="from_dir",
            val_split_mode="from_test",
            seed=42,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )
    else:
        if not args.mvtec_category:
            logging.error("--mvtec_category is required for dataset_type=mvtec")
            raise SystemExit(2)
        datamodule = MVTecAD(
            root=str(args.dataset_root),
            category=args.mvtec_category,
            task="segmentation",
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )

    # Build model (PatchCore)
    model = Patchcore(
        backbone=args.backbone,
        layers=args.layers,  # type: ignore[arg-type]
        pre_trained=True,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=args.num_neighbors,
    )

    # Trainer via Engine
    engine = Engine()

    logging.info(
        "Starting training | dataset=%s root=%s model=PatchCore epochs=%d",
        args.dataset_type,
        args.dataset_root,
        args.max_epochs,
    )

    # Configure some trainer options if available via Engine
    # Engine.fit can accept 'trainer' kwargs in recent versions; fallback to defaults otherwise.
    try:
        engine.fit(
            datamodule=datamodule,
            model=model,
            trainer={
                "max_epochs": args.max_epochs,
                "accelerator": args.accelerator,
                "devices": args.devices,
                "log_every_n_steps": 1,
                "enable_checkpointing": True,
            },
        )
    except TypeError:
        # Older Engine signature without trainer dict support
        logging.warning("Engine.fit signature does not accept trainer kwargs; using defaults.")
        engine.fit(datamodule=datamodule, model=model)

    logging.info("Training finished.")


if __name__ == "__main__":
    main()
