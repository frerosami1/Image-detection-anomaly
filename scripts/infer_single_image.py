"""
This script was removed as part of repository cleanup.

It is intentionally left as a stub to avoid broken imports. If you need
single-image inference again, ask to restore the feature and tasks.
"""

import sys

print("infer_single_image.py was removed from the project. Nothing to do.")
sys.exit(0)
"""Single-image anomaly inference using Anomalib CLI.

This script:
- Creates a temporary Folder-style dataset around the given image
- Calls `anomalib predict` with the provided PatchCore (or other) config and checkpoint
- Saves outputs under the provided workdir

Usage (PowerShell):
  .\.venv\Scripts\python.exe scripts\infer_single_image.py ^
    --image path\to\image.jpg ^
    --ckpt models\patchcore_ready.ckpt ^
    --config configs\patchcore_ready.yaml ^
    --workdir outputs\single_image

Notes:
- Requires an existing trained checkpoint.
- data.root is overridden at runtime to the temp dataset path.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import tempfile
import yaml


def setup_logger() -> None:
    """Configure logging for the script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def create_temp_dataset(image_path: Path, workdir: Path) -> Path:
    """Create a temporary folder dataset around the single image.

    Layout mirrors Folder datamodule expectations. Only test split is needed
    for prediction with a trained checkpoint, but we also create the empty
    abnormal directory referenced in config to avoid missing-path issues.

    Args:
        image_path: Path to the image file to analyze.
        workdir: Working/output directory.

    Returns:
        Root path of the temporary dataset.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = workdir / f"single_image_dataset_{ts}"
    train_normal_dir = dataset_root / "train" / "normal"
    test_normal_dir = dataset_root / "test" / "normal"
    test_abnormal_dir = dataset_root / "test" / "abnormal"
    train_normal_dir.mkdir(parents=True, exist_ok=True)
    test_normal_dir.mkdir(parents=True, exist_ok=True)
    test_abnormal_dir.mkdir(parents=True, exist_ok=True)
    # Copy the image into test/normal and also into train/normal to satisfy
    # Folder datamodule checks that expect at least one training image.
    dst_test = test_normal_dir / image_path.name
    shutil.copy2(image_path, dst_test)
    dst_train = train_normal_dir / (image_path.stem + "_traincopy" + image_path.suffix)
    shutil.copy2(image_path, dst_train)
    # Also place a copy in test/abnormal to satisfy datamodule expectations when
    # abnormal_dir is configured in the YAML.
    dst_abnormal = test_abnormal_dir / (image_path.stem + "_abcopy" + image_path.suffix)
    shutil.copy2(image_path, dst_abnormal)
    logging.info("Created temp dataset at: %s", dataset_root)
    logging.info("Copied image to test: %s", dst_test)
    logging.info("Copied image to train: %s", dst_train)
    logging.info("Copied image to test abnormal: %s", dst_abnormal)
    return dataset_root


def run_anomalib_predict(config: Path, ckpt: Path, data_root: Path, output_dir: Path) -> int:
    """Run `anomalib predict` using a temporary YAML with updated paths.

    Some Anomalib CLI versions do not accept command-line Hydra overrides for
    nested init_args. We clone the YAML config, update paths, and run predict.

    Args:
        config: Config YAML (PatchCore or other model) compatible with CLI.
        ckpt: Trained checkpoint.
        data_root: Temporary dataset root path.
        output_dir: Desired output directory for predictions and artifacts.

    Returns:
        Subprocess return code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Safely update config fields
    try:
        cfg.setdefault("data", {}).setdefault("init_args", {})
        cfg["data"]["init_args"]["root"] = str(data_root)
        cfg["data"]["init_args"]["name"] = "single_image"
        # Avoid validation split for single-image predict
        cfg["data"]["init_args"]["val_split_mode"] = "none"
        cfg["data"]["init_args"]["val_split_ratio"] = 0.0
    except Exception as exc:
        logging.warning("Could not update data.init_args in config: %s", exc)

    try:
        cfg.setdefault("trainer", {})
        cfg["trainer"]["default_root_dir"] = str(output_dir)
    except Exception as exc:
        logging.warning("Could not update trainer.default_root_dir: %s", exc)

    # Write temp config next to output_dir
    temp_cfg_path = output_dir / "single_image_config.yaml"
    with temp_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    python_dir = Path(sys.executable).parent
    anomalib_exe = python_dir / ("anomalib.exe" if sys.platform.startswith("win") else "anomalib")

    if anomalib_exe.exists():
        cmd = [
            str(anomalib_exe),
            "predict",
            "--config",
            str(temp_cfg_path),
            "--ckpt_path",
            str(ckpt),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "anomalib",
            "predict",
            "--config",
            str(temp_cfg_path),
            "--ckpt_path",
            str(ckpt),
        ]

    logging.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent), check=False)
    return proc.returncode


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image anomaly inference (Folder/PatchCore)")
    parser.add_argument("--image", type=Path, required=True, help="Path to image to analyze.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to trained checkpoint (.ckpt).")
    parser.add_argument("--config", type=Path, required=True, help="Path to Anomalib config YAML.")
    parser.add_argument("--workdir", type=Path, default=Path("outputs") / "single_image", help="Working/output directory.")
    return parser.parse_args(argv)


def main() -> None:
    setup_logger()
    args = parse_args()

    if not args.image.exists():
        logging.error("Image not found: %s", args.image)
        sys.exit(2)
    if not args.ckpt.exists():
        logging.error("Checkpoint not found: %s", args.ckpt)
        sys.exit(2)
    if not args.config.exists():
        logging.error("Config not found: %s", args.config)
        sys.exit(2)

    dataset_root = create_temp_dataset(args.image, args.workdir)
    code = run_anomalib_predict(args.config, args.ckpt, dataset_root, args.workdir)
    if code == 0:
        logging.info("Inference complete. Outputs: %s", args.workdir.resolve())
    else:
        logging.error("Inference failed with code: %s", code)
    sys.exit(code)


if __name__ == "__main__":  # pragma: no cover
    main()
