"""
This script was removed as part of repository cleanup.

It is intentionally left as a stub to avoid broken imports. If you need
single-image visualization again, ask to restore the feature and tasks.
"""

import sys

print("visualize_single_image.py was removed from the project. Nothing to do.")
sys.exit(0)
"""Single-image anomaly visualization via Anomalib CLI artifacts.

Workflow:
1. Build a temporary Folder-style dataset around the provided image.
2. Clone and modify PatchCore config to point to that dataset and disable val split.
3. Run `anomalib predict` to generate anomaly heatmap & overlay images.
4. Locate generated artifacts (original, *_heatmap, *_overlay) and compose a panel.

Outputs written to the --output directory:
    original.png
    heatmap.png
    overlay.png
    example_inference.png   (composite panel: original | heatmap | overlay)

Requires:
    - Trained Lightning checkpoint (.ckpt), e.g. models/patchcore_ready.ckpt
    - Anomalib installed

Usage (PowerShell):
    .\.venv\Scripts\python.exe scripts\visualize_single_image.py \
        --image img\examples_images.png \
        --ckpt models\patchcore_ready.ckpt \
        --config configs\patchcore_ready.yaml \
        --output outputs\single_image \
        --threshold 0.5
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import shutil
import yaml
import subprocess


def setup_logger(level: int = logging.INFO) -> None:
    """Configure root logger."""
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def load_image_bgr(path: Path) -> np.ndarray:
    """Load an image in BGR format using OpenCV."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def normalize_map(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1]."""
    arr = arr.astype(np.float32)
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max <= a_min:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - a_min) / (a_max - a_min + 1e-12)


def colorize_heatmap(norm_map: np.ndarray) -> np.ndarray:
    """Convert a [0,1] heatmap to color (BGR) using JET colormap."""
    heat_u8 = (np.clip(norm_map, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)


def draw_label(image_bgr: np.ndarray, text: str, *, font_scale: float = 0.6) -> np.ndarray:
    """Draw an outlined text label on the image."""
    out = image_bgr.copy()
    org = (12, 28)
    color_fg = (255, 255, 255)  # white
    color_bg = (0, 0, 0)        # black
    thickness_fg, thickness_bg = 1, 3
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bg, thickness_bg, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_fg, thickness_fg, cv2.LINE_AA)
    return out


def compose_panel(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Horizontally concatenate three equally-sized images (BGR)."""
    return cv2.hconcat([left, middle, right])


def build_temp_dataset(image: Path, workdir: Path) -> Path:
    """Create minimal folder dataset with copies for required splits."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = workdir / f"single_image_panel_dataset_{ts}"
    for sub in ["train/normal", "test/normal", "test/abnormal"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
        dst = root / sub / image.name
        shutil.copy2(image, dst)
    return root


def run_cli_predict(config: Path, ckpt: Path, data_root: Path, workdir: Path, dataset_name: str) -> Path:
    """Clone config, modify paths, run anomalib predict, return results images dir."""
    with config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("data", {}).setdefault("init_args", {})
    cfg["data"]["init_args"]["root"] = str(data_root)
    cfg["data"]["init_args"]["name"] = dataset_name
    cfg["data"]["init_args"]["val_split_mode"] = "none"
    cfg["data"]["init_args"]["val_split_ratio"] = 0.0
    # Ensure trainer default_root_dir stable
    cfg.setdefault("trainer", {})
    cfg["trainer"]["default_root_dir"] = str(workdir)
    temp_cfg = workdir / "panel_config.yaml"
    workdir.mkdir(parents=True, exist_ok=True)
    with temp_cfg.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    python_dir = Path(sys.executable).parent
    anomalib_exe = python_dir / ("anomalib.exe" if sys.platform.startswith("win") else "anomalib")
    if anomalib_exe.exists():
        cmd = [str(anomalib_exe), "predict", "--config", str(temp_cfg), "--ckpt_path", str(ckpt)]
    else:
        cmd = [sys.executable, "-m", "anomalib", "predict", "--config", str(temp_cfg), "--ckpt_path", str(ckpt)]
    logging.info("Running CLI predict: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"CLI predict failed (code {proc.returncode})")
    # Results path convention: results/Patchcore/<dataset_name>/latest/images/test/<class>
    results_root = Path("results") / "Patchcore" / dataset_name / "latest" / "images"
    return results_root


def locate_cli_artifacts(results_root: Path, stem: str) -> Tuple[Path, Path, Path]:
    """Find original, heatmap, and overlay images produced by CLI for given stem."""
    candidates = list(results_root.rglob(f"{stem}*"))
    # Heuristics to pick files
    heatmap = next((p for p in candidates if "heatmap" in p.stem.lower()), None)
    overlay = next((p for p in candidates if "overlay" in p.stem.lower()), None)
    # Prefer an 'original'-like file; otherwise first png matching stem
    original = next((p for p in candidates if p.stem.lower() == stem.lower()), None)
    if original is None:
        original = next((p for p in candidates if p.suffix.lower() in (".png", ".jpg", ".jpeg")), None)
    if not (original and heatmap and overlay):
        raise FileNotFoundError("Could not locate CLI artifact images (original/heatmap/overlay).")
    return original, heatmap, overlay


def compose_and_save_panel(original: Path, heatmap: Path, overlay: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    orig_img = cv2.imread(str(original), cv2.IMREAD_COLOR)
    heat_img = cv2.imread(str(heatmap), cv2.IMREAD_COLOR)
    over_img = cv2.imread(str(overlay), cv2.IMREAD_COLOR)
    if orig_img is None or heat_img is None or over_img is None:
        raise RuntimeError("Failed to read artifact images from CLI.")
    panel = compose_panel(draw_label(orig_img, "Original"), draw_label(heat_img, "Heatmap"), draw_label(over_img, "Overlay"))
    cv2.imwrite(str(out_dir / "original.png"), orig_img)
    cv2.imwrite(str(out_dir / "heatmap.png"), heat_img)
    cv2.imwrite(str(out_dir / "overlay.png"), over_img)
    panel_path = out_dir / "example_inference.png"
    cv2.imwrite(str(panel_path), panel)
    return panel_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize anomaly heatmap and overlay for a single image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the image (png/jpg/tif...).")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to trained anomalib checkpoint (.ckpt).")
    parser.add_argument("--config", type=Path, required=True, help="Path to PatchCore config YAML.")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "single_image", help="Output directory.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for labeling.")
    return parser.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    logging.info("Loading image: %s", args.image)
    img_bgr = load_image_bgr(args.image)

    logging.info("Running CLI predict to produce heatmap/overlay artifacts...")
    dataset_root = build_temp_dataset(args.image, args.output)
    logging.info("Temp dataset root: %s", dataset_root)
    results_root = run_cli_predict(args.config, args.ckpt, dataset_root, args.output, dataset_name="single_image_panel")
    logging.info("CLI results root: %s", results_root)
    stem = Path(args.image).stem
    original, heatmap, overlay = locate_cli_artifacts(results_root, stem)
    logging.info("Artifacts:\n  original: %s\n  heatmap: %s\n  overlay: %s", original, heatmap, overlay)
    panel_path = compose_and_save_panel(original, heatmap, overlay, args.output)
    logging.info("Saved composite panel: %s", panel_path.resolve())
    logging.info("Done.")


if __name__ == "__main__":
    main()
