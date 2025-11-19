# -*- coding: utf-8 -*-
"""
Post-process anomaly heatmaps: blur, normalize, threshold, and morphological closing.
Generates overlay images on top of originals.

Usage:
  .\.venv\Scripts\python.exe scripts\postprocess_heatmap.py --heatmap results/.../anomaly_map.png --image path/to/orig.png --out outputs/overlays

It also accepts a directory of heatmaps and images and will match by filename.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def normalize(img: np.ndarray) -> np.ndarray:
    mn = img.min()
    mx = img.max()
    if mx - mn <= 0:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def postprocess_map(hmap: np.ndarray, blur_ksize: int = 7, threshold: float = 0.5, morph_k: int = 5) -> np.ndarray:
    # Blur
    if blur_ksize > 1:
        hmap = cv2.GaussianBlur(hmap, (blur_ksize, blur_ksize), 0)

    # Normalize to [0,1]
    hmap = normalize(hmap.astype(np.float32))

    # Threshold
    th_mask = (hmap >= threshold).astype("uint8") * 255

    # Morph closing to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    th_mask = cv2.morphologyEx(th_mask, cv2.MORPH_CLOSE, kernel)

    return th_mask


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    color_mask = np.zeros_like(image)
    color_mask[:, :, 2] = mask  # red channel
    overlayed = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return overlayed


def find_pairs(heatmap_dir: Path, image_dir: Path) -> Iterable[tuple[Path, Path]]:
    # yield (heatmap_path, image_path) pairs matched by stem
    hmaps = {p.stem: p for p in heatmap_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    imgs = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    for k, h in hmaps.items():
        if k in imgs:
            yield h, imgs[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heatmap", type=Path, help="Path to heatmap image or directory")
    ap.add_argument("--image", type=Path, help="Path to original image or directory")
    ap.add_argument("--out", type=Path, default=Path("outputs/overlays"), help="Output directory")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--blur", type=int, default=7)
    ap.add_argument("--morph_k", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.6)
    args = ap.parse_args()

    if args.heatmap is None or args.image is None:
        print("Both --heatmap and --image are required (file or directory)")
        raise SystemExit(2)

    args.out.mkdir(parents=True, exist_ok=True)

    if args.heatmap.is_dir() and args.image.is_dir():
        pairs = list(find_pairs(args.heatmap, args.image))
    elif args.heatmap.is_file() and args.image.is_file():
        pairs = [(args.heatmap, args.image)]
    else:
        print("Heatmap and image should both be files or both be directories")
        raise SystemExit(2)

    for hpath, ipath in pairs:
        h = cv2.imread(str(hpath), cv2.IMREAD_UNCHANGED)
        if h is None:
            print(f"Failed to read heatmap: {hpath}")
            continue
        if h.ndim == 3:
            # convert to grayscale
            h = cv2.cvtColor(h, cv2.COLOR_BGR2GRAY)
        img = cv2.imread(str(ipath), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to read image: {ipath}")
            continue
        # Resize heatmap to match original image size if necessary
        if h.shape != img.shape[:2]:
            # cv2.resize expects (width, height)
            h = cv2.resize(h, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        mask = postprocess_map(h.astype(np.float32), blur_ksize=args.blur, threshold=args.threshold, morph_k=args.morph_k)
        ov = overlay(img, mask, alpha=args.alpha)

        out_img = args.out / (hpath.stem + "_overlay.png")
        out_mask = args.out / (hpath.stem + "_mask.png")
        cv2.imwrite(str(out_img), ov)
        cv2.imwrite(str(out_mask), mask)
        print(f"Wrote: {out_img} and {out_mask}")


if __name__ == "__main__":
    main()
