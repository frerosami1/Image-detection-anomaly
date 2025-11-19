# -*- coding: utf-8 -*-
"""
Compute best threshold (Youden J) from a CSV of image scores and binary labels.

CSV expected format (no header): image_path,score,label
label: 1 = anomaly, 0 = normal

Outputs JSON with best_threshold and ROC AUC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import roc_curve, auc


def load_scores(path: Path) -> List[Tuple[float, int]]:
    items: List[Tuple[float, int]] = []
    text = path.read_text(encoding="utf8").strip()
    if not text:
        return items
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            score = float(parts[1])
            label = int(parts[2])
        except Exception:
            continue
        items.append((score, label))
    return items


def find_best_threshold(scores_labels: List[Tuple[float, int]]) -> Tuple[float, float]:
    y_scores = np.array([s for s, _ in scores_labels])
    y_true = np.array([l for _, l in scores_labels])
    if len(y_true) == 0:
        raise ValueError("No scores found")
    fpr, tpr, th = roc_curve(y_true, y_scores)
    youden = tpr - fpr
    idx = int(np.nanargmax(youden))
    best_thr = float(th[idx])
    roc_auc = float(auc(fpr, tpr))
    return best_thr, roc_auc


def main() -> None:
    ap = argparse.ArgumentParser(description="Find optimal threshold from CSV")
    ap.add_argument("--csv", type=Path, required=True, help="CSV file (image_path,score,label)")
    ap.add_argument("--out", type=Path, default=Path("outputs/predictions/threshold.json"), help="Output JSON path")
    args = ap.parse_args()

    items = load_scores(args.csv)
    if not items:
        print(f"No valid scores found in {args.csv}")
        raise SystemExit(2)

    best_thr, roc_auc = find_best_threshold(items)
    out = {"best_threshold": best_thr, "roc_auc": roc_auc, "n_samples": len(items)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
