# -*- coding: utf-8 -*-
"""
Extract per-image predictions from Anomalib predict results and save as CSV.
Creates a CSV with: image_path, score, label (0=normal, 1=anomaly)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import torch
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import MVTecAD


def extract_predictions(
    ckpt_path: Path, 
    dataset_root: Path, 
    category: str, 
    output_csv: Path
) -> None:
    """Extract predictions and save to CSV."""
    
    # Load model and data
    model = Patchcore.load_from_checkpoint(str(ckpt_path))
    datamodule = MVTecAD(
        root=str(dataset_root),
        category=category,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0
    )
    
    engine = Engine()
    
    # Get predictions
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        return_predictions=True
    )
    
    # Extract scores and labels
    results: List[Tuple[str, float, int]] = []
    
    for pred in predictions:
        if hasattr(pred, 'image_path'):
            image_path = str(pred.image_path)
        else:
            # Fallback if no path in prediction
            image_path = "unknown"
            
        # Get anomaly score
        if hasattr(pred, 'pred_score'):
            score = float(pred.pred_score)
        elif hasattr(pred, 'anomaly_score'):
            score = float(pred.anomaly_score)
        else:
            score = 0.0
            
        # Determine label from path (MVTec convention)
        if 'good' in image_path or 'normal' in image_path:
            label = 0  # normal
        else:
            label = 1  # anomaly
            
        results.append((image_path, score, label))
    
    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'score', 'label'])
        writer.writerows(results)
    
    print(f"Saved {len(results)} predictions to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="Model checkpoint")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Dataset root")
    parser.add_argument("--category", type=str, required=True, help="MVTec category")
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions/scores.csv"))
    
    args = parser.parse_args()
    
    extract_predictions(args.ckpt, args.dataset_root, args.category, args.output)


if __name__ == "__main__":
    main()