"""
Training script for anomaly detection models.
Usage: python scripts/train_model.py --config configs/padim.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import ConfigManager
from src.training import train_model_from_config
from src.utils import setup_logging
import subprocess
import shutil
import glob


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Override dataset path from config"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override max epochs from config"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    print(f"🚀 Starting training with config: {args.config}")
    
    try:
        # Prefer native training via our wrapper (Anomalib API)
        trainer = train_model_from_config(args.config)
        print("✅ Training completed successfully!")
        
        # Save model
        model_name = Path(args.config).stem
        save_path = f"models/{model_name}_trained.ckpt"
        trainer.save_model(save_path)
        print(f"💾 Model saved to: {save_path}")
        
    except Exception as e:
        # Fallback for Anomalib v2 API changes: use CLI to train and then copy the produced checkpoint
        print(f"⚠️ Native training failed, falling back to CLI: {e}")
        try:
            python_scripts = Path(sys.executable).parent
            anomalib_exe = python_scripts / ("anomalib.exe" if sys.platform.startswith("win") else "anomalib")
            if not anomalib_exe.exists():
                raise FileNotFoundError(f"Anomalib CLI not found at {anomalib_exe}")

            # Run CLI training
            cmd = [str(anomalib_exe), "train", "--config", args.config]
            print(f"▶️  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent), check=True)

            # Find latest checkpoint under results/**/weights/lightning/model.ckpt
            results_dir = Path("results")
            candidates = sorted(results_dir.glob("**/weights/lightning/model.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise FileNotFoundError("No checkpoint found under results/**/weights/lightning/")

            latest_ckpt = candidates[0]
            model_name = Path(args.config).stem
            save_path = Path("models") / f"{model_name}_trained.ckpt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(latest_ckpt, save_path)
            print(f"💾 Model saved to: {save_path}")
            print("✅ CLI training and export completed successfully!")

        except subprocess.CalledProcessError as se:
            print(f"❌ CLI training failed with return code {se.returncode}")
            sys.exit(se.returncode or 1)
        except Exception as ie:
            print(f"❌ Training failed: {ie}")
            sys.exit(1)


if __name__ == "__main__":
    main()