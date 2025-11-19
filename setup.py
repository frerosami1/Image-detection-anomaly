# -*- coding: utf-8 -*-
"""
Setup script for the Image Anomaly Detection project.
Run this script to initialize the project environment.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"[INFO] {description}...")
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[OK] {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up Image Anomaly Detection Project")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("[WARN] Some packages may have failed to install. Check the output above.")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed",
        "data/datasets", 
        "experiments",
        "models",
        "logs",
        "outputs/visualizations",
        "outputs/predictions"
    ]
    
    print("\nCreating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created: {directory}")
    
    # Optionally create a sample dataset structure (disabled by default)
    # Enable by setting environment variable CREATE_SAMPLE_DATASET=1
    if os.getenv("CREATE_SAMPLE_DATASET", "0").lower() in {"1", "true", "yes"}:
        from src.data import create_sample_dataset
        try:
            sample_path = create_sample_dataset(Path("data/datasets"), "sample")
            print(f"[OK] Sample dataset created at: {sample_path}")
        except Exception as e:
            print(f"[WARN] Could not create sample dataset: {e}")
    else:
        print("[INFO] Skipping sample dataset creation. Set CREATE_SAMPLE_DATASET=1 to enable.")
    
    # Verify installation
    print("\nVerifying installation...")
    try:
        import torch
        import anomalib
        print(f"[OK] PyTorch: {torch.__version__}")
        print(f"[OK] Anomalib: {anomalib.__version__}")
        
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("[INFO] CUDA not available, using CPU")
            
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Open notebooks/01_getting_started.ipynb to begin")
    print("2. Place your image datasets in data/datasets/")
    print("3. Configure model settings in configs/")
    print("4. Start training your anomaly detection models!")
    
    return True


if __name__ == "__main__":
    main()