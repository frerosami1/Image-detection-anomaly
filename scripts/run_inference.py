"""
Inference script for anomaly detection.
Usage: python scripts/run_inference.py --model models/padim_trained.ckpt --input data/test_images/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.inference import AnomalyPredictor
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run anomaly detection inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions",
        help="Output directory for results"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Anomaly threshold for classification"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    print(f"🔍 Running inference with model: {args.model}")
    print(f"📂 Input: {args.input}")
    
    try:
        # Initialize predictor
        predictor = AnomalyPredictor()
        predictor.load_model(args.model)
        
        # Run prediction
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single image
            result = predictor.predict_image(input_path)
            results = [result]
            print(f"🎯 Prediction for {input_path.name}:")
            print(f"   Score: {result['anomaly_score']:.4f}")
            print(f"   Anomaly: {result['is_anomaly']}")
        else:
            # Directory of images
            results = predictor.predict_directory(input_path)
            print(f"🎯 Processed {len(results)} images")
        
        # Create visualizations
        output_dir = Path(args.output)
        predictor.visualize_results(results, output_dir, args.threshold)
        
        print(f"✅ Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()