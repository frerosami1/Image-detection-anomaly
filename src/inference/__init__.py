"""Inference utilities for anomaly detection."""

import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class AnomalyPredictor:
    """Handles inference for trained anomaly detection models."""
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model checkpoint
        """
        self.model_path = model_path
        self.model = None
        self.device = "cpu"
        self.transforms = None
        
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a trained anomaly detection model.
        
        Args:
            model_path: Path to the model checkpoint
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            import torch

            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            if self.model_path.suffix == ".ckpt":
                # Load checkpoint and try to infer model type
                ckpt = torch.load(self.model_path, map_location=self.device)
                hparams = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}

                def _detect_model_type() -> str:
                    n = hparams.get("model") or hparams.get("model_name")
                    if isinstance(n, str):
                        return n.lower()
                    sd = ckpt.get("state_dict", {}) if isinstance(ckpt, dict) else {}
                    keys = " ".join(sd.keys()).lower()
                    if ("patchcore" in keys) or ("coreset" in keys) or ("memory_bank" in keys) or ("nearest" in keys):
                        return "patchcore"
                    if "padim" in keys or "gaussian" in keys:
                        return "padim"
                    return "padim"

                model_type = _detect_model_type()

                if model_type == "patchcore":
                    from anomalib.models.image.patchcore.lightning_model import Patchcore
                    self.model = Patchcore.load_from_checkpoint(str(self.model_path), map_location=self.device)
                elif model_type == "padim":
                    try:
                        from anomalib.models.image.padim.lightning_model import Padim
                        self.model = Padim.load_from_checkpoint(str(self.model_path), map_location=self.device)
                    except Exception:
                        # Fallback to generic
                        from anomalib.models import get_model
                        self.model = get_model("padim")
                        if isinstance(ckpt, dict) and "state_dict" in ckpt:
                            self.model.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
                else:
                    from anomalib.models import get_model
                    self.model = get_model(model_type)
                    if isinstance(ckpt, dict) and "state_dict" in ckpt:
                        self.model.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
            else:
                # Assume it's a regular state dict for PaDiM by default
                from anomalib.models import get_model
                self.model = get_model("padim")
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

            # Try moving to device & eval
            try:
                self.model.to(self.device)  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                self.model.eval()  # type: ignore[union-attr]
            except Exception:
                pass

            logger.info(f"Model loaded from {self.model_path}")

        except ImportError:
            logger.error("Anomalib not installed. Install with: pip install anomalib")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Predict anomaly for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            import torch
            from torchvision import transforms
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Default transforms (adjust based on your training setup)
            if self.transforms is None:
                self.transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            
            # Apply transforms
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            # Perform inference
            with torch.no_grad():
                try:
                    outputs = self.model(input_tensor)  # type: ignore[operator]
                except TypeError:
                    batch = {"image": input_tensor}
                    if hasattr(self.model, "forward"):
                        outputs = self.model.forward(batch)  # type: ignore[union-attr]
                    elif hasattr(self.model, "predict_step"):
                        outputs = self.model.predict_step(batch, 0)  # type: ignore[union-attr]
                    else:
                        outputs = None

            # Normalize outputs
            score: float = 0.0
            label: int = 0
            anomaly_map_np: Optional[np.ndarray] = None

            if isinstance(outputs, dict):
                score = float(outputs.get("pred_score", 0.0))
                label = int(outputs.get("pred_label", 0))
                if "anomaly_map" in outputs:
                    try:
                        anomaly_map_np = outputs["anomaly_map"].detach().cpu().numpy()  # type: ignore[union-attr]
                    except Exception:
                        pass
            elif outputs is not None:
                try:
                    if hasattr(outputs, "mean"):
                        score = float(outputs.mean().item())  # type: ignore[union-attr]
                    elif isinstance(outputs, (list, tuple)) and len(outputs) > 0 and hasattr(outputs[0], "mean"):
                        score = float(outputs[0].mean().item())  # type: ignore[union-attr]
                except Exception:
                    pass
                label = int(score > 0.5)

            results = {
                "image_path": str(image_path),
                "anomaly_score": score,
                "prediction": label,
                "is_anomaly": bool(label),
            }

            if anomaly_map_np is not None:
                results["anomaly_map"] = anomaly_map_np
                results["anomaly_map_shape"] = anomaly_map_np.shape
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting image: {e}")
            raise
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Predict anomalies for a batch of images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    "image_path": str(image_path),
                    "error": str(e)
                })
        
        return results
    
    def predict_directory(
        self, 
        directory_path: Union[str, Path],
        extensions: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Predict anomalies for all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            extensions: List of file extensions to process
            
        Returns:
            List of prediction results
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(directory_path.rglob(f"*{ext}"))
            image_files.extend(directory_path.rglob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} images in {directory_path}")
        
        return self.predict_batch(image_files)
    
    def visualize_results(
        self, 
        results: List[Dict[str, Any]], 
        output_dir: Union[str, Path],
        threshold: float = 0.5
    ) -> None:
        """
        Create visualizations of the prediction results.
        
        Args:
            results: List of prediction results
            output_dir: Directory to save visualizations
            threshold: Threshold for anomaly classification
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            
            # Create summary plot
            scores = [r.get("anomaly_score", 0.0) for r in results if "error" not in r]
            # Apply provided threshold to determine decisions for visualization/summary
            predictions = [float(s) >= float(threshold) for s in scores]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Score distribution
            ax1.hist(scores, bins=20, alpha=0.7, edgecolor='black')
            ax1.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
            ax1.set_xlabel('Anomaly Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Anomaly Score Distribution')
            ax1.legend()
            
            # Prediction counts
            normal_count = sum(1 for p in predictions if not p)
            anomaly_count = sum(1 for p in predictions if p)
            
            ax2.bar(['Normal', 'Anomaly'], [normal_count, anomaly_count], 
                   color=['green', 'red'], alpha=0.7)
            ax2.set_ylabel('Count')
            ax2.set_title('Prediction Summary')
            
            plt.tight_layout()
            plt.savefig(output_dir / "prediction_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save detailed results
            with open(output_dir / "results.txt", 'w') as f:
                f.write("Image Anomaly Detection Results\n")
                f.write("=" * 40 + "\n\n")
                
                for result in results:
                    if "error" not in result:
                        score = float(result.get("anomaly_score", 0.0))
                        decision = score >= float(threshold)
                        f.write(f"Image: {result['image_path']}\n")
                        f.write(f"Score: {score:.4f}\n")
                        f.write(f"Anomaly(thr={threshold}): {decision}\n")
                        f.write("-" * 30 + "\n")
                    else:
                        f.write(f"Error processing {result['image_path']}: {result['error']}\n")
                        f.write("-" * 30 + "\n")
            
            logger.info(f"Visualizations saved to {output_dir}")
            
        except ImportError:
            logger.error("Matplotlib not installed. Install with: pip install matplotlib")
            raise
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise


def load_and_predict(
    model_path: Union[str, Path],
    image_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Convenience function to load model and predict single image.
    
    Args:
        model_path: Path to trained model
        image_path: Path to image for prediction
        
    Returns:
        Prediction results
    """
    predictor = AnomalyPredictor()
    predictor.load_model(model_path)
    return predictor.predict_image(image_path)