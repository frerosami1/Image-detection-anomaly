"""Custom models and model utilities."""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating and configuring anomaly detection models."""
    
    SUPPORTED_MODELS = {
        'padim': 'PaDiM (Patch Distribution Modeling)',
        'stfpm': 'STFPM (Student-Teacher Feature Pyramid Matching)', 
        'patchcore': 'PatchCore',
        'fastflow': 'FastFlow',
        'cflow': 'C-Flow',
        'dfm': 'Deep Feature Modeling',
        'dfkde': 'Deep Feature Kernel Density Estimation',
        'ganomaly': 'GANomaly',
        'skip_ganomaly': 'Skip-GANomaly',
        'draem': 'DRÆM'
    }
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """
        Get list of available models.
        
        Returns:
            Dictionary of model names and descriptions
        """
        return cls.SUPPORTED_MODELS.copy()
    
    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> Any:
        """
        Create an anomaly detection model.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Model-specific parameters
            
        Returns:
            Configured model instance
        """
        if model_name not in cls.SUPPORTED_MODELS:
            available = ', '.join(cls.SUPPORTED_MODELS.keys())
            raise ValueError(f"Unsupported model: {model_name}. Available: {available}")
        
        try:
            from anomalib.models import get_model
            
            model = get_model(model_name)
            logger.info(f"Created {model_name} model")
            return model
            
        except ImportError:
            logger.error("Anomalib not installed. Install with: pip install anomalib")
            raise
        except Exception as e:
            logger.error(f"Error creating model {model_name}: {e}")
            raise
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """
        Get default configuration for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Default configuration dictionary
        """
        configs = {
            'padim': {
                'backbone': 'resnet18',
                'pre_trained': True,
                'layers': ['layer1', 'layer2', 'layer3']
            },
            'stfpm': {
                'backbone': 'resnet18',
                'pre_trained': True,
                'layers': ['layer1', 'layer2', 'layer3']
            },
            'patchcore': {
                'backbone': 'wide_resnet50_2',
                'pre_trained': True,
                'layers': ['layer2', 'layer3'],
                'coreset_sampling_ratio': 0.1,
                'num_neighbors': 9
            },
            'fastflow': {
                'backbone': 'resnet18',
                'pre_trained': True,
                'flow_steps': 8,
                'conv3x3_only': False,
                'hidden_ratio': 1.0
            },
            'cflow': {
                'backbone': 'wide_resnet50_2',
                'pre_trained': True,
                'layers': ['layer1', 'layer2', 'layer3'],
                'decoder': 'freia-cflow',
                'pool_layers': 3
            }
        }
        
        return configs.get(model_name, {})


class ModelEvaluator:
    """Utilities for evaluating trained models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def compute_metrics(self, predictions, ground_truth, threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            threshold: Classification threshold
            
        Returns:
            Dictionary of computed metrics
        """
        try:
            import numpy as np
            from sklearn.metrics import (
                roc_auc_score, precision_score, recall_score, 
                f1_score, accuracy_score, confusion_matrix
            )
            
            # Convert to numpy arrays
            y_true = np.array(ground_truth)
            y_scores = np.array(predictions)
            y_pred = (y_scores > threshold).astype(int)
            
            # Compute metrics
            metrics = {
                'auroc': roc_auc_score(y_true, y_scores),
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'threshold': threshold
            }
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics.update({
                'true_positives': int(tp),
                'true_negatives': int(tn), 
                'false_positives': int(fp),
                'false_negatives': int(fn)
            })
            
            return metrics
            
        except ImportError:
            logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
            raise
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            raise
    
    def find_optimal_threshold(self, predictions, ground_truth) -> float:
        """
        Find optimal threshold using F1 score.
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            
        Returns:
            Optimal threshold value
        """
        try:
            import numpy as np
            from sklearn.metrics import f1_score
            
            thresholds = np.linspace(0, 1, 100)
            f1_scores = []
            
            for threshold in thresholds:
                y_pred = (np.array(predictions) > threshold).astype(int)
                f1 = f1_score(ground_truth, y_pred, zero_division=0)
                f1_scores.append(f1)
            
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            logger.info(f"Optimal threshold: {optimal_threshold:.3f} (F1: {f1_scores[optimal_idx]:.3f})")
            return float(optimal_threshold)
            
        except ImportError:
            logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
            raise
        except Exception as e:
            logger.error(f"Error finding optimal threshold: {e}")
            raise


def list_available_models() -> None:
    """Print available models and their descriptions."""
    models = ModelFactory.get_available_models()
    
    print("Available Anomaly Detection Models:")
    print("=" * 50)
    
    for name, description in models.items():
        print(f"{name:15} - {description}")
    
    print(f"\nTotal: {len(models)} models available")


def create_model_from_config(config: Dict[str, Any]) -> Any:
    """
    Create a model from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured model instance
    """
    model_name = config.get('name', 'padim')
    model_params = config.get('parameters', {})
    
    return ModelFactory.create_model(model_name, **model_params)