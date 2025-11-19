"""Training utilities for anomaly detection models."""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
import torch

logger = logging.getLogger(__name__)


class AnomalibTrainer:
    """Wrapper for training anomaly detection models with Anomalib."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.model = None
        self.datamodule = None
        self.trainer = None
        
    def setup_model(self, model_name: str = "padim", **kwargs) -> None:
        """
        Setup the anomaly detection model.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional model parameters
        """
        try:
            # Import Anomalib components
            from anomalib.models import get_model
            
            # Create model
            self.model = get_model(model_name)
            logger.info(f"Initialized {model_name} model")
            
        except ImportError:
            logger.error("Anomalib not installed. Install with: pip install anomalib")
            raise
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise
    
    def setup_data(self, data_path: Union[str, Path], **kwargs) -> None:
        """
        Setup the data module for training.
        
        Args:
            data_path: Path to dataset
            **kwargs: Additional data parameters
        """
        try:
            from anomalib.data import get_datamodule
            
            # Default data configuration
            data_config = {
                "class_path": "anomalib.data.Folder",
                "init_args": {
                    "root": str(data_path),
                    "normal_dir": "normal",
                    "abnormal_dir": "abnormal",
                    "task": "segmentation",
                    "image_size": [224, 224],
                    "train_batch_size": 32,
                    "eval_batch_size": 32,
                    "num_workers": 4
                }
            }
            
            # Update with provided kwargs
            data_config["init_args"].update(kwargs)
            
            self.datamodule = get_datamodule(data_config)
            logger.info(f"Setup data module for {data_path}")
            
        except ImportError:
            logger.error("Anomalib not installed. Install with: pip install anomalib")
            raise
        except Exception as e:
            logger.error(f"Error setting up data: {e}")
            raise
    
    def train(
        self,
        max_epochs: int = 10,
        accelerator: str = "auto",
        devices: int = 1,
        **trainer_kwargs
    ) -> None:
        """
        Train the anomaly detection model.
        
        Args:
            max_epochs: Maximum number of training epochs
            accelerator: Training accelerator (auto, gpu, cpu)
            devices: Number of devices to use
            **trainer_kwargs: Additional trainer arguments
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        if self.datamodule is None:
            raise ValueError("Data module not initialized. Call setup_data() first.")
        
        try:
            import pytorch_lightning as pl
            
            # Setup trainer
            trainer_config = {
                "max_epochs": max_epochs,
                "accelerator": accelerator,
                "devices": devices,
                "enable_checkpointing": True,
                "default_root_dir": "./experiments",
                **trainer_kwargs
            }
            
            self.trainer = pl.Trainer(**trainer_config)
            
            # Start training
            logger.info("Starting training...")
            self.trainer.fit(self.model, self.datamodule)
            logger.info("Training completed successfully")
            
        except ImportError:
            logger.error("PyTorch Lightning not installed. Install Anomalib with: pip install anomalib")
            raise
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.trainer is None or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            logger.info("Starting evaluation...")
            results = self.trainer.test(self.model, self.datamodule)
            logger.info("Evaluation completed")
            return results[0] if results else {}
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def save_model(self, save_path: Union[str, Path]) -> None:
        """
        Save the trained model.
        
        Args:
            save_path: Path where to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model checkpoint
            if hasattr(self.trainer, 'save_checkpoint'):
                self.trainer.save_checkpoint(save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
            
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the saved model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            if self.model is None:
                raise ValueError("Model not initialized. Call setup_model() first.")
            
            # Load model weights
            if model_path.suffix == '.ckpt':
                # PyTorch Lightning checkpoint
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # Regular PyTorch state dict
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


def train_model_from_config(config_path: Union[str, Path]) -> AnomalibTrainer:
    """
    Train a model using configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Trained AnomalibTrainer instance
    """
    try:
        from anomalib.config import get_configurable_parameters
        from anomalib.data import get_datamodule
        from anomalib.models import get_model
        import pytorch_lightning as pl
        
        # Load configuration
        config = get_configurable_parameters(config_path)
        
        # Setup components
        datamodule = get_datamodule(config)
        model = get_model(config)
        
        # Setup trainer
        trainer = pl.Trainer(**config.trainer)
        
        # Train
        trainer.fit(model, datamodule)
        
        # Create trainer instance and populate
        anomalib_trainer = AnomalibTrainer()
        anomalib_trainer.model = model
        anomalib_trainer.datamodule = datamodule
        anomalib_trainer.trainer = trainer
        
        return anomalib_trainer
        
    except ImportError:
        logger.error("Anomalib not installed. Install with: pip install anomalib")
        raise
    except Exception as e:
        logger.error(f"Error training model from config: {e}")
        raise