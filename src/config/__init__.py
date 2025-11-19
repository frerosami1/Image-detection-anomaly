"""Configuration utilities for anomaly detection models."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig


class ConfigManager:
    """Manages configuration files and settings for anomaly detection models."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or Path(__file__).parent.parent.parent / "configs"
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self, config_path: str | Path) -> DictConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration as OmegaConf object
        """
        if isinstance(config_path, str):
            config_path = Path(config_path)
            
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
            
        return OmegaConf.load(config_path)
    
    def save_config(self, config: DictConfig, save_path: str | Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration object to save
            save_path: Path where to save the configuration
        """
        if isinstance(save_path, str):
            save_path = Path(save_path)
            
        if not save_path.is_absolute():
            save_path = self.config_dir / save_path
            
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            OmegaConf.save(config, f)
    
    def create_default_config(self, model_name: str = "padim") -> DictConfig:
        """
        Create a default configuration for a given model.
        
        Args:
            model_name: Name of the anomaly detection model
            
        Returns:
            Default configuration
        """
        default_configs = {
            "padim": {
                "model": {
                    "name": "padim",
                    "backbone": "resnet18",
                    "pre_trained": True,
                    "layers": ["layer1", "layer2", "layer3"]
                },
                "dataset": {
                    "name": "custom",
                    "format": "folder",
                    "path": "./data/datasets",
                    "normal_dir": "normal",
                    "abnormal_dir": "abnormal",
                    "task": "segmentation",
                    "image_size": [224, 224],
                    "train_batch_size": 32,
                    "eval_batch_size": 32,
                    "num_workers": 8
                },
                "metrics": {
                    "image": ["AUROC", "F1Score"],
                    "pixel": ["AUROC", "F1Score"],
                    "threshold": {
                        "method": "adaptive"
                    }
                },
                "trainer": {
                    "enable_checkpointing": True,
                    "default_root_dir": "./experiments",
                    "gpus": 1,
                    "max_epochs": 1,
                    "val_check_interval": 1.0,
                    "check_val_every_n_epoch": 1
                },
                "optimization": {
                    "export_mode": "torch"
                },
                "logging": {
                    "logger": ["tensorboard", "wandb"],
                    "log_graph": False
                }
            }
        }
        
        return OmegaConf.create(default_configs.get(model_name, default_configs["padim"]))
    
    def update_config(self, base_config: DictConfig, updates: Dict[str, Any]) -> DictConfig:
        """
        Update configuration with new values.
        
        Args:
            base_config: Base configuration
            updates: Dictionary of updates to apply
            
        Returns:
            Updated configuration
        """
        return OmegaConf.merge(base_config, updates)


def get_config_manager() -> ConfigManager:
    """Get a singleton instance of ConfigManager."""
    if not hasattr(get_config_manager, '_instance'):
        get_config_manager._instance = ConfigManager()
    return get_config_manager._instance