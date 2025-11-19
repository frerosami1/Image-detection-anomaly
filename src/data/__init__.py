"""Data handling utilities for anomaly detection."""

import os
import shutil
from pathlib import Path
from typing import Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DatasetOrganizer:
    """Organizes datasets according to Anomalib's expected structure."""
    
    def __init__(self, root_dir: Union[str, Path]):
        """
        Initialize the dataset organizer.
        
        Args:
            root_dir: Root directory for datasets
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dataset_structure(self, dataset_name: str) -> Path:
        """
        Create the standard Anomalib dataset structure.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to the created dataset directory
        """
        dataset_path = self.root_dir / dataset_name
        
        # Create directory structure
        directories = [
            dataset_path / "train" / "normal",
            dataset_path / "test" / "normal", 
            dataset_path / "test" / "abnormal",
            dataset_path / "ground_truth" / "abnormal"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Created dataset structure at {dataset_path}")
        return dataset_path
    
    def organize_images(
        self, 
        source_dir: Union[str, Path],
        dataset_name: str,
        train_split: float = 0.8,
        normal_patterns: List[str] = None,
        abnormal_patterns: List[str] = None
    ) -> None:
        """
        Organize images from a source directory into Anomalib structure.
        
        Args:
            source_dir: Directory containing source images
            dataset_name: Target dataset name
            train_split: Fraction of normal images for training
            normal_patterns: File patterns for normal images
            abnormal_patterns: File patterns for abnormal images
        """
        source_dir = Path(source_dir)
        dataset_path = self.create_dataset_structure(dataset_name)
        
        if normal_patterns is None:
            normal_patterns = ["*normal*", "*good*", "*ok*"]
        if abnormal_patterns is None:
            abnormal_patterns = ["*abnormal*", "*defect*", "*bad*", "*anomaly*"]
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        all_files = []
        for ext in image_extensions:
            all_files.extend(source_dir.rglob(f"*{ext}"))
            all_files.extend(source_dir.rglob(f"*{ext.upper()}"))
        
        normal_images = []
        abnormal_images = []
        
        # Classify images based on patterns
        for file_path in all_files:
            file_name = file_path.name.lower()
            
            if any(pattern.replace("*", "") in file_name for pattern in normal_patterns):
                normal_images.append(file_path)
            elif any(pattern.replace("*", "") in file_name for pattern in abnormal_patterns):
                abnormal_images.append(file_path)
            else:
                # Default to normal if no pattern matches
                normal_images.append(file_path)
        
        # Split normal images for training and testing
        split_idx = int(len(normal_images) * train_split)
        train_normal = normal_images[:split_idx]
        test_normal = normal_images[split_idx:]
        
        # Copy files to appropriate directories
        self._copy_files(train_normal, dataset_path / "train" / "normal")
        self._copy_files(test_normal, dataset_path / "test" / "normal")
        self._copy_files(abnormal_images, dataset_path / "test" / "abnormal")
        
        logger.info(f"Organized {len(train_normal)} training images")
        logger.info(f"Organized {len(test_normal)} normal test images")
        logger.info(f"Organized {len(abnormal_images)} abnormal test images")
    
    def _copy_files(self, file_list: List[Path], target_dir: Path) -> None:
        """Copy files to target directory."""
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in file_list:
            target_path = target_dir / file_path.name
            shutil.copy2(file_path, target_path)
    
    def validate_dataset(self, dataset_name: str) -> bool:
        """
        Validate that dataset follows Anomalib structure.
        
        Args:
            dataset_name: Name of dataset to validate
            
        Returns:
            True if dataset structure is valid
        """
        dataset_path = self.root_dir / dataset_name
        
        required_dirs = [
            dataset_path / "train" / "normal",
            dataset_path / "test" / "normal",
            dataset_path / "test" / "abnormal"
        ]
        
        for required_dir in required_dirs:
            if not required_dir.exists():
                logger.error(f"Missing required directory: {required_dir}")
                return False
            
            # Check if directory has images
            image_files = list(required_dir.glob("*.jpg")) + list(required_dir.glob("*.png"))
            if not image_files:
                logger.warning(f"No images found in: {required_dir}")
        
        logger.info(f"Dataset {dataset_name} structure is valid")
        return True
    
    def get_dataset_info(self, dataset_name: str) -> dict:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        dataset_path = self.root_dir / dataset_name
        
        if not dataset_path.exists():
            return {"error": "Dataset not found"}
        
        info = {}
        
        # Count files in each directory
        for split in ["train", "test"]:
            info[split] = {}
            split_path = dataset_path / split
            
            for category in ["normal", "abnormal"]:
                category_path = split_path / category
                if category_path.exists():
                    image_files = []
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        image_files.extend(category_path.glob(f"*{ext}"))
                        image_files.extend(category_path.glob(f"*{ext.upper()}"))
                    info[split][category] = len(image_files)
                else:
                    info[split][category] = 0
        
        return info


def create_sample_dataset(output_dir: Union[str, Path], dataset_name: str = "sample") -> Path:
    """
    Create a sample dataset for testing purposes.
    
    Args:
        output_dir: Output directory for the dataset
        dataset_name: Name of the sample dataset
        
    Returns:
        Path to the created dataset
    """
    organizer = DatasetOrganizer(output_dir)
    dataset_path = organizer.create_dataset_structure(dataset_name)
    
    # Create placeholder files to indicate structure
    placeholder_dirs = [
        dataset_path / "train" / "normal",
        dataset_path / "test" / "normal",
        dataset_path / "test" / "abnormal",
        dataset_path / "ground_truth" / "abnormal"
    ]
    
    for directory in placeholder_dirs:
        placeholder_file = directory / "README.md"
        with open(placeholder_file, 'w') as f:
            f.write(f"# {directory.name} Images\n\n")
            f.write(f"Place {directory.name} images in this directory.\n")
            f.write(f"Directory: {directory}\n")
    
    return dataset_path