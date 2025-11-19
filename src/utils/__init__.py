"""General utilities for the anomaly detection project."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_format: Custom log format string
        
    Returns:
        Configured logger
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")
    
    return logger


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        directory: Directory path to ensure
        
    Returns:
        Path object of the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file safely.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path where to save JSON file
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def get_image_files(
    directory: Union[str, Path],
    extensions: List[str] = None,
    recursive: bool = True
) -> List[Path]:
    """
    Get all image files from a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        recursive: Whether to search recursively
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    directory = Path(directory)
    image_files = []
    
    search_pattern = "**/*" if recursive else "*"
    
    for ext in extensions:
        image_files.extend(directory.glob(f"{search_pattern}{ext}"))
        image_files.extend(directory.glob(f"{search_pattern}{ext.upper()}"))
    
    return sorted(image_files)


def calculate_dataset_stats(dataset_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Calculate statistics for an Anomalib-structured dataset.
    
    Args:
        dataset_dir: Path to dataset directory
        
    Returns:
        Dictionary with dataset statistics
    """
    dataset_dir = Path(dataset_dir)
    stats = {
        'dataset_path': str(dataset_dir),
        'train': {'normal': 0, 'abnormal': 0},
        'test': {'normal': 0, 'abnormal': 0},
        'validation': {'normal': 0, 'abnormal': 0},
        'total_images': 0
    }
    
    # Count images in each split and category
    for split in ['train', 'test', 'validation', 'val']:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
            
        # Map 'val' to 'validation' for consistency
        split_key = 'validation' if split == 'val' else split
        
        for category in ['normal', 'abnormal']:
            category_dir = split_dir / category
            if category_dir.exists():
                image_files = get_image_files(category_dir, recursive=False)
                count = len(image_files)
                stats[split_key][category] = count
                stats['total_images'] += count
    
    return stats


def format_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"


def get_directory_size(directory: Union[str, Path]) -> int:
    """
    Get total size of directory in bytes.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
    """
    directory = Path(directory)
    total_size = 0
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, description: str = "Progress"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            description: Description of the operation
        """
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logging.getLogger(__name__)
    
    def update(self, increment: int = 1) -> None:
        """
        Update progress.
        
        Args:
            increment: Number of items processed
        """
        self.current += increment
        percentage = (self.current / self.total) * 100
        
        if self.current % max(1, self.total // 10) == 0 or self.current == self.total:
            self.logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def finish(self) -> None:
        """Mark progress as complete."""
        self.current = self.total
        self.logger.info(f"{self.description}: Completed ({self.total} items)")


def validate_environment() -> Dict[str, bool]:
    """
    Validate that required libraries are available.
    
    Returns:
        Dictionary with availability status of key libraries
    """
    libraries = {
        'torch': False,
        'torchvision': False,
        'anomalib': False,
        'pytorch_lightning': False,
        'opencv': False,
        'pillow': False,
        'numpy': False,
        'matplotlib': False,
        'sklearn': False
    }
    
    # Test imports
    try:
        import torch
        libraries['torch'] = True
    except ImportError:
        pass
    
    try:
        import torchvision
        libraries['torchvision'] = True
    except ImportError:
        pass
    
    try:
        import anomalib
        libraries['anomalib'] = True
    except ImportError:
        pass
    
    try:
        import pytorch_lightning
        libraries['pytorch_lightning'] = True
    except ImportError:
        pass
    
    try:
        import cv2
        libraries['opencv'] = True
    except ImportError:
        pass
    
    try:
        from PIL import Image
        libraries['pillow'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        libraries['numpy'] = True
    except ImportError:
        pass
    
    try:
        import matplotlib
        libraries['matplotlib'] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        libraries['sklearn'] = True
    except ImportError:
        pass
    
    return libraries


def print_environment_status() -> None:
    """Print status of required libraries."""
    libraries = validate_environment()
    
    print("Environment Status")
    print("=" * 40)
    
    for lib, available in libraries.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"{lib:20} - {status}")
    
    missing = [lib for lib, available in libraries.items() if not available]
    
    if missing:
        print(f"\nMissing libraries: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
    else:
        print("\n✓ All required libraries are available!")