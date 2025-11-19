# Image Anomaly Detection Project with Anomalib

This project focuses on image anomaly detection using the Anomalib library. The workspace is configured for deep learning workflows with proper data management, experimentation notebooks, and model training capabilities.

## Project Context
- **Framework**: Anomalib (PyTorch-based anomaly detection library)
- **Task**: Image anomaly detection and localization
- **Data**: Industrial/medical images for defect detection
- **Models**: Support for various anomaly detection algorithms (PaDiM, STFPM, PatchCore, etc.)

## Development Guidelines
- Use type hints for all Python functions
- Follow PEP 8 style guidelines
- Include comprehensive docstrings for functions and classes
- Use configuration files (YAML) for model parameters
- Implement proper logging for training and inference
- Structure code for modularity and reusability
- Use data versioning for dataset management
- Include unit tests for custom functions

## Anomalib-Specific Practices
- Use Anomalib's configuration system for model setup
- Follow Anomalib's data organization structure
- Utilize Anomalib's visualization tools for result analysis
- Implement custom callbacks for training monitoring
- Use Anomalib's metrics for evaluation
- Follow the library's conventions for custom model implementation