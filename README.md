# Image Anomaly Detection with Anomalib

A comprehensive image anomaly detection project using the Anomalib library for industrial defect detection and medical image analysis.

## Project Overview

This project implements state-of-the-art anomaly detection algorithms for image data using Anomalib, a PyTorch-based library for anomaly detection. It supports various algorithms including PaDiM, STFPM, PatchCore, and more.

## Features

- **Multiple Algorithms**: Support for various anomaly detection models
- **Easy Configuration**: YAML-based configuration system
- **Data Management**: Structured data organization following Anomalib conventions
- **Visualization**: Built-in tools for result analysis and visualization
- **Extensible**: Custom model and callback implementations
- **Production Ready**: Model optimization and deployment utilities

## Project Structure

```
├── .github/                    # GitHub configuration and Copilot instructions
├── configs/                    # Model and training configurations
├── data/                      # Dataset directory (organized by Anomalib structure)
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Preprocessed data
│   └── datasets/              # Anomalib dataset structure
├── experiments/               # Experiment results and logs
├── models/                    # Trained models and checkpoints
├── notebooks/                 # Jupyter notebooks for exploration
├── src/                       # Source code
│   ├── __init__.py
│   ├── config/                # Configuration utilities
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Custom models and wrappers
│   ├── training/              # Training scripts and utilities
│   ├── inference/             # Inference and prediction utilities
│   └── utils/                 # General utilities
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Image-Anomaly
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Anomalib:
```bash
pip install anomalib
```

### Basic Usage

1. **Data Preparation**: Organize your data following Anomalib's structure in the `data/datasets/` directory
2. **Configuration**: Modify or create configuration files in `configs/`
3. **Training**: Use the training scripts in `src/training/`
4. **Inference**: Run inference using scripts in `src/inference/`

## Dataset Structure

Follow Anomalib's standard dataset structure:

```
data/datasets/<dataset_name>/
├── train/
│   └── normal/          # Normal/good images
├── test/
│   ├── normal/          # Normal test images
│   └── abnormal/        # Anomalous/defective images
└── ground_truth/
    └── abnormal/        # Ground truth masks for anomalous images
```

## Configuration

All model and training configurations are stored in YAML files in the `configs/` directory. Modify these files to:

- Change model parameters
- Adjust training settings
- Configure data paths
- Set up logging and callbacks

## Models Supported

- **PaDiM**: Patch Distribution Modeling
- **STFPM**: Student-Teacher Feature Pyramid Matching
- **PatchCore**: Coreset-based anomaly detection
- **FastFlow**: Fast normalizing flow-based detection
- **And more**: Easily extensible to other Anomalib models

## Development

### Adding Custom Models

1. Create new model classes in `src/models/`
2. Follow Anomalib's model interface
3. Add configuration files in `configs/`
4. Test with unit tests in `tests/`

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Include comprehensive docstrings
- Write unit tests for new functionality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the coding guidelines
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Anomalib](https://github.com/openvinotoolkit/anomalib) - The core anomaly detection library
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenVINO](https://openvino.ai/) - Model optimization and deployment