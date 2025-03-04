# Resurrection Physics Informed Super-resolution

This project aims to enhance image resolution using physics-informed super-resolution techniques. The repository contains various scripts and results related to the project.

## Project Structure

### Source Code (`src/`)

- **`compare_methods.py`**: Contains methods to compare different super-resolution techniques.
- **`data_generation.py`**: Responsible for generating synthetic data for training and testing.
- **`large_scale_320 copy.py`**: A script for handling large-scale data processing.
- **`models.py`**: Defines the machine learning models used for super-resolution.
- **`resolution_comparison.py`**: Compares the resolution of images before and after applying super-resolution.
- **`resolution_comparison_enhanced copy.py`**: An enhanced version of the resolution comparison script.
- **`test_out_of_sample.py`**: Tests the model's performance on out-of-sample data.
- **`train.py`**: Script to train the super-resolution models.
- **`utils.py`**: Utility functions used across different scripts.
- **`visualization.py`**: Contains functions for visualizing results.

### Results (`results/`)

- **`run_20250303_173409/`**: Contains the results of a specific run.
  - **`config.json`**: Configuration file for the run.
  - **`best_model.pth`**: The best-performing model saved during the run.
  - **`resolution_comparison_results/`**: Contains images and metrics comparing resolutions.
    - **`comparison_80x80.png`**, **`comparison_160x160.png`**, **`comparison_320x320.png`**, **`comparison_640x640.png`**: Images showing resolution comparisons at different scales.
    - **`error_distribution_80x80.png`**, **`error_distribution_160x160.png`**, **`error_distribution_320x320.png`**, **`error_distribution_640x640.png`**: Images showing error distributions at different scales.
    - **`resolution_comparison_metrics.png`**: Metrics related to resolution comparison.

## Getting Started

To get started with the project, clone the repository and follow the instructions in the `train.py` script to train the models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 