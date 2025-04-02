#!/bin/bash

# Activate the virtual environment
source /u/tawal/venv/bin/activate

# Change to the source directory
cd /u/tawal/Resurrection-Physics-Informed-Superresolution/src

# Run the resolution comparison with the improved model
python resolution_comparison.py --model_path results/run_20250401_174958/best_model.pth
