#!/bin/bash

# Activate virtual environment
source /u/tawal/venv/bin/activate

# Run the simple resolution comparison script with the specified model
MODEL_PATH="/u/tawal/Resurrection-Physics-Informed-Superresolution/src/results/run_20250401_174958/best_model.pth"

echo "Using model: $MODEL_PATH"
python /u/tawal/Resurrection-Physics-Informed-Superresolution/src/simple_resolution_comparison.py --model_path $MODEL_PATH
