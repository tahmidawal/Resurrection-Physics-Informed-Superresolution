#!/bin/bash

# Request GPU allocation
# Use GPU 0 if available
export CUDA_VISIBLE_DEVICES=0

# Activate the virtual environment
source /u/tawal/venv/bin/activate

# Change to the src directory
cd /u/tawal/Resurrection-Physics-Informed-Superresolution/src

# Run the resolution comparison with k1=k2=9.0 
# Print CUDA availability for debugging
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())" 

# Run the actual comparison script
python resolution_comparison.py --model_path results/run_20250326_205334/best_model.pth

# Output message upon completion
echo "Resolution comparison with k1=k2=9.0 completed."
