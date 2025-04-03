#!/bin/bash
#SBATCH --job-name=resolution_comparison_k9
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=resolution_comparison_k9_%j.out
#SBATCH --account=YOUR_ACCOUNT_HERE

# Activate the virtual environment
source /u/tawal/venv/bin/activate

# Change to the source directory
cd /u/tawal/Resurrection-Physics-Informed-Superresolution/src

# Print CUDA availability for debugging
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())" 

# Run the resolution comparison script with the fixed k=9.0 model 
python resolution_comparison.py --model_path results/run_20250326_205334/best_model.pth
