#!/bin/bash
#SBATCH --job-name=data_generation_k_values
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=data_generation_k_values_%j.out
#SBATCH --account=bcqs-delta-gpu

# Activate the virtual environment
source /u/tawal/venv/bin/activate

# Change to the source directory
cd /u/tawal/Resurrection-Physics-Informed-Superresolution/src

# Run the data generation script with the specified k values
python data_generation_subdomains.py

# Output information on completion
echo "Data generation completed. Dataset saved to data/pde_dataset_subdomains.npz"
