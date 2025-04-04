#!/bin/bash
#SBATCH --job-name=accuracy_comparison
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=Output/accuracy_comparison_%j.out
#SBATCH --account=bcqs-delta-gpu

# Create Output directory if it doesn't exist
mkdir -p Output/accuracy_comparison

# Activate the virtual environment
source /u/tawal/venv/bin/activate

# Change to the project directory
cd /u/tawal/20250403-fixing-subdomains

# Set PYTHONPATH to fix import issues
export PYTHONPATH=/u/tawal/20250403-fixing-subdomains:$PYTHONPATH

# Run the accuracy comparison script with parameters that match our dataset
python src/accurate_resolution_comparison.py --k1=8.0 --k2=6.0 --output-dir=Output/accuracy_comparison

# Output information on completion
echo "Accuracy comparison completed. Results saved to Output/accuracy_comparison/" 