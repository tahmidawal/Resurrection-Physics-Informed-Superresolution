#!/bin/bash
#SBATCH --job-name=inference_test
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=Output/inference_test_%j.out
#SBATCH --account=bcqs-delta-gpu
#SBATCH --mem=32G

# Create Output directory if it doesn't exist
mkdir -p Output/inference_results

# Load CUDA module
module load cuda/11.8.0

# Activate the virtual environment
source /u/tawal/venv/bin/activate

# Create a temporary directory for pip install
export TMPDIR=/tmp

# Change to the project directory
cd /u/tawal/20250402-subdomain

# Print GPU information
nvidia-smi

# Verify PyTorch CUDA availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}');" | tee -a Output/inference_gpu_info.log

# Run the inference test with 10 test samples
PYTHONPATH=$PWD python src/test_inference.py --num_samples 10 --output_dir Output/inference_results --model_path results/run_improved_20250403_091429/best_model.pth 2>&1 | tee -a Output/inference_log.out

# Output information on completion
echo "Inference testing completed. Results saved in Output/inference_results directory." | tee -a Output/inference_status.log 