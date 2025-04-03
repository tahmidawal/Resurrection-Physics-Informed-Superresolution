#!/bin/bash
#SBATCH --job-name=train_24x24_48x48
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:00
#SBATCH --output=Output/train_24x24_48x48_%j.out
#SBATCH --account=bcqs-delta-gpu
#SBATCH --mem=32G

# Create Output directory if it doesn't exist
mkdir -p Output

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
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}');" | tee -a Output/training_gpu_info.log

# Check if dataset exists, if not, generate it
if [ ! -f "data/pde_dataset_overlapping.npz" ]; then
    echo "Dataset not found, generating dataset first..."
    PYTHONPATH=$PWD python src/data_generation_overlapping.py
fi

# Run the training with 100 epochs
PYTHONPATH=$PWD python src/train_24x24_to_48x48.py --epochs 100 --batch_size 64 --learning_rate 0.001 2>&1 | tee -a Output/training_log.out

# After training, run inference test on the best model
echo "Training completed. Running inference test..."
# Find the most recent model directory
latest_model_dir=$(find results -type d -name "run_24x24_48x48_*" -printf '%T+ %p\n' | sort -r | head -n 1 | cut -d' ' -f2-)
best_model="$latest_model_dir/best_model.pth"

# Run inference
PYTHONPATH=$PWD python src/test_inference.py --num_samples 10 --output_dir Output/inference_results_24x48 --model_path "$best_model" 2>&1 | tee -a Output/inference_log.out

# Output information on completion
echo "Training and inference completed. Results saved in Output directories." | tee -a Output/training_status.log 