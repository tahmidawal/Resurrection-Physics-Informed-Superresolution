#!/bin/bash
#SBATCH --job-name=train_improved_boundary
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=Output/train_improved_boundary_%j.out
#SBATCH --account=bcqs-delta-gpu
#SBATCH --mem=64G

# Create Output directory if it doesn't exist
mkdir -p Output

# Load CUDA module
module load cuda/11.8.0

# Activate the virtual environment
source /u/tawal/venv/bin/activate

# Create a temporary directory for pip install
export TMPDIR=/tmp

# Install CUDA-enabled PyTorch in the existing virtual environment
pip install --no-cache-dir torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Change to the source directory
cd /u/tawal/20250402-subdomain

# Print GPU information
nvidia-smi

# Verify PyTorch CUDA availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}');" | tee -a Output/torch_info.log

# Run the training script with the enhanced model and boundary-aware loss
PYTHONPATH=$PWD python src/train_improved.py 2>&1 | tee -a Output/training_log.out

# Output information on completion
echo "Training completed. Model saved in the results directory." | tee -a Output/completion_status.log
