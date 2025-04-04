#!/bin/bash
#SBATCH --job-name=train_improved_model
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=Output/train_improved_model_%j.out
#SBATCH --account=bcqs-delta-gpu

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
cd /u/tawal/20250403-fixing-subdomains

# Print GPU information
nvidia-smi

# Verify PyTorch CUDA availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}');" | tee -a Output/torch_info.log

# Set PYTHONPATH to fix import issues
export PYTHONPATH=/u/tawal/20250403-fixing-subdomains:$PYTHONPATH

# Run the training script from the project root directory
cd /u/tawal/20250403-fixing-subdomains
python src/train_improved.py

# Output information on completion
echo "Training completed. Results saved to results/run_improved_*"
