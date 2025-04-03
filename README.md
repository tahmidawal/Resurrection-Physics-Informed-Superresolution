# PDE Solution Refinement with Context-Aware UNet

This repository contains code for training and testing a Context-Aware UNet model for refining coarse PDE solutions to finer resolutions while maintaining accuracy at overlapping boundaries.

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install required packages:
```bash
pip install torch numpy matplotlib
```

## Data Generation

The dataset consists of PDE solutions at different resolutions with overlapping boundaries. To generate the dataset:

1. Run the data generation script:
```bash
python src/generate_data.py --num_samples 1000 --output_path data/pde_dataset_overlapping.npz
```

This will create a dataset with:
- Coarse solutions (24x24)
- Fine solutions (48x48)
- Forcing functions at both resolutions
- Material parameters k₁ and k₂

## Training

To train the model:

```bash
python src/train.py --epochs 100 --batch_size 32 --learning_rate 0.001
```

The training script will:
- Save checkpoints in `results/run_improved_[timestamp]/`
- Track metrics using tensorboard
- Save the best model based on validation loss

## Testing

For inference testing:

```bash
python src/test_inference.py --num_samples 10 --output_dir Output/inference_results
```

### Latest Test Results

Average metrics across 10 test samples:
- Bilinear MAE: 0.000075
- ML Model MAE: 0.000013
- Average Improvement: 5.73x

Individual sample improvements ranged from 4.94x to 11.44x, demonstrating consistent performance across different test cases.

Sample-specific results:
```
Sample 1: 9.34x improvement (MAE: 0.000005)
Sample 2: 5.50x improvement (MAE: 0.000017)
Sample 3: 11.44x improvement (MAE: 0.000005)
Sample 4: 5.50x improvement (MAE: 0.000009)
```

The model consistently outperforms bilinear interpolation, with particularly strong performance on samples with complex features or sharp gradients.

## Quick Testing

For a quick test on a single sample:

```bash
python src/quick_test.py
```

This will generate visualizations comparing:
- Input coarse solution
- Ground truth fine solution
- Model prediction
- Bilinear upsampling baseline
- Error distributions

## Model Architecture

The Context-Aware UNet architecture features:
- Input channels: 2 (solution + forcing function)
- Context padding: 2 cells
- Overlapping boundary handling
- Skip connections for feature preservation

## Batch Job Support

For HPC environments, use the provided SLURM script:

```bash
sbatch src/run_inference_test.sbatch
```

This will run the inference test on a GPU node and save results in the Output directory. 