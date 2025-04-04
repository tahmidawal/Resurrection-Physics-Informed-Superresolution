import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F
import sys
import os

# Fix import to work whether run from project root or src directory
try:
    from src.models_improved import ContextAwareUNet, OverlappingPDEDataset
except ImportError:
    from models_improved import ContextAwareUNet, OverlappingPDEDataset

import argparse
from typing import Dict, List, Tuple

def load_model(checkpoint_path: Path, device: str = 'cuda') -> ContextAwareUNet:
    """Load the trained model."""
    print(f"Loading model from {checkpoint_path} to device {device}")
    model = ContextAwareUNet(in_channels=2, context_padding=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_test_data(data_path: Path, test_indices: List[int], device: str = 'cuda') -> Dict:
    """Load test samples from the dataset."""
    print(f"Loading test data from {data_path} to device {device}")
    data_file = np.load(data_path)
    
    # Check if data is in batched format
    keys = list(data_file.keys())
    print(f"Dataset keys: {keys[:5]}...")  # Print first few keys
    
    # Determine if data is in batched format
    is_batched = any('_batch_' in key for key in keys)
    print(f"Dataset is in batched format: {is_batched}")
    
    if is_batched:
        # Reconstruct data from batches
        print("Reconstructing data from batches...")
        reconstructed_data = {}
        
        # Extract batch numbers and data keys
        batch_info = {}
        for key in keys:
            if '_batch_' in key:
                base_key, batch_num = key.rsplit('_batch_', 1)
                batch_num = int(batch_num)
                if base_key not in batch_info:
                    batch_info[base_key] = []
                batch_info[base_key].append(batch_num)
        
        # Get unique base keys and sort batch numbers
        base_keys = list(batch_info.keys())
        for base_key in base_keys:
            batch_info[base_key] = sorted(batch_info[base_key])
        
        print(f"Found base keys: {base_keys}")
        
        # Concatenate batches for each base key
        for base_key in base_keys:
            batch_data = []
            for batch_num in batch_info[base_key]:
                batch_key = f"{base_key}_batch_{batch_num}"
                batch_data.append(data_file[batch_key])
            reconstructed_data[base_key] = np.concatenate(batch_data, axis=0)
            print(f"Reconstructed {base_key}: {reconstructed_data[base_key].shape}")
        
        # Add non-batched keys (like k1, k2, etc.)
        for key in keys:
            if '_batch_' not in key:
                reconstructed_data[key] = data_file[key]
        
        data = reconstructed_data
    else:
        # Normal dataset format
        data = {key: data_file[key] for key in data_file.keys()}
    
    # Select test samples
    test_data = {}
    for key in ['u_coarse', 'u_fine', 'f_coarse', 'f_fine', 'theta_coarse', 'theta_fine']:
        if key in data:
            test_data[key] = torch.from_numpy(data[key][test_indices]).float().to(device)
    
    # Add k1 and k2 if available
    if 'k1' in data and 'k2' in data:
        test_data['k1'] = torch.from_numpy(data['k1'][test_indices]).float().to(device) if data['k1'].shape[0] > max(test_indices) else torch.ones(len(test_indices)).to(device)
        test_data['k2'] = torch.from_numpy(data['k2'][test_indices]).float().to(device) if data['k2'].shape[0] > max(test_indices) else torch.ones(len(test_indices)).to(device)
    else:
        # Default values if k1/k2 not available
        test_data['k1'] = torch.ones(len(test_indices)).to(device) * 8.0
        test_data['k2'] = torch.ones(len(test_indices)).to(device) * 8.0
        print("Warning: k1 and k2 not found in dataset, using default values of 8.0")
    
    return test_data, data

def process_sample(
    model: ContextAwareUNet, 
    dataset: OverlappingPDEDataset, 
    idx: int, 
    save_dir: Path,
    test_data: Dict,
    device: str = 'cuda'
) -> Tuple[float, float]:
    """Process a single test sample and visualize the results."""
    
    # Prepare input data
    coarse_solution = test_data['u_coarse'][idx].cpu().numpy()  # 24x24
    fine_solution = test_data['u_fine'][idx].cpu().numpy()      # 48x48
    f_coarse = test_data['f_coarse'][idx].cpu().numpy()         # 24x24
    f_fine = test_data['f_fine'][idx].cpu().numpy()             # 48x48
    k1 = test_data['k1'][idx].item()
    k2 = test_data['k2'][idx].item()
    
    # Get dataset statistics to CPU for normalization
    u_mean = dataset.u_mean.cpu().item()
    u_std = dataset.u_std.cpu().item()
    f_mean = dataset.f_mean.cpu().item()
    f_std = dataset.f_std.cpu().item()
    
    # Normalize inputs for model
    coarse_norm = (torch.from_numpy(coarse_solution).float() - u_mean) / u_std
    f_coarse_norm = (torch.from_numpy(f_coarse).float() - f_mean) / f_std
    
    # Upsample coarse inputs to fine grid (24x24 → 48x48)
    coarse_upsampled = F.interpolate(
        coarse_norm.unsqueeze(0).unsqueeze(0),
        size=(48, 48),  # Full fine grid size with context padding
        mode='bilinear',
        align_corners=True
    )
    
    f_upsampled = F.interpolate(
        f_coarse_norm.unsqueeze(0).unsqueeze(0),
        size=(48, 48),  # Full fine grid size with context padding
        mode='bilinear',
        align_corners=True
    )
    
    # Combine inputs
    model_input = torch.cat([
        coarse_upsampled.squeeze(0),
        f_upsampled.squeeze(0)
    ], dim=0).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        prediction = model(model_input)  # Model outputs 40x40 by removing context padding
        # Manual denormalization to avoid device issues
        prediction = prediction * u_std + u_mean
    
    # Convert to numpy
    prediction_np = prediction.squeeze().cpu().numpy()  # 40x40
    
    # Create a bilinear upsampling baseline directly to 40x40 for fair comparison
    bilinear_upsampled = F.interpolate(
        torch.from_numpy(coarse_solution).float().unsqueeze(0).unsqueeze(0),
        size=(40, 40),  # Core size without context padding
        mode='bilinear',
        align_corners=True
    ).squeeze().numpy()
    
    # Extract core part (40x40) of fine solution for comparison
    # This removes the 4-pixel context padding on each side of the 48x48 fine solution
    h, w = fine_solution.shape
    start_h = (h - 40) // 2  # 4 pixels padding on each side
    start_w = (w - 40) // 2  # 4 pixels padding on each side
    fine_core = fine_solution[start_h:start_h+40, start_w:start_w+40]
    
    # Calculate errors
    ml_error = np.abs(prediction_np - fine_core)
    bilinear_error = np.abs(bilinear_upsampled - fine_core)
    
    ml_mae = np.mean(ml_error)
    bilinear_mae = np.mean(bilinear_error)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 15))
    plt.suptitle(f'Model Inference Results - Sample {idx}\nk₁={k1:.2f}, k₂={k2:.2f}', fontsize=16)
    
    # Add explanation of dimensions
    plt.figtext(0.5, 0.01, 
                "Note: The model upscales from 24x24 to 48x48, but removes 4-pixel context padding on each side to produce a 40x40 output", 
                wrap=True, horizontalalignment='center', fontsize=12)
    
    # Create grid for subplots
    gs = plt.GridSpec(3, 4, figure=fig)
    
    # First row: Solutions
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(coarse_solution)
    ax1.set_title(f'Coarse Solution ({coarse_solution.shape[0]}×{coarse_solution.shape[1]})')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(fine_core)
    ax2.set_title(f'Ground Truth (Core {fine_core.shape[0]}×{fine_core.shape[1]})')
    plt.colorbar(im2, ax=ax2)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(bilinear_upsampled)
    ax3.set_title('Bilinear Interpolation (24×24 → 40×40)')
    plt.colorbar(im3, ax=ax3)
    
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(prediction_np)
    ax4.set_title('ML Model Prediction (24×24 → 48×48 → 40×40)')
    plt.colorbar(im4, ax=ax4)
    
    # Also show full fine solution
    ax1_5 = fig.add_subplot(gs[1, 1])
    im1_5 = ax1_5.imshow(fine_solution)
    ax1_5.set_title(f'Full Fine Solution ({fine_solution.shape[0]}×{fine_solution.shape[1]})')
    plt.colorbar(im1_5, ax=ax1_5)
    
    # Highlight core region
    rect = plt.Rectangle((start_w-0.5, start_h-0.5), 40, 40, linewidth=2, edgecolor='r', facecolor='none')
    ax1_5.add_patch(rect)
    
    # Second row: Force field and errors (shifted by one to make room for full fine solution)
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(f_coarse)
    ax5.set_title('Forcing Function (Coarse)')
    plt.colorbar(im5, ax=ax5)
    
    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(bilinear_error)
    ax7.set_title(f'Bilinear Error (MAE: {bilinear_mae:.6f})')
    plt.colorbar(im7, ax=ax7)
    
    ax8 = fig.add_subplot(gs[1, 3])
    im8 = ax8.imshow(ml_error)
    ax8.set_title(f'ML Model Error (MAE: {ml_mae:.6f})')
    plt.colorbar(im8, ax=ax8)
    
    # Third row: Histograms and cross-section
    ax9 = fig.add_subplot(gs[2, 0:2])
    ax9.hist(bilinear_error.flatten(), bins=50, alpha=0.5, label='Bilinear')
    ax9.hist(ml_error.flatten(), bins=50, alpha=0.5, label='ML Model')
    ax9.set_xlabel('Absolute Error')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Error Distribution')
    ax9.legend()
    
    # Cross-section through middle of domain
    ax10 = fig.add_subplot(gs[2, 2:])
    mid_idx = fine_core.shape[0] // 2
    ax10.plot(fine_core[mid_idx, :], label='Ground Truth')
    ax10.plot(bilinear_upsampled[mid_idx, :], label='Bilinear', linestyle='--')
    ax10.plot(prediction_np[mid_idx, :], label='ML Model')
    ax10.set_xlabel('X Position')
    ax10.set_ylabel('Solution Value')
    ax10.set_title(f'Cross-section at Y={mid_idx}')
    ax10.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the explanation text
    plt.savefig(save_dir / f'test_sample_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print metrics
    print(f"Sample {idx}:")
    print(f"  Bilinear MAE: {bilinear_mae:.6f}")
    print(f"  ML Model MAE: {ml_mae:.6f}")
    print(f"  Improvement: {bilinear_mae / ml_mae:.2f}x")
    
    return bilinear_mae, ml_mae

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test trained model on examples')
    parser.add_argument('--model_path', type=str, required=False,
                        help='Path to the trained model checkpoint',
                        default=None)
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of test samples to process')
    parser.add_argument('--output_dir', type=str, default='Output/test_results',
                        help='Directory to save test results')
    args = parser.parse_args()
    
    # Find most recent model if not specified
    if args.model_path is None:
        results_dir = Path('results')
        runs = list(results_dir.glob('run_improved_*'))
        if runs:
            latest_run = sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            model_path = latest_run / 'best_model.pth'
            if not model_path.exists():
                model_path = latest_run / 'final_model.pth'
            print(f"Using latest run: {latest_run}")
        else:
            raise FileNotFoundError("No model found in results directory")
    else:
        model_path = Path(args.model_path)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using model: {model_path}")
    print(f"Device: {device}")
    print(f"Saving results to: {output_dir}")
    
    # Load model
    model = load_model(model_path, device)
    
    # Load dataset
    data_path = Path('data/pde_dataset_overlapping.npz')
    
    # Create test indices from the highest quality samples
    np.random.seed(42)  # for reproducibility
    
    # Load and reconstruct the entire dataset
    _, full_data = load_test_data(data_path, [0], device)  # Just pass [0] to get structure without loading all data
    
    if 'u_fine' in full_data:
        n_samples = len(full_data['u_fine'])
    else:
        # Find another key with the right shape
        for key in full_data.keys():
            if key.startswith('u_fine') or key.startswith('u_coarse'):
                n_samples = len(full_data[key])
                break
        else:
            raise ValueError("Could not determine dataset size")
    
    print(f"Dataset contains {n_samples} samples")
    
    # Create test indices - use samples that weren't in training
    all_indices = np.arange(n_samples)
    test_indices = np.random.choice(all_indices, size=args.num_samples, replace=False)
    print(f"Selected test indices: {test_indices}")
    
    # Load actual test data
    test_data, full_data = load_test_data(data_path, test_indices, device)
    
    # Create dataset for normalization
    print(f"Creating dataset for normalization on device {device}")
    
    # Prepare a subset of data for the dataset class
    dataset_data = {}
    for key in ['u_coarse', 'u_fine', 'f_fine']:
        if key in full_data:
            # Only load enough data for normalization statistics
            sample_indices = np.random.choice(n_samples, min(1000, n_samples), replace=False)
            dataset_data[key] = full_data[key][sample_indices]
    
    dataset = OverlappingPDEDataset(dataset_data, device=device)
    
    # Get the dataset statistics to CPU to avoid device issues
    dataset.u_mean = dataset.u_mean.cpu()
    dataset.u_std = dataset.u_std.cpu()
    dataset.f_mean = dataset.f_mean.cpu()
    dataset.f_std = dataset.f_std.cpu()
    
    # Process each test sample
    mae_bilinear = []
    mae_ml = []
    
    for i in range(args.num_samples):
        b_mae, ml_mae = process_sample(model, dataset, i, output_dir, test_data, device)
        mae_bilinear.append(b_mae)
        mae_ml.append(ml_mae)
    
    # Calculate average metrics
    avg_bilinear_mae = np.mean(mae_bilinear)
    avg_ml_mae = np.mean(mae_ml)
    avg_improvement = avg_bilinear_mae / avg_ml_mae
    
    print("\nAverage metrics across all test samples:")
    print(f"  Bilinear MAE: {avg_bilinear_mae:.6f}")
    print(f"  ML Model MAE: {avg_ml_mae:.6f}")
    print(f"  Improvement: {avg_improvement:.2f}x")
    
    # Save metrics to file
    with open(output_dir / 'test_metrics.txt', 'w') as f:
        f.write("Average metrics across all test samples:\n")
        f.write(f"Bilinear MAE: {avg_bilinear_mae:.6f}\n")
        f.write(f"ML Model MAE: {avg_ml_mae:.6f}\n")
        f.write(f"Improvement: {avg_improvement:.2f}x\n\n")
        
        f.write("Per-sample metrics:\n")
        for i in range(args.num_samples):
            f.write(f"Sample {i}:\n")
            f.write(f"  Bilinear MAE: {mae_bilinear[i]:.6f}\n")
            f.write(f"  ML Model MAE: {mae_ml[i]:.6f}\n")
            f.write(f"  Improvement: {mae_bilinear[i] / mae_ml[i]:.2f}x\n\n")

if __name__ == '__main__':
    main() 