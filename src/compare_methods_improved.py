import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
from src.models_improved import ContextAwareUNet, OverlappingPDEDataset
from src.data_generation_overlapping import PoissonSolver
import seaborn as sns
from datetime import datetime
from typing import Tuple, List

def load_model(checkpoint_path: Path, device: str = 'cuda') -> ContextAwareUNet:
    """Load the trained model."""
    model = ContextAwareUNet(in_channels=2, context_padding=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def extract_overlapping_subdomains(field: np.ndarray, core_size: int, context_size: int, grid_size: int) -> List[np.ndarray]:
    """
    Extract overlapping subdomains for prediction.
    
    Args:
        field: The field to extract subdomains from
        core_size: Size of the core region
        context_size: Size of the context window
        grid_size: Number of subdomains per dimension
        
    Returns:
        List of subdomain arrays with context
    """
    total_size = field.shape[0]
    
    # Calculate step size
    step_size = total_size // grid_size
    
    # Padding needed for context
    padding = (context_size - core_size) // 2
    
    subdomains = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the center of the subdomain
            i_center = i * step_size + step_size // 2
            j_center = j * step_size + step_size // 2
            
            # Calculate start and end indices with padding
            i_start = max(0, i_center - context_size // 2)
            j_start = max(0, j_center - context_size // 2)
            
            # Adjust if we're too close to the edge
            i_start = min(i_start, total_size - context_size)
            j_start = min(j_start, total_size - context_size)
            
            # Extract subdomain with context
            i_end = i_start + context_size
            j_end = j_start + context_size
            
            subdomain = field[i_start:i_end, j_start:j_end]
            subdomains.append(subdomain)
    
    return subdomains

def stitch_predictions(predictions: List[np.ndarray], grid_size: int, output_size: int, overlap: int = 4) -> np.ndarray:
    """
    Stitch together predictions from overlapping subdomains.
    
    Args:
        predictions: List of predictions (each corresponding to one subdomain)
        grid_size: Number of subdomains per dimension
        output_size: Size of the output grid
        overlap: Number of pixels to overlap when stitching
        
    Returns:
        Stitched solution
    """
    # Determine dimensions of one prediction
    core_h, core_w = predictions[0].shape
    
    # Initialize output and weight arrays
    stitched = np.zeros((output_size, output_size))
    weights = np.zeros((output_size, output_size))
    
    # Create a distance weight function (higher in center, lower at edges)
    def create_weight_mask(h, w):
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        xx, yy = np.meshgrid(x, y)
        # Radial distance from center (0 at center, 1 at corners)
        dist = np.sqrt(xx**2 + yy**2) / np.sqrt(2)
        # Convert to weight (1 at center, 0.1 at corners)
        return 1.0 - 0.9 * dist
    
    # Create weight mask for one subdomain
    weight_mask = create_weight_mask(core_h, core_w)
    
    # Calculate step size
    step_size = (output_size - core_h) // (grid_size - 1) + core_h
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate starting position
            start_h = i * (step_size - overlap)
            start_w = j * (step_size - overlap)
            
            # Ensure we don't go past the edges
            start_h = min(start_h, output_size - core_h)
            start_w = min(start_w, output_size - core_w)
            
            # Add weighted prediction
            stitched[start_h:start_h+core_h, start_w:start_w+core_w] += predictions[idx] * weight_mask
            weights[start_h:start_h+core_h, start_w:start_w+core_w] += weight_mask
            
            idx += 1
    
    # Average overlapping regions
    valid_mask = weights > 0
    stitched[valid_mask] = stitched[valid_mask] / weights[valid_mask]
    
    return stitched

def upscale_with_model(model: ContextAwareUNet, coarse_solution: np.ndarray, f_coarse: np.ndarray, 
                      dataset: OverlappingPDEDataset, grid_size: int) -> np.ndarray:
    """
    Upscale a solution using the trained model with overlapping subdomains.
    
    Args:
        model: Trained neural network
        coarse_solution: Coarse solution to upscale
        f_coarse: Coarse forcing term
        dataset: Dataset object for normalization
        grid_size: Number of subdomains per dimension
        
    Returns:
        Upscaled solution
    """
    device = next(model.parameters()).device
    
    # Extract overlapping subdomains
    subdomains_u = extract_overlapping_subdomains(coarse_solution, 20, 24, grid_size)
    subdomains_f = extract_overlapping_subdomains(f_coarse, 20, 24, grid_size)
    
    # Convert to tensors and normalize
    subdomains_u_tensor = []
    subdomains_f_tensor = []
    
    for u, f in zip(subdomains_u, subdomains_f):
        u_norm = (torch.from_numpy(u).float() - dataset.u_mean) / dataset.u_std
        f_norm = (torch.from_numpy(f).float() - dataset.f_mean) / dataset.f_std
        subdomains_u_tensor.append(u_norm)
        subdomains_f_tensor.append(f_norm)
    
    # Process each subdomain
    predictions = []
    
    for u_tensor, f_tensor in zip(subdomains_u_tensor, subdomains_f_tensor):
        # Upsample coarse solution
        u_upsampled = F.interpolate(
            u_tensor.unsqueeze(0).unsqueeze(0),
            size=(48, 48),  # Upsample to context size
            mode='bilinear',
            align_corners=True
        ).squeeze(0)
        
        # Upsample forcing term
        f_upsampled = F.interpolate(
            f_tensor.unsqueeze(0).unsqueeze(0),
            size=(48, 48),  # Upsample to context size
            mode='bilinear',
            align_corners=True
        ).squeeze(0)
        
        # Combine inputs
        model_input = torch.cat([u_upsampled, f_upsampled], dim=0).unsqueeze(0).to(device)
        
        # Run model
        with torch.no_grad():
            prediction = model(model_input)
            
        # Denormalize
        prediction = dataset.denormalize(prediction)
        
        # Convert to numpy and add to list
        predictions.append(prediction.squeeze().cpu().numpy())
    
    # Stitch predictions together
    output_size = coarse_solution.shape[0] * 2  # Double the size
    upscaled = stitch_predictions(predictions, grid_size, output_size)
    
    return upscaled

def plot_comparison(
    coarse_solution: np.ndarray,
    fine_solution: np.ndarray,
    bilinear_upscaled: np.ndarray,
    ml_upscaled: np.ndarray,
    save_path: Path,
    sample_idx: int,
    k1: float,
    k2: float,
    theta_fine: np.ndarray
):
    """
    Create a comprehensive comparison plot of different methods.
    
    Args:
        coarse_solution: Original coarse solution
        fine_solution: Ground truth fine solution
        bilinear_upscaled: Bilinearly interpolated solution
        ml_upscaled: ML model's prediction
        save_path: Path to save the plot
        sample_idx: Index of the sample being plotted
        k1, k2: Wave numbers used for this sample
        theta_fine: Fine grid diffusion coefficient
    """
    fig = plt.figure(figsize=(20, 15))
    plt.suptitle(f'Comparison of Upscaling Methods with Overlapping Subdomains - Sample {sample_idx}\n' + 
                f'k₁={k1:.2f}, k₂={k2:.2f}', fontsize=16)
    
    # Create grid for subplots
    gs = plt.GridSpec(3, 4, figure=fig)
    
    # First row: Solutions
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(coarse_solution)
    ax1.set_title(f'Coarse Solution ({coarse_solution.shape[0]}×{coarse_solution.shape[1]})')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(fine_solution)
    ax2.set_title(f'Ground Truth ({fine_solution.shape[0]}×{fine_solution.shape[1]})')
    plt.colorbar(im2, ax=ax2)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(bilinear_upscaled)
    ax3.set_title('Bilinear Interpolation')
    plt.colorbar(im3, ax=ax3)
    
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(ml_upscaled)
    ax4.set_title('ML Model Prediction')
    plt.colorbar(im4, ax=ax4)
    
    # Second row: Absolute Errors
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(theta_fine)
    ax5.set_title('θ (Diffusion Coefficient)')
    plt.colorbar(im5, ax=ax5)
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(np.zeros_like(fine_solution))
    ax6.set_title('Ground Truth Error (Zero)')
    plt.colorbar(im6, ax=ax6)
    
    bilinear_error = np.abs(bilinear_upscaled - fine_solution)
    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(bilinear_error)
    ax7.set_title(f'Bilinear Error (MAE: {np.mean(bilinear_error):.4f})')
    plt.colorbar(im7, ax=ax7)
    
    ml_error = np.abs(ml_upscaled - fine_solution)
    ax8 = fig.add_subplot(gs[1, 3])
    im8 = ax8.imshow(ml_error)
    ax8.set_title(f'ML Model Error (MAE: {np.mean(ml_error):.4f})')
    plt.colorbar(im8, ax=ax8)
    
    # Third row: Error histograms
    ax9 = fig.add_subplot(gs[2, :2])
    ax9.hist(bilinear_error.flatten(), bins=50, alpha=0.5, label='Bilinear')
    ax9.hist(ml_error.flatten(), bins=50, alpha=0.5, label='ML Model')
    ax9.set_title('Error Distribution')
    ax9.set_xlabel('Absolute Error')
    ax9.set_ylabel('Frequency')
    ax9.legend()
    
    # Add error statistics
    stats_text = (
        f'Bilinear Interpolation:\n'
        f'  MAE: {np.mean(bilinear_error):.6f}\n'
        f'  Max Error: {np.max(bilinear_error):.6f}\n'
        f'  RMSE: {np.sqrt(np.mean(bilinear_error**2)):.6f}\n\n'
        f'ML Model:\n'
        f'  MAE: {np.mean(ml_error):.6f}\n'
        f'  Max Error: {np.max(ml_error):.6f}\n'
        f'  RMSE: {np.sqrt(np.mean(ml_error**2)):.6f}\n\n'
        f'Improvement:\n'
        f'  MAE: {np.mean(bilinear_error) / np.mean(ml_error):.2f}x\n'
        f'  RMSE: {np.sqrt(np.mean(bilinear_error**2)) / np.sqrt(np.mean(ml_error**2)):.2f}x'
    )
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=10)
    ax10.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path / f'comparison_improved_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_samples': 5,  # Number of samples to visualize
        'grid_size': 2     # Number of subdomains per dimension
    }
    
    # Create results directory
    results_dir = Path('results')
    runs = list(results_dir.glob('run_improved_*'))
    
    if runs:
        # Find the run with the trained model
        for run_dir in sorted(runs, reverse=True):
            model_path = run_dir / 'best_model.pth'
            if model_path.exists():
                break
        else:
            raise FileNotFoundError("No trained model found in any run directory")
    else:
        raise FileNotFoundError("No run directories found")
    
    comparison_dir = run_dir / 'method_comparison_improved'
    comparison_dir.mkdir(exist_ok=True)
    
    # Load trained model
    model = load_model(model_path, device=config['device'])
    
    # Load dataset
    data = np.load('data/pde_dataset_overlapping.npz')
    dataset = OverlappingPDEDataset(data, device=config['device'])
    
    # Process multiple samples
    for idx in range(config['num_samples']):
        print(f'Processing sample {idx+1}/{config["num_samples"]}')
        
        # Get data for this sample
        coarse_solution = data['u_coarse'][idx]
        fine_solution = data['u_fine'][idx]
        f_coarse = data['f_coarse'][idx]
        f_fine = data['f_fine'][idx]
        theta_fine = data['theta_fine'][idx]
        k1 = data['k1'][idx]
        k2 = data['k2'][idx]
        
        # Bilinear interpolation
        bilinear_upscaled = F.interpolate(
            torch.from_numpy(coarse_solution).float().unsqueeze(0).unsqueeze(0),
            size=fine_solution.shape,
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()
        
        # ML model prediction with overlapping subdomains
        ml_upscaled = upscale_with_model(model, coarse_solution, f_coarse, dataset, config['grid_size'])
        
        # Create comparison plot
        plot_comparison(
            coarse_solution=coarse_solution,
            fine_solution=fine_solution,
            bilinear_upscaled=bilinear_upscaled,
            ml_upscaled=ml_upscaled,
            save_path=comparison_dir,
            sample_idx=idx,
            k1=k1,
            k2=k2,
            theta_fine=theta_fine
        )
        
        # Print metrics
        print(f'\nMetrics for sample {idx}:')
        bilinear_mae = np.mean(np.abs(bilinear_upscaled - fine_solution))
        ml_mae = np.mean(np.abs(ml_upscaled - fine_solution))
        print(f'Bilinear MAE: {bilinear_mae:.6f}')
        print(f'ML Model MAE: {ml_mae:.6f}')
        print(f'Improvement: {bilinear_mae / ml_mae:.2f}x')
        
        # Save numerical results
        metrics = {
            'bilinear_mae': float(bilinear_mae),
            'bilinear_rmse': float(np.sqrt(np.mean((bilinear_upscaled - fine_solution)**2))),
            'bilinear_max_error': float(np.max(np.abs(bilinear_upscaled - fine_solution))),
            'ml_model_mae': float(ml_mae),
            'ml_model_rmse': float(np.sqrt(np.mean((ml_upscaled - fine_solution)**2))),
            'ml_model_max_error': float(np.max(np.abs(ml_upscaled - fine_solution))),
            'improvement_mae': float(bilinear_mae / ml_mae),
            'improvement_rmse': float(np.sqrt(np.mean((bilinear_upscaled - fine_solution)**2)) / 
                                    np.sqrt(np.mean((ml_upscaled - fine_solution)**2))),
            'k1': float(k1),
            'k2': float(k2)
        }
        
        with open(comparison_dir / f'metrics_improved_sample_{idx}.txt', 'w') as f:
            for name, value in metrics.items():
                f.write(f'{name}: {value:.6f}\n')
    
    # Generate overall summary
    print("\nGenerating overall summary...")
    all_metrics = []
    for idx in range(config['num_samples']):
        with open(comparison_dir / f'metrics_improved_sample_{idx}.txt', 'r') as f:
            metrics = {}
            for line in f:
                name, value = line.strip().split(': ')
                metrics[name] = float(value)
            all_metrics.append(metrics)
    
    # Calculate average metrics
    avg_metrics = {name: np.mean([m[name] for m in all_metrics]) for name in all_metrics[0].keys()}
    
    with open(comparison_dir / 'summary_metrics.txt', 'w') as f:
        f.write("Average metrics across all samples:\n")
        for name, value in avg_metrics.items():
            f.write(f'{name}: {value:.6f}\n')
    
    print("Done! Comparison results saved to:", comparison_dir)

if __name__ == '__main__':
    main() 