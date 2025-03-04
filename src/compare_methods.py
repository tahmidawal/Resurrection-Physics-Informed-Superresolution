import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
from models import UNet, PDEDataset
from data_generation import PoissonSolver
import seaborn as sns
from datetime import datetime

def load_model(checkpoint_path: Path, device: str = 'cuda') -> UNet:
    """Load the trained model."""
    model = UNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

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
        coarse_solution: Original 20×20 solution
        fine_solution: Ground truth 40×40 solution
        bilinear_upscaled: Bilinearly interpolated solution
        ml_upscaled: ML model's prediction
        save_path: Path to save the plot
        sample_idx: Index of the sample being plotted
        k1, k2: Wave numbers used for this sample
        theta_fine: Fine grid diffusion coefficient
    """
    fig = plt.figure(figsize=(20, 15))
    plt.suptitle(f'Comparison of Upscaling Methods - Sample {sample_idx}\n' + 
                f'k₁={k1:.2f}, k₂={k2:.2f}', fontsize=16)
    
    # Create grid for subplots
    gs = plt.GridSpec(3, 4, figure=fig)
    
    # First row: Solutions
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(coarse_solution)
    ax1.set_title('Coarse Solution (20×20)')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(fine_solution)
    ax2.set_title('Ground Truth (40×40)')
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
        f'  RMSE: {np.sqrt(np.mean(ml_error**2)):.6f}'
    )
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=10)
    ax10.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path / f'comparison_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_samples': 5  # Number of samples to visualize
    }
    
    # Create results directory
    results_dir = Path('results')
    runs = list(results_dir.glob('run_*'))
    
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
    
    comparison_dir = run_dir / 'method_comparison'
    comparison_dir.mkdir(exist_ok=True)
    
    # Load trained model
    model = load_model(model_path, device=config['device'])
    
    # Load dataset
    data = np.load('data/pde_dataset.npz')
    dataset = PDEDataset(data, device=config['device'])
    
    # Process multiple samples
    for idx in range(config['num_samples']):
        print(f'Processing sample {idx+1}/{config["num_samples"]}')
        
        # Get data for this sample
        coarse_solution = data['u_coarse'][idx]
        fine_solution = data['u_fine'][idx]
        theta_fine = data['theta_fine'][idx]
        k1 = data['k1'][idx]
        k2 = data['k2'][idx]
        
        # Bilinear interpolation
        bilinear_upscaled = F.interpolate(
            torch.from_numpy(coarse_solution).float().unsqueeze(0).unsqueeze(0),
            size=(40, 40),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()
        
        # ML model prediction
        inputs, _ = dataset[idx]
        with torch.no_grad():
            ml_upscaled = model(inputs.unsqueeze(0))
            # Denormalize the output
            ml_upscaled = dataset.denormalize(ml_upscaled)
            ml_upscaled = ml_upscaled.squeeze().cpu().numpy()
        
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
        print(f'Bilinear MAE: {np.mean(np.abs(bilinear_upscaled - fine_solution)):.6f}')
        print(f'ML Model MAE: {np.mean(np.abs(ml_upscaled - fine_solution)):.6f}')
        
        # Save numerical results
        metrics = {
            'bilinear_mae': float(np.mean(np.abs(bilinear_upscaled - fine_solution))),
            'bilinear_rmse': float(np.sqrt(np.mean((bilinear_upscaled - fine_solution)**2))),
            'bilinear_max_error': float(np.max(np.abs(bilinear_upscaled - fine_solution))),
            'ml_model_mae': float(np.mean(np.abs(ml_upscaled - fine_solution))),
            'ml_model_rmse': float(np.sqrt(np.mean((ml_upscaled - fine_solution)**2))),
            'ml_model_max_error': float(np.max(np.abs(ml_upscaled - fine_solution))),
            'k1': float(k1),
            'k2': float(k2)
        }
        
        with open(comparison_dir / f'metrics_sample_{idx}.txt', 'w') as f:
            for name, value in metrics.items():
                f.write(f'{name}: {value:.6f}\n')

if __name__ == '__main__':
    main() 