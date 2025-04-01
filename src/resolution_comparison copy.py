import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from data_generation import PoissonSolver
from models import UNet, PDEDataset
from compare_methods import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List, Tuple

def solve_multi_resolution(n_coarse: int = 40, resolutions: List[int] = [80, 160, 320, 640]):
    """
    Solve PDE at multiple resolutions, starting from n_coarse.
    Returns solutions at all resolutions for comparison.
    """
    print("\nGenerating multi-resolution test case...")
    
    # Create solvers for each resolution
    solvers = {}
    for n_fine in resolutions:
        n_coarse_for_solver = n_fine // 2
        solvers[n_fine] = PoissonSolver(n_coarse=n_coarse_for_solver, n_fine=n_fine)
    
    # Generate random k values (higher frequency for challenging test)
    k1 = np.random.uniform(11.0, 12.0)
    k2 = np.random.uniform(11.0, 12.0)
    print(f"Using wave numbers: k₁={k1:.2f}, k₂={k2:.2f}")
    
    # Generate fields on finest grid (640x640)
    n_finest = max(resolutions)
    x = np.linspace(0, 1, n_finest)
    y = np.linspace(0, 1, n_finest)
    X, Y = np.meshgrid(x, y)
    f_finest = np.sin(k1 * 2 * np.pi * X) * np.sin(k2 * 2 * np.pi * Y)
    theta_finest = np.random.uniform(0.5, 2.0, size=(n_finest, n_finest))
    
    # Initialize data dictionary
    data = {
        'k1': k1,
        'k2': k2,
        'f': {},
        'theta': {},
        'u': {}
    }
    
    # Downsample and solve for each resolution
    print("\nSolving Poisson equation at all resolutions...")
    for res in [n_coarse] + resolutions:
        # Downsample f and theta
        if res == n_finest:
            data['f'][res] = f_finest
            data['theta'][res] = theta_finest
        else:
            step = n_finest // res
            data['f'][res] = f_finest[::step, ::step]
            data['theta'][res] = theta_finest[::step, ::step]
        
        # Solve PDE
        if res == n_coarse:
            solver = PoissonSolver(n_coarse=res//2, n_fine=res)
            data['u'][res] = solver.solve_poisson(
                data['f'][res], 
                data['theta'][res], 
                'fine'
            )
        else:
            data['u'][res] = solvers[res].solve_poisson(
                data['f'][res],
                data['theta'][res],
                'fine'
            )
        
        print(f"\nSolution statistics for {res}x{res}:")
        print(f"u_{res} - min: {data['u'][res].min():.6f}, max: {data['u'][res].max():.6f}")
    
    return data

def upscale_subdomain(model: torch.nn.Module, u_coarse: np.ndarray, 
                     f_fine: np.ndarray, theta_fine: np.ndarray,
                     global_norm, device: str) -> np.ndarray:
    """
    Apply the ML model to upscale a single subdomain using global normalization.
    """
    # Convert inputs to tensors
    u_coarse = torch.from_numpy(u_coarse).float().to(device)
    f_fine = torch.from_numpy(f_fine).float().to(device)
    theta_fine = torch.from_numpy(theta_fine).float().to(device)
    
    # Normalize using global statistics
    u_coarse_norm = (u_coarse - global_norm.u_mean) / global_norm.u_std
    f_fine_norm = (f_fine - global_norm.f_mean) / global_norm.f_std
    
    if global_norm.theta_is_constant:
        theta_fine_norm = theta_fine
    else:
        theta_fine_norm = (theta_fine - global_norm.theta_mean) / global_norm.theta_std
    
    # Upsample coarse solution
    u_coarse_upsampled = F.interpolate(
        u_coarse_norm.unsqueeze(0).unsqueeze(0),
        size=(40, 40),
        mode='bilinear',
        align_corners=True
    )
    
    # Combine inputs
    inputs = torch.cat([
        u_coarse_upsampled.squeeze(0),
        theta_fine_norm.unsqueeze(0),
        f_fine_norm.unsqueeze(0)
    ], dim=0)
    
    # Get model prediction
    with torch.no_grad():
        prediction = model(inputs.unsqueeze(0))
        prediction = prediction * global_norm.u_std + global_norm.u_mean
        prediction = prediction.squeeze().cpu().numpy()
    
    return prediction

def split_into_subdomains(array: np.ndarray, subdomain_size: int) -> list:
    """Split a 2D array into subdomains."""
    n_subdomains = array.shape[0] // subdomain_size
    subdomains = []
    
    for i in range(n_subdomains):
        row = []
        for j in range(n_subdomains):
            start_i = i * subdomain_size
            start_j = j * subdomain_size
            end_i = start_i + subdomain_size
            end_j = start_j + subdomain_size
            subdomain = array[start_i:end_i, start_j:end_j]
            row.append(subdomain)
        subdomains.append(row)
    
    return subdomains

def stitch_subdomains(subdomains: list) -> np.ndarray:
    """Stitch subdomains back together."""
    n_rows = len(subdomains)
    n_cols = len(subdomains[0])
    subdomain_size = subdomains[0][0].shape[0]
    full_size = n_rows * subdomain_size
    
    result = np.zeros((full_size, full_size))
    
    for i in range(n_rows):
        for j in range(n_cols):
            start_i = i * subdomain_size
            start_j = j * subdomain_size
            end_i = start_i + subdomain_size
            end_j = start_j + subdomain_size
            result[start_i:end_i, start_j:end_j] = subdomains[i][j]
    
    return result

class GlobalNormalization:
    """Compute and store global normalization statistics."""
    def __init__(self, u_fine, u_coarse, f_fine, theta_fine):
        # Convert to tensors
        u_fine = torch.from_numpy(u_fine).float()
        u_coarse = torch.from_numpy(u_coarse).float()
        f_fine = torch.from_numpy(f_fine).float()
        theta_fine = torch.from_numpy(theta_fine).float()
        
        # Compute statistics
        self.u_mean = u_fine.mean()
        self.u_std = u_fine.std()
        self.f_mean = f_fine.mean()
        self.f_std = f_fine.std()
        
        self.theta_is_constant = (theta_fine.std() < 1e-6)
        if self.theta_is_constant:
            self.theta_mean = 0
            self.theta_std = 1
        else:
            self.theta_mean = theta_fine.mean()
            self.theta_std = theta_fine.std()

def ml_multi_level_upscale(model: torch.nn.Module, data: dict, 
                          target_resolution: int, device: str) -> np.ndarray:
    """
    Perform multi-level upscaling using ML model from 40x40 to target resolution.
    """
    current_res = 40
    current_solution = data['u'][current_res]
    
    while current_res < target_resolution:
        next_res = current_res * 2
        print(f"\nUpscaling {current_res}x{current_res} → {next_res}x{next_res}")
        
        # Prepare normalization
        global_norm = GlobalNormalization(
            data['u'][next_res],  # fine solution (ground truth)
            current_solution,      # coarse solution (current)
            data['f'][next_res],  # fine forcing
            data['theta'][next_res]  # fine theta
        )
        
        # Split into 20x20 subdomains
        n_subdomains = current_res // 20
        coarse_subdomains = split_into_subdomains(current_solution, 20)
        f_subdomains = split_into_subdomains(data['f'][next_res], 40)
        theta_subdomains = split_into_subdomains(data['theta'][next_res], 40)
        
        # Process each subdomain
        upscaled_subdomains = []
        for i in range(n_subdomains):
            row = []
            for j in range(n_subdomains):
                prediction = upscale_subdomain(
                    model,
                    coarse_subdomains[i][j],
                    f_subdomains[i][j],
                    theta_subdomains[i][j],
                    global_norm,
                    device
                )
                row.append(prediction)
            upscaled_subdomains.append(row)
        
        # Update for next iteration
        current_solution = stitch_subdomains(upscaled_subdomains)
        current_res = next_res
    
    return current_solution

def plot_resolution_comparison(data: dict, ml_solutions: dict, 
                             bilinear_solutions: dict, save_dir: Path):
    """
    Create comparison plots for each resolution.
    """
    resolutions = sorted([res for res in ml_solutions.keys()])
    
    # Plot error metrics vs resolution
    plt.figure(figsize=(12, 8))
    plt.title('Error Metrics vs Resolution', fontsize=14)
    
    ml_maes = []
    ml_rmses = []
    bilinear_maes = []
    bilinear_rmses = []
    
    for res in resolutions:
        # ML metrics
        ml_error = np.abs(ml_solutions[res] - data['u'][res])
        ml_mae = np.mean(ml_error)
        ml_rmse = np.sqrt(np.mean(ml_error**2))
        ml_maes.append(ml_mae)
        ml_rmses.append(ml_rmse)
        
        # Bilinear metrics
        bl_error = np.abs(bilinear_solutions[res] - data['u'][res])
        bl_mae = np.mean(bl_error)
        bl_rmse = np.sqrt(np.mean(bl_error**2))
        bilinear_maes.append(bl_mae)
        bilinear_rmses.append(bl_rmse)
    
    # Plot metrics
    plt.plot(resolutions, ml_maes, 'bo-', label='ML MAE', linewidth=2)
    plt.plot(resolutions, ml_rmses, 'b^--', label='ML RMSE', linewidth=2)
    plt.plot(resolutions, bilinear_maes, 'ro-', label='Bilinear MAE', linewidth=2)
    plt.plot(resolutions, bilinear_rmses, 'r^--', label='Bilinear RMSE', linewidth=2)
    
    plt.xlabel('Resolution', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.xticks(resolutions, [f'{r}x{r}' for r in resolutions])
    
    # Add value labels
    for i, res in enumerate(resolutions):
        plt.text(res, ml_maes[i], f'{ml_maes[i]:.6f}', 
                verticalalignment='bottom', horizontalalignment='right')
        plt.text(res, bilinear_maes[i], f'{bilinear_maes[i]:.6f}',
                verticalalignment='bottom', horizontalalignment='right')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'resolution_comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed comparison plots for each resolution
    for res in resolutions:
        fig = plt.figure(figsize=(20, 20))
        plt.suptitle(f'Solution Comparison at {res}x{res}', fontsize=16)
        
        # Use consistent normalization
        vmin = min(data['u'][res].min(), ml_solutions[res].min(), bilinear_solutions[res].min())
        vmax = max(data['u'][res].max(), ml_solutions[res].max(), bilinear_solutions[res].max())
        
        # Ground truth
        ax1 = plt.subplot(331)
        im1 = ax1.imshow(data['u'][res], vmin=vmin, vmax=vmax)
        ax1.set_title(f'Ground Truth ({res}x{res})')
        plt.colorbar(im1, ax=ax1)
        
        # ML solution
        ax2 = plt.subplot(332)
        im2 = ax2.imshow(ml_solutions[res], vmin=vmin, vmax=vmax)
        ml_mae = np.mean(np.abs(ml_solutions[res] - data['u'][res]))
        ax2.set_title(f'ML Multi-level\nMAE: {ml_mae:.6f}')
        plt.colorbar(im2, ax=ax2)
        
        # Bilinear solution
        ax3 = plt.subplot(333)
        im3 = ax3.imshow(bilinear_solutions[res], vmin=vmin, vmax=vmax)
        bl_mae = np.mean(np.abs(bilinear_solutions[res] - data['u'][res]))
        ax3.set_title(f'Direct Bilinear\nMAE: {bl_mae:.6f}')
        plt.colorbar(im3, ax=ax3)
        
        # Error plots
        ax4 = plt.subplot(334)
        error_ml = np.abs(ml_solutions[res] - data['u'][res])
        im4 = ax4.imshow(error_ml)
        ax4.set_title('ML Error')
        plt.colorbar(im4, ax=ax4)
        
        ax5 = plt.subplot(335)
        error_bl = np.abs(bilinear_solutions[res] - data['u'][res])
        im5 = ax5.imshow(error_bl)
        ax5.set_title('Bilinear Error')
        plt.colorbar(im5, ax=ax5)
        
        # Error difference
        ax6 = plt.subplot(336)
        error_diff = error_ml - error_bl
        im6 = ax6.imshow(error_diff, cmap='RdBu')
        ax6.set_title('Error Difference\n(Blue: ML better)')
        plt.colorbar(im6, ax=ax6)

        # Theta plot
        ax7 = plt.subplot(337)
        im7 = ax7.imshow(data['theta'][res])
        ax7.set_title(f'Theta ({res}x{res})')
        plt.colorbar(im7, ax=ax7)

        # Forcing term plot
        ax8 = plt.subplot(338)
        im8 = ax8.imshow(data['f'][res])
        ax8.set_title(f'Forcing Term ({res}x{res})')
        plt.colorbar(im8, ax=ax8)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'comparison_{res}x{res}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create error distribution plot
        plt.figure(figsize=(12, 8))
        plt.title(f'Error Distribution at {res}x{res}', fontsize=14)
        
        sns.kdeplot(data=error_ml.flatten(), 
                   label=f'ML Multi-level (MAE: {ml_mae:.6f})',
                   fill=True, alpha=0.5)
        sns.kdeplot(data=error_bl.flatten(),
                   label=f'Direct Bilinear (MAE: {bl_mae:.6f})',
                   fill=True, alpha=0.5)
        
        plt.xlabel('Absolute Error', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add statistical information
        ml_std = np.std(error_ml)
        bl_std = np.std(error_bl)
        plt.text(0.98, 0.95,
                f'ML Std: {ml_std:.6f}\nBilinear Std: {bl_std:.6f}',
                transform=plt.gca().transAxes,
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_dir / f'error_distribution_{res}x{res}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Multi-resolution upscaling comparison')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model file')
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at path: {model_path}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    model.eval()
    
    # Create results directory
    results_dir = model_path.parent / 'resolution_comparison_results'
    results_dir.mkdir(exist_ok=True)
    
    # Define resolutions to test
    resolutions = [80, 160, 320, 640]
    
    # Generate test data
    data = solve_multi_resolution(n_coarse=40, resolutions=resolutions)
    
    # Initialize solution dictionaries
    ml_solutions = {}
    bilinear_solutions = {}
    
    # Perform upscaling for each target resolution
    for res in resolutions:
        print(f"\n=== Testing upscaling to {res}x{res} ===")
        
        # ML multi-level upscaling
        print("\nPerforming ML multi-level upscaling...")
        ml_solutions[res] = ml_multi_level_upscale(
            model, data, res, device
        )
        
        # Direct bilinear upscaling
        print("\nPerforming direct bilinear upscaling...")
        bilinear_solutions[res] = F.interpolate(
            torch.from_numpy(data['u'][40]).float().unsqueeze(0).unsqueeze(0),
            size=(res, res),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()
        
        # Calculate metrics
        ml_mae = np.mean(np.abs(ml_solutions[res] - data['u'][res]))
        ml_rmse = np.sqrt(np.mean((ml_solutions[res] - data['u'][res])**2))
        
        bl_mae = np.mean(np.abs(bilinear_solutions[res] - data['u'][res]))
        bl_rmse = np.sqrt(np.mean((bilinear_solutions[res] - data['u'][res])**2))
        
        print(f"\nResults for {res}x{res}:")
        print(f"ML Multi-level - MAE: {ml_mae:.6f}, RMSE: {ml_rmse:.6f}")
        print(f"Direct Bilinear - MAE: {bl_mae:.6f}, RMSE: {bl_rmse:.6f}")
    
    # Create comparison plots
    plot_resolution_comparison(data, ml_solutions, bilinear_solutions, results_dir)

if __name__ == '__main__':
    main() 