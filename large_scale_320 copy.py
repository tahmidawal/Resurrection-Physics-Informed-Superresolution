import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from data_generation import PoissonSolver
from models import UNet, PDEDataset
from compare_methods import load_model
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

def split_into_subdomains(array: np.ndarray, subdomain_size: int) -> list:
    """
    Split a 2D array into subdomains.
    Returns a 2D list of subdomains.
    """
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
    """
    Stitch subdomains back together into a single array.
    """
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
    def __init__(self, data: dict):
        """
        Compute global normalization statistics from the full dataset.
        """
        # Convert to tensors for computation
        u_fine = torch.from_numpy(data['u_fine']).float()
        u_coarse = torch.from_numpy(data['u_coarse']).float()
        f_fine = torch.from_numpy(data['f_fine']).float()
        theta_fine = torch.from_numpy(data['theta_fine']).float()
        
        # Compute global statistics
        self.u_mean = u_fine.mean()
        self.u_std = u_fine.std()
        self.f_mean = f_fine.mean()
        self.f_std = f_fine.std()
        
        # For theta, if it's constant, don't normalize
        self.theta_is_constant = (theta_fine.std() < 1e-6)
        if self.theta_is_constant:
            self.theta_mean = 0
            self.theta_std = 1
            print("Detected constant theta field, skipping normalization")
        else:
            self.theta_mean = theta_fine.mean()
            self.theta_std = theta_fine.std()
        
        print("\nGlobal normalization statistics:")
        print(f"u_mean: {self.u_mean:.6f}, u_std: {self.u_std:.6f}")
        print(f"f_mean: {self.f_mean:.6f}, f_std: {self.f_std:.6f}")
        print(f"theta_mean: {self.theta_mean:.6f}, theta_std: {self.theta_std:.6f}")

def upscale_subdomain(model: torch.nn.Module, u_coarse: np.ndarray, 
                     f_fine: np.ndarray, theta_fine: np.ndarray,
                     global_norm: GlobalNormalization,
                     device: str) -> np.ndarray:
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
    
    # Print input stats for debugging
    print(f"Input stats - min: {inputs.min().item():.6f}, max: {inputs.max().item():.6f}")
    
    # Get model prediction
    with torch.no_grad():
        prediction = model(inputs.unsqueeze(0))
        print(f"Prediction stats (before denorm) - min: {prediction.min().item():.6f}, max: {prediction.max().item():.6f}")
        
        # Denormalize
        prediction = prediction * global_norm.u_std + global_norm.u_mean
        print(f"Prediction stats (after denorm) - min: {prediction.min().item():.6f}, max: {prediction.max().item():.6f}")
        
        prediction = prediction.squeeze().cpu().numpy()
    
    return prediction

def solve_320x320(n_coarse: int = 160, n_fine: int = 320):
    """
    Solve PDE on 320x320 grid with 160x160 coarse grid.
    """
    print("\nGenerating 320x320 test case...")
    solver = PoissonSolver(n_coarse=n_coarse, n_fine=n_fine)
    
    # Generate random k values from higher range (5.0, 8.0)
    k1 = np.random.uniform(5.0, 8.0)
    k2 = np.random.uniform(5.0, 8.0)
    print(f"Using wave numbers: k₁={k1:.2f}, k₂={k2:.2f}")
    
    # Generate source term (f) on fine grid
    x = np.linspace(0, 1, n_fine)
    y = np.linspace(0, 1, n_fine)
    X, Y = np.meshgrid(x, y)
    f_fine = np.sin(k1 * 2 * np.pi * X) * np.sin(k2 * 2 * np.pi * Y)
    
    # Generate theta field on fine grid
    theta_fine = np.random.uniform(0.5, 2.0, size=(n_fine, n_fine))
    
    # Downsample f and theta to coarse grid
    f_coarse = f_fine[::2, ::2]
    theta_coarse = theta_fine[::2, ::2]
    
    # Solve on both grids
    print("\nSolving Poisson equation on both grids...")
    u_fine = solver.solve_poisson(f_fine, theta_fine, 'fine')
    u_coarse = solver.solve_poisson(f_coarse, theta_coarse, 'coarse')
    
    print(f"\nSolution statistics:")
    print(f"u_fine (320x320) - min: {u_fine.min():.6f}, max: {u_fine.max():.6f}")
    print(f"u_coarse (160x160) - min: {u_coarse.min():.6f}, max: {u_coarse.max():.6f}")
    
    return {
        'u_fine': u_fine,
        'u_coarse': u_coarse,
        'f_fine': f_fine,
        'f_coarse': f_coarse,
        'theta_fine': theta_fine,
        'theta_coarse': theta_coarse,
        'k1': k1,
        'k2': k2
    }

def direct_upscale_320x320(model: torch.nn.Module, data: dict, device: str) -> np.ndarray:
    """
    Direct upscaling from 160x160 to 320x320 using the ML model.
    The ML model operates on 20x20 → 40x40 patches, so we:
    1. Split the 160x160 into 8x8 subdomains of size 20x20
    2. Upscale each subdomain to 40x40
    3. Stitch together the 40x40 patches to form 320x320
    """
    # First split the 160x160 grid into 8x8 subdomains of size 20x20
    coarse_subdomains = []
    f_fine_subdomains = []
    theta_fine_subdomains = []
    
    for i in range(8):
        coarse_row = []
        f_row = []
        theta_row = []
        for j in range(8):
            # Extract 20x20 patches from coarse grid
            start_i = i * 20
            start_j = j * 20
            end_i = start_i + 20
            end_j = start_j + 20
            
            coarse_patch = data['u_coarse'][start_i:end_i, start_j:end_j]
            coarse_row.append(coarse_patch)
            
            # Extract corresponding 40x40 patches from fine grid
            start_i_fine = i * 40
            start_j_fine = j * 40
            end_i_fine = start_i_fine + 40
            end_j_fine = start_j_fine + 40
            
            f_patch = data['f_fine'][start_i_fine:end_i_fine, start_j_fine:end_j_fine]
            theta_patch = data['theta_fine'][start_i_fine:end_i_fine, start_j_fine:end_j_fine]
            
            f_row.append(f_patch)
            theta_row.append(theta_patch)
        
        coarse_subdomains.append(coarse_row)
        f_fine_subdomains.append(f_row)
        theta_fine_subdomains.append(theta_row)
    
    # Global normalization using the full dataset
    global_norm = GlobalNormalization(data)
    
    # Process each subdomain
    upscaled_subdomains = []
    print("\nProcessing subdomains for direct upscaling...")
    for i in range(8):
        row = []
        for j in range(8):
            print(f"Processing subdomain ({i+1}, {j+1})")
            prediction = upscale_subdomain(
                model,
                coarse_subdomains[i][j],  # 20x20
                f_fine_subdomains[i][j],  # 40x40
                theta_fine_subdomains[i][j],  # 40x40
                global_norm,
                device
            )  # Returns 40x40
            row.append(prediction)
        upscaled_subdomains.append(row)
    
    # Stitch together the 40x40 patches to form the final 320x320 solution
    return stitch_subdomains(upscaled_subdomains)

def solve_multi_level_320(n_coarse: int = 40, n_mid1: int = 80, n_mid2: int = 160, n_fine: int = 320):
    """
    Solve PDE with multiple resolution levels up to 320x320.
    """
    print("\nGenerating multi-level test case (up to 320x320)...")
    
    # Create solvers for each resolution
    solver_fine = PoissonSolver(n_coarse=n_mid2, n_fine=n_fine)     # 160x320
    solver_mid2 = PoissonSolver(n_coarse=n_mid1, n_fine=n_mid2)     # 80x160
    solver_mid1 = PoissonSolver(n_coarse=n_coarse, n_fine=n_mid1)   # 40x80
    
    # Generate random k values
    k1 = np.random.uniform(5.0, 8.0)
    k2 = np.random.uniform(5.0, 8.0)
    print(f"Using wave numbers: k₁={k1:.2f}, k₂={k2:.2f}")
    
    # Generate fields on finest grid
    x = np.linspace(0, 1, n_fine)
    y = np.linspace(0, 1, n_fine)
    X, Y = np.meshgrid(x, y)
    f_fine = np.sin(k1 * 2 * np.pi * X) * np.sin(k2 * 2 * np.pi * Y)
    theta_fine = np.random.uniform(0.5, 2.0, size=(n_fine, n_fine))
    
    # Downsample to other resolutions
    f_mid2 = f_fine[::2, ::2]      # 160x160
    f_mid1 = f_mid2[::2, ::2]      # 80x80
    f_coarse = f_mid1[::2, ::2]    # 40x40
    
    theta_mid2 = theta_fine[::2, ::2]
    theta_mid1 = theta_mid2[::2, ::2]
    theta_coarse = theta_mid1[::2, ::2]
    
    # Solve on all grids
    print("\nSolving Poisson equation on all grids...")
    u_fine = solver_fine.solve_poisson(f_fine, theta_fine, 'fine')
    u_mid2 = solver_mid2.solve_poisson(f_mid2, theta_mid2, 'fine')
    u_mid1 = solver_mid1.solve_poisson(f_mid1, theta_mid1, 'fine')
    u_coarse = solver_mid1.solve_poisson(f_coarse, theta_coarse, 'coarse')
    
    print(f"\nSolution statistics:")
    print(f"u_fine (320x320) - min: {u_fine.min():.6f}, max: {u_fine.max():.6f}")
    print(f"u_mid2 (160x160) - min: {u_mid2.min():.6f}, max: {u_mid2.max():.6f}")
    print(f"u_mid1 (80x80) - min: {u_mid1.min():.6f}, max: {u_mid1.max():.6f}")
    print(f"u_coarse (40x40) - min: {u_coarse.min():.6f}, max: {u_coarse.max():.6f}")
    
    return {
        'u_fine': u_fine,
        'u_mid2': u_mid2,
        'u_mid1': u_mid1,
        'u_coarse': u_coarse,
        'f_fine': f_fine,
        'f_mid2': f_mid2,
        'f_mid1': f_mid1,
        'f_coarse': f_coarse,
        'theta_fine': theta_fine,
        'theta_mid2': theta_mid2,
        'theta_mid1': theta_mid1,
        'theta_coarse': theta_coarse,
        'k1': k1,
        'k2': k2
    }

def plot_320x320_comparison(data: dict, ml_solution: np.ndarray, 
                          bilinear_solution: np.ndarray, save_dir: Path):
    """
    Create detailed comparison plots for 320x320 results.
    """
    fig = plt.figure(figsize=(20, 15))
    plt.suptitle('Large-scale Solution Comparison (320×320)', fontsize=16)
    
    # Use consistent normalization
    vmin = min(data['u_coarse'].min(), data['u_fine'].min(), 
              bilinear_solution.min(), ml_solution.min())
    vmax = max(data['u_coarse'].max(), data['u_fine'].max(), 
              bilinear_solution.max(), ml_solution.max())
    
    # Original solutions
    ax1 = plt.subplot(231)
    im1 = ax1.imshow(data['u_coarse'], vmin=vmin, vmax=vmax)
    ax1.set_title('Coarse Solution (160×160)')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = plt.subplot(232)
    im2 = ax2.imshow(data['u_fine'], vmin=vmin, vmax=vmax)
    ax2.set_title('Ground Truth (320×320)')
    plt.colorbar(im2, ax=ax2)
    
    ax3 = plt.subplot(233)
    im3 = ax3.imshow(data['theta_fine'], vmin=0.5, vmax=2.0)
    ax3.set_title('θ (Diffusion Coefficient)')
    plt.colorbar(im3, ax=ax3)
    
    # Upscaled solutions and errors
    ax4 = plt.subplot(234)
    im4 = ax4.imshow(bilinear_solution, vmin=vmin, vmax=vmax)
    bilinear_mae = np.mean(np.abs(bilinear_solution - data['u_fine']))
    ax4.set_title(f'Bilinear (MAE: {bilinear_mae:.6f})')
    plt.colorbar(im4, ax=ax4)
    
    ax5 = plt.subplot(235)
    im5 = ax5.imshow(ml_solution, vmin=vmin, vmax=vmax)
    ml_mae = np.mean(np.abs(ml_solution - data['u_fine']))
    ax5.set_title(f'ML Solution (MAE: {ml_mae:.6f})')
    plt.colorbar(im5, ax=ax5)
    
    # Error difference
    ax6 = plt.subplot(236)
    error_diff = np.abs(ml_solution - data['u_fine']) - np.abs(bilinear_solution - data['u_fine'])
    im6 = ax6.imshow(error_diff, cmap='RdBu')
    ax6.set_title('Error Difference\n(Blue: ML better, Red: Bilinear better)')
    plt.colorbar(im6, ax=ax6)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'large_scale_320_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot error distributions
    plt.figure(figsize=(12, 6))
    plt.title('Error Distribution')
    
    ml_errors = np.abs(ml_solution - data['u_fine']).flatten()
    bilinear_errors = np.abs(bilinear_solution - data['u_fine']).flatten()
    
    plt.hist(ml_errors, bins=50, alpha=0.5, label=f'ML (MAE: {ml_mae:.6f})', density=True)
    plt.hist(bilinear_errors, bins=50, alpha=0.5, label=f'Bilinear (MAE: {bilinear_mae:.6f})', density=True)
    
    plt.xlabel('Absolute Error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'large_scale_320_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot cross-sections
    plt.figure(figsize=(15, 5))
    plt.suptitle('Solution Cross-sections at y=160', fontsize=12)
    
    mid_idx = data['u_fine'].shape[0] // 2
    x = np.linspace(0, 1, data['u_fine'].shape[1])
    
    plt.plot(x, data['u_fine'][mid_idx, :], 'k-', label='Ground Truth', linewidth=2)
    plt.plot(x, bilinear_solution[mid_idx, :], 'r--', label='Bilinear', linewidth=2)
    plt.plot(x, ml_solution[mid_idx, :], 'b--', label='ML Model', linewidth=2)
    
    plt.xlabel('x coordinate')
    plt.ylabel('Solution value')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'large_scale_320_cross_section.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_multi_level_320_comparison(data: dict, ml_solutions: dict, save_dir: Path):
    """
    Create comparison plots for multi-level upscaling results.
    """
    # Add direct bilinear upscaling from 40x40 to 320x320
    bilinear_direct_320 = F.interpolate(
        torch.from_numpy(data['u_coarse']).float().unsqueeze(0).unsqueeze(0),
        size=(320, 320),
        mode='bilinear',
        align_corners=True
    ).squeeze().numpy()
    
    # Calculate direct bilinear error
    bilinear_direct_mae = np.mean(np.abs(bilinear_direct_320 - data['u_fine']))
    bilinear_direct_rmse = np.sqrt(np.mean((bilinear_direct_320 - data['u_fine'])**2))
    
    print("\nDirect Bilinear (40x40 → 320x320) Results:")
    print(f"MAE: {bilinear_direct_mae:.6f}, RMSE: {bilinear_direct_rmse:.6f}")
    
    # Plot solutions at each resolution level
    resolutions = ['40x40', '80x80', '160x160', '320x320']
    solutions = [data['u_coarse'], data['u_mid1'], data['u_mid2'], data['u_fine']]
    ml_upscaled = [None, ml_solutions['80x80'], ml_solutions['160x160'], ml_solutions['320x320']]
    
    for idx, (res, true_sol, ml_sol) in enumerate(zip(
        resolutions[1:], solutions[1:], ml_upscaled[1:])):
        
        fig = plt.figure(figsize=(15, 10))
        plt.suptitle(f'Solution Comparison at {res}', fontsize=16)
        
        # Use consistent normalization
        vmin = min(true_sol.min(), ml_sol.min())
        vmax = max(true_sol.max(), ml_sol.max())
        
        # Plot solutions
        ax1 = plt.subplot(221)
        im1 = ax1.imshow(true_sol, vmin=vmin, vmax=vmax)
        ax1.set_title(f'Ground Truth ({res})')
        plt.colorbar(im1, ax=ax1)
        
        ax2 = plt.subplot(222)
        im2 = ax2.imshow(ml_sol, vmin=vmin, vmax=vmax)
        ml_mae = np.mean(np.abs(ml_sol - true_sol))
        ax2.set_title(f'ML Solution\nMAE: {ml_mae:.6f}')
        plt.colorbar(im2, ax=ax2)
        
        # Plot errors
        ax3 = plt.subplot(223)
        error_ml = np.abs(ml_sol - true_sol)
        im3 = ax3.imshow(error_ml)
        ax3.set_title('ML Error')
        plt.colorbar(im3, ax=ax3)
        
        # Plot error histogram
        ax4 = plt.subplot(224)
        ax4.hist(error_ml.flatten(), bins=50, density=True)
        ax4.set_title('Error Distribution')
        ax4.set_xlabel('Absolute Error')
        ax4.set_ylabel('Density')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'multi_level_320_comparison_{res}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Final comparison with direct bilinear
    fig = plt.figure(figsize=(20, 10))
    plt.suptitle('ML Multi-level vs Direct Bilinear Upscaling (40x40 → 320x320)', fontsize=16)
    
    vmin = min(data['u_fine'].min(), ml_solutions['320x320'].min(), bilinear_direct_320.min())
    vmax = max(data['u_fine'].max(), ml_solutions['320x320'].max(), bilinear_direct_320.max())
    
    # Ground truth
    ax1 = plt.subplot(231)
    im1 = ax1.imshow(data['u_fine'], vmin=vmin, vmax=vmax)
    ax1.set_title('Ground Truth (320x320)')
    plt.colorbar(im1, ax=ax1)
    
    # ML multi-level
    ax2 = plt.subplot(232)
    im2 = ax2.imshow(ml_solutions['320x320'], vmin=vmin, vmax=vmax)
    ml_mae = np.mean(np.abs(ml_solutions['320x320'] - data['u_fine']))
    ax2.set_title(f'ML Multi-level\nMAE: {ml_mae:.6f}')
    plt.colorbar(im2, ax=ax2)
    
    # Direct bilinear
    ax3 = plt.subplot(233)
    im3 = ax3.imshow(bilinear_direct_320, vmin=vmin, vmax=vmax)
    ax3.set_title(f'Direct Bilinear\nMAE: {bilinear_direct_mae:.6f}')
    plt.colorbar(im3, ax=ax3)
    
    # ML error
    ax4 = plt.subplot(234)
    error_ml = np.abs(ml_solutions['320x320'] - data['u_fine'])
    im4 = ax4.imshow(error_ml)
    ax4.set_title('ML Error')
    plt.colorbar(im4, ax=ax4)
    
    # Bilinear error
    ax5 = plt.subplot(235)
    error_bl = np.abs(bilinear_direct_320 - data['u_fine'])
    im5 = ax5.imshow(error_bl)
    ax5.set_title('Bilinear Error')
    plt.colorbar(im5, ax=ax5)
    
    # Error difference
    ax6 = plt.subplot(236)
    error_diff = error_ml - error_bl
    im6 = ax6.imshow(error_diff, cmap='RdBu')
    ax6.set_title('Error Difference\n(Blue: ML better)')
    plt.colorbar(im6, ax=ax6)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'multi_level_320_final_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot error progression for ML model
    plt.figure(figsize=(10, 6))
    plt.title('ML Model Error Progression Across Resolutions')
    
    resolutions = ['80x80', '160x160', '320x320']
    ml_maes = [np.mean(np.abs(ml_solutions[res] - sol)) 
               for res, sol in zip(resolutions, solutions[1:])]
    
    plt.plot(resolutions, ml_maes, 'bo-', label='ML Model')
    plt.axhline(y=bilinear_direct_mae, color='r', linestyle='--', 
                label='Direct Bilinear (40x40 → 320x320)')
    
    plt.xlabel('Resolution')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'multi_level_320_error_progression.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create detailed error distribution comparison
    plt.figure(figsize=(12, 8))
    plt.title('Error Distribution Comparison (320x320)', fontsize=14)
    
    # Get error data
    ml_errors = error_ml.flatten()
    bilinear_errors = error_bl.flatten()
    
    # Plot histograms with kernel density estimation
    sns.kdeplot(data=ml_errors, label=f'ML Multi-level (MAE: {ml_mae:.6f})', 
                fill=True, alpha=0.5)
    sns.kdeplot(data=bilinear_errors, label=f'Direct Bilinear (MAE: {bilinear_direct_mae:.6f})', 
                fill=True, alpha=0.5)
    
    plt.xlabel('Absolute Error', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistical information
    ml_std = np.std(ml_errors)
    bilinear_std = np.std(bilinear_errors)
    plt.text(0.98, 0.95, 
             f'ML Std: {ml_std:.6f}\nBilinear Std: {bilinear_std:.6f}',
             transform=plt.gca().transAxes,
             horizontalalignment='right',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'multi_level_320_error_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Large-scale upscaling to 320x320')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
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
    results_dir = model_path.parent / 'large_scale_320_results'
    results_dir.mkdir(exist_ok=True)
    
    # Test multi-level upscaling
    print("\n=== Testing Multi-level Upscaling to 320x320 ===")
    multi_data = solve_multi_level_320()
    
    # Initialize dictionary to store ML solutions
    ml_solutions = {}
    
    # Stage 1: 40x40 → 80x80
    global_norm_stage1 = GlobalNormalization({
        'u_fine': multi_data['u_mid1'],
        'u_coarse': multi_data['u_coarse'],
        'f_fine': multi_data['f_mid1'],
        'theta_fine': multi_data['theta_mid1']
    })
    
    print("\nStage 1: 40x40 → 80x80")
    stage1_subdomains = []
    for i in range(2):
        row = []
        for j in range(2):
            prediction = upscale_subdomain(
                model,
                split_into_subdomains(multi_data['u_coarse'], 20)[i][j],
                split_into_subdomains(multi_data['f_mid1'], 40)[i][j],
                split_into_subdomains(multi_data['theta_mid1'], 40)[i][j],
                global_norm_stage1,
                device
            )
            row.append(prediction)
        stage1_subdomains.append(row)
    
    ml_solutions['80x80'] = stitch_subdomains(stage1_subdomains)
    
    # Stage 2: 80x80 → 160x160
    global_norm_stage2 = GlobalNormalization({
        'u_fine': multi_data['u_mid2'],
        'u_coarse': ml_solutions['80x80'],
        'f_fine': multi_data['f_mid2'],
        'theta_fine': multi_data['theta_mid2']
    })
    
    print("\nStage 2: 80x80 → 160x160")
    stage2_subdomains = []
    for i in range(4):
        row = []
        for j in range(4):
            prediction = upscale_subdomain(
                model,
                split_into_subdomains(ml_solutions['80x80'], 20)[i][j],
                split_into_subdomains(multi_data['f_mid2'], 40)[i][j],
                split_into_subdomains(multi_data['theta_mid2'], 40)[i][j],
                global_norm_stage2,
                device
            )
            row.append(prediction)
        stage2_subdomains.append(row)
    
    ml_solutions['160x160'] = stitch_subdomains(stage2_subdomains)
    
    # Stage 3: 160x160 → 320x320
    global_norm_stage3 = GlobalNormalization({
        'u_fine': multi_data['u_fine'],
        'u_coarse': ml_solutions['160x160'],
        'f_fine': multi_data['f_fine'],
        'theta_fine': multi_data['theta_fine']
    })
    
    print("\nStage 3: 160x160 → 320x320")
    stage3_subdomains = []
    for i in range(8):
        row = []
        for j in range(8):
            prediction = upscale_subdomain(
                model,
                split_into_subdomains(ml_solutions['160x160'], 20)[i][j],
                split_into_subdomains(multi_data['f_fine'], 40)[i][j],
                split_into_subdomains(multi_data['theta_fine'], 40)[i][j],
                global_norm_stage3,
                device
            )
            row.append(prediction)
        stage3_subdomains.append(row)
    
    ml_solutions['320x320'] = stitch_subdomains(stage3_subdomains)
    
    # Calculate final metrics for ML multi-level
    ml_final_mae = np.mean(np.abs(ml_solutions['320x320'] - multi_data['u_fine']))
    ml_final_rmse = np.sqrt(np.mean((ml_solutions['320x320'] - multi_data['u_fine'])**2))
    
    # Direct bilinear upscaling from 40x40 to 320x320
    bilinear_direct = F.interpolate(
        torch.from_numpy(multi_data['u_coarse']).float().unsqueeze(0).unsqueeze(0),
        size=(320, 320),
        mode='bilinear',
        align_corners=True
    ).squeeze().numpy()
    
    # Calculate direct bilinear metrics
    bilinear_mae = np.mean(np.abs(bilinear_direct - multi_data['u_fine']))
    bilinear_rmse = np.sqrt(np.mean((bilinear_direct - multi_data['u_fine'])**2))
    
    print("\nMulti-level Upscaling Results:")
    print(f"ML Model - MAE: {ml_final_mae:.6f}, RMSE: {ml_final_rmse:.6f}")
    print(f"\nDirect Bilinear (40x40 → 320x320) Results:")
    print(f"MAE: {bilinear_mae:.6f}, RMSE: {bilinear_rmse:.6f}")
    
    # Create comparison plots
    plot_multi_level_320_comparison(multi_data, ml_solutions, results_dir)

if __name__ == '__main__':
    main() 