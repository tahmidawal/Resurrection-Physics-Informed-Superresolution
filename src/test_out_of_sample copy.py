import numpy as np
import torch
from pathlib import Path
from data_generation import PoissonSolver
from models import UNet, PDEDataset
from compare_methods import plot_comparison, load_model
import json
import matplotlib.pyplot as plt
import seaborn as sns

def generate_out_of_sample_data(n_samples: int = 10):
    """
    Generate out-of-sample test cases with different parameter ranges.
    """
    print("Generating out-of-sample test data...")
    solver = PoissonSolver()
    
    # Use different k ranges than training
    k_range = (5.0, 8.0)  # Higher frequency range
    print(f"Using k range: {k_range}")
    
    # Generate dataset with new parameters
    dataset = solver.generate_dataset(
        n_samples=n_samples,
        k_range=k_range
    )
    
    # Save the out-of-sample dataset
    save_path = Path('data/out_of_sample_dataset.npz')
    np.savez(save_path, **dataset)
    print(f"Saved out-of-sample dataset to {save_path}")
    
    return dataset

def plot_detailed_comparison(
    coarse_solution: np.ndarray,
    fine_solution: np.ndarray,
    bilinear_upscaled: np.ndarray,
    ml_upscaled: np.ndarray,
    theta_fine: np.ndarray,
    k1: float,
    k2: float,
    save_dir: Path,
    sample_idx: int
):
    """
    Create and save detailed comparison plots.
    """
    # Create plots directory
    plots_dir = save_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Main comparison plot (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.suptitle(f'Solution Comparison - Sample {sample_idx}\n' + 
                f'k₁={k1:.2f}, k₂={k2:.2f}', fontsize=16)
    
    # First row: Solutions
    im1 = axes[0,0].imshow(coarse_solution)
    axes[0,0].set_title('Coarse Solution (20×20)')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(fine_solution)
    axes[0,1].set_title('Ground Truth (40×40)')
    plt.colorbar(im2, ax=axes[0,1])
    
    im3 = axes[0,2].imshow(theta_fine)
    axes[0,2].set_title('θ (Diffusion Coefficient)')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Second row: Upscaled solutions and error
    im4 = axes[1,0].imshow(bilinear_upscaled)
    axes[1,0].set_title('Bilinear Interpolation')
    plt.colorbar(im4, ax=axes[1,0])
    
    im5 = axes[1,1].imshow(ml_upscaled)
    axes[1,1].set_title('ML Model Prediction')
    plt.colorbar(im5, ax=axes[1,1])
    
    # Error comparison
    error_diff = np.abs(ml_upscaled - fine_solution) - np.abs(bilinear_upscaled - fine_solution)
    im6 = axes[1,2].imshow(error_diff, cmap='RdBu')
    axes[1,2].set_title('Error Difference\n(Blue: ML better, Red: Bilinear better)')
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'comparison_full_{sample_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Error distribution plot
    plt.figure(figsize=(10, 6))
    plt.title(f'Error Distribution - Sample {sample_idx}')
    
    ml_errors = np.abs(ml_upscaled - fine_solution).flatten()
    bilinear_errors = np.abs(bilinear_upscaled - fine_solution).flatten()
    
    sns.kdeplot(data=ml_errors, label='ML Model', fill=True, alpha=0.5)
    sns.kdeplot(data=bilinear_errors, label='Bilinear', fill=True, alpha=0.5)
    
    plt.xlabel('Absolute Error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / f'error_distribution_{sample_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Cross-section plot
    mid_idx = fine_solution.shape[0] // 2
    plt.figure(figsize=(12, 6))
    plt.title(f'Solution Cross-section at y={mid_idx} - Sample {sample_idx}')
    
    plt.plot(fine_solution[mid_idx, :], label='Ground Truth', linewidth=2)
    plt.plot(bilinear_upscaled[mid_idx, :], '--', label='Bilinear', linewidth=2)
    plt.plot(ml_upscaled[mid_idx, :], '--', label='ML Model', linewidth=2)
    
    plt.xlabel('x coordinate')
    plt.ylabel('Solution value')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / f'cross_section_{sample_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_out_of_sample(model_path: Path, device: str = 'cuda'):
    """
    Evaluate the model on out-of-sample data.
    """
    print("\nEvaluating model on out-of-sample data...")
    
    # Load or generate out-of-sample data
    data_path = Path('data/out_of_sample_dataset.npz')
    if data_path.exists():
        print("Loading existing out-of-sample dataset...")
        data = np.load(data_path)
    else:
        print("Generating new out-of-sample dataset...")
        data = generate_out_of_sample_data()
    
    # Create dataset
    dataset = PDEDataset(data, device=device)
    
    # Load model
    model = load_model(model_path, device)
    model.eval()
    
    # Create results directory
    results_dir = model_path.parent / 'out_of_sample_comparison'
    results_dir.mkdir(exist_ok=True)
    
    # Process each sample
    all_metrics = []
    n_samples = len(dataset)
    
    print(f"\nProcessing {n_samples} out-of-sample test cases...")
    for idx in range(n_samples):
        print(f"\nProcessing sample {idx+1}/{n_samples}")
        
        # Get data for this sample
        coarse_solution = data['u_coarse'][idx]
        fine_solution = data['u_fine'][idx]
        theta_fine = data['theta_fine'][idx]
        k1 = data['k1'][idx]
        k2 = data['k2'][idx]
        
        # Get model prediction
        inputs, _ = dataset[idx]
        with torch.no_grad():
            ml_upscaled = model(inputs.unsqueeze(0))
            ml_upscaled = dataset.denormalize(ml_upscaled)
            ml_upscaled = ml_upscaled.squeeze().cpu().numpy()
        
        # Bilinear interpolation
        bilinear_upscaled = torch.nn.functional.interpolate(
            torch.from_numpy(coarse_solution).float().unsqueeze(0).unsqueeze(0),
            size=(40, 40),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()
        
        # Create detailed comparison plots
        plot_detailed_comparison(
            coarse_solution=coarse_solution,
            fine_solution=fine_solution,
            bilinear_upscaled=bilinear_upscaled,
            ml_upscaled=ml_upscaled,
            theta_fine=theta_fine,
            k1=k1,
            k2=k2,
            save_dir=results_dir,
            sample_idx=idx
        )
        
        # Calculate metrics
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
        
        all_metrics.append(metrics)
        
        # Save individual metrics
        with open(results_dir / f'metrics_sample_{idx}.txt', 'w') as f:
            for name, value in metrics.items():
                f.write(f'{name}: {value:.6f}\n')
    
    # Calculate and save average metrics
    avg_metrics = {
        metric: np.mean([m[metric] for m in all_metrics])
        for metric in all_metrics[0].keys()
        if metric not in ['k1', 'k2']
    }
    
    print("\nAverage Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.6f}")
    
    with open(results_dir / 'average_metrics.json', 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    
    # Plot error statistics across all samples
    plt.figure(figsize=(12, 6))
    plt.title('Error Statistics Across Samples')
    
    sample_indices = range(len(all_metrics))
    ml_maes = [m['ml_model_mae'] for m in all_metrics]
    bilinear_maes = [m['bilinear_mae'] for m in all_metrics]
    
    plt.plot(sample_indices, ml_maes, 'o-', label='ML Model MAE')
    plt.plot(sample_indices, bilinear_maes, 'o-', label='Bilinear MAE')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / 'error_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Find the most recent run directory
    results_dir = Path('results')
    latest_run = max(results_dir.glob('run_*'))
    model_path = latest_run / 'best_model.pth'
    
    if not model_path.exists():
        raise FileNotFoundError("No trained model found!")
    
    print(f"Using model from: {model_path}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run evaluation
    evaluate_out_of_sample(model_path, device)

if __name__ == '__main__':
    main() 