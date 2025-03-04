import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models import UNet, PDEDataset

def load_model(checkpoint_path: Path, device: str = 'cuda') -> UNet:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    model = UNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def plot_solution_comparison(
    true_solution: np.ndarray,
    predicted_solution: np.ndarray,
    coarse_solution: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Plot comparison between true, predicted, and coarse solutions.
    
    Args:
        true_solution: Ground truth solution
        predicted_solution: Model prediction
        coarse_solution: Coarse grid solution
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot true solution
    im0 = axes[0, 0].imshow(true_solution)
    axes[0, 0].set_title('True Solution (40×40)')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Plot predicted solution
    im1 = axes[0, 1].imshow(predicted_solution)
    axes[0, 1].set_title('Predicted Solution (40×40)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Plot coarse solution
    im2 = axes[1, 0].imshow(coarse_solution)
    axes[1, 0].set_title('Coarse Solution (20×20)')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot error
    error = np.abs(true_solution - predicted_solution)
    im3 = axes[1, 1].imshow(error)
    axes[1, 1].set_title('Absolute Error')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()

def compute_metrics(
    true_solution: np.ndarray,
    predicted_solution: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute various error metrics between true and predicted solutions.
    
    Args:
        true_solution: Ground truth solution
        predicted_solution: Model prediction
        
    Returns:
        Tuple of (MSE, RMSE, MAE)
    """
    mse = mean_squared_error(true_solution.flatten(), predicted_solution.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_solution.flatten(), predicted_solution.flatten())
    
    return mse, rmse, mae

def evaluate_model(
    model: UNet,
    dataset: PDEDataset,
    idx: int,
    save_dir: Path,
    show_plots: bool = True
) -> dict:
    """
    Evaluate model performance on a single example.
    
    Args:
        model: Trained model
        dataset: Dataset containing the example
        idx: Index of the example to evaluate
        save_dir: Directory to save results
        show_plots: Whether to display plots
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Get example data
    inputs, target = dataset[idx]
    inputs = inputs.unsqueeze(0)  # Add batch dimension
    
    # Get model prediction
    with torch.no_grad():
        prediction = model(inputs)
    
    # Convert to numpy arrays
    true_solution = target.squeeze().cpu().numpy()
    predicted_solution = prediction.squeeze().cpu().numpy()
    coarse_solution = dataset.u_coarse[idx].cpu().numpy()
    
    # Compute metrics
    mse, rmse, mae = compute_metrics(true_solution, predicted_solution)
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae)
    }
    
    # Save metrics
    with open(save_dir / f'metrics_sample_{idx}.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot and save comparison
    plot_solution_comparison(
        true_solution,
        predicted_solution,
        coarse_solution,
        save_path=save_dir / f'comparison_sample_{idx}.png',
        show=show_plots
    )
    
    return metrics

def main():
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_samples': 5  # Number of samples to evaluate
    }
    
    # Create results directory
    results_dir = Path('../results')
    latest_run = max(results_dir.glob('run_*'))  # Get most recent run
    eval_dir = latest_run / 'evaluation'
    eval_dir.mkdir(exist_ok=True)
    
    # Load model
    model = load_model(latest_run / 'best_model.pth', device=config['device'])
    
    # Load dataset
    data = np.load('../data/pde_dataset.npz')
    dataset = PDEDataset(data, device=config['device'])
    
    # Evaluate model on multiple samples
    all_metrics = []
    for idx in range(config['num_samples']):
        print(f'Evaluating sample {idx+1}/{config["num_samples"]}')
        metrics = evaluate_model(model, dataset, idx, eval_dir, show_plots=False)
        all_metrics.append(metrics)
    
    # Compute and save average metrics
    avg_metrics = {
        metric: float(np.mean([m[metric] for m in all_metrics]))
        for metric in all_metrics[0].keys()
    }
    
    with open(eval_dir / 'average_metrics.json', 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    
    print('\nAverage Metrics:')
    for metric, value in avg_metrics.items():
        print(f'{metric.upper()}: {value:.6f}')

if __name__ == '__main__':
    main() 