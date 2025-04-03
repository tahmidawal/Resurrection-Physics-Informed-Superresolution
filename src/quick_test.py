import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F
from src.models_improved import ContextAwareUNet, OverlappingPDEDataset

def simple_test(model_path='results/run_improved_20250403_091429/best_model.pth'):
    """Run a simple test on a single example."""
    # Force CPU mode
    device = 'cpu'
    print(f"Running simple test on {device}")
    
    # Create output directory
    output_dir = Path('Output/quick_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}")
    model = ContextAwareUNet(in_channels=2, context_padding=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    data_path = Path('data/pde_dataset_overlapping.npz')
    print(f"Loading data from {data_path}")
    data = np.load(data_path)
    
    # Pick a single sample
    idx = 42  # Using a random but fixed index
    
    # Create dataset for normalization
    dataset = OverlappingPDEDataset(data, device=device)
    
    # Get data statistics
    u_mean = dataset.u_mean.item()
    u_std = dataset.u_std.item()
    f_mean = dataset.f_mean.item()
    f_std = dataset.f_std.item()
    
    # Get the specific data we need
    coarse_solution = data['u_coarse'][idx]
    fine_solution = data['u_fine'][idx]
    f_coarse = data['f_coarse'][idx]
    k1 = data['k1'][idx]
    k2 = data['k2'][idx]
    
    # Prepare the input
    coarse_norm = (torch.from_numpy(coarse_solution).float() - u_mean) / u_std
    f_coarse_norm = (torch.from_numpy(f_coarse).float() - f_mean) / f_std
    
    # Upsample to the fine grid
    coarse_upsampled = F.interpolate(
        coarse_norm.unsqueeze(0).unsqueeze(0),
        size=(48, 48),
        mode='bilinear',
        align_corners=True
    )
    
    f_upsampled = F.interpolate(
        f_coarse_norm.unsqueeze(0).unsqueeze(0),
        size=(48, 48),
        mode='bilinear',
        align_corners=True
    )
    
    # Combine inputs
    model_input = torch.cat([
        coarse_upsampled.squeeze(0),
        f_upsampled.squeeze(0)
    ], dim=0).unsqueeze(0)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        prediction = model(model_input)
        prediction = prediction * u_std + u_mean
    
    # Create baseline comparison
    bilinear_upsampled = F.interpolate(
        torch.from_numpy(coarse_solution).float().unsqueeze(0).unsqueeze(0),
        size=(40, 40),
        mode='bilinear',
        align_corners=True
    ).squeeze().numpy()
    
    # Get core of fine solution
    h, w = fine_solution.shape
    start_h = (h - 40) // 2
    start_w = (w - 40) // 2
    fine_core = fine_solution[start_h:start_h+40, start_w:start_w+40]
    
    # Create visualizations
    print("Creating visualizations...")
    prediction_np = prediction.squeeze().numpy()
    
    # Calculate errors
    bilinear_error = np.abs(bilinear_upsampled - fine_core)
    ml_error = np.abs(prediction_np - fine_core)
    
    bilinear_mae = np.mean(bilinear_error)
    ml_mae = np.mean(ml_error)
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    plt.subplot(231)
    plt.imshow(coarse_solution)
    plt.colorbar()
    plt.title('Coarse Solution (Input)')
    
    plt.subplot(232)
    plt.imshow(fine_core)
    plt.colorbar()
    plt.title('Fine Solution (Ground Truth)')
    
    plt.subplot(233)
    plt.imshow(prediction_np)
    plt.colorbar()
    plt.title('ML Prediction')
    
    plt.subplot(234)
    plt.imshow(bilinear_upsampled)
    plt.colorbar()
    plt.title('Bilinear Upsampling')
    
    plt.subplot(235)
    plt.imshow(bilinear_error)
    plt.colorbar()
    plt.title(f'Bilinear Error (MAE: {bilinear_mae:.6f})')
    
    plt.subplot(236)
    plt.imshow(ml_error)
    plt.colorbar()
    plt.title(f'ML Error (MAE: {ml_mae:.6f})')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quick_test_results.png')
    
    print(f"Results:")
    print(f"Bilinear MAE: {bilinear_mae:.6f}")
    print(f"ML Model MAE: {ml_mae:.6f}")
    print(f"Improvement: {bilinear_mae / ml_mae:.2f}x")
    print(f"Visualization saved to {output_dir / 'quick_test_results.png'}")

if __name__ == '__main__':
    simple_test() 