import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys
import os

# Fix import to work whether run from project root or src directory
try:
    from src.models_improved import ContextAwareUNet, OverlappingPDEDataset, init_weights, BoundaryAttentionModule
except ImportError:
    from models_improved import ContextAwareUNet, OverlappingPDEDataset, init_weights, BoundaryAttentionModule

# Define a custom loss function that focuses more on boundaries
class BoundaryAwareLoss(nn.Module):
    def __init__(self, lambda_boundary: float = 2.0):
        super().__init__()
        self.lambda_boundary = lambda_boundary
        self.mse = nn.MSELoss()
        
        # Define Sobel filters for edge detection
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Regular MSE loss
        mse_loss = self.mse(pred, target)
        
        # Move Sobel filters to same device as input
        device = pred.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        
        # Compute gradients using Sobel filters
        pred_grad_x = torch.nn.functional.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = torch.nn.functional.conv2d(pred, sobel_y, padding=1)
        target_grad_x = torch.nn.functional.conv2d(target, sobel_x, padding=1)
        target_grad_y = torch.nn.functional.conv2d(target, sobel_y, padding=1)
        
        # Compute gradient magnitude
        pred_grad_magnitude = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
        target_grad_magnitude = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
        
        # Compute boundary loss (MSE on gradients)
        boundary_loss = self.mse(pred_grad_magnitude, target_grad_magnitude)
        
        # Combine losses
        total_loss = mse_loss + self.lambda_boundary * boundary_loss
        
        return total_loss

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    warmup_scheduler: torch.optim.lr_scheduler.LambdaLR,
    num_epochs: int,
    device: str,
    save_dir: Path,
    writer: SummaryWriter,
    grad_clip: float = 0.1,
    early_stopping_patience: int = 30
) -> dict:
    """
    Train the model and save checkpoints.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Main learning rate scheduler
        warmup_scheduler: Warmup learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
        save_dir: Directory to save checkpoints
        writer: TensorBoard writer
        grad_clip: Maximum norm of gradients
        early_stopping_patience: Number of epochs to wait before early stopping
        
    Returns:
        Dictionary containing training history
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'val_mse': [],
        'best_val_loss': float('inf')
    }
    
    # MSE for evaluation
    mse_criterion = nn.MSELoss()
    
    # Early stopping variables
    early_stopping_counter = 0
    early_stopping_min_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Check shapes and log if there's a mismatch
            if outputs.shape != targets.shape:
                print(f"Shape mismatch: outputs {outputs.shape}, targets {targets.shape}")
                # Ensure compatible shapes - either resize targets or crop outputs
                if outputs.shape[2] > targets.shape[2] or outputs.shape[3] > targets.shape[3]:
                    # Crop outputs to match targets
                    diff_h = (outputs.shape[2] - targets.shape[2]) // 2
                    diff_w = (outputs.shape[3] - targets.shape[3]) // 2
                    outputs = outputs[:, :, 
                                    diff_h:diff_h+targets.shape[2], 
                                    diff_w:diff_w+targets.shape[3]]
                elif outputs.shape[2] < targets.shape[2] or outputs.shape[3] < targets.shape[3]:
                    # Crop targets to match outputs
                    diff_h = (targets.shape[2] - outputs.shape[2]) // 2
                    diff_w = (targets.shape[3] - outputs.shape[3]) // 2
                    targets = targets[:, :, 
                                    diff_h:diff_h+outputs.shape[2], 
                                    diff_w:diff_w+outputs.shape[3]]
            
            loss = criterion(outputs, targets)
            mse = mse_criterion(outputs, targets)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += mse.item()
        
        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Check shapes and log if there's a mismatch
                if outputs.shape != targets.shape:
                    print(f"Shape mismatch in validation: outputs {outputs.shape}, targets {targets.shape}")
                    # Ensure compatible shapes - either resize targets or crop outputs
                    if outputs.shape[2] > targets.shape[2] or outputs.shape[3] > targets.shape[3]:
                        # Crop outputs to match targets
                        diff_h = (outputs.shape[2] - targets.shape[2]) // 2
                        diff_w = (outputs.shape[3] - targets.shape[3]) // 2
                        outputs = outputs[:, :, 
                                        diff_h:diff_h+targets.shape[2], 
                                        diff_w:diff_w+targets.shape[3]]
                    elif outputs.shape[2] < targets.shape[2] or outputs.shape[3] < targets.shape[3]:
                        # Crop targets to match outputs
                        diff_h = (targets.shape[2] - outputs.shape[2]) // 2
                        diff_w = (targets.shape[3] - outputs.shape[3]) // 2
                        targets = targets[:, :, 
                                        diff_h:diff_h+outputs.shape[2], 
                                        diff_w:diff_w+outputs.shape[3]]
                
                loss = criterion(outputs, targets)
                mse = mse_criterion(outputs, targets)
                
                val_loss += loss.item()
                val_mse += mse.item()
        
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        
        # Step the schedulers
        if epoch < 5:  # During warmup
            warmup_scheduler.step()
        else:  # After warmup
            scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MSE/train', train_mse, epoch)
        writer.add_scalar('MSE/val', val_mse, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.6f}, MSE: {train_mse:.6f}')
        print(f'Val Loss: {val_loss:.6f}, MSE: {val_mse:.6f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save checkpoint if validation loss improved
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mse': train_mse,
                'val_mse': val_mse
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            
            # Save a sample prediction as an image
            if epoch % 10 == 0:
                visualize_prediction(model, val_loader, device, save_dir, epoch)
                
            # Reset early stopping counter since we improved
            early_stopping_counter = 0
            early_stopping_min_val_loss = val_loss
        else:
            # Early stopping logic
            early_stopping_counter += 1
            print(f'EarlyStopping: {early_stopping_counter}/{early_stopping_patience}')
            
            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                # Save final checkpoint
                final_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_mse': train_mse,
                    'val_mse': val_mse,
                    'early_stopped': True
                }
                torch.save(final_checkpoint, save_dir / 'final_model.pth')
                break
    
    return history

def visualize_prediction(model: nn.Module, data_loader: DataLoader, device: str, save_dir: Path, epoch: int):
    """Generate a visualization of model predictions during training."""
    model.eval()
    
    # Get a batch
    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Choose a random sample from the batch
    idx = np.random.randint(0, inputs.shape[0])
    
    with torch.no_grad():
        prediction = model(inputs[idx:idx+1])
        
        # Check for shape mismatch
        if prediction.shape != targets[idx:idx+1].shape:
            print(f"Shape mismatch in visualization: prediction {prediction.shape}, target {targets[idx:idx+1].shape}")
            # Handle different cases
            if prediction.shape[2] > targets.shape[2] or prediction.shape[3] > targets.shape[3]:
                # Crop prediction to match target
                diff_h = (prediction.shape[2] - targets.shape[2]) // 2
                diff_w = (prediction.shape[3] - targets.shape[3]) // 2
                prediction = prediction[:, :, 
                                     diff_h:diff_h+targets.shape[2], 
                                     diff_w:diff_w+targets.shape[3]]
    
    # Convert to numpy for visualization
    input_img = inputs[idx, 0].cpu().numpy()  # First channel (coarse solution)
    target_img = targets[idx, 0].cpu().numpy()
    pred_img = prediction[0, 0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(input_img)
    axes[0].set_title('Input (Upsampled Coarse)')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(target_img)
    axes[1].set_title('Target (Fine)')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(pred_img)
    axes[2].set_title('Prediction')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_dir / f'prediction_epoch_{epoch}.png')
    plt.close()

def plot_losses(history: dict, save_dir: Path):
    """
    Plot training and validation losses.
    
    Args:
        history: Dictionary containing loss history
        save_dir: Directory to save the plot
    """
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History - Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'training_history_loss.png')
    plt.close()
    
    # Plot MSE
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_mse'], label='Training MSE')
    plt.plot(history['val_mse'], label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training History - MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'training_history_mse.png')
    plt.close()

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    
    # Configuration
    config = {
        'batch_size': 16,
        'num_epochs': 500,
        'learning_rate': 1e-4,
        'min_lr': 1e-7,
        'patience': 15,
        'val_split': 0.2,
        'grad_clip': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'pin_memory': True,
        'lambda_boundary': 2.0,  # Weight for boundary loss
        'context_padding': 2,    # Padding size for context (24->20, 48->40)
        'early_stopping_patience': 30  # Number of epochs to wait before early stopping
    }
    
    # Create directories
    base_dir = Path('results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = base_dir / f'run_improved_{timestamp}'
    save_dir.mkdir(parents=True)
    
    # Save configuration
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(save_dir / 'tensorboard'))
    
    # Load data
    print("Loading dataset...")
    data_file = np.load('data/pde_dataset_overlapping.npz')
    
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
        print(f"Batch numbers: {[batch_info[base_keys[0]]]}")
        
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
    
    # Create a random permutation of indices
    if 'u_fine' in data:
        n_samples = len(data['u_fine'])
    else:
        # If u_fine is not available, try to use another data key
        for key in data.keys():
            if key.startswith('u_fine'):
                n_samples = len(data[key])
                break
        else:
            # If no u_fine or u_fine_batch_* keys found
            sample_key = list(data.keys())[0]
            n_samples = len(data[sample_key])
            print(f"Using {sample_key} for sample count: {n_samples}")
    
    indices = np.random.permutation(n_samples)
    
    # Split indices into train and validation
    val_size = int(n_samples * config['val_split'])
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Create train and validation datasets with the split indices
    train_data = {}
    val_data = {}
    
    # Only add data arrays with the right shape (exclude 'k1', 'k2', 'total_samples', etc.)
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            # Check if the array has the correct first dimension
            if data[key].shape and data[key].shape[0] == n_samples:
                # These are the main data arrays (u_coarse, u_fine, etc.)
                train_data[key] = data[key][train_indices]
                val_data[key] = data[key][val_indices]
            else:
                # These might be metadata like k1, k2 - we'll skip them for datasets
                print(f"Skipping key {key} with shape {data[key].shape} for train/val split")
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # Debug: Print keys in train_data to verify
    print(f"Keys in train_data: {list(train_data.keys())}")
    
    # Create datasets
    train_dataset = OverlappingPDEDataset(
        train_data, 
        device='cpu',  # Initialize on CPU
        core_size=(20, 40),
        context_size=(24, 48)
    )
    
    val_dataset = OverlappingPDEDataset(
        val_data, 
        device='cpu',  # Initialize on CPU
        core_size=(20, 40),
        context_size=(24, 48)
    )
    
    # Create data loaders with shuffling for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    # Initialize model
    model = ContextAwareUNet(
        in_channels=2, 
        context_padding=config['context_padding']
    ).to(config['device'])
    
    model.apply(init_weights)
    
    # Loss function and optimizer with modified parameters
    criterion = BoundaryAwareLoss(lambda_boundary=config['lambda_boundary'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-6,
        betas=(0.9, 0.99)
    )
    
    # Learning rate scheduler with larger reduction factor
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,
        patience=config['patience'],
        min_lr=config['min_lr'],
        verbose=True
    )
    
    # Warmup scheduler (corrected lambda function)
    def warmup_lambda(epoch):
        if epoch < 5:
            return epoch / 5  # Linear warmup
        return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=warmup_lambda
    )
    
    # Train model
    print("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        num_epochs=config['num_epochs'],
        device=config['device'],
        save_dir=save_dir,
        writer=writer,
        grad_clip=config['grad_clip'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    # Plot and save training history
    plot_losses(history, save_dir)
    
    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main() 