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
import argparse

from src.models_improved import ContextAwareUNet, OverlappingPDEDataset, init_weights, BoundaryAttentionModule, BoundaryAwareLoss

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
    grad_clip: float = 0.1
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
    
    # MSE loss for monitoring
    mse_loss = nn.MSELoss()
    
    # For visualization, store predictions periodically
    if num_epochs >= 100:
        vis_epochs = [0, 10, 20, 60, 100]
    else:
        vis_epochs = [0, num_epochs // 5, num_epochs // 2, num_epochs - 1]
    
    # Get a fixed validation sample for visualization
    val_inputs, val_targets = next(iter(val_loader))
    vis_input = val_inputs[0:1].to(device)
    vis_target = val_targets[0:1].to(device)
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_mses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Also calculate MSE for monitoring
            mse = mse_loss(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            optimizer.step()
            
            train_losses.append(loss.item())
            train_mses.append(mse.item())
            
            pbar.set_postfix({"loss": f"{loss.item():.6f}", "mse": f"{mse.item():.6f}"})
        
        # Apply warmup scheduler in early epochs
        if epoch < 5:
            warmup_scheduler.step()
        
        # Validation phase
        model.eval()
        val_losses = []
        val_mses = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                mse = mse_loss(outputs, targets)
                
                val_losses.append(loss.item())
                val_mses.append(mse.item())
        
        # Calculate average metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_train_mse = np.mean(train_mses)
        avg_val_mse = np.mean(val_mses)
        
        # Apply main scheduler based on validation loss
        scheduler.step(avg_val_loss)
        
        # Log metrics
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mse'].append(avg_train_mse)
        history['val_mse'].append(avg_val_mse)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('MSE/train', avg_train_mse, epoch)
        writer.add_scalar('MSE/val', avg_val_mse, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Train MSE: {avg_train_mse:.6f}, Val MSE: {avg_val_mse:.6f}")
        
        # Visualization of prediction
        if epoch in vis_epochs:
            with torch.no_grad():
                prediction = model(vis_input)
                
                # Denormalize
                dataset = train_loader.dataset
                input_denorm = dataset.denormalize(vis_input[:, 0:1])
                target_denorm = dataset.denormalize(vis_target)
                pred_denorm = dataset.denormalize(prediction)
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(input_denorm[0, 0].cpu().numpy())
                axes[0].set_title("Input (Upsampled Coarse)")
                
                axes[1].imshow(pred_denorm[0, 0].cpu().numpy())
                axes[1].set_title("Prediction")
                
                axes[2].imshow(target_denorm[0, 0].cpu().numpy())
                axes[2].set_title("Target (Fine)")
                
                plt.tight_layout()
                plt.savefig(save_dir / f"prediction_epoch_{epoch}.png")
                plt.close()
        
        # Save best model
        if avg_val_loss < history['best_val_loss']:
            history['best_val_loss'] = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, save_dir / 'best_model.pth')
            print(f"  Saved best model with val_loss: {avg_val_loss:.6f}")
    
    # Save final model
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': history['val_loss'][-1],
    }, save_dir / 'final_model.pth')
    
    return history

def plot_losses(history: dict, save_dir: Path):
    """Plot training and validation loss."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history_loss.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_mse'], label='Train MSE')
    plt.plot(history['val_mse'], label='Validation MSE')
    plt.title('Training History - MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history_mse.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train model for 24x24 to 48x48 upscaling')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='data/pde_dataset_overlapping.npz', help='Path to dataset')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'model_name': 'ContextAwareUNet',
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'valid_split': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4 if torch.cuda.is_available() else 0,
        'pin_memory': torch.cuda.is_available(),
        'patience': 10,
        'min_lr': 1e-7,
        'grad_clip': 0.5,
        'lambda_boundary': 2.0
    }
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f'results/run_24x24_48x48_{timestamp}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # TensorBoard writer
    writer = SummaryWriter(save_dir / 'tensorboard')
    
    # Load dataset
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run data_generation_overlapping.py first.")
    
    print(f"Loading dataset from {data_path}")
    data = np.load(data_path)
    
    # Verify shapes for 24x24 -> 48x48 upscaling
    sample_coarse = data['u_coarse'][0]
    sample_fine = data['u_fine'][0]
    print(f"Coarse shape: {sample_coarse.shape}, Fine shape: {sample_fine.shape}")
    
    # Create dataset
    full_dataset = OverlappingPDEDataset(data, device=config['device'])
    
    # Create train/val split
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * config['valid_split'])
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Total samples: {dataset_size}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
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
    
    # Initialize model for 24x24 -> 48x48 upscaling
    model = ContextAwareUNet(in_channels=2, context_padding=2).to(config['device'])
    model.apply(init_weights)
    
    # Custom loss function with boundary awareness
    criterion = BoundaryAwareLoss(lambda_boundary=config['lambda_boundary']).to(config['device'])
    
    # Optimizer with weight decay
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
    
    # Warmup scheduler
    def warmup_lambda(epoch):
        if epoch < 5:
            return epoch / 5  # Linear warmup
        return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=warmup_lambda
    )
    
    # Train model
    print(f"Starting training on {config['device']}...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        num_epochs=config['epochs'],
        device=config['device'],
        save_dir=save_dir,
        writer=writer,
        grad_clip=config['grad_clip']
    )
    
    # Plot and save training history
    plot_losses(history, save_dir)
    
    print(f"Training completed. Results saved in {save_dir}")

if __name__ == '__main__':
    main() 