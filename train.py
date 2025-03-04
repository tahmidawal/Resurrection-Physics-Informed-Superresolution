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

from models import UNet, PDEDataset, init_weights

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: str,
    save_dir: Path,
    writer: SummaryWriter,
    grad_clip: float = 1.0
) -> dict:
    """
    Train the model and save checkpoints.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
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
        'best_val_loss': float('inf')
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Step the scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss: {val_loss:.6f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save checkpoint if validation loss improved
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
    
    return history

def plot_losses(history: dict, save_dir: Path):
    """
    Plot training and validation losses.
    
    Args:
        history: Dictionary containing loss history
        save_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'training_history.png')
    plt.close()

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    
    # Configuration
    config = {
        'batch_size': 32,
        'num_epochs': 500,
        'learning_rate': 2e-4,
        'min_lr': 1e-6,
        'patience': 10,
        'val_split': 0.2,
        'grad_clip': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,  # Number of workers for data loading
        'pin_memory': True  # Faster data transfer to GPU
    }
    
    # Create directories
    base_dir = Path('results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = base_dir / f'run_{timestamp}'
    save_dir.mkdir(parents=True)
    
    # Save configuration
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(save_dir / 'tensorboard'))
    
    # Load data
    print("Loading dataset...")
    data = np.load('data/pde_dataset.npz')
    
    # Create a random permutation of indices
    n_samples = len(data['u_fine'])
    indices = np.random.permutation(n_samples)
    
    # Split indices into train and validation
    val_size = int(n_samples * config['val_split'])
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Create train and validation datasets with the split indices
    train_data = {key: data[key][train_indices] for key in data.files}
    val_data = {key: data[key][val_indices] for key in data.files}
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # Create datasets
    train_dataset = PDEDataset(train_data, device=config['device'])
    val_dataset = PDEDataset(val_data, device=config['device'])
    
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
        shuffle=False,  # No need to shuffle validation data
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    # Initialize model
    model = UNet().to(config['device'])
    model.apply(init_weights)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config['patience'],
        min_lr=config['min_lr'],
        verbose=True
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
        num_epochs=config['num_epochs'],
        device=config['device'],
        save_dir=save_dir,
        writer=writer,
        grad_clip=config['grad_clip']
    )
    
    # Plot and save training history
    plot_losses(history, save_dir)
    
    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main() 