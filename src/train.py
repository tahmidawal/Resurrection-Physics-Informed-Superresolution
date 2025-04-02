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

class BoundaryAwareLoss(nn.Module):
    def __init__(self, boundary_weight=2.0, subdomain_size=40, boundary_width=3):
        """Loss function that gives higher weight to errors at subdomain boundaries.
        
        Args:
            boundary_weight: Weight multiplier for boundary regions
            subdomain_size: Size of subdomains used during inference
            boundary_width: Width of boundary region on each side of subdomain
        """
        super().__init__()
        self.boundary_weight = boundary_weight
        self.subdomain_size = subdomain_size
        self.boundary_width = boundary_width
        self.base_criterion = nn.MSELoss(reduction='none')
        
    def forward(self, pred, target):
        # Calculate base loss (pixel-wise MSE)
        base_loss = self.base_criterion(pred, target)
        
        # Create boundary weight mask - higher weights near boundaries
        mask = torch.ones_like(pred)
        
        # Increase weight for pixels near subdomain boundaries
        for i in range(0, pred.shape[2], self.subdomain_size):
            if i > 0:  # Skip first boundary
                # Left boundary of subdomain
                start_idx = max(0, i - self.boundary_width)
                end_idx = min(pred.shape[2], i + self.boundary_width)
                mask[:, :, start_idx:end_idx, :] *= self.boundary_weight
                
        for j in range(0, pred.shape[3], self.subdomain_size):
            if j > 0:  # Skip first boundary
                # Top boundary of subdomain
                start_idx = max(0, j - self.boundary_width)
                end_idx = min(pred.shape[3], j + self.boundary_width)
                mask[:, :, :, start_idx:end_idx] *= self.boundary_weight
        
        # Apply mask and reduce
        weighted_loss = (base_loss * mask).mean()
        return weighted_loss

class MultiScaleConsistencyLoss(nn.Module):
    def __init__(self, alpha=0.2):
        """Multi-scale consistency loss.
        
        Args:
            alpha: Weight for consistency loss component
        """
        super().__init__()
        self.alpha = alpha
        self.base_criterion = nn.MSELoss()
        
    def forward(self, pred, target, coarse_input):
        # Base reconstruction loss
        recon_loss = self.base_criterion(pred, target)
        
        # Downsample the prediction to coarse resolution
        downsampled = F.avg_pool2d(pred, kernel_size=2, stride=2)
        
        # Consistency loss between downsampled prediction and coarse input
        consistency_loss = self.base_criterion(downsampled, coarse_input[:, 0:1])
        
        # Combined loss
        return recon_loss + self.alpha * consistency_loss

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
            
            # Uncomment for multi-scale consistency loss (optional enhancement)
            # if isinstance(criterion, MultiScaleConsistencyLoss):
            #     loss = criterion(outputs, targets, inputs)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            # Free up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
        
        # Step the schedulers
        if epoch < 5:  # During warmup
            warmup_scheduler.step()
        else:  # After warmup
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
                'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
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
    
    # Check and print GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    if device == 'cuda':
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.manual_seed(42)
    else:
        print("WARNING: No GPU detected, training will be slow!")
    
    # Configuration
    config = {
        'batch_size': 4,  # Reduced batch size for GPU memory efficiency
        'num_epochs': 500,
        'learning_rate': 5e-5,
        'min_lr': 1e-7,
        'patience': 15,
        'val_split': 0.2,
        'grad_clip': 0.1,
        'device': device,  # Use the already validated device
        'num_workers': 1,  # Single worker to avoid memory issues
        'pin_memory': True if device == 'cuda' else False  # Only pin memory for GPU
    }
    
    # Enable cuDNN benchmarking for faster training on GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
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
    try:
        data = np.load('data/pde_dataset_subdomains.npz')
        print(f"Dataset loaded successfully, size: {sum(data[key].nbytes for key in data.files) / 1e6:.2f} MB")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
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
    
    # Create datasets (always initialize on CPU to avoid OOM during loading)
    train_dataset = PDEDataset(train_data, device='cpu')  
    val_dataset = PDEDataset(val_data, device='cpu')
    print("Datasets created successfully")
    
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
    print(f"\nInitializing UNet model on {config['device']}...")
    model = UNet(in_channels=2).to(config['device'])  # Updated to 2 input channels
    model.apply(init_weights)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {trainable_params:,} trainable parameters (total: {total_params:,})")
    
    # Enhanced loss function for better boundary handling
    criterion = BoundaryAwareLoss(boundary_weight=2.0, subdomain_size=40, boundary_width=3)
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
        grad_clip=config['grad_clip']
    )
    
    # Plot and save training history
    plot_losses(history, save_dir)
    
    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main() 