#!/usr/bin/env python
# Memory-optimized version of train.py for GPU training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json
import time
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import gc

# Import the model and custom loss
from models import UNet, init_weights
from loss import BoundaryAwareLoss

class PDEDataset(TensorDataset):
    """Dataset for PDE data"""
    def __init__(self, data_dict, device='cpu'):
        # Extract input fields (coarse grid and k value) and target (fine grid)
        u_coarse = torch.tensor(data_dict['u_coarse'], dtype=torch.float32).to(device)
        k_values = torch.tensor(data_dict['k_values'], dtype=torch.float32).to(device)
        
        # Stack coarse grid and k values as input channels
        k_channel = k_values.unsqueeze(1).unsqueeze(2).expand(-1, u_coarse.shape[1], u_coarse.shape[2])
        inputs = torch.stack([u_coarse, k_channel], dim=1)  # [B, 2, H, W]
        
        targets = torch.tensor(data_dict['u_fine'], dtype=torch.float32).unsqueeze(1).to(device)
        
        super().__init__(inputs, targets)

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, scheduler, warmup_scheduler, writer, 
                save_dir, device, grad_clip=None):
    """Train the model with memory optimizations for GPU"""
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        # Use tqdm for progress bar
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for inputs, targets in train_bar:
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)  # Memory optimization
            
            # Move to GPU
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            optimizer.step()
            
            # Add batch loss
            train_loss += loss.item()
            
            # Update progress bar
            train_bar.set_postfix(loss=loss.item())
            
            # Clear GPU memory
            del inputs, targets, outputs, loss
            torch.cuda.empty_cache()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move to GPU
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Clear GPU memory
                del inputs, targets, outputs, loss
                torch.cuda.empty_cache()
        
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
        history['lr'].append(current_lr)
        
        # Print update
        print(f'Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.6f} - Validation Loss: {val_loss:.6f} - LR: {current_lr:.2e}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'learning_rate': current_lr,
            }, save_dir / 'best_model.pth')
            print(f'Saved new best model with validation loss: {val_loss:.6f}')
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'learning_rate': current_lr,
                'history': history,
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
            
            # Save current history plot
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.semilogy(history['train_loss'], label='Train Loss')
            plt.semilogy(history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.semilogy(history['lr'], label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(save_dir / 'training_history.png')
            plt.close()
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
    
    # Final save
    torch.save({
        'epoch': num_epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'learning_rate': current_lr,
        'history': history,
    }, save_dir / 'final_model.pth')
    
    # Create plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.semilogy(history['train_loss'], label='Train Loss')
    plt.semilogy(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.semilogy(history['lr'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    
    plt.tight_layout()
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
        torch.backends.cudnn.benchmark = True  # Speed up training
    else:
        print("WARNING: No GPU detected, training will be slow!")
    
    # Configuration
    config = {
        'batch_size': 4,  # Smaller batch size to reduce memory usage
        'num_epochs': 500,
        'learning_rate': 5e-5,
        'min_lr': 1e-7,
        'patience': 15,
        'val_split': 0.2,
        'grad_clip': 0.1,
        'device': device,
        'num_workers': 2,  # Reasonable number of workers
        'pin_memory': True if device == 'cuda' else False
    }
    
    # Create directories
    base_dir = Path('results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = base_dir / f'run_{timestamp}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(save_dir / 'config.json', 'w') as f:
        # Convert non-serializable items to strings
        serializable_config = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v 
                              for k, v in config.items()}
        json.dump(serializable_config, f, indent=4)
    
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
    print(f"Total samples: {n_samples}")
    
    indices = np.random.permutation(n_samples)
    val_size = int(n_samples * config['val_split'])
    
    # Split into train and validation
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Extract data for train and validation
    train_data = {k: v[train_indices] for k, v in data.items()}
    val_data = {k: v[val_indices] for k, v in data.items()}
    
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
        pin_memory=config['pin_memory'],
        drop_last=True,  # Drop last batch to avoid OOM issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=True,  # Drop last batch to avoid OOM issues
    )
    
    # Initialize model
    print(f"\nInitializing UNet model on {config['device']}...")
    model = UNet(in_channels=2).to(config['device'])
    model.apply(init_weights)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {trainable_params:,} trainable parameters (total: {total_params:,})")
    
    # Enhanced loss function for better boundary handling
    criterion = BoundaryAwareLoss(boundary_weight=2.0, subdomain_size=40, boundary_width=3).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-5
    )
    
    # Learning rate schedulers
    # Warmup scheduler for first 5 epochs
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=5
    )
    
    # Reducer scheduler after warmup
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config['patience'],
        min_lr=config['min_lr'],
        verbose=True
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=save_dir / 'tensorboard')
    
    # Train model
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['num_epochs'],
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        writer=writer,
        save_dir=save_dir,
        device=config['device'],
        grad_clip=config['grad_clip']
    )
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"Training completed. Model saved in {save_dir}")

if __name__ == "__main__":
    main()
