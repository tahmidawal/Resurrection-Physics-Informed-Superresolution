import torch
import numpy as np
from pathlib import Path

print("Testing imports...")

# Test data loading
try:
    data_path = Path('data/pde_dataset_overlapping.npz')
    print(f"Data file exists: {data_path.exists()}")
    if data_path.exists():
        data = np.load(data_path)
        print(f"Data keys: {data.files}")
        print(f"u_coarse shape: {data['u_coarse'].shape}")
        print(f"u_fine shape: {data['u_fine'].shape}")
except Exception as e:
    print(f"Error loading data: {e}")

# Test model imports
try:
    from src.models_improved import ContextAwareUNet, OverlappingPDEDataset
    print("Successfully imported model classes")
    
    # Try to create model
    model = ContextAwareUNet(in_channels=2, context_padding=2)
    print(f"Created model: {type(model)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
except Exception as e:
    print(f"Error importing or creating model: {e}")

print("Test complete") 