import torch
import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import interp2d

def upsample_solution(
    solution: np.ndarray,
    target_size: Tuple[int, int],
    method: str = 'bilinear'
) -> np.ndarray:
    """
    Upsample a solution using interpolation.
    
    Args:
        solution: Input solution array
        target_size: Target size (height, width)
        method: Interpolation method ('bilinear' or 'bicubic')
        
    Returns:
        Upsampled solution
    """
    h, w = solution.shape
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    
    # Create interpolation function
    f = interp2d(x, y, solution, kind=method[2:])
    
    # Create target grid
    x_new = np.linspace(0, 1, target_size[1])
    y_new = np.linspace(0, 1, target_size[0])
    
    # Interpolate
    return f(x_new, y_new)

def compute_relative_error(
    true_solution: np.ndarray,
    predicted_solution: np.ndarray,
    eps: float = 1e-8
) -> float:
    """
    Compute relative L2 error between true and predicted solutions.
    
    Args:
        true_solution: Ground truth solution
        predicted_solution: Predicted solution
        eps: Small constant to avoid division by zero
        
    Returns:
        Relative L2 error
    """
    error = np.linalg.norm(true_solution - predicted_solution)
    norm = np.linalg.norm(true_solution) + eps
    return error / norm

def create_grid(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D grid of n×n points in [0,1]×[0,1].
    
    Args:
        n: Number of points in each dimension
        
    Returns:
        Tuple of meshgrid arrays (X, Y)
    """
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    return np.meshgrid(x, y)

def save_tensor(
    tensor: torch.Tensor,
    path: str,
    normalize: bool = False
) -> None:
    """
    Save a tensor as a numpy array.
    
    Args:
        tensor: Input tensor
        path: Save path
        normalize: Whether to normalize the tensor to [0,1]
    """
    array = tensor.detach().cpu().numpy()
    if normalize:
        array = (array - array.min()) / (array.max() - array.min() + 1e-8)
    np.save(path, array)

def load_tensor(
    path: str,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Load a tensor from a numpy array.
    
    Args:
        path: Load path
        device: Device to load tensor to
        
    Returns:
        Loaded tensor
    """
    array = np.load(path)
    tensor = torch.from_numpy(array)
    if device is not None:
        tensor = tensor.to(device)
    return tensor 