import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import os
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import random

class PoissonSolver:
    def __init__(self, n: int):
        """
        Initialize the Poisson equation solver.
        
        Args:
            n: Number of grid points in each dimension
        """
        self.n = n
        
        # Initialize grid
        self.x = np.linspace(0, 1, n)
        self.y = np.linspace(0, 1, n)
        
        # Create meshgrid
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize Laplacian matrix
        self.L = self._create_laplacian(n)

    def _create_laplacian(self, n: int) -> np.ndarray:
        """
        Create the 2D Laplacian matrix using sparse matrices.
        
        Args:
            n: Number of grid points in each dimension
            
        Returns:
            Sparse matrix representing the 2D Laplacian operator
        """
        h = 1.0 / (n - 1)
        n2 = n * n
        
        # Create 1D Laplacian
        main_diag = -4 * np.ones(n2)
        off_diag = np.ones(n2-1)
        off_diag[np.arange(n-1, n2-1, n)] = 0  # Remove connections across boundary
        
        # Construct sparse matrix
        diagonals = [main_diag, off_diag, off_diag, np.ones(n*(n-1)), np.ones(n*(n-1))]
        offsets = [0, 1, -1, n, -n]
        L = diags(diagonals, offsets, shape=(n2, n2))
        
        return L / (h * h)

    def generate_forcing_term(self, k1: float, k2: float) -> np.ndarray:
        """
        Generate the forcing term f(x,y) = sin(k₁ * 2πx) * sin(k₂ * 2πy).
        
        Args:
            k1: First wave number
            k2: Second wave number
            
        Returns:
            2D array containing the forcing term values
        """
        return np.sin(2 * np.pi * k1 * self.X) * np.sin(2 * np.pi * k2 * self.Y)

    def solve_poisson(self, f: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Solve the Poisson equation -∇·(θ∇u) = f with zero Dirichlet boundary conditions.
        
        Args:
            f: Forcing term
            theta: Diffusion coefficient field
            
        Returns:
            Solution u
        """
        # Reshape inputs to 1D arrays
        f_flat = f.reshape(-1)
        theta_flat = theta.reshape(-1)
        
        # Modify Laplacian with theta
        L_theta = diags(theta_flat) @ self.L
        
        # Solve the system
        u_flat = spsolve(L_theta, f_flat)
        
        return u_flat.reshape((self.n, self.n))

def extract_subdomains(field: np.ndarray, grid_size: int, num_subdomains: int) -> List[np.ndarray]:
    """
    Extract subdomains from a field.
    
    Args:
        field: The field to extract subdomains from
        grid_size: The size of each subdomain (assuming square)
        num_subdomains: Number of subdomains per dimension (total = num_subdomains^2)
    
    Returns:
        List of subdomain arrays
    """
    total_size = field.shape[0]
    step_size = total_size // num_subdomains
    
    subdomains = []
    for i in range(num_subdomains):
        for j in range(num_subdomains):
            i_start = i * step_size
            j_start = j * step_size
            subdomain = field[i_start:i_start+grid_size, j_start:j_start+grid_size]
            subdomains.append(subdomain)
    
    return subdomains

def generate_dataset_part1(n_samples: int, k_values: List[float] = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]) -> Dict:
    """
    Generate dataset with the first approach:
    1. Generate u and f in 40x40 (coarse) and 80x80 (fine)
    2. Take 4 subdomains of each grid (20x20 to 40x40)
    
    Args:
        n_samples: Number of full grid samples to generate
        k_range: Range for random wave numbers
    
    Returns:
        Dictionary containing the dataset
    """
    # Initialize solvers
    solver_coarse = PoissonSolver(40)  # 40x40 grid
    solver_fine = PoissonSolver(80)    # 80x80 grid
    
    # Initialize dataset
    dataset = {
        'u_coarse': [],  # Will be 20x20
        'u_fine': [],    # Will be 40x40
        'f_coarse': [],  # Will be 20x20
        'f_fine': [],    # Will be 40x40
        'theta_coarse': [],  # Will be 20x20
        'theta_fine': [],    # Will be 40x40
        'k1': [],
        'k2': []
    }
    
    # Generate samples
    for _ in range(n_samples):
        # Select k values from the predefined list
        k_idx = _ % len(k_values)  # Cycle through k values
        k1 = k_values[k_idx]
        k2 = k_values[k_idx]  # Using same k value for both dimensions
        
        # Set constant theta
        theta_coarse = np.ones((40, 40))
        theta_fine = np.ones((80, 80))
        
        # Generate forcing terms
        f_coarse = solver_coarse.generate_forcing_term(k1, k2)
        f_fine = solver_fine.generate_forcing_term(k1, k2)
        
        # Solve PDE
        u_coarse = solver_coarse.solve_poisson(f_coarse, theta_coarse)
        u_fine = solver_fine.solve_poisson(f_fine, theta_fine)
        
        # Extract 4 subdomains (2x2 grid of subdomains)
        u_coarse_subdomains = extract_subdomains(u_coarse, 20, 2)  # 4 subdomains of 20x20
        u_fine_subdomains = extract_subdomains(u_fine, 40, 2)      # 4 subdomains of 40x40
        f_coarse_subdomains = extract_subdomains(f_coarse, 20, 2)  # 4 subdomains of 20x20
        f_fine_subdomains = extract_subdomains(f_fine, 40, 2)      # 4 subdomains of 40x40
        theta_coarse_subdomains = extract_subdomains(theta_coarse, 20, 2)  # 4 subdomains of 20x20
        theta_fine_subdomains = extract_subdomains(theta_fine, 40, 2)     # 4 subdomains of 40x40
        
        # Store each subdomain as a separate sample
        for i in range(4):
            dataset['u_coarse'].append(u_coarse_subdomains[i])
            dataset['u_fine'].append(u_fine_subdomains[i])
            dataset['f_coarse'].append(f_coarse_subdomains[i])
            dataset['f_fine'].append(f_fine_subdomains[i])
            dataset['theta_coarse'].append(theta_coarse_subdomains[i])
            dataset['theta_fine'].append(theta_fine_subdomains[i])
            dataset['k1'].append(k1)
            dataset['k2'].append(k2)
    
    # Convert lists to arrays
    for key in dataset:
        dataset[key] = np.array(dataset[key])
        
    return dataset

def generate_dataset_part2(n_samples: int, k_values: List[float] = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]) -> Dict:
    """
    Generate dataset with the second approach:
    1. Generate u and f in 80x80 and 160x160
    2. Take 16 subdomains of each grid (20x20 to 40x40)
    
    Args:
        n_samples: Number of full grid samples to generate
        k_range: Range for random wave numbers
    
    Returns:
        Dictionary containing the dataset
    """
    # Initialize solvers
    solver_coarse = PoissonSolver(80)   # 80x80 grid
    solver_fine = PoissonSolver(160)    # 160x160 grid
    
    # Initialize dataset
    dataset = {
        'u_coarse': [],  # Will be 20x20
        'u_fine': [],    # Will be 40x40
        'f_coarse': [],  # Will be 20x20
        'f_fine': [],    # Will be 40x40
        'theta_coarse': [],  # Will be 20x20
        'theta_fine': [],    # Will be 40x40
        'k1': [],
        'k2': []
    }
    
    # Generate samples
    for _ in range(n_samples):
        # Select k values from the predefined list
        k_idx = _ % len(k_values)  # Cycle through k values
        k1 = k_values[k_idx]
        k2 = k_values[k_idx]  # Using same k value for both dimensions
        
        # Set constant theta
        theta_coarse = np.ones((80, 80))
        theta_fine = np.ones((160, 160))
        
        # Generate forcing terms
        f_coarse = solver_coarse.generate_forcing_term(k1, k2)
        f_fine = solver_fine.generate_forcing_term(k1, k2)
        
        # Solve PDE
        u_coarse = solver_coarse.solve_poisson(f_coarse, theta_coarse)
        u_fine = solver_fine.solve_poisson(f_fine, theta_fine)
        
        # Extract 16 subdomains (4x4 grid of subdomains)
        u_coarse_subdomains = extract_subdomains(u_coarse, 20, 4)  # 16 subdomains of 20x20
        u_fine_subdomains = extract_subdomains(u_fine, 40, 4)      # 16 subdomains of 40x40
        f_coarse_subdomains = extract_subdomains(f_coarse, 20, 4)  # 16 subdomains of 20x20
        f_fine_subdomains = extract_subdomains(f_fine, 40, 4)      # 16 subdomains of 40x40
        theta_coarse_subdomains = extract_subdomains(theta_coarse, 20, 4)  # 16 subdomains of 20x20
        theta_fine_subdomains = extract_subdomains(theta_fine, 40, 4)     # 16 subdomains of 40x40
        
        # Store each subdomain as a separate sample
        for i in range(16):
            dataset['u_coarse'].append(u_coarse_subdomains[i])
            dataset['u_fine'].append(u_fine_subdomains[i])
            dataset['f_coarse'].append(f_coarse_subdomains[i])
            dataset['f_fine'].append(f_fine_subdomains[i])
            dataset['theta_coarse'].append(theta_coarse_subdomains[i])
            dataset['theta_fine'].append(theta_fine_subdomains[i])
            dataset['k1'].append(k1)
            dataset['k2'].append(k2)
    
    # Convert lists to arrays
    for key in dataset:
        dataset[key] = np.array(dataset[key])
        
    return dataset

def combine_datasets(dataset1: Dict, dataset2: Dict) -> Dict:
    """
    Combine two datasets.
    
    Args:
        dataset1: First dataset
        dataset2: Second dataset
    
    Returns:
        Combined dataset
    """
    combined_dataset = {}
    for key in dataset1.keys():
        combined_dataset[key] = np.concatenate((dataset1[key], dataset2[key]), axis=0)
    
    return combined_dataset

def save_dataset(dataset: Dict, path: str = 'data'):
    """
    Save the generated dataset.
    
    Args:
        dataset: Dictionary containing the dataset
        path: Path to save the dataset
    """
    save_path = Path(path)
    if not save_path.exists():
        save_path.mkdir(parents=True)
        
    np.savez(
        save_path / 'pde_dataset_subdomains.npz',
        **dataset
    )

def plot_samples(dataset: Dict, n_samples: int = 5, save_dir: str = 'dataset_samples'):
    """
    Plot a few samples from the dataset.
    
    Args:
        dataset: Dictionary containing the dataset
        n_samples: Number of samples to plot
        save_dir: Directory to save the plots
    """
    # Create directory if it doesn't exist
    save_path = Path(save_dir)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    
    # Get random sample indices
    sample_indices = random.sample(range(len(dataset['u_fine'])), n_samples)
    
    for i, idx in enumerate(sample_indices):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(231)
        plt.imshow(dataset['u_coarse'][idx])
        plt.colorbar()
        plt.title(f'Coarse Solution (Sample {idx})')
        
        plt.subplot(232)
        plt.imshow(dataset['u_fine'][idx])
        plt.colorbar()
        plt.title(f'Fine Solution (Sample {idx})')
        
        plt.subplot(233)
        plt.imshow(dataset['f_coarse'][idx])
        plt.colorbar()
        plt.title(f'Forcing Term Coarse (Sample {idx})')
        
        plt.subplot(234)
        plt.imshow(dataset['f_fine'][idx])
        plt.colorbar()
        plt.title(f'Forcing Term Fine (Sample {idx})')
        
        plt.subplot(235)
        plt.imshow(dataset['theta_coarse'][idx])
        plt.colorbar()
        plt.title(f'Diffusion Coeff Coarse (Sample {idx})')
        
        plt.subplot(236)
        plt.imshow(dataset['theta_fine'][idx])
        plt.colorbar()
        plt.title(f'Diffusion Coeff Fine (Sample {idx})')
        
        plt.tight_layout()
        plt.savefig(save_path / f'subdomain_sample_{idx}.png')
        plt.close()

if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Part 1: Generate datasets from 40x40 and 80x80 grids
    n_samples_part1 = 625  # 625 × 4 subdomains = 2500 samples
    k_values = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    
    print(f"Part 1: Generating {n_samples_part1} samples (40x40 → 80x80) with k_values {k_values}...")
    print(f"This will produce {n_samples_part1 * 4} subdomain samples (20x20 → 40x40)")
    dataset_part1 = generate_dataset_part1(n_samples_part1, k_values)
    
    # Part 2: Generate datasets from 80x80 and 160x160 grids
    n_samples_part2 = 219  # 219 × 16 subdomains ≈ 3500 samples (actually 3504)
    
    print(f"Part 2: Generating {n_samples_part2} samples (80x80 → 160x160) with k_values {k_values}...")
    print(f"This will produce {n_samples_part2 * 16} subdomain samples (20x20 → 40x40)")
    dataset_part2 = generate_dataset_part2(n_samples_part2, k_values)
    
    # Combine datasets
    combined_dataset = combine_datasets(dataset_part1, dataset_part2)
    print(f"Combined dataset contains {len(combined_dataset['u_fine'])} samples")
    
    # Verify shapes
    print(f"Coarse solution shape: {combined_dataset['u_coarse'][0].shape}")
    print(f"Fine solution shape: {combined_dataset['u_fine'][0].shape}")
    
    # Save dataset
    save_dataset(combined_dataset)
    print("Dataset saved successfully!")
    
    # Plot a few samples
    plot_samples(combined_dataset, n_samples=10)
    print("Sample plots saved successfully!") 