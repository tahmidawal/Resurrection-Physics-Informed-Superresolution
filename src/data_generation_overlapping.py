import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import os
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import random
import argparse
import gc  # Garbage collector

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

def extract_overlapping_subdomains(field: np.ndarray, subdomain_size: int, context_size: int, num_subdomains: int) -> List[np.ndarray]:
    """
    Extract overlapping subdomains from a field.
    
    Args:
        field: The field to extract subdomains from
        subdomain_size: The size of each subdomain core (assuming square)
        context_size: The size of the context window (will extract context_size x context_size)
        num_subdomains: Number of subdomains per dimension (total = num_subdomains^2)
    
    Returns:
        List of subdomain arrays with context
    """
    total_size = field.shape[0]
    
    # Calculate step size for subdomain cores
    core_step_size = total_size // num_subdomains
    
    # Padding needed for context
    padding = (context_size - subdomain_size) // 2
    
    subdomains = []
    for i in range(num_subdomains):
        for j in range(num_subdomains):
            # Calculate the center of the subdomain
            i_center = i * core_step_size + core_step_size // 2
            j_center = j * core_step_size + core_step_size // 2
            
            # Calculate start and end indices with padding
            i_start = max(0, i_center - context_size // 2)
            j_start = max(0, j_center - context_size // 2)
            
            # Adjust if we're too close to the edge
            i_start = min(i_start, total_size - context_size)
            j_start = min(j_start, total_size - context_size)
            
            # Extract subdomain with context
            i_end = i_start + context_size
            j_end = j_start + context_size
            
            subdomain = field[i_start:i_end, j_start:j_end]
            subdomains.append(subdomain)
    
    return subdomains

def generate_dataset_part1(n_samples: int, k_range: Tuple[float, float] = (8.0, 10.0)) -> Dict:
    """
    Generate dataset with overlapping subdomains, first approach:
    1. Generate u and f in 48x48 (coarse) and 96x96 (fine)
    2. Take 4 subdomains of each grid (24x24 to 48x48)
    
    Args:
        n_samples: Number of full grid samples to generate
        k_range: Range for random wave numbers
    
    Returns:
        Dictionary containing the dataset
    """
    # Initialize solvers
    solver_coarse = PoissonSolver(48)  # 48x48 grid
    solver_fine = PoissonSolver(96)    # 96x96 grid
    
    # Initialize dataset
    dataset = {
        'u_coarse': [],  # Will be 24x24
        'u_fine': [],    # Will be 48x48
        'f_coarse': [],  # Will be 24x24
        'f_fine': [],    # Will be 48x48
        'theta_coarse': [],  # Will be 24x24
        'theta_fine': [],    # Will be 48x48
        'k1': [],
        'k2': []
    }
    
    # Generate samples
    for _ in range(n_samples):
        # Generate random wave numbers
        k1 = np.random.uniform(*k_range)
        k2 = np.random.uniform(*k_range)
        
        # Set constant theta
        theta_coarse = np.ones((48, 48))
        theta_fine = np.ones((96, 96))
        
        # Generate forcing terms
        f_coarse = solver_coarse.generate_forcing_term(k1, k2)
        f_fine = solver_fine.generate_forcing_term(k1, k2)
        
        # Solve PDE
        u_coarse = solver_coarse.solve_poisson(f_coarse, theta_coarse)
        u_fine = solver_fine.solve_poisson(f_fine, theta_fine)
        
        # Extract 4 overlapping subdomains (2x2 grid of subdomains)
        u_coarse_subdomains = extract_overlapping_subdomains(u_coarse, 20, 24, 2)  # 4 subdomains of 24x24
        u_fine_subdomains = extract_overlapping_subdomains(u_fine, 40, 48, 2)      # 4 subdomains of 48x48
        f_coarse_subdomains = extract_overlapping_subdomains(f_coarse, 20, 24, 2)  # 4 subdomains of 24x24
        f_fine_subdomains = extract_overlapping_subdomains(f_fine, 40, 48, 2)      # 4 subdomains of 48x48
        theta_coarse_subdomains = extract_overlapping_subdomains(theta_coarse, 20, 24, 2)  # 4 subdomains of 24x24
        theta_fine_subdomains = extract_overlapping_subdomains(theta_fine, 40, 48, 2)     # 4 subdomains of 48x48
        
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

def generate_dataset_part2(n_samples: int, k_range: Tuple[float, float] = (8.0, 10.0)) -> Dict:
    """
    Generate dataset with overlapping subdomains, second approach:
    1. Generate u and f in 96x96 and 192x192
    2. Take 16 subdomains of each grid (24x24 to 48x48)
    
    Args:
        n_samples: Number of full grid samples to generate
        k_range: Range for random wave numbers
    
    Returns:
        Dictionary containing the dataset
    """
    # Initialize solvers
    solver_coarse = PoissonSolver(96)   # 96x96 grid
    solver_fine = PoissonSolver(192)    # 192x192 grid
    
    # Initialize dataset
    dataset = {
        'u_coarse': [],  # Will be 24x24
        'u_fine': [],    # Will be 48x48
        'f_coarse': [],  # Will be 24x24
        'f_fine': [],    # Will be 48x48
        'theta_coarse': [],  # Will be 24x24
        'theta_fine': [],    # Will be 48x48
        'k1': [],
        'k2': []
    }
    
    # Generate samples
    for _ in range(n_samples):
        # Generate random wave numbers
        k1 = np.random.uniform(*k_range)
        k2 = np.random.uniform(*k_range)
        
        # Set constant theta
        theta_coarse = np.ones((96, 96))
        theta_fine = np.ones((192, 192))
        
        # Generate forcing terms
        f_coarse = solver_coarse.generate_forcing_term(k1, k2)
        f_fine = solver_fine.generate_forcing_term(k1, k2)
        
        # Solve PDE
        u_coarse = solver_coarse.solve_poisson(f_coarse, theta_coarse)
        u_fine = solver_fine.solve_poisson(f_fine, theta_fine)
        
        # Extract 16 overlapping subdomains (4x4 grid of subdomains)
        u_coarse_subdomains = extract_overlapping_subdomains(u_coarse, 20, 24, 4)  # 16 subdomains of 24x24
        u_fine_subdomains = extract_overlapping_subdomains(u_fine, 40, 48, 4)      # 16 subdomains of 48x48
        f_coarse_subdomains = extract_overlapping_subdomains(f_coarse, 20, 24, 4)  # 16 subdomains of 24x24
        f_fine_subdomains = extract_overlapping_subdomains(f_fine, 40, 48, 4)      # 16 subdomains of 48x48
        theta_coarse_subdomains = extract_overlapping_subdomains(theta_coarse, 20, 24, 4)  # 16 subdomains of 24x24
        theta_fine_subdomains = extract_overlapping_subdomains(theta_fine, 40, 48, 4)     # 16 subdomains of 48x48
        
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

def save_dataset_in_batches(dataset: Dict, path: str = 'data', batch_size: int = 1000):
    """
    Save the generated dataset in batches to avoid memory issues.
    
    Args:
        dataset: Dictionary containing the dataset
        path: Path to save the dataset
        batch_size: Number of samples to save in each batch
    """
    save_path = Path(path)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    
    total_samples = len(dataset['u_fine'])
    
    # Save in batches
    if total_samples > batch_size:
        print(f"Dataset is large ({total_samples} samples). Saving in batches of {batch_size}...")
        
        # First save metadata and create file
        init_data = {
            'total_samples': total_samples,
            'k1': dataset['k1'],
            'k2': dataset['k2']
        }
        np.savez(save_path / 'pde_dataset_overlapping.npz', **init_data)
        
        # Then save each batch
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            print(f"Saving batch {start_idx//batch_size + 1}: samples {start_idx} to {end_idx-1}")
            
            batch_data = {}
            for key in ['u_coarse', 'u_fine', 'f_coarse', 'f_fine', 'theta_coarse', 'theta_fine']:
                batch_data[f'{key}_batch_{start_idx//batch_size}'] = dataset[key][start_idx:end_idx]
            
            # Append to the npz file
            with np.load(save_path / 'pde_dataset_overlapping.npz') as data:
                # Convert loaded npz to dict
                existing_data = {key: data[key] for key in data.files}
                
                # Combine with new batch data
                combined_data = {**existing_data, **batch_data}
                
                # Save combined data
                np.savez(save_path / 'pde_dataset_overlapping.npz', **combined_data)
            
            # Clear memory
            del batch_data
            gc.collect()
    else:
        # Save as single file if small enough
        np.savez(save_path / 'pde_dataset_overlapping.npz', **dataset)
    
    print(f"Dataset saved to {save_path / 'pde_dataset_overlapping.npz'}")

def generate_dataset_in_batches(n_samples_part1: int, n_samples_part2: int, k_range: Tuple[float, float], batch_size: int = 500):
    """
    Generate the dataset in batches to save memory.
    
    Args:
        n_samples_part1: Number of samples for part 1
        n_samples_part2: Number of samples for part 2
        k_range: Range of wave numbers
        batch_size: Number of samples to generate in each batch
    """
    # Initialize dataset
    dataset = {
        'u_coarse': [],
        'u_fine': [],
        'f_coarse': [],
        'f_fine': [],
        'theta_coarse': [],
        'theta_fine': [],
        'k1': [],
        'k2': []
    }
    
    # Part 1: Generate datasets from 48x48 and 96x96 grids in batches
    print(f"Part 1: Generating {n_samples_part1} samples (48x48 → 96x96) with k_range {k_range}...")
    for start_idx in range(0, n_samples_part1, batch_size):
        end_idx = min(start_idx + batch_size, n_samples_part1)
        print(f"  Generating batch {start_idx//batch_size + 1}: samples {start_idx} to {end_idx-1}")
        
        # Generate this batch
        batch_samples = end_idx - start_idx
        batch_data = generate_dataset_part1(batch_samples, k_range)
        
        # Add to overall dataset
        for key in dataset:
            dataset[key].extend(batch_data[key])
        
        # Clear memory
        del batch_data
        gc.collect()
    
    # Part 2: Generate datasets from 96x96 and 192x192 grids in batches
    print(f"Part 2: Generating {n_samples_part2} samples (96x96 → 192x192) with k_range {k_range}...")
    for start_idx in range(0, n_samples_part2, batch_size // 4):  # Smaller batches for larger grids
        end_idx = min(start_idx + batch_size // 4, n_samples_part2)
        print(f"  Generating batch {start_idx//(batch_size//4) + 1}: samples {start_idx} to {end_idx-1}")
        
        # Generate this batch
        batch_samples = end_idx - start_idx
        batch_data = generate_dataset_part2(batch_samples, k_range)
        
        # Add to overall dataset
        for key in dataset:
            dataset[key].extend(batch_data[key])
        
        # Clear memory
        del batch_data
        gc.collect()
    
    # Convert lists to arrays
    for key in dataset:
        dataset[key] = np.array(dataset[key])
    
    print(f"Combined dataset contains {len(dataset['u_fine'])} samples")
    return dataset

def plot_samples(dataset: Dict, n_samples: int = 5, save_dir: str = 'dataset_samples_overlapping'):
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
        plt.title(f'Coarse Solution (24x24, Sample {idx})')
        
        plt.subplot(232)
        plt.imshow(dataset['u_fine'][idx])
        plt.colorbar()
        plt.title(f'Fine Solution (48x48, Sample {idx})')
        
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate dataset of PDE solutions with overlapping subdomains')
    parser.add_argument('--samples_part1', type=int, default=625, 
                        help='Number of samples for part 1 (48x48 to 96x96 grids)')
    parser.add_argument('--samples_part2', type=int, default=219, 
                        help='Number of samples for part 2 (96x96 to 192x192 grids)')
    parser.add_argument('--k_range_min', type=float, default=8.0,
                        help='Minimum value for wave number range')
    parser.add_argument('--k_range_max', type=float, default=10.0,
                        help='Maximum value for wave number range')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size for generating and saving data')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Get arguments
    n_samples_part1 = args.samples_part1
    n_samples_part2 = args.samples_part2
    k_range = (args.k_range_min, args.k_range_max)
    batch_size = args.batch_size
    
    print(f"Generating dataset with:")
    print(f"  - Part 1: {n_samples_part1} samples x 4 subdomains = {n_samples_part1 * 4} subdomain samples")
    print(f"  - Part 2: {n_samples_part2} samples x 16 subdomains = {n_samples_part2 * 16} subdomain samples")
    print(f"  - Total: ~{n_samples_part1 * 4 + n_samples_part2 * 16} subdomain samples")
    print(f"  - k-range: {k_range}")
    print(f"  - Batch size: {batch_size}")
    
    # Generate dataset in batches
    dataset = generate_dataset_in_batches(n_samples_part1, n_samples_part2, k_range, batch_size)
    
    # Verify shapes
    print(f"Coarse solution shape: {dataset['u_coarse'][0].shape}")
    print(f"Fine solution shape: {dataset['u_fine'][0].shape}")
    
    # Save dataset in batches
    save_dataset_in_batches(dataset, batch_size=batch_size)
    
    # Plot a few samples (if there are enough samples)
    n_plot_samples = min(10, len(dataset['u_fine']))
    if n_plot_samples > 0:
        plot_samples(dataset, n_samples=n_plot_samples)
        print(f"Sample plots saved successfully! ({n_plot_samples} samples)")
    
    print("Process completed successfully!") 