import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import os
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt

class PoissonSolver:
    def __init__(self, n_coarse: int = 20, n_fine: int = 40):
        """
        Initialize the Poisson equation solver.
        
        Args:
            n_coarse: Number of grid points in each dimension for coarse grid
            n_fine: Number of grid points in each dimension for fine grid
        """
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        
        # Initialize grids
        self.x_coarse = np.linspace(0, 1, n_coarse)
        self.y_coarse = np.linspace(0, 1, n_coarse)
        self.x_fine = np.linspace(0, 1, n_fine)
        self.y_fine = np.linspace(0, 1, n_fine)
        
        # Create meshgrids
        self.X_coarse, self.Y_coarse = np.meshgrid(self.x_coarse, self.y_coarse)
        self.X_fine, self.Y_fine = np.meshgrid(self.x_fine, self.y_fine)
        
        # Initialize Laplacian matrices
        self.L_coarse = self._create_laplacian(n_coarse)
        self.L_fine = self._create_laplacian(n_fine)

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

    def generate_forcing_term(self, k1: float, k2: float, grid: str = 'fine') -> np.ndarray:
        """
        Generate the forcing term f(x,y) = sin(k₁ * 2πx) * sin(k₂ * 2πy).
        
        Args:
            k1: First wave number
            k2: Second wave number
            grid: 'fine' or 'coarse' grid
            
        Returns:
            2D array containing the forcing term values
        """
        if grid == 'fine':
            X, Y = self.X_fine, self.Y_fine
        else:
            X, Y = self.X_coarse, self.Y_coarse
            
        return np.sin(2 * np.pi * k1 * X) * np.sin(2 * np.pi * k2 * Y)

    def solve_poisson(self, f: np.ndarray, theta: np.ndarray, grid: str = 'fine') -> np.ndarray:
        """
        Solve the Poisson equation -∇·(θ∇u) = f with zero Dirichlet boundary conditions.
        
        Args:
            f: Forcing term
            theta: Diffusion coefficient field
            grid: 'fine' or 'coarse' grid
            
        Returns:
            Solution u
        """
        n = self.n_fine if grid == 'fine' else self.n_coarse
        L = self.L_fine if grid == 'fine' else self.L_coarse
        
        # Reshape inputs to 1D arrays
        f_flat = f.reshape(-1)
        theta_flat = theta.reshape(-1)
        
        # Modify Laplacian with theta
        L_theta = diags(theta_flat) @ L
        
        # Solve the system
        u_flat = spsolve(L_theta, f_flat)
        
        return u_flat.reshape((n, n))

    def generate_dataset(self, n_samples: int, k_range: Tuple[float, float] = (1, 5)) -> dict:
        """
        Generate a dataset of PDE solutions.
        
        Args:
            n_samples: Number of samples to generate
            k_range: Range for random wave numbers
            
        Returns:
            Dictionary containing the dataset
        """
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
        
        for _ in range(n_samples):
            # Generate random wave numbers
            k1 = np.random.uniform(*k_range)
            k2 = np.random.uniform(*k_range)
            
            # Generate random theta fields
            theta_fine = np.random.uniform(0.5, 2.0, size=(self.n_fine, self.n_fine))
            theta_coarse = theta_fine[::2, ::2]  # Downsample for coarse grid
            
            # Generate forcing terms
            f_fine = self.generate_forcing_term(k1, k2, 'fine')
            f_coarse = self.generate_forcing_term(k1, k2, 'coarse')
            
            # Solve PDE on both grids
            u_fine = self.solve_poisson(f_fine, theta_fine, 'fine')
            u_coarse = self.solve_poisson(f_coarse, theta_coarse, 'coarse')
            
            # Store results
            dataset['u_coarse'].append(u_coarse)
            dataset['u_fine'].append(u_fine)
            dataset['f_coarse'].append(f_coarse)
            dataset['f_fine'].append(f_fine)
            dataset['theta_coarse'].append(theta_coarse)
            dataset['theta_fine'].append(theta_fine)
            dataset['k1'].append(k1)
            dataset['k2'].append(k2)
        
        # Convert lists to arrays
        for key in dataset:
            dataset[key] = np.array(dataset[key])
            
        return dataset

    def save_dataset(self, dataset: dict, path: str = 'data'):
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
            save_path / 'pde_dataset.npz',
            **dataset
        )

if __name__ == '__main__':
    # Example usage
    solver = PoissonSolver()
    
    # Generate a larger dataset
    n_samples = 2000  # Increased from 100 to 1000
    print(f"Generating {n_samples} samples...")
    dataset = solver.generate_dataset(n_samples=n_samples, k_range=(0.5, 6.0))  # Wider k_range
    
    # Save the dataset
    solver.save_dataset(dataset)
    print("Dataset saved successfully!")
    
    # Create results directory if it doesn't exist
    results_dir = Path('results')
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    
    # Plot an example solution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(dataset['u_coarse'][0])
    plt.colorbar()
    plt.title('Coarse Solution')
    
    plt.subplot(132)
    plt.imshow(dataset['u_fine'][0])
    plt.colorbar()
    plt.title('Fine Solution')
    
    plt.subplot(133)
    plt.imshow(dataset['f_fine'][0])
    plt.colorbar()
    plt.title('Forcing Term (Fine)')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'example_solution.png')
    plt.close() 