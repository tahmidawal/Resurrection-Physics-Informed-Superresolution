import numpy as np
from scipy.sparse import diags, linalg
import matplotlib.pyplot as plt
from pathlib import Path
import time
import argparse
import os

class PoissonSolver:
    def __init__(self, n: int):
        """
        Initialize the Poisson equation solver with a specific grid size.
        
        Args:
            n: Number of grid points in each dimension
        """
        self.n = n
        self.h = 1.0 / (n - 1)  # Grid spacing
        
        # Initialize grid coordinates
        self.x = np.linspace(0, 1, n)
        self.y = np.linspace(0, 1, n)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Create the Laplacian matrix for finite difference
        self.L = self._create_laplacian()
    
    def _create_laplacian(self):
        """Create the 2D Laplacian finite difference matrix."""
        n = self.n
        n2 = n * n
        h2 = self.h * self.h
        
        # Main diagonal: -4
        main_diag = -4 * np.ones(n2)
        
        # Off diagonals: 1
        off_diag1 = np.ones(n2-1)
        # Remove connections across horizontal boundaries
        off_diag1[np.arange(n-1, n2-1, n)] = 0
        
        # Off diagonals for vertical connections
        off_diag_n = np.ones(n2-n)
        
        # Assemble the matrix using scipy's diags
        diagonals = [main_diag, off_diag1, off_diag1, off_diag_n, off_diag_n]
        offsets = [0, 1, -1, n, -n]
        laplacian = diags(diagonals, offsets, shape=(n2, n2))
        
        return laplacian / h2
    
    def generate_forcing_term(self, k1, k2):
        """
        Generate the forcing term f(x,y) = sin(k₁ * 2πx) * sin(k₂ * 2πy).
        
        Args:
            k1: First wave number parameter
            k2: Second wave number parameter
        
        Returns:
            2D array of forcing term values
        """
        return np.sin(2 * np.pi * k1 * self.X) * np.sin(2 * np.pi * k2 * self.Y)
    
    def solve(self, f):
        """
        Solve the Poisson equation -∇²u = f with Dirichlet boundary conditions.
        
        Args:
            f: 2D array of forcing term values
            
        Returns:
            2D array of solution values
        """
        # Flatten f for the linear solver
        f_flat = f.flatten()
        
        # Apply boundary conditions (Dirichlet u=0 on boundaries)
        # This is already handled in our Laplacian construction
        
        # Solve the linear system
        u_flat = linalg.spsolve(self.L, f_flat)
        
        # Reshape back to 2D
        u = u_flat.reshape((self.n, self.n))
        
        return u

def compute_reference_solution(k1, k2, n_ref=640):
    """
    Compute a high-resolution reference solution to use as ground truth.
    
    Args:
        k1, k2: Wave number parameters
        n_ref: Resolution for reference solution
        
    Returns:
        Reference solution on a high-resolution grid
    """
    print(f"Computing reference solution at {n_ref}x{n_ref} resolution...")
    start_time = time.time()
    
    solver = PoissonSolver(n_ref)
    f = solver.generate_forcing_term(k1, k2)
    u_ref = solver.solve(f)
    
    elapsed = time.time() - start_time
    print(f"Reference solution computed in {elapsed:.2f} seconds")
    
    return u_ref, solver.x, solver.y

def interpolate_to_reference_grid(u, x, y, x_ref, y_ref):
    """
    Interpolate a solution to the reference grid for comparison.
    
    Args:
        u: Lower resolution solution
        x, y: Lower resolution grid coordinates
        x_ref, y_ref: Reference grid coordinates
        
    Returns:
        Solution interpolated to reference grid
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # Create interpolation function
    interp_func = RegularGridInterpolator((y, x), u, bounds_error=False, fill_value=0)
    
    # Create grid of points to evaluate
    XX_ref, YY_ref = np.meshgrid(x_ref, y_ref)
    points = np.column_stack((YY_ref.flatten(), XX_ref.flatten()))
    
    # Interpolate
    u_interp = interp_func(points).reshape(XX_ref.shape)
    
    return u_interp

def compute_errors(u_test, u_ref):
    """
    Compute error metrics between test and reference solutions.
    
    Args:
        u_test: Test solution (interpolated to reference grid)
        u_ref: Reference solution
        
    Returns:
        Dictionary of error metrics
    """
    # Compute absolute error
    abs_error = np.abs(u_test - u_ref)
    
    # Error metrics
    mae = np.mean(abs_error)
    max_error = np.max(abs_error)
    rmse = np.sqrt(np.mean(np.square(abs_error)))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'Max Error': max_error
    }

def run_comparison(k1, k2, resolutions, output_dir):
    """
    Run comparison of solutions at different resolutions.
    
    Args:
        k1, k2: Wave number parameters
        resolutions: List of grid resolutions to test
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute reference (high-resolution) solution
    u_ref, x_ref, y_ref = compute_reference_solution(k1, k2)
    
    # Store results
    results = {}
    solution_data = {}
    
    # Compute solutions at each resolution
    for n in resolutions:
        print(f"\nSolving at {n}x{n} resolution...")
        start_time = time.time()
        
        # Create solver and solve
        solver = PoissonSolver(n)
        f = solver.generate_forcing_term(k1, k2)
        u = solver.solve(f)
        
        # Measure time
        elapsed = time.time() - start_time
        print(f"Solution computed in {elapsed:.2f} seconds")
        
        # Interpolate to reference grid for comparison
        u_interp = interpolate_to_reference_grid(u, solver.x, solver.y, x_ref, y_ref)
        
        # Compute errors
        errors = compute_errors(u_interp, u_ref)
        print(f"Errors: MAE={errors['MAE']:.6f}, RMSE={errors['RMSE']:.6f}, Max={errors['Max Error']:.6f}")
        
        # Store results
        results[n] = {
            'time': elapsed,
            'errors': errors
        }
        solution_data[n] = {
            'u': u,
            'x': solver.x,
            'y': solver.y
        }
    
    # Plot and save results
    plot_results(results, solution_data, k1, k2, output_dir)
    save_results(results, k1, k2, output_dir)
    
    return results

def plot_results(results, solution_data, k1, k2, output_dir):
    """Plot and save visualizations of the results."""
    # Plot errors vs resolution
    plt.figure(figsize=(12, 8))
    resolutions = sorted(results.keys())
    
    # Plot MAE
    maes = [results[n]['errors']['MAE'] for n in resolutions]
    plt.loglog(resolutions, maes, 'o-', label='Mean Absolute Error')
    
    # Plot RMSE
    rmses = [results[n]['errors']['RMSE'] for n in resolutions]
    plt.loglog(resolutions, rmses, 's-', label='Root Mean Square Error')
    
    # Plot Max Error
    max_errors = [results[n]['errors']['Max Error'] for n in resolutions]
    plt.loglog(resolutions, max_errors, '^-', label='Maximum Error')
    
    # Add theoretical second-order convergence line
    ref_line = maes[0] * (np.array(resolutions) / resolutions[0])**(-2)
    plt.loglog(resolutions, ref_line, 'k--', label='Second-order convergence')
    
    plt.xlabel('Resolution N (grid points per dimension)')
    plt.ylabel('Error')
    plt.title(f'Error vs Resolution for Poisson Equation\n(k₁={k1}, k₂={k2})')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_vs_resolution.png", dpi=300)
    
    # Plot solution comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot for each resolution
    for i, n in enumerate(sorted(solution_data.keys())):
        if i >= len(axes):
            break
            
        data = solution_data[n]
        im = axes[i].imshow(data['u'], origin='lower', extent=[0, 1, 0, 1])
        axes[i].set_title(f"{n}x{n} Resolution\nMAE: {results[n]['errors']['MAE']:.6f}")
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/solution_comparison.png", dpi=300)
    
    # Plot error distribution
    plt.figure(figsize=(12, 8))
    for n in sorted(solution_data.keys()):
        # Calculate error histogram for this resolution
        u = solution_data[n]['u']
        u_interp = interpolate_to_reference_grid(
            u, solution_data[n]['x'], solution_data[n]['y'], 
            solution_data[max(solution_data.keys())]['x'], solution_data[max(solution_data.keys())]['y']
        )
        error = np.abs(u_interp - solution_data[max(solution_data.keys())]['u'])
        
        # Plot histogram
        plt.hist(error.flatten(), bins=50, alpha=0.5, density=True, label=f"{n}x{n}")
    
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency (density)')
    plt.title('Error Distribution by Resolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_distribution.png", dpi=300)

def save_results(results, k1, k2, output_dir):
    """Save numerical results to a text file."""
    with open(f"{output_dir}/results.txt", 'w') as f:
        f.write(f"Accuracy Results for Poisson Equation with k₁={k1}, k₂={k2}\n")
        f.write("=" * 80 + "\n\n")
        
        # Table header
        f.write(f"{'Resolution':<12} {'Time (s)':<12} {'MAE':<15} {'RMSE':<15} {'Max Error':<15}\n")
        f.write("-" * 70 + "\n")
        
        # Table rows
        for n in sorted(results.keys()):
            f.write(f"{n:<12} {results[n]['time']:<12.4f} {results[n]['errors']['MAE']:<15.8f} "
                    f"{results[n]['errors']['RMSE']:<15.8f} {results[n]['errors']['Max Error']:<15.8f}\n")
        
        # Calculate convergence rates
        f.write("\nConvergence Rates (between consecutive resolutions):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Resolutions':<15} {'MAE Rate':<15} {'RMSE Rate':<15} {'Max Error Rate':<15}\n")
        
        resolutions = sorted(results.keys())
        for i in range(1, len(resolutions)):
            n1, n2 = resolutions[i-1], resolutions[i]
            ratio = n2 / n1
            
            mae_rate = np.log(results[n1]['errors']['MAE'] / results[n2]['errors']['MAE']) / np.log(ratio)
            rmse_rate = np.log(results[n1]['errors']['RMSE'] / results[n2]['errors']['RMSE']) / np.log(ratio)
            max_rate = np.log(results[n1]['errors']['Max Error'] / results[n2]['errors']['Max Error']) / np.log(ratio)
            
            f.write(f"{n1} → {n2:<10} {mae_rate:<15.4f} {rmse_rate:<15.4f} {max_rate:<15.4f}\n")
        
        # Save raw data as numpy array for later use
        np.savez(f"{output_dir}/result_data.npz", 
                 resolutions=resolutions,
                 times=[results[n]['time'] for n in resolutions],
                 maes=[results[n]['errors']['MAE'] for n in resolutions],
                 rmses=[results[n]['errors']['RMSE'] for n in resolutions],
                 max_errors=[results[n]['errors']['Max Error'] for n in resolutions])

def main():
    parser = argparse.ArgumentParser(description='Compare PDE solution accuracy at different resolutions')
    parser.add_argument('--k1', type=float, default=8.0, help='First wave number parameter')
    parser.add_argument('--k2', type=float, default=6.0, help='Second wave number parameter')
    parser.add_argument('--output-dir', type=str, default='Output/accuracy_comparison', 
                        help='Directory to save results')
    args = parser.parse_args()
    
    # List of resolutions to test
    resolutions = [20, 40, 80, 160, 320]
    
    print(f"Starting accuracy comparison for Poisson equation with k₁={args.k1}, k₂={args.k2}")
    print(f"Testing resolutions: {resolutions}")
    print(f"Results will be saved to: {args.output_dir}")
    
    # Run comparison
    run_comparison(args.k1, args.k2, resolutions, args.output_dir)
    
    print("\nComparison complete!")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 