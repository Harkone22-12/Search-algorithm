"""
3D Surface Plotting Utility for 2D Optimization Functions.

This script can be run directly to generate and save 3D plots for the
Sphere and Rastrigin benchmark functions.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, List
import sys

# Add project root to path to allow importing problem classes
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import problem classes for consistency
from Source.Problems.Continuous.Functions.Sphere import SphereProblem
from Source.Problems.Continuous.Functions.Rastrigin import RastriginProblem


def plot_3d_surface(func: Callable[[np.ndarray], float], func_name: str, bounds: list, output_dir: Path):
    """
    Generates and saves a 3D surface plot of a given 2D function.
    
    Args:
        func (Callable): The 2D function to plot.
        func_name (str): The name of the function for the plot title.
        bounds (list): A list `[min_val, max_val]` for the x and y axes.
        output_dir (Path): The directory to save the plot image.
    """
    print(f"Generating 3D surface plot for {func_name}...")
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([func(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f'3D Surface of {func_name} Function', fontsize=16, fontweight='bold')
    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_zlabel('f(x1, x2)', fontsize=12)
    
    filepath = output_dir / f"surface_plot_{func_name}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Plot saved to {filepath}")

# --- Main Execution ---
def main():
    """Generates plots for standard benchmark functions when the script is run directly."""
    # The output directory will be the same directory as this script.
    output_dir = Path(__file__).parent
    
    # Instantiate problem classes (must be 2D for plotting)
    sphere_problem = SphereProblem(dimensions=2)
    rastrigin_problem = RastriginProblem(dimensions=2)

    problems = {
        "Sphere": {"problem": sphere_problem, "bounds": sphere_problem.bounds[0]},
        "Rastrigin": {"problem": rastrigin_problem, "bounds": rastrigin_problem.bounds[0]}
    }

    print("="*50)
    print("Generating 3D Surface Plots...")
    print("="*50)

    for name, details in problems.items():
        problem_instance = details["problem"]
        # The plot function expects a callable that takes a numpy array.
        # The problem's evaluate_state method takes a list.
        # This lambda bridges the two.
        plot_func = lambda arr: problem_instance.evaluate_state(list(arr))

        plot_3d_surface(
            func=plot_func,
            func_name=name,
            bounds=details["bounds"],
            output_dir=output_dir
        )
    
    print("\nAll plots generated successfully.")

if __name__ == "__main__":
    main()