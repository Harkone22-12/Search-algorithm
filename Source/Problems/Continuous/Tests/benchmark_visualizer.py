"""
Visualization component for the Benchmarking Framework.

Contains functions to generate various plots for algorithm comparison,
single algorithm analysis, and parameter sensitivity.
This module separates plotting logic from the main benchmarking and analysis classes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .benchmark_framework import BenchmarkResult


# --- Single Algorithm Analysis Plots ---

def plot_single_solution_quality(result: 'BenchmarkResult', algo_name: str, problem_name: str, output_dir: str):
    """Plot solution quality distribution for a single algorithm."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = result.get_metrics()
    best_solutions = result.best_solutions
    
    # Left plot: Histogram of best solutions
    ax1.hist(best_solutions, bins=10, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(metrics['best_found'], color='#2ecc71', linestyle='--', linewidth=2, label=f"Best: {metrics['best_found']:.6f}")
    ax1.axvline(metrics['average_best'], color='#e74c3c', linestyle='--', linewidth=2, label=f"Mean: {metrics['average_best']:.6f}")
    ax1.set_xlabel('Fitness Value', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title(f'{algo_name} on {problem_name}: Solution Quality', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Right plot: Run-by-run best solutions
    runs = range(1, result.num_runs + 1)
    ax2.plot(runs, best_solutions, marker='o', linestyle='-', color='#3498db', linewidth=2, markersize=6)
    ax2.axhline(metrics['average_best'], color='#e74c3c', linestyle='--', linewidth=2, label=f"Mean: {metrics['average_best']:.6f}")
    ax2.fill_between(runs, 
                     np.array(best_solutions) - np.std(best_solutions),
                     np.array(best_solutions) + np.std(best_solutions),
                     alpha=0.2, color='#3498db')
    ax2.set_xlabel('Run Number', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Best Fitness Found', fontsize=11, fontweight='bold')
    ax2.set_title(f'{algo_name}: Quality per Run', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / f"{problem_name}_{algo_name}_01_solution_quality.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path.name}")

def plot_single_execution_time(result: 'BenchmarkResult', algo_name: str, problem_name: str, output_dir: str):
    """Plot execution time for a single algorithm."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = result.get_metrics()
    times = result.times
    
    # Left plot: Histogram of execution times
    ax1.hist(times, bins=10, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax1.axvline(metrics['mean_time'], color='#2ecc71', linestyle='--', linewidth=2, label=f"Mean: {metrics['mean_time']:.4f}s")
    ax1.set_xlabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title(f'{algo_name} on {problem_name}: Execution Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Right plot: Run-by-run execution times
    runs = range(1, result.num_runs + 1)
    ax2.bar(runs, times, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.axhline(metrics['mean_time'], color='#2ecc71', linestyle='--', linewidth=2, label=f"Mean: {metrics['mean_time']:.4f}s")
    ax2.set_xlabel('Run Number', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title(f'{algo_name}: Time per Run', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / f"{problem_name}_{algo_name}_02_execution_time.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path.name}")

def plot_single_convergence_curves(result: 'BenchmarkResult', algo_name: str, problem_name: str, output_dir: str):
    """Plot convergence curves for a single algorithm."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    convergence_curves = result.convergence_curves
    
    # Plot all convergence curves with transparency
    for idx, curve in enumerate(convergence_curves):
        ax.plot(curve, alpha=0.3, color='#3498db', linewidth=1)
    
    # Plot mean convergence curve
    mean_curve = np.mean(convergence_curves, axis=0)
    ax.plot(mean_curve, color='#2ecc71', linewidth=3, label='Mean Convergence', zorder=10)
    
    # Plot best and worst curves
    best_idx = np.argmin([curve[-1] for curve in convergence_curves])
    worst_idx = np.argmax([curve[-1] for curve in convergence_curves])
    ax.plot(convergence_curves[best_idx], color='#27ae60', linestyle='--', linewidth=2, label='Best Run', zorder=5)
    ax.plot(convergence_curves[worst_idx], color='#c0392b', linestyle='--', linewidth=2, label='Worst Run', zorder=5)
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitness Value (log scale)', fontsize=12, fontweight='bold')
    ax.set_title(f'{algo_name} on {problem_name}: Convergence ({result.num_runs} runs)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_path = Path(output_dir) / f"{problem_name}_{algo_name}_03_convergence_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path.name}")

def plot_single_robustness(result: 'BenchmarkResult', algo_name: str, problem_name: str, output_dir: str):
    """Plot robustness analysis (box plot) for a single algorithm."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = result.get_metrics()
    best_solutions = result.best_solutions
    
    # Left plot: Box plot
    bp = ax1.boxplot([best_solutions], labels=[algo_name], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][0].set_alpha(0.7)
    ax1.set_ylabel('Fitness Value', fontsize=11, fontweight='bold')
    ax1.set_title(f'{algo_name} on {problem_name}: Robustness', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = (f"Mean: {metrics['mean_fitness']:.6f}\n"
                 f"Std: {metrics['std_fitness']:.6f}\n"
                 f"CV: {metrics['cv_fitness']:.4f}\n"
                 f"Min: {min(best_solutions):.6f}\n"
                 f"Max: {max(best_solutions):.6f}")
    ax1.text(1.15, np.median(best_solutions), stats_text, fontsize=10, 
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right plot: Violin plot with individual points
    parts = ax2.violinplot([best_solutions], positions=[1], showmeans=True, showmedians=True)
    ax2.scatter([1] * len(best_solutions), best_solutions, alpha=0.5, color='#e74c3c', s=50)
    ax2.set_ylabel('Fitness Value', fontsize=11, fontweight='bold')
    ax2.set_xticks([1])
    ax2.set_xticklabels([algo_name])
    ax2.set_title(f'{algo_name}: Solution Distribution', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / f"{problem_name}_{algo_name}_04_robustness_boxplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path.name}")

def plot_single_algorithm_analysis(result: 'BenchmarkResult', algo_name: str, problem_name: str, output_dir: str):
    """Create detailed plots for a single algorithm."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    plot_single_solution_quality(result, algo_name, problem_name, output_dir)
    plot_single_execution_time(result, algo_name, problem_name, output_dir)
    plot_single_convergence_curves(result, algo_name, problem_name, output_dir)
    plot_single_robustness(result, algo_name, problem_name, output_dir)
    
    print(f"\nDetailed plots for {algo_name} saved to '{output_dir}/' directory")


# --- Comparison Plots (Multiple Algorithms) ---

def plot_solution_quality_comparison(results: Dict, problem_name: str, output_dir: str, file_prefix: str = ""):
    """Plot best solution quality for each algorithm."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algo_names = sorted(list(results.keys()))
    best_values = [results[name].get_metrics()['best_found'] for name in algo_names]
    mean_values = [results[name].get_metrics()['average_best'] for name in algo_names]
    
    x = np.arange(len(algo_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, best_values, width, label='Best Found', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, mean_values, width, label='Mean Found', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitness Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Solution Quality Comparison - {problem_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename_prefix = f"{file_prefix}_" if file_prefix else ""
    output_path = Path(output_dir) / f"{filename_prefix}01_solution_quality.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_execution_time_comparison(results: Dict, problem_name: str, output_dir: str, file_prefix: str = ""):
    """Plot execution time for each algorithm."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algo_names = sorted(list(results.keys()))
    times = [results[name].get_metrics()['mean_time'] for name in algo_names]
    stds = [results[name].get_metrics()['std_time'] for name in algo_names]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(algo_names)))
    bars = ax.bar(algo_names, times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.errorbar(algo_names, times, yerr=stds, fmt='none', ecolor='black', capsize=5, capthick=2)
    
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'Execution Time Comparison - {problem_name}', fontsize=14, fontweight='bold')
    ax.set_xticklabels(algo_names, rotation=45, ha="right")
    ax.grid(axis='y', alpha=0.3)
    
    for bar, time_val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., time_val,
                f'{time_val:.4f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filename_prefix = f"{file_prefix}_" if file_prefix else ""
    output_path = Path(output_dir) / f"{filename_prefix}02_execution_time.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_convergence_curves_comparison(results: Dict, max_iterations: int, problem_name: str, output_dir: str, file_prefix: str = ""):
    """Plot convergence curves for each algorithm on a single plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    algo_names = sorted(list(results.keys()))
    # Sử dụng colormap có độ tương phản tốt cho các đường line
    colors = plt.cm.get_cmap('tab10', len(algo_names))
    
    for idx, algo_name in enumerate(algo_names):
        result = results[algo_name]
        
        # Chuẩn hóa độ dài các đường cong
        max_len = max_iterations + 1
        standardized_curves = []
        for curve in result.convergence_curves:
            if not curve: continue
            # Thêm padding nếu thuật toán kết thúc sớm
            padding = [curve[-1]] * (max_len - len(curve)) if len(curve) < max_len else []
            standardized_curves.append(curve[:max_len] + padding)

        if not standardized_curves: continue

        # Tính đường cong hội tụ trung bình
        avg_curve = np.mean(standardized_curves, axis=0)
        
        # Vẽ đường cong trung bình cho thuật toán này
        ax.plot(avg_curve, color=colors(idx), linewidth=2, label=algo_name)
        
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitness Value (log scale)', fontsize=12, fontweight='bold')
    ax.set_title(f'Convergence Curve Comparison on {problem_name}', fontsize=14, fontweight='bold')
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    filename_prefix = f"{file_prefix}_" if file_prefix else ""
    output_path = Path(output_dir) / f"{filename_prefix}03_convergence_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_robustness_comparison(results: Dict, problem_name: str, output_dir: str, file_prefix: str = ""):
    """Plot robustness comparison (box plot)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algo_names = sorted(list(results.keys()))
    data = [results[name].best_solutions for name in algo_names]
    
    bp = ax.boxplot(data, labels=algo_names, patch_artist=True, showmeans=True, meanline=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Best Fitness Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Algorithm Robustness - {problem_name}', fontsize=14, fontweight='bold')
    ax.set_xticklabels(algo_names, rotation=45, ha="right")
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename_prefix = f"{file_prefix}_" if file_prefix else ""
    output_path = Path(output_dir) / f"{filename_prefix}04_robustness_boxplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_algorithm_comparison(results: Dict, max_iterations: int, problem_name: str, output_dir: str, file_prefix: str = ""):
    """Create comparison plots for all algorithms."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    if not results:
        print("No results to plot")
        return
    
    print(f"\n>>> Generating comparison plots for {problem_name}...")
    plot_solution_quality_comparison(results, problem_name, output_dir, file_prefix)
    plot_execution_time_comparison(results, problem_name, output_dir, file_prefix)
    plot_convergence_curves_comparison(results, max_iterations, problem_name, output_dir, file_prefix)
    plot_robustness_comparison(results, problem_name, output_dir, file_prefix)
    print(f">>> Comparison plots saved to directory: {output_dir}")


# --- Sensitivity Analysis Plots ---

def plot_parameter_sensitivity(param_name: str, results: Dict, output_dir: str):
    """Create plots for parameter sensitivity showing 3 key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    param_values = results['param_values']
    mean_fitness = results['mean_fitness']
    best_fitness = results.get('best_fitness', [])
    robustness = results.get('robustness', [])
    exploration_score = results.get('exploration_score', [])
    exploitation_score = results.get('exploitation_score', [])
    exec_times = results['exec_times']
    
    # Plot 1: Best/Average Quality
    ax1 = axes[0, 0]
    ax1.plot(range(len(param_values)), best_fitness, 'o-', color='#2ecc71', 
             linewidth=2.5, markersize=8, label='Best Fitness')
    ax1.plot(range(len(param_values)), mean_fitness, 's--', color='#3498db', 
             linewidth=2.5, markersize=7, label='Mean Fitness')
    
    ax1.set_xlabel(f'{param_name}', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Fitness Value', fontsize=11, fontweight='bold')
    ax1.set_title('1. Best/Average Quality (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(param_values)))
    ax1.set_xticklabels([f'{v}' for v in param_values], rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    
    # Plot 2: Robustness
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(param_values)), robustness, color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel(f'{param_name}', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Std Deviation', fontsize=11, fontweight='bold')
    ax2.set_title('2. Robustness: Consistency (Lower Std is Better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(param_values)))
    ax2.set_xticklabels([f'{v}' for v in param_values], rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    for bar, rob_val in zip(bars, robustness):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rob_val:.2e}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Exploration vs Exploitation
    ax3 = axes[1, 0]
    x_pos = np.arange(len(param_values))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, exploration_score, width, label='Exploration (Early Improvement)', 
                    color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax3.bar(x_pos + width/2, exploitation_score, width, label='Exploitation (Convergence)', 
                    color='#f39c12', alpha=0.8, edgecolor='black')
    
    ax3.set_xlabel(f'{param_name}', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax3.set_title('3. Exploration vs Exploitation Balance', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{v}' for v in param_values], rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(fontsize=10)
    ax3.set_ylim([0, 1.1])
    
    # Plot 4: Execution Time
    ax4 = axes[1, 1]
    bars = ax4.bar(range(len(param_values)), exec_times, color='#9b59b6', alpha=0.7, edgecolor='black')
    
    ax4.set_xlabel(f'{param_name}', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_title('Computation Cost', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(param_values)))
    ax4.set_xticklabels([f'{v}' for v in param_values], rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, time_val in zip(bars, exec_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.3f}s', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle(f'Parameter Sensitivity Analysis: {param_name}', fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    safe_param_name = param_name.replace(' ', '_').lower()
    output_path = Path(output_dir) / f"sensitivity_{safe_param_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Plot saved: sensitivity_{safe_param_name}.png")

def plot_sensitivity_heatmap(sensitivity_results: Dict, output_dir: str, base_params: Dict = None):
    """Generate heatmaps for each parameter showing 3 key metrics.
    
    Metrics displayed:
    1. Best/Average Quality: Lower fitness is better
    2. Robustness: Lower std deviation is better (more consistent)
    3. Exploration vs Exploitation: Balance between early improvement and convergence
    
    Args:
        sensitivity_results: Dict with sensitivity analysis results for each parameter
        output_dir: Directory to save the heatmaps
        base_params: Base parameters used in the analysis (for display purposes)
    """
    if not sensitivity_results:
        print("No sensitivity results to plot")
        return
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    param_names = list(sensitivity_results.keys())
    num_params = len(param_names)
    
    # Create a figure with 3 rows (one for each metric) and columns for each parameter
    cols = num_params
    rows = 3  # Three metrics: Quality, Robustness, Exploration vs Exploitation
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    
    # Handle single parameter case
    if num_params == 1:
        axes = axes.reshape(3, 1)
    
    # Create title with base parameters information
    title = 'Parameter Sensitivity Analysis: 3 Key Metrics'
    if base_params:
        fixed_params = {k: v for k, v in base_params.items() if k not in param_names}
        if fixed_params:
            fixed_str = ", ".join([f"{k}={v}" for k, v in fixed_params.items()])
            title += f"\nBase Parameters: {fixed_str}"
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # Metric configurations: (metric_key, title, cmap, lower_is_better)
    metrics = [
        ('best_fitness', 'Best/Average Quality (Lower is Better)', 'RdYlGn_r', True),
        ('robustness', 'Robustness: Consistency (Lower Std is Better)', 'RdYlGn_r', True),
        ('exploration_score', 'Exploration Rate (Early Improvement)', 'YlGn', False),
    ]
    
    # Plot each metric as a row
    for metric_idx, (metric_key, metric_title, cmap, lower_better) in enumerate(metrics):
        for param_idx, param_name in enumerate(param_names):
            ax = axes[metric_idx, param_idx] if num_params > 1 else axes[metric_idx]
            results = sensitivity_results[param_name]
            param_values = results['param_values']
            
            # Get metric values
            if metric_key in results:
                metric_values = results[metric_key]
            else:
                metric_values = [0] * len(param_values)
            
            # Create a 1D heatmap
            matrix = np.array([metric_values])
            
            # Display heatmap
            im = ax.imshow(matrix, cmap=cmap, aspect='auto')
            
            # Set axes labels
            ax.set_xticks(np.arange(len(param_values)))
            ax.set_xticklabels([f'{v}' for v in param_values], fontsize=9, fontweight='bold')
            
            if metric_idx == 0:
                ax.set_yticks([0])
                ax.set_yticklabels([param_name], fontsize=10, fontweight='bold')
            else:
                ax.set_yticks([])
            
            if metric_idx == 2:  # Only bottom row shows x-axis label
                ax.set_xlabel(f'{param_name} Values', fontsize=10, fontweight='bold')
            
            if param_idx == 0:  # Only left column shows metric title
                ax.set_ylabel(metric_title.split(':')[0], fontsize=10, fontweight='bold')
            
            # Add metric values as text
            for j, value in enumerate(metric_values):
                ax.text(j, 0, f'{value:.2e}', ha="center", va="center", 
                       color="black", fontsize=8, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Value', fontsize=8)
            
            # Add grid
            ax.set_xticks(np.arange(len(param_values)) - 0.5, minor=True)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=2, axis='x')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "sensitivity_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nHeatmap saved: sensitivity_heatmap.png")
    print("\nMetric Interpretation:")
    print("  • Best/Average Quality: Lower fitness values indicate better solutions")
    print("  • Robustness: Lower standard deviation indicates more consistent performance")
    print("  • Exploration Rate: Higher early improvement indicates better exploration tendency")
    
    # Also print base parameters info to console
    if base_params:
        fixed_params = {k: v for k, v in base_params.items() if k not in param_names}
        if fixed_params:
            print("\nBase Parameters (kept constant during analysis):")
            for param, value in fixed_params.items():
                print(f"  {param}: {value}")