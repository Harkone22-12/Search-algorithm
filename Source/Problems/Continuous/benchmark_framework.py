"""
Benchmarking Framework for Continuous Optimization Algorithms

Evaluates algorithms based on:
1. Convergence Speed
2. Solution Quality (Best/Average)
3. Computational Complexity (Time & Space)
4. Robustness (Mean ± Std)
5. Scalability (Performance with dimensions)
6. Exploration vs Exploitation
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Callable
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, algorithm_name: str, runs: List[Dict]):
        self.algorithm_name = algorithm_name
        self.runs = runs
        self.num_runs = len(runs)
        
        # Extract metrics from runs
        self.best_solutions = [r['best_cost'] for r in runs]
        self.final_costs = [r['final_cost'] for r in runs]
        self.times = [r['time'] for r in runs]
        self.convergence_curves = [r['history'] for r in runs]
        
    def get_metrics(self) -> Dict:
        """Calculate all performance metrics."""
        metrics = {}
        
        # 1. Convergence Speed
        metrics['convergence_iterations'] = [
            self._iterations_to_convergence(curve) 
            for curve in self.convergence_curves
        ]
        metrics['mean_convergence_iterations'] = np.mean(metrics['convergence_iterations'])
        metrics['std_convergence_iterations'] = np.std(metrics['convergence_iterations'])
        
        # 2. Solution Quality
        metrics['best_found'] = min(self.best_solutions)
        metrics['average_best'] = np.mean(self.best_solutions)
        metrics['worst_found'] = max(self.best_solutions)
        metrics['median_best'] = np.median(self.best_solutions)
        
        # 3. Robustness
        metrics['mean_fitness'] = np.mean(self.final_costs)
        metrics['std_fitness'] = np.std(self.final_costs)
        metrics['cv_fitness'] = metrics['std_fitness'] / abs(metrics['mean_fitness']) if metrics['mean_fitness'] != 0 else 0
        
        # 4. Computational Complexity
        metrics['mean_time'] = np.mean(self.times)
        metrics['std_time'] = np.std(self.times)
        metrics['total_time'] = sum(self.times)
        
        # 5. Efficiency
        metrics['iterations_per_second'] = [
            len(self.convergence_curves[i]) / self.times[i] if self.times[i] > 0 else 0
            for i in range(self.num_runs)
        ]
        metrics['mean_iterations_per_second'] = np.mean(metrics['iterations_per_second'])
        
        # 6. Exploration vs Exploitation (from convergence curve)
        metrics['exploration_score'] = self._calculate_exploration_score()
        
        return metrics
    
    @staticmethod
    def _iterations_to_convergence(curve: List[float], threshold: float = 1e-3) -> int:
        """Find iterations to reach convergence threshold."""
        if not curve:
            return 0
        
        initial = curve[0]
        for i, val in enumerate(curve):
            improvement = initial - val
            if improvement >= threshold or i == len(curve) - 1:
                return i + 1
        return len(curve)
    
    def _calculate_exploration_score(self) -> float:
        """Calculate exploration score from convergence curves."""
        scores = []
        for curve in self.convergence_curves:
            if len(curve) < 2:
                scores.append(0)
                continue
            
            # Early convergence (first 30%) improvement vs late (last 30%)
            n = len(curve)
            early_phase = int(n * 0.3)
            late_phase = n - int(n * 0.3)
            
            early_improvement = curve[0] - (curve[early_phase] if early_phase > 0 else curve[0])
            late_improvement = curve[late_phase] - curve[-1] if late_phase < n else 0
            
            # High early/low late = good exploration; low early/high late = good exploitation
            exploration_ratio = early_improvement / (late_improvement + 1e-10)
            scores.append(min(exploration_ratio, 10))  # Cap at 10
        
        return np.mean(scores)


class AlgorithmBenchmark:
    """Framework for benchmarking algorithms."""
    
    def __init__(self, problem_class, dimensions: int = 5, max_iterations: int = 100, num_runs: int = 30):
        self.problem_class = problem_class
        self.dimensions = dimensions
        self.max_iterations = max_iterations
        self.num_runs = num_runs
        self.results: Dict[str, BenchmarkResult] = {}
    
    def run_algorithm(
        self, 
        algorithm_class, 
        algorithm_name: str,
        **algorithm_params
    ) -> BenchmarkResult:
        """
        Run algorithm multiple times and collect results.
        
        Args:
            algorithm_class: Algorithm class
            algorithm_name: Name for the algorithm
            **algorithm_params: Parameters for algorithm initialization
        
        Returns:
            BenchmarkResult object
        """
        runs = []
        
        for run_id in range(self.num_runs):
            # Create problem instance
            problem = self.problem_class(dimensions=self.dimensions)
            
            # Create algorithm with params
            algorithm = algorithm_class(
                max_iterations=self.max_iterations,
                **algorithm_params
            )
            
            # Run algorithm
            start_time = time.time()
            result = algorithm.search(problem)
            elapsed_time = time.time() - start_time
            
            # Ensure history exists
            history = result.get('history', [])
            if not history:
                history = [result.get('cost', float('inf'))]
            
            # Build best-so-far curve
            best_so_far = [history[0]]
            best = history[0]
            for fitness in history[1:]:
                best = min(best, fitness)
                best_so_far.append(best)
            
            run_data = {
                'run_id': run_id + 1,
                'best_cost': result.get('cost', float('inf')),
                'final_cost': history[-1] if history else float('inf'),
                'time': elapsed_time,
                'history': best_so_far,
                'expanded_nodes': result.get('expanded_nodes', 0)
            }
            runs.append(run_data)
            
            print(f"{algorithm_name} Run {run_id + 1}/{self.num_runs}: "
                  f"Best={run_data['best_cost']:.6f}, Time={elapsed_time:.4f}s")
        
        result = BenchmarkResult(algorithm_name, runs)
        self.results[algorithm_name] = result
        return result
    
    def print_metrics(self, algorithm_name: str):
        """Print metrics for an algorithm."""
        if algorithm_name not in self.results:
            print(f"No results for {algorithm_name}")
            return
        
        result = self.results[algorithm_name]
        metrics = result.get_metrics()
        
        print(f"\n{'='*70}")
        print(f"ALGORITHM: {algorithm_name}")
        print(f"{'='*70}")
        
        print(f"\n1. CONVERGENCE SPEED:")
        print(f"   Iterations to convergence: {metrics['mean_convergence_iterations']:.1f} ± {metrics['std_convergence_iterations']:.1f}")
        print(f"   Mean time per run: {metrics['mean_time']:.4f}s ± {metrics['std_time']:.4f}s")
        
        print(f"\n2. SOLUTION QUALITY:")
        print(f"   Best solution found: {metrics['best_found']:.6f}")
        print(f"   Average solution: {metrics['average_best']:.6f}")
        print(f"   Worst solution found: {metrics['worst_found']:.6f}")
        print(f"   Median solution: {metrics['median_best']:.6f}")
        
        print(f"\n3. ROBUSTNESS:")
        print(f"   Mean fitness: {metrics['mean_fitness']:.6f}")
        print(f"   Std deviation: {metrics['std_fitness']:.6f}")
        print(f"   Coefficient of variation: {metrics['cv_fitness']:.4f}")
        
        print(f"\n4. COMPUTATIONAL COMPLEXITY:")
        print(f"   Mean execution time: {metrics['mean_time']:.4f}s")
        print(f"   Iterations per second: {metrics['mean_iterations_per_second']:.2f}")
        print(f"   Total time for {self.num_runs} runs: {metrics['total_time']:.2f}s")
        
        print(f"\n5. EXPLORATION vs EXPLOITATION:")
        print(f"   Exploration score (early/late): {metrics['exploration_score']:.2f}")
        if metrics['exploration_score'] > 5:
            print(f"   --> Good explorer (explores more than exploits)")
        elif metrics['exploration_score'] < 2:
            print(f"   --> Good exploiter (exploits more than explores)")
        else:
            print(f"   --> Balanced (good exploration-exploitation)")
        
        print(f"\n6. RUN STATISTICS:")
        print(f"   Number of runs: {result.num_runs}")
        print(f"   {'='*70}\n")
    
    def compare_algorithms(self, *algorithm_names):
        """Compare multiple algorithms."""
        print(f"\n{'='*80}")
        print(f"ALGORITHM COMPARISON (Sphere Function - {self.dimensions}D)")
        print(f"{'='*80}")
        print(f"{'Algorithm':<20} {'Best':<12} {'Mean':<12} {'Std':<12} {'Time(s)':<12} {'Conv.Iter':<12}")
        print(f"{'-'*80}")
        
        for name in algorithm_names:
            if name not in self.results:
                continue
            result = self.results[name]
            metrics = result.get_metrics()
            
            print(f"{name:<20} {metrics['best_found']:<12.6f} {metrics['average_best']:<12.6f} "
                  f"{metrics['std_fitness']:<12.6f} {metrics['mean_time']:<12.4f} "
                  f"{metrics['mean_convergence_iterations']:<12.1f}")
        
        print(f"{'='*80}\n")
    
    def hypothesis_test(self, algo1: str, algo2: str, alpha: float = 0.05) -> Dict:
        """
        Perform t-test between two algorithms.
        
        Returns:
            Dictionary with test results and p-value
        """
        if algo1 not in self.results or algo2 not in self.results:
            return {}
        
        data1 = np.array(self.results[algo1].best_solutions)
        data2 = np.array(self.results[algo2].best_solutions)
        
        # Shapiro-Wilk test for normality
        stat1, p1 = stats.shapiro(data1)
        stat2, p2 = stats.shapiro(data2)
        
        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(data1, data2)
        
        # Wilcoxon test (non-parametric)
        w_stat, w_pval = stats.wilcoxon(data1, data2)
        
        return {
            'algorithm1': algo1,
            'algorithm2': algo2,
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_pvalue': w_pval,
            'significant': t_pval < alpha,
            'better_algo': algo1 if t_stat > 0 else algo2
        }
    
    def print_hypothesis_test(self, algo1: str, algo2: str, alpha: float = 0.05):
        """Print hypothesis testing results."""
        result = self.hypothesis_test(algo1, algo2, alpha)
        
        if not result:
            print(f"Cannot compare - missing results")
            return
        
        print(f"\n{'='*70}")
        print(f"STATISTICAL HYPOTHESIS TEST: {algo1} vs {algo2}")
        print(f"{'='*70}")
        
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {result['t_statistic']:.6f}")
        print(f"  p-value: {result['t_pvalue']:.6f}")
        print(f"  Significant (alpha={alpha}): {result['significant']}")
        
        if result['significant']:
            print(f"  --> {result['better_algo']} is significantly better (p < {alpha})")
        else:
            print(f"  --> No significant difference (p >= {alpha})")
        
        print(f"\nWilcoxon signed-rank test (non-parametric):")
        print(f"  W-statistic: {result['wilcoxon_statistic']:.6f}")
        print(f"  p-value: {result['wilcoxon_pvalue']:.6f}")
        print(f"{'='*70}\n")

    def plot_comparison(self, output_dir: str = "benchmark_plots"):
        """Create comparison plots for all algorithms."""
        Path(output_dir).mkdir(exist_ok=True)
        
        if not self.results:
            print("No results to plot")
            return
        
        # If only 1 algorithm, use single algorithm plots
        if len(self.results) == 1:
            algo_name = list(self.results.keys())[0]
            self.plot_single_algorithm(algo_name, output_dir)
            return
        
        # Plot 1: Solution Quality Comparison
        self._plot_solution_quality(output_dir)
        
        # Plot 2: Execution Time Comparison
        self._plot_execution_time(output_dir)
        
        # Plot 3: Convergence Curves
        self._plot_convergence_curves(output_dir)
        
        # Plot 4: Robustness (Box plot)
        self._plot_robustness(output_dir)
    
    def plot_single_algorithm(self, algorithm_name: str, output_dir: str = "benchmark_plots"):
        """Create detailed plots for a single algorithm."""
        Path(output_dir).mkdir(exist_ok=True)
        
        if algorithm_name not in self.results:
            print(f"No results for {algorithm_name}")
            return
        
        result = self.results[algorithm_name]
        metrics = result.get_metrics()
        
        # Plot 1: Solution Quality (Distribution)
        self._plot_single_solution_quality(algorithm_name, result, output_dir)
        
        # Plot 2: Execution Time Distribution
        self._plot_single_execution_time(algorithm_name, result, output_dir)
        
        # Plot 3: Convergence Curves
        self._plot_single_convergence_curves(algorithm_name, result, output_dir)
        
        # Plot 4: Robustness (Box plot)
        self._plot_single_robustness(algorithm_name, result, output_dir)
        
        print(f"\nDetailed plots for {algorithm_name} saved to '{output_dir}/' directory")
    
    def _plot_single_solution_quality(self, algo_name: str, result: 'BenchmarkResult', output_dir: str):
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
        ax1.set_title(f'{algo_name}: Solution Quality Distribution', fontsize=12, fontweight='bold')
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
        plt.savefig(f"{output_dir}/01_solution_quality.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: 01_solution_quality.png")
    
    def _plot_single_execution_time(self, algo_name: str, result: 'BenchmarkResult', output_dir: str):
        """Plot execution time for a single algorithm."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        metrics = result.get_metrics()
        times = result.times
        
        # Left plot: Histogram of execution times
        ax1.hist(times, bins=10, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax1.axvline(metrics['mean_time'], color='#2ecc71', linestyle='--', linewidth=2, label=f"Mean: {metrics['mean_time']:.4f}s")
        ax1.set_xlabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title(f'{algo_name}: Execution Time Distribution', fontsize=12, fontweight='bold')
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
        plt.savefig(f"{output_dir}/02_execution_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: 02_execution_time.png")
    
    def _plot_single_convergence_curves(self, algo_name: str, result: 'BenchmarkResult', output_dir: str):
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
        ax.set_title(f'{algo_name}: Convergence Curves ({result.num_runs} runs)', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/03_convergence_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: 03_convergence_curves.png")
    
    def _plot_single_robustness(self, algo_name: str, result: 'BenchmarkResult', output_dir: str):
        """Plot robustness analysis (box plot) for a single algorithm."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        metrics = result.get_metrics()
        best_solutions = result.best_solutions
        
        # Left plot: Box plot
        bp = ax1.boxplot([best_solutions], labels=[algo_name], patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][0].set_alpha(0.7)
        ax1.set_ylabel('Fitness Value', fontsize=11, fontweight='bold')
        ax1.set_title(f'{algo_name}: Robustness (Box Plot)', fontsize=12, fontweight='bold')
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
        ax2.set_title(f'{algo_name}: Solution Distribution (Violin Plot)', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/04_robustness_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: 04_robustness_boxplot.png")
        
        print(f"\nPlots saved to '{output_dir}/' directory")
    
    def _plot_solution_quality(self, output_dir: str):
        """Plot best solution quality for each algorithm."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algo_names = list(self.results.keys())
        best_values = [self.results[name].get_metrics()['best_found'] for name in algo_names]
        mean_values = [self.results[name].get_metrics()['average_best'] for name in algo_names]
        
        x = np.arange(len(algo_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, best_values, width, label='Best Found', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width/2, mean_values, width, label='Mean Found', color='#3498db', alpha=0.8)
        
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fitness Value', fontsize=12, fontweight='bold')
        ax.set_title('Solution Quality Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algo_names)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2e}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/01_solution_quality.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_execution_time(self, output_dir: str):
        """Plot execution time for each algorithm."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algo_names = list(self.results.keys())
        times = [self.results[name].get_metrics()['mean_time'] for name in algo_names]
        stds = [self.results[name].get_metrics()['std_time'] for name in algo_names]
        
        colors = ['#e74c3c', '#f39c12', '#9b59b6']
        bars = ax.bar(algo_names, times, color=colors[:len(algo_names)], alpha=0.7, 
                     edgecolor='black', linewidth=1.5)
        ax.errorbar(algo_names, times, yerr=stds, fmt='none', ecolor='black', capsize=5, capthick=2)
        
        ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Execution Time Comparison (30 runs)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, time) in enumerate(zip(bars, times)):
            ax.text(bar.get_x() + bar.get_width()/2., time,
                    f'{time:.4f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/02_execution_time.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_convergence_curves(self, output_dir: str):
        """Plot convergence curves for each algorithm."""
        fig, axes = plt.subplots(1, len(self.results), figsize=(5*len(self.results), 5), sharey=True)
        
        if len(self.results) == 1:
            axes = [axes]
        
        algo_names = list(self.results.keys())
        colors_line = ['#2ecc71', '#3498db', '#e74c3c']
        
        for idx, algo_name in enumerate(algo_names):
            ax = axes[idx]
            result = self.results[algo_name]
            
            # Plot convergence curves for first 5 runs
            for run_idx in range(min(5, len(result.convergence_curves))):
                curve = result.convergence_curves[run_idx]
                ax.plot(curve, alpha=0.5, linewidth=1, color=colors_line[idx % len(colors_line)])
            
            # Plot average curve
            avg_curve = np.mean([c for c in result.convergence_curves], axis=0)
            ax.plot(avg_curve, 'k-', linewidth=2.5, label='Average')
            
            ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('Fitness Value', fontsize=11, fontweight='bold')
            
            ax.set_title(f'{algo_name} Convergence', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/03_convergence_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_robustness(self, output_dir: str):
        """Plot robustness comparison (box plot)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algo_names = list(self.results.keys())
        data = [self.results[name].best_solutions for name in algo_names]
        
        bp = ax.boxplot(data, labels=algo_names, patch_artist=True, 
                       showmeans=True, meanline=True)
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Best Fitness Value', fontsize=12, fontweight='bold')
        ax.set_title('Algorithm Robustness (30 runs)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/04_robustness_boxplot.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_scalability(self, scalability_results: Dict, output_dir: str = "benchmark_plots"):
        """Plot scalability analysis across dimensions."""
        Path(output_dir).mkdir(exist_ok=True)
        
        if not scalability_results:
            print("No scalability results to plot")
            return
        
        # Determine grid size based on number of algorithms
        num_algos = len(scalability_results)
        if num_algos == 1:
            rows, cols = 1, 1
            figsize = (8, 6)
        elif num_algos == 2:
            rows, cols = 1, 2
            figsize = (14, 5)
        elif num_algos == 3:
            rows, cols = 2, 2
            figsize = (14, 10)
        else:
            rows, cols = 2, 2
            figsize = (14, 10)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Handle single subplot case (axes is 1D array instead of 2D)
        if num_algos == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (algo_name, dim_data) in enumerate(scalability_results.items()):
            if idx >= num_algos:
                break
            
            ax = axes[idx]
            
            dimensions = sorted(dim_data.keys())
            best_values = [dim_data[d]['best'] for d in dimensions]
            mean_values = [dim_data[d]['mean'] for d in dimensions]
            times = [dim_data[d]['time'] for d in dimensions]
            
            ax_twin = ax.twinx()
            
            # Plot solution quality
            line1 = ax.plot(dimensions, best_values, 'o-', color='#2ecc71', linewidth=2.5, 
                           markersize=8, label='Best Found')
            line2 = ax.plot(dimensions, mean_values, 's--', color='#3498db', linewidth=2, 
                           markersize=6, label='Mean Found')
            
            # Plot execution time
            line3 = ax_twin.plot(dimensions, times, '^-', color='#e74c3c', linewidth=2, 
                                markersize=6, label='Time (s)')
            
            ax.set_xlabel('Dimension', fontsize=11, fontweight='bold')
            ax.set_ylabel('Fitness Value', fontsize=11, fontweight='bold', color='#2c3e50')
            ax_twin.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold', color='#e74c3c')
            ax.set_title(f'{algo_name} Scalability', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            # Combine legends
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', fontsize=9)
        
        # Hide empty subplots if less than 4 algorithms
        for idx in range(num_algos, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/05_scalability_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()


class ParameterSensitivityAnalyzer:
    """Analyze parameter sensitivity of an optimization algorithm."""
    
    def __init__(self, problem_class, dimensions: int = 5, num_runs: int = 5):
        self.problem_class = problem_class
        self.dimensions = dimensions
        self.num_runs = num_runs
        self.sensitivity_results = {}
    
    def analyze_parameter(self, algorithm_class, param_name: str, param_values: List, 
                         base_params: Dict, output_dir: str = "sensitivity_plots") -> Dict:
        """
        Analyze how a single parameter affects algorithm performance.
        
        Args:
            algorithm_class: Algorithm class to test
            param_name: Name of parameter to vary
            param_values: List of values to test for the parameter
            base_params: Base parameters for the algorithm
            output_dir: Directory to save plots
            
        Returns:
            Dictionary with results for each parameter value
        """
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {
            'param_name': param_name,
            'param_values': param_values,
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'exec_times': []
        }
        
        print(f"\n{'='*70}")
        print(f"PARAMETER SENSITIVITY ANALYSIS: {param_name}")
        print(f"{'='*70}")
        print(f"Testing values: {param_values}\n")
        
        for param_value in param_values:
            # Create custom parameters with varied value
            custom_params = base_params.copy()
            custom_params[param_name] = param_value
            
            # Run algorithm multiple times
            best_values = []
            mean_values = []
            times = []
            
            for run in range(self.num_runs):
                try:
                    problem = self.problem_class(self.dimensions)
                    algorithm = algorithm_class(max_iterations=100, **custom_params)
                    
                    start_time = time.time()
                    result = algorithm.search(problem)
                    elapsed_time = time.time() - start_time
                    
                    best_cost = result.get('cost', result.get('best_cost', float('inf')))
                    best_values.append(best_cost)
                    mean_values.append(best_cost)
                    times.append(elapsed_time)
                    
                except Exception as e:
                    print(f"  Error with {param_name}={param_value}: {str(e)}")
                    continue
            
            if best_values:
                results['best_fitness'].append(np.min(best_values))
                results['mean_fitness'].append(np.mean(best_values))
                results['std_fitness'].append(np.std(best_values))
                results['exec_times'].append(np.mean(times))
                
                print(f"{param_name}={param_value}: "
                      f"Best={np.min(best_values):.6e}, "
                      f"Mean={np.mean(best_values):.6e}, "
                      f"Time={np.mean(times):.4f}s")
        
        self.sensitivity_results[param_name] = results
        self._plot_sensitivity(param_name, results, output_dir)
        
        return results
    
    def analyze_multiple_parameters(self, algorithm_class, param_grid: Dict[str, List],
                                   base_params: Dict, output_dir: str = "sensitivity_plots"):
        """
        Analyze multiple parameters sequentially.
        
        Args:
            algorithm_class: Algorithm class to test
            param_grid: Dictionary of parameter names and their values
            base_params: Base parameters for the algorithm
            output_dir: Directory to save plots
        """
        print("\n" + "="*70)
        print("PARAMETER SENSITIVITY ANALYSIS (Multiple Parameters)")
        print("="*70)
        
        for param_name, param_values in param_grid.items():
            self.analyze_parameter(algorithm_class, param_name, param_values, 
                                 base_params, output_dir)
    
    def _plot_sensitivity(self, param_name: str, results: Dict, output_dir: str):
        """Create plots for parameter sensitivity."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        param_values = results['param_values']
        best_fitness = results['best_fitness']
        mean_fitness = results['mean_fitness']
        std_fitness = results['std_fitness']
        
        # Plot 1: Fitness vs Parameter Value
        ax1 = axes[0]
        ax1.errorbar(range(len(param_values)), mean_fitness, yerr=std_fitness, 
                    fmt='o-', color='#3498db', ecolor='#95a5a6', capsize=5, 
                    capthick=2, linewidth=2, markersize=8, label='Mean ± Std')
        ax1.plot(range(len(param_values)), best_fitness, 's--', color='#2ecc71', 
                linewidth=2, markersize=6, label='Best')
        
        ax1.set_xlabel(f'{param_name}', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Fitness Value', fontsize=12, fontweight='bold')
        ax1.set_title(f'Solution Quality vs {param_name}', fontsize=13, fontweight='bold')
        ax1.set_xticks(range(len(param_values)))
        ax1.set_xticklabels([f'{v}' for v in param_values], rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.legend(fontsize=10)
        
        # Plot 2: Execution Time vs Parameter Value
        ax2 = axes[1]
        exec_times = results['exec_times']
        bars = ax2.bar(range(len(param_values)), exec_times, color='#e74c3c', alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel(f'{param_name}', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Execution Time vs {param_name}', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(param_values)))
        ax2.set_xticklabels([f'{v}' for v in param_values], rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, time_val in zip(bars, exec_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        safe_param_name = param_name.replace(' ', '_').lower()
        plt.savefig(f"{output_dir}/sensitivity_{safe_param_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Plot saved: sensitivity_{safe_param_name}.png")
    
    def generate_heatmap(self, output_dir: str = "sensitivity_plots"):
        """Generate heatmap of parameter sensitivity across multiple parameters."""
        if not self.sensitivity_results:
            print("No sensitivity results to plot")
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        param_names = list(self.sensitivity_results.keys())
        max_values = max(len(v['param_values']) for v in self.sensitivity_results.values())
        
        # Create matrix
        matrix = np.zeros((len(param_names), max_values))
        labels_x = []
        
        for i, param_name in enumerate(param_names):
            results = self.sensitivity_results[param_name]
            fitness_values = results['best_fitness']
            
            for j, fitness in enumerate(fitness_values):
                matrix[i, j] = fitness
            
            if i == 0:
                labels_x = [f'{v}' for v in results['param_values']]
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
        
        ax.set_yticks(range(len(param_names)))
        ax.set_yticklabels(param_names, fontsize=11)
        ax.set_xticks(range(len(labels_x)))
        ax.set_xticklabels(labels_x, rotation=45, ha='right')
        
        ax.set_xlabel('Parameter Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Parameter Name', fontsize=12, fontweight='bold')
        ax.set_title('Parameter Sensitivity Heatmap', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Best Fitness Value (log scale)', fontsize=11)
        
        # Add values to cells
        for i in range(len(param_names)):
            for j in range(len(labels_x)):
                if matrix[i, j] > 0:
                    text = ax.text(j, i, f'{matrix[i, j]:.2e}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sensitivity_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nHeatmap saved: sensitivity_heatmap.png")
