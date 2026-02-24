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
from pathlib import Path

from . import benchmark_visualizer as viz


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


class AlgorithmBenchmark:
    """Framework for benchmarking algorithms."""
    
    def __init__(self, problem_class, dimensions: int = 5, max_iterations: int = 100, num_runs: int = 30, problem_name: str = ""):
        self.problem_class = problem_class
        self.problem_name = problem_name or problem_class.__name__.replace("Problem", "")
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
    
    def compare_algorithms(self, *algorithm_names):
        """Compare multiple algorithms."""
        print(f"\n{'='*80}")
        print(f"ALGORITHM COMPARISON ({self.problem_name} - {self.dimensions}D)")
        print(f"{'='*80}")
        print(f"{'Algorithm':<20} {'Best':<12} {'Mean':<12} {'Std':<12} {'Time(s)':<12} {'Conv.Iter':<12}")
        print(f"{'-'*80}")
        
        # Ensure we only print unique algorithms, sorted for consistency
        unique_names = sorted(list(set(algorithm_names)))

        for name in unique_names:
            if name not in self.results:
                continue
            result = self.results[name]
            metrics = result.get_metrics()
            
            print(f"{name:<20} {metrics['best_found']:<12.6f} {metrics['average_best']:<12.6f} "
                  f"{metrics['std_fitness']:<12.6f} {metrics['mean_time']:<12.4f} "
                  f"{metrics['mean_convergence_iterations']:<12.1f}")
        
        print(f"{'='*80}\n")
    
    def plot_comparison(self, output_dir: str = "benchmark_plots", file_prefix: str = ""):
        """Create comparison plots for all algorithms by calling the visualizer."""
        Path(output_dir).mkdir(exist_ok=True)
        
        if not self.results:
            print("No results to plot")
            return
        
        # Use problem_name for titles, which is often the same as file_prefix
        plot_title_prefix = file_prefix or self.problem_name

        if len(self.results) == 1:
            algo_name = list(self.results.keys())[0]
            result = self.results[algo_name]
            viz.plot_single_algorithm_analysis(result, algo_name, plot_title_prefix, output_dir)
        else:
            viz.plot_algorithm_comparison(self.results, self.max_iterations, plot_title_prefix, output_dir, file_prefix)


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

                    # Prepare parameters, ensuring max_iterations is set without duplication.
                    algo_params = custom_params.copy()
                    if 'max_iterations' not in algo_params:
                        algo_params['max_iterations'] = 100
                    algorithm = algorithm_class(**algo_params)

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
        viz.plot_parameter_sensitivity(param_name, results, output_dir)
        
        return results
    
    def analyze_multiple_parameters(self, algorithm_class, param_grid: Dict[str, List],
                                   base_params: Dict, output_dir: str = "sensitivity_plots"):
        """Analyze multiple parameters sequentially."""
        print("\n" + "="*70)
        print("PARAMETER SENSITIVITY ANALYSIS (Multiple Parameters)")
        print("="*70)
        
        for param_name, param_values in param_grid.items():
            self.analyze_parameter(algorithm_class, param_name, param_values, 
                                 base_params, output_dir)
    
    def generate_heatmap(self, output_dir: str = "sensitivity_plots"):
        """Generate heatmap of parameter sensitivity by calling the visualizer."""
        viz.plot_sensitivity_heatmap(self.sensitivity_results, output_dir)
