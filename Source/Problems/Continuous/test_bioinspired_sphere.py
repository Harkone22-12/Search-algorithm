"""
Test Suite for Bio-inspired Algorithms on Sphere Function

Benchmarks:
- Artificial Bee Colony (ABC)
- Firefly Algorithm (FA)
- Cuckoo Search (CS)- Simulated Annealing (SA)
Using evaluation criteria:
1. Convergence Speed
2. Solution Quality
3. Computational Complexity
4. Robustness (Mean ± Std)
5. Scalability
6. Exploration vs Exploitation
"""

import numpy as np
from Source.Problems.Continuous.Sphere import SphereProblem
from Source.Problems.Continuous.benchmark_framework import AlgorithmBenchmark
from Source.Search.Search import SearchAlgorithm
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem

# Import algorithms - using direct module import to handle special filenames
import sys
import importlib.util

def load_algorithm_from_file(filepath, module_name):
    """Load algorithm from file with special characters in name."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load algorithms
abc_path = "Source/Search/Nature_Inspired/Biology-Based/ABC.py"
fa_path = "Source/Search/Nature_Inspired/Biology-Based/FA.py"
cs_path = "Source/Search/Nature_Inspired/Biology-Based/Cuckoo Search.py"
sa_path = "Source/Search/Nature_Inspired/Physics-Based/Stimulated Annealing.py"

abc_module = load_algorithm_from_file(abc_path, "abc_module")
fa_module = load_algorithm_from_file(fa_path, "fa_module")
cs_module = load_algorithm_from_file(cs_path, "cs_module")
sa_module = load_algorithm_from_file(sa_path, "sa_module")

ArtificialBeeColony = abc_module.ArtificialBeeColony
FireflyAlgorithm = fa_module.FireflyAlgorithm
CuckooSearch = cs_module.CuckooSearch
SimulatedAnnealing = sa_module.SimulatedAnnealing

# Algorithm definitions
ALGORITHMS = {
    "ABC": {
        "class": ArtificialBeeColony,
        "display_name": "Artificial Bee Colony",
        "params": {"colony_size": 30, "limit": 100, "seed": 42}
    },
    "Firefly": {
        "class": FireflyAlgorithm,
        "display_name": "Firefly Algorithm",
        "params": {"population_size": 30, "alpha": 0.5, "beta0": 1.0, "gamma": 0.01, "seed": 42}
    },
    "Cuckoo": {
        "class": CuckooSearch,
        "display_name": "Cuckoo Search",
        "params": {"population_size": 25, "pa": 0.25, "seed": 42}
    },
    "SA": {
        "class": SimulatedAnnealing,
        "display_name": "Simulated Annealing",
        "params": {"initial_temperature": 100.0, "cooling_rate": 0.95, "min_temperature": 0.01, "seed": 42}
    }
}


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*80)
    print("TEST SUITE FOR BIO-INSPIRED ALGORITHMS ON SPHERE FUNCTION")
    print("="*80)
    print("\nUsage:")
    print("  python -m Source.Problems.Continuous.test_bioinspired_sphere")
    print("    → Run all algorithms (ABC, Firefly, Cuckoo, SA)")
    print("\n  python -m Source.Problems.Continuous.test_bioinspired_sphere ABC Firefly")
    print("    → Run only ABC and Firefly")
    print("\n  python -m Source.Problems.Continuous.test_bioinspired_sphere ABC")
    print("    → Run only ABC")
    print("\nAvailable Algorithms:")
    for algo_name, algo_info in ALGORITHMS.items():
        print(f"  {algo_name:12} - {algo_info['display_name']}")
    print("="*80 + "\n")


def parse_arguments():
    """Parse command line arguments and return list of algorithms to test."""
    if len(sys.argv) > 1:
        selected_algos = [arg for arg in sys.argv[1:]]
        
        # Validate algorithm names (case-insensitive)
        valid_algos = list(ALGORITHMS.keys())
        normalized_algos = []
        invalid_algos = []
        
        for algo in selected_algos:
            # Find matching algorithm (case-insensitive)
            match = next((a for a in valid_algos if a.upper() == algo.upper()), None)
            if match:
                normalized_algos.append(match)
            else:
                invalid_algos.append(algo)
        
        if invalid_algos:
            print(f"\nError: Unknown algorithm(s): {', '.join(invalid_algos)}")
            print(f"Valid algorithms: {', '.join(valid_algos)}")
            sys.exit(1)
        
        return normalized_algos
    else:
        # Default: run all algorithms
        return list(ALGORITHMS.keys())


def run_algorithms(benchmark, selected_algos):
    """Run selected algorithms and return their names."""
    print("\n" + "="*80)
    print("RUNNING ALGORITHMS")
    print("="*80)
    
    algo_list = [algo for algo in ALGORITHMS.keys() if algo in selected_algos]
    
    for idx, algo_name in enumerate(algo_list, 1):
        algo_info = ALGORITHMS[algo_name]
        print(f"\n[{idx}/{len(algo_list)}] Running {algo_info['display_name']}...")
        benchmark.run_algorithm(
            algo_info["class"],
            algo_name,
            **algo_info["params"]
        )
    
    return algo_list


def print_metrics(benchmark, selected_algos):
    """Print detailed metrics for selected algorithms."""
    print("\n" + "="*80)
    print("DETAILED METRICS")
    print("="*80)
    
    for algo_name in selected_algos:
        benchmark.print_metrics(algo_name)


def run_hypothesis_tests(benchmark, selected_algos):
    """Run statistical hypothesis tests between selected algorithms."""
    if len(selected_algos) < 2:
        print("\nSkipping hypothesis testing (need at least 2 algorithms)")
        return
    
    print("\n" + "="*80)
    print("STATISTICAL HYPOTHESIS TESTING")
    print("="*80)
    
    # Run pairwise tests
    test_count = 1
    for i in range(len(selected_algos)):
        for j in range(i + 1, len(selected_algos)):
            algo1 = selected_algos[i]
            algo2 = selected_algos[j]
            print(f"\n{test_count}. {algo1} vs {algo2}")
            benchmark.print_hypothesis_test(algo1, algo2)
            test_count += 1


def run_scalability_analysis(selected_algos):
    """Run scalability analysis for selected algorithms."""
    print("\n" + "="*80)
    print("SCALABILITY ANALYSIS")
    print("="*80)
    
    scalability_data = {algo: {} for algo in selected_algos}
    dimensions_list = [5, 10, 20, 30]
    
    algo_info_list = [(algo, ALGORITHMS[algo]) for algo in selected_algos]
    
    for dim in dimensions_list:
        print(f"\nTesting with {dim} dimensions...")
        
        for algo_name, algo_info in algo_info_list:
            benchmark_scale = AlgorithmBenchmark(
                problem_class=SphereProblem,
                dimensions=dim,
                max_iterations=100,
                num_runs=10
            )
            
            benchmark_scale.run_algorithm(algo_info["class"], f"{algo_name}-{dim}D", **algo_info["params"])
            
            result = benchmark_scale.results[f"{algo_name}-{dim}D"]
            metrics = result.get_metrics()
            
            scalability_data[algo_name][dim] = {
                'best': metrics['best_found'],
                'mean': metrics['average_best'],
                'time': metrics['mean_time']
            }
            
            print(f"  {algo_name} ({dim}D): Best={metrics['best_found']:.6f}, "
                  f"Mean={metrics['average_best']:.6f}, "
                  f"Time={metrics['mean_time']:.4f}s")
    
    return scalability_data


def main():
    """Run comprehensive benchmark of bio-inspired algorithms."""
    
    # Parse command line arguments
    selected_algos = parse_arguments()
    
    print("="*80)
    print("BENCHMARKING BIO-INSPIRED ALGORITHMS ON SPHERE FUNCTION")
    print("="*80)
    print(f"\nSelected Algorithms: {', '.join(selected_algos)}")
    
    # Configuration
    dimensions = 5
    max_iterations = 100
    num_runs = 10
    
    print(f"\nConfiguration:")
    print(f"  Problem: Sphere Function (f(x) = sum(x_i^2))")
    print(f"  Dimensions: {dimensions}D")
    print(f"  Max Iterations: {max_iterations}")
    print(f"  Number of Runs: {num_runs}")
    print(f"  Search Space: [-5.12, 5.12]")
    print(f"  Optimal Value: 0.0 at (0, 0, ..., 0)")
    
    # Create benchmark
    benchmark = AlgorithmBenchmark(
        problem_class=SphereProblem,
        dimensions=dimensions,
        max_iterations=max_iterations,
        num_runs=num_runs
    )
    
    # Run selected algorithms
    tested_algos = run_algorithms(benchmark, selected_algos)
    
    # Print metrics for selected algorithms
    print_metrics(benchmark, tested_algos)
    
    # Compare algorithms (only if more than 1)
    if len(tested_algos) > 1:
        print("\n" + "="*80)
        print("ALGORITHM COMPARISON")
        print("="*80)
        benchmark.compare_algorithms(*tested_algos)
    
    # Run hypothesis tests
    run_hypothesis_tests(benchmark, tested_algos)
    
    # Generate comparison plots (works for 1+ algorithms now)
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80)
    benchmark.plot_comparison("Source/Problems/Continuous/benchmark_plots")
    
    # Scalability analysis
    scalability_data = run_scalability_analysis(tested_algos)
    
    # Plot scalability
    print("\n" + "="*80)
    print("GENERATING SCALABILITY PLOTS")
    print("="*80)
    benchmark.plot_scalability(scalability_data, "Source/Problems/Continuous/benchmark_plots")
    
    print("\n" + "="*80)
    print("BENCHMARKING COMPLETE")
    print("="*80)
    print(f"\nTested Algorithms: {', '.join(tested_algos)}")
    print(f"Results saved to: Source/Problems/Continuous/benchmark_plots/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
