"""
Parameter Sensitivity Analysis Tool

Allows testing a single algorithm with varying parameters to understand
how parameter values affect algorithm performance.

Usage:
    python -m Source.Problems.Continuous.test_parameter_sensitivity ABC
    python -m Source.Problems.Continuous.test_parameter_sensitivity Firefly
    python -m Source.Problems.Continuous.test_parameter_sensitivity Cuckoo
    python -m Source.Problems.Continuous.test_parameter_sensitivity SA
"""

import sys
import importlib.util
import numpy as np
from Source.Problems.Continuous.Sphere import SphereProblem
from Source.Problems.Continuous.benchmark_framework import ParameterSensitivityAnalyzer


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


def test_abc():
    """Parameter sensitivity for ABC."""
    print("\n" + "="*80)
    print("ABC PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    analyzer = ParameterSensitivityAnalyzer(
        problem_class=SphereProblem,
        dimensions=5,
        num_runs=5
    )
    
    base_params = {
        "colony_size": 30,
        "limit": 100,
        "seed": 42
    }
    
    # Parameter grid for ABC
    param_grid = {
        "colony_size": [10, 20, 30, 40, 50],
        "limit": [50, 100, 150, 200]
    }
    
    analyzer.analyze_multiple_parameters(
        ArtificialBeeColony,
        param_grid,
        base_params,
        output_dir="Source/Problems/Continuous/sensitivity_plots/ABC"
    )
    
    print("\nGenerating sensitivity heatmap...")
    analyzer.generate_heatmap("Source/Problems/Continuous/sensitivity_plots/ABC")


def test_firefly():
    """Parameter sensitivity for Firefly Algorithm."""
    print("\n" + "="*80)
    print("FIREFLY ALGORITHM PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    analyzer = ParameterSensitivityAnalyzer(
        problem_class=SphereProblem,
        dimensions=5,
        num_runs=5
    )
    
    base_params = {
        "population_size": 30,
        "alpha": 0.5,
        "beta0": 1.0,
        "gamma": 0.01,
        "seed": 42
    }
    
    # Parameter grid for Firefly
    param_grid = {
        "population_size": [15, 25, 30, 40, 50],
        "alpha": [0.1, 0.3, 0.5, 0.7, 0.9],
        "gamma": [0.001, 0.01, 0.05, 0.1, 0.2]
    }
    
    analyzer.analyze_multiple_parameters(
        FireflyAlgorithm,
        param_grid,
        base_params,
        output_dir="Source/Problems/Continuous/sensitivity_plots/Firefly"
    )
    
    print("\nGenerating sensitivity heatmap...")
    analyzer.generate_heatmap("Source/Problems/Continuous/sensitivity_plots/Firefly")


def test_cuckoo():
    """Parameter sensitivity for Cuckoo Search."""
    print("\n" + "="*80)
    print("CUCKOO SEARCH PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    analyzer = ParameterSensitivityAnalyzer(
        problem_class=SphereProblem,
        dimensions=5,
        num_runs=5
    )
    
    base_params = {
        "population_size": 25,
        "pa": 0.25,
        "seed": 42
    }
    
    # Parameter grid for Cuckoo Search
    param_grid = {
        "population_size": [15, 20, 25, 35, 50],
        "pa": [0.1, 0.2, 0.25, 0.3, 0.4]
    }
    
    analyzer.analyze_multiple_parameters(
        CuckooSearch,
        param_grid,
        base_params,
        output_dir="Source/Problems/Continuous/sensitivity_plots/Cuckoo"
    )
    
    print("\nGenerating sensitivity heatmap...")
    analyzer.generate_heatmap("Source/Problems/Continuous/sensitivity_plots/Cuckoo")


def test_sa():
    """Parameter sensitivity for Simulated Annealing."""
    print("\n" + "="*80)
    print("SIMULATED ANNEALING PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    analyzer = ParameterSensitivityAnalyzer(
        problem_class=SphereProblem,
        dimensions=5,
        num_runs=5
    )
    
    base_params = {
        "initial_temperature": 100.0,
        "cooling_rate": 0.95,
        "min_temperature": 0.01,
        "seed": 42
    }
    
    # Parameter grid for SA
    param_grid = {
        "initial_temperature": [10.0, 50.0, 100.0, 500.0, 1000.0],
        "cooling_rate": [0.85, 0.90, 0.95, 0.98, 0.99],
        "min_temperature": [0.001, 0.01, 0.05, 0.1]
    }
    
    analyzer.analyze_multiple_parameters(
        SimulatedAnnealing,
        param_grid,
        base_params,
        output_dir="Source/Problems/Continuous/sensitivity_plots/SA"
    )
    
    print("\nGenerating sensitivity heatmap...")
    analyzer.generate_heatmap("Source/Problems/Continuous/sensitivity_plots/SA")


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY ANALYSIS TOOL")
    print("="*80)
    print("\nUsage:")
    print("  python -m Source.Problems.Continuous.test_parameter_sensitivity ABC")
    print("  python -m Source.Problems.Continuous.test_parameter_sensitivity Firefly")
    print("  python -m Source.Problems.Continuous.test_parameter_sensitivity Cuckoo")
    print("  python -m Source.Problems.Continuous.test_parameter_sensitivity SA")
    print("\nAlgorithms:")
    print("  ABC        - Artificial Bee Colony")
    print("  Firefly    - Firefly Algorithm")
    print("  Cuckoo     - Cuckoo Search")
    print("  SA         - Simulated Annealing")
    print("\nOutput:")
    print("  Sensitivity plots are saved to: Source/Problems/Continuous/sensitivity_plots/<Algorithm>/")
    print("  Files include:")
    print("    - sensitivity_<parameter>.png (one for each parameter)")
    print("    - sensitivity_heatmap.png (overview of all parameters)")
    print("="*80 + "\n")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print_usage()
        print("Error: Please specify an algorithm name")
        print("Example: python -m Source.Problems.Continuous.test_parameter_sensitivity ABC")
        sys.exit(1)
    
    algorithm = sys.argv[1].upper()
    
    print("\n" + "="*80)
    print("OPTIMIZATION ALGORITHM PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"\nTesting algorithm: {algorithm}")
    print(f"Problem: Sphere Function (5D)")
    print(f"Runs per parameter value: 5")
    print(f"Output directory: Source/Problems/Continuous/sensitivity_plots/{algorithm}/")
    
    if algorithm == "ABC":
        test_abc()
    elif algorithm == "FIREFLY":
        test_firefly()
    elif algorithm == "CUCKOO":
        test_cuckoo()
    elif algorithm == "SA":
        test_sa()
    else:
        print(f"Unknown algorithm: {algorithm}")
        print_usage()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: sensitivity_plots/{algorithm}/")


if __name__ == "__main__":
    main()
