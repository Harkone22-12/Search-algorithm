"""
Lab 2: Particle Swarm Optimization (PSO) Sensitivity Analysis

Mục tiêu:
Phân tích độ nhạy của các tham số trong thuật toán PSO trên hai hàm mục tiêu
kinh điển là Sphere và Rastrigin, sử dụng benchmark framework có sẵn.

Quy trình thực hiện:
1. Chuẩn bị:
   - Import các class Problem (Sphere, Rastrigin) và Algorithm (PSO) có sẵn.
   - Import ParameterSensitivityAnalyzer từ benchmark framework.

2. Chạy phân tích:
   - Lặp qua từng bài toán (Sphere, Rastrigin).
   - Khởi tạo ParameterSensitivityAnalyzer cho mỗi bài toán.
   - Định nghĩa các tham số cơ sở và lưới tham số (parameter grid) để quét.
   - Gọi phương thức `analyze_multiple_parameters` để tự động chạy thử nghiệm,
     thu thập dữ liệu và tạo biểu đồ phân tích độ nhạy.
   - Tạo heatmap để trực quan hóa tổng quan độ nhạy.

3. Tổng kết:
   - Kết quả và biểu đồ được tự động lưu vào thư mục output.
"""

import sys
from pathlib import Path

# Add project root to path to resolve imports
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import framework, problems, and algorithm for consistency
from Source.Problems.Continuous.Tests.benchmark_framework import ParameterSensitivityAnalyzer
from Source.Problems.Continuous.Functions.Sphere import SphereProblem
from Source.Problems.Continuous.Functions.Rastrigin import RastriginProblem
from Source.Search.Nature_Inspired.Biology_Based.PSO import ParticleSwarmOptimization

# --- MAIN EXECUTION ---
def main():
    """
    Runs a parameter sensitivity analysis for PSO on Sphere and Rastrigin functions
    using the project's benchmarking framework.
    """
    # --- Configuration ---
    DIMENSIONS = 10
    NUM_RUNS = 30
    MAX_ITER = 100
    OUTPUT_DIR_BASE = Path("Source/Problems/Continuous/Tests/Lab2_Results")

    # --- Problems to Test ---
    problems = {
        "Sphere": SphereProblem,
        "Rastrigin": RastriginProblem
    }

    # --- PSO Parameters for Sensitivity Analysis ---
    # Base parameters matching the original experiment
    base_params = {
        'population_size': 50, 
        'w': 0.729, 
        'c1': 1.494, 
        'c2': 1.494,
        'max_iterations': MAX_ITER # Use 'max_iterations' for the framework's algorithm classes
    }
    
    # Grid of parameter values to test
    param_grid = {
        "w": [0.4, 0.6, 0.729, 0.9, 1.2], 
        "population_size": [10, 20, 50, 100, 200]
    }

    # --- Run Analysis for Each Problem ---
    for func_name, problem_class in problems.items():
        print("\n" + "="*80)
        print(f"ANALYZING PSO SENSITIVITY ON: {func_name.upper()} FUNCTION")
        print("="*80)

        # Create a dedicated output directory for this problem's results
        output_dir = OUTPUT_DIR_BASE / func_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved to: {output_dir}")

        # Initialize the analyzer for the current problem
        analyzer = ParameterSensitivityAnalyzer(
            problem_class=problem_class,
            dimensions=DIMENSIONS,
            num_runs=NUM_RUNS
        )

        # Run the analysis for all parameters in the grid.
        # This will automatically generate and save plots for each parameter.
        analyzer.analyze_multiple_parameters(
            algorithm_class=ParticleSwarmOptimization,
            param_grid=param_grid,
            base_params=base_params,
            output_dir=str(output_dir)
        )
        
        # Optionally, generate a heatmap for a visual overview of all tested parameters
        print(f"\nGenerating sensitivity heatmap for {func_name}...")
        analyzer.generate_heatmap(str(output_dir))

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETE.")
    print(f"-> All results and plots saved in subdirectories under: {OUTPUT_DIR_BASE}")

if __name__ == "__main__":
    main()