import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to the Python path to resolve imports when running the script directly.
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import Problems từ đúng đường dẫn thư mục Functions của bạn
from Source.Problems.Continuous.Functions.Sphere import SphereProblem
from Source.Problems.Continuous.Functions.Rastrigin import RastriginProblem

# Import Benchmark Framework của bạn
from Source.Problems.Continuous.Tests.benchmark_framework import AlgorithmBenchmark

# Import các thuật toán thực tế đã có trong source code
from Source.Search.Nature_Inspired.Biology_Based.ABC import ArtificialBeeColony
from Source.Search.Nature_Inspired.Biology_Based.FA import FireflyAlgorithm
from Source.Search.Nature_Inspired.Biology_Based.PSO import ParticleSwarmOptimization
from Source.Search.Nature_Inspired.Biology_Based.Cuckoo_Search import CuckooSearch
from Source.Search.Nature_Inspired.Evolution_Based.GA import GeneticAlgorithm
from Source.Search.Nature_Inspired.Evolution_Based.DE import DifferentialEvolution
from Source.Search.Nature_Inspired.Physics_Based.Hill_Climbing import HillClimbing
from Source.Search.Nature_Inspired.Physics_Based.Stimulated_Annealing import SimulatedAnnealing



def run_lab1():
    # 1. Thiết lập thông số thực nghiệm
    dimensions = 5
    max_iter = 100
    num_runs = 10  # Số lần chạy để tính Mean và Std
    output_dir = "Source/Problems/Continuous/Tests/Lab1_Results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 2. Danh sách các bài toán (Sử dụng đúng class từ Sphere.py và Rastrigin.py)
    problems = [
        {"name": "Sphere", "class": SphereProblem},
        {"name": "Rastrigin", "class": RastriginProblem}
    ]

    # 3. Danh sách các thuật toán (Sử dụng đúng class từ source của bạn)
    # Các tham số truyền vào đúng với hàm __init__ của từng class
    algorithms = [
        {
            "name": "ABC", 
            "class": ArtificialBeeColony, 
            "params": {"colony_size": 40, "limit": 50}
        },
        {
            "name": "Firefly", 
            "class": FireflyAlgorithm, 
            "params": {"population_size": 40, "alpha": 0.2, "beta0": 1.0, "gamma": 1.0}
        },
        {
            "name": "SimulatedAnnealing", 
            "class": SimulatedAnnealing, 
            "params": {"initial_temperature": 1000.0, "cooling_rate": 0.95}
        },
        {
            "name": "GA",
            "class": GeneticAlgorithm,
            "params": {"population_size": 40, "mutation_rate": 0.1, "crossover_rate": 0.8}
        },
        {
            "name": "DE",
            "class": DifferentialEvolution,
            "params": {"population_size": 40, "F": 0.8, "CR": 0.9}
        }
        ,
        {
            "name": "PSO",
            "class": ParticleSwarmOptimization,
            "params": {"population_size": 40, "w": 0.7, "c1": 1.5, "c2": 1.5}
        },
        {
            "name": "CuckooSearch",
            "class": CuckooSearch,
            "params": {"population_size": 40, "pa": 0.25}
        },
        {
            "name": "HillClimbing",
            "class": HillClimbing,
            "params": {}
        }
    ]

    for p_info in problems:
        print(f"\n>>> Đang thực hiện Lab 1 trên hàm: {p_info['name']}")
        
        # Khởi tạo benchmark cho hàm mục tiêu hiện tại
        benchmark = AlgorithmBenchmark(
            problem_class=p_info['class'],
            problem_name=p_info['name'],
            dimensions=dimensions,
            max_iterations=max_iter,
            num_runs=num_runs
        )

        # Chạy từng thuật toán đã định nghĩa
        for algo in algorithms:
            print(f"    Đang chạy {algo['name']}...")
            benchmark.run_algorithm(
                algo['class'], 
                algo['name'], 
                **algo['params']
            )

        # Xuất kết quả so sánh định lượng ra Terminal (Best, Mean, Std, Time)
        benchmark.compare_algorithms(*[a['name'] for a in algorithms])

        # Vẽ và lưu biểu đồ hội tụ vào thư mục kết quả
        print(f"\n>>> Generating plots for {p_info['name']}...")
        benchmark.plot_comparison(
            output_dir=output_dir, file_prefix=p_info['name']
        )
        print(f">>> Plots saved to directory: {output_dir}")

if __name__ == "__main__":
    run_lab1()