import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import Problems từ đúng đường dẫn thư mục Functions của bạn
from Source.Problems.Continuous.Functions.Sphere import SphereProblem
from Source.Problems.Continuous.Functions.Rastrigin import RastriginProblem

# Import Benchmark Framework của bạn
from Source.Problems.Continuous.Tests.benchmark_framework import AlgorithmBenchmark

# Import các thuật toán thực tế đã có trong source code
from Source.Search.Nature_Inspired.Biology_Based.ABC import ArtificialBeeColony
from Source.Search.Nature_Inspired.Biology_Based.FA import FireflyAlgorithm
from Source.Search.Nature_Inspired.Physics_Based.Stimulated_Annealing import SimulatedAnnealing

def run_lab1():
    # 1. Thiết lập thông số thực nghiệm
    dimensions = 5
    max_iter = 100
    num_runs = 10  # Số lần chạy để tính Mean và Std
    output_dir = "Source/Problems/Continuous/Lab1_Results"
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
        }
    ]

    for p_info in problems:
        print(f"\n>>> Đang thực hiện Lab 1 trên hàm: {p_info['name']}")
        
        # Khởi tạo benchmark cho hàm mục tiêu hiện tại
        benchmark = AlgorithmBenchmark(
            problem_class=p_info['class'],
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
        save_path = f"{output_dir}/{p_info['name']}_convergence.png"
        benchmark.plot_comparison()
        plt.title(f"Convergence Comparison on {p_info['name']} Function")
        plt.savefig(save_path)
        plt.close()
        print(f">>> Đã lưu biểu đồ tại: {save_path}")

if __name__ == "__main__":
    run_lab1()