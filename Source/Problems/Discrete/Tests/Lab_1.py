import random
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Cấu hình hiển thị Pandas đẹp hơn
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

from benchmark_framework import BenchmarkFramework
from benchmark_visualizer import BenchmarkVisualizer

# Dùng 2 dấu gạch dưới cho __file__
project_root = Path(__file__).resolve().parents[4] 
sys.path.append(str(project_root))

from Source.Problems.Discrete.Functions.Knapsack_Problem import KnapsackProblem
from Source.Problems.Discrete.Functions.TSP import TSPProblem

from Source.Search.Classical.A_Star import AStarSearch
from Source.Search.Classical.BFS import BFS
from Source.Search.Classical.DFS import DFS
from Source.Search.Nature_Inspired.Biology_Based.ABC import ArtificialBeeColony
from Source.Search.Nature_Inspired.Biology_Based.ACO import AntColonyOptimization
from Source.Search.Nature_Inspired.Evolution_Based.GA import GeneticAlgorithm
from Source.Search.Nature_Inspired.Physics_Based.Hill_Climbing import HillClimbing
from Source.Search.Nature_Inspired.Physics_Based.Stimulated_Annealing import SimulatedAnnealing

def main():
    framework = BenchmarkFramework()
    # THIẾT LẬP THÔNG SỐ (Tăng num_runs lên 30 khi chạy để lấy số liệu vào báo cáo chính thức)
    num_runs = 30 
    
    random.seed(42)
    np.random.seed(42) 

    # ==========================================
    # 1. KNAPSACK PROBLEM - Kiểm tra Scalability
    # Chạy qua 2 size: 10 và 15
    # ==========================================
    knapsack_sizes = [10, 15]
    for size in knapsack_sizes:
        kp_problem = KnapsackProblem(dimensions=size, capacity=50.0)
        
        kp_algorithms = {
            "GA": GeneticAlgorithm(population_size=30, max_iterations=50),
            "ABC": ArtificialBeeColony(colony_size=30, max_iterations=50),
            "Hill Climbing": HillClimbing(variant='steepest', max_iterations=200)
        }
        # Chỉ chạy thuật toán cổ điển ở size nhỏ để tránh bị treo máy
        if size <= 15:
            kp_algorithms["A*"] = AStarSearch()
            kp_algorithms["BFS"] = BFS()
            kp_algorithms["DFS"] = DFS()
            
        framework.run_suite("Knapsack", size, kp_problem, kp_algorithms, num_runs)

    # ==========================================
    # 2. TSP PROBLEM - Kiểm tra Scalability
    # Chạy qua 2 size: 6 và 8 (Do 10! rất lớn, có thể làm BFS chạy lâu)
    # ==========================================
    tsp_sizes = [6, 8]
    for size in tsp_sizes:
        tsp_problem = TSPProblem(dimensions=size) 
        
        tsp_algorithms = {
            "ACO": AntColonyOptimization(population_size=30, max_iterations=50),
            "GA": GeneticAlgorithm(population_size=30, max_iterations=50),
            "SA": SimulatedAnnealing(initial_temperature=100.0, cooling_rate=0.9, max_iterations=50),
            "Hill Climbing": HillClimbing(variant='steepest', max_iterations=200)
        }
        if size <= 8:
            tsp_algorithms["A*"] = AStarSearch()
            tsp_algorithms["BFS"] = BFS()
            tsp_algorithms["DFS"] = DFS()

        framework.run_suite("TSP", size, tsp_problem, tsp_algorithms, num_runs)

    # ==========================================
    # 3. KẾT XUẤT VÀ TRỰC QUAN HÓA
    # ==========================================
    df_results = framework.get_dataframe()
    print("\n" + "="*80)
    print("BẢNG KẾT QUẢ BENCHMARK (ĐÃ LOẠI BỎ RAW DATA ĐỂ HIỂN THỊ):")
    print("="*80)
    
    # In bảng ra console (bỏ cột Raw_Costs đi cho gọn)
    display_df = df_results.drop(columns=['Raw_Costs'])
    print(display_df.to_string(index=False))
    
    visualizer = BenchmarkVisualizer(output_dir="Lab1_results")
    visualizer.generate_all_plots(df_results, framework.convergence_data)
    print("\n-> Đã tạo xong biểu đồ cực xịn! Vui lòng kiểm tra thư mục 'Lab1_results'.")

if __name__ == "__main__":
    import scipy # Kiểm tra xem thư viện scipy đã cài chưa
    main()