import time
import numpy as np
import pandas as pd
from scipy import stats

class BenchmarkFramework:
    def __init__(self):
        self.results = []
        self.convergence_data = {}

    def run_suite(self, problem_type, size, problem, algorithms, num_runs=5):
        problem_name = f"{problem_type}_{size}"
        print(f"\n{'='*60}")
        print(f"Bắt đầu Benchmark: {problem_name} (Số lần chạy: {num_runs})")
        print(f"{'='*60}")
        
        for algo_name, algo in algorithms.items():
            print(f"Đang chạy {algo_name}...")
            costs, times, nodes_list, histories = [], [], [], []

            for _ in range(num_runs):
                start_time = time.time()
                try:
                    res = algo.search(problem)
                except Exception as e:
                    print(f"  -> Lỗi khi chạy {algo_name}: {e}")
                    res = None
                end_time = time.time()

                cost = float('inf')
                history = []
                
                # Trích xuất kết quả
                if isinstance(res, tuple):
                    if res[0] is not None:
                        cost = res[1]
                elif isinstance(res, dict):
                    cost = res.get('cost', float('inf'))
                    history = res.get('history', [])
                
                if not history:
                    history = getattr(algo, 'history', [])

                nodes = getattr(algo, 'expanded_nodes', getattr(algo, 'expanded_nodes_', 0))

                costs.append(cost)
                times.append(end_time - start_time)
                nodes_list.append(nodes)
                if history:
                    histories.append(history)

            # Tính toán thống kê
            valid_costs = [c for c in costs if c != float('inf')]
            mean_cost = np.mean(valid_costs) if valid_costs else float('inf')
            std_cost = np.std(valid_costs) if valid_costs else 0.0
            best_cost = np.min(valid_costs) if valid_costs else float('inf')
            
            mean_time = np.mean(times)
            mean_nodes = np.mean(nodes_list)

            self.results.append({
                'Problem_Type': problem_type,
                'Size': size,
                'Problem': problem_name,
                'Algorithm': algo_name,
                'Best_Cost': best_cost,
                'Mean_Cost': mean_cost,
                'Std_Cost': std_cost,
                'Mean_Time': mean_time,
                'Mean_Nodes': mean_nodes,
                'Raw_Costs': costs # Giữ lại dữ liệu thô để vẽ Boxplot
            })

            # Trung bình hóa lịch sử hội tụ (Convergence)
            if histories:
                min_len = min(len(h) for h in histories)
                avg_history = np.mean([h[:min_len] for h in histories], axis=0)
                self.convergence_data[f"{problem_name}_{algo_name}"] = avg_history

    def get_dataframe(self):
        return pd.DataFrame(self.results)