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
        
        # Track tất cả histories để pad về độ dài cùng nhau sau (cho convergence plots)
        all_algo_histories = {}
        
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
                
                # Nếu không lấy được history từ result dict, lấy từ attribute của algo
                if not history:
                    history = getattr(algo, 'history', [])

                nodes = getattr(algo, 'expanded_nodes', getattr(algo, 'expanded_nodes_', 0))

                costs.append(cost)
                times.append(end_time - start_time)
                nodes_list.append(nodes)
                # Thêm history ngay cả khi rỗng để tránh skip
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

            # Lưu histories để xử lý sau (pad để cùng độ dài)
            all_algo_histories[algo_name] = histories
        
        # AFTER xử lý tất cả algorithms, pad histories để cùng độ dài
        # Tìm max_len từ TẤT CẢ algorithms
        all_valid_histories = []
        for histories in all_algo_histories.values():
            all_valid_histories.extend([h for h in histories if len(h) > 0])
        
        if all_valid_histories:
            global_max_len = max(len(h) for h in all_valid_histories)
            
            # Giờ pad và tính convergence data cho mỗi algorithm
            for algo_name, histories in all_algo_histories.items():
                valid_histories = [h for h in histories if len(h) > 0]
                if valid_histories:
                    # Pad tất cả lên global_max_len
                    padded_histories = []
                    for h in valid_histories:
                        if len(h) < global_max_len:
                            padded = list(h) + [h[-1]] * (global_max_len - len(h))
                            padded_histories.append(padded)
                        else:
                            padded_histories.append(h)
                    
                    avg_history = np.mean(padded_histories, axis=0)
                    self.convergence_data[f"{problem_name}_{algo_name}"] = avg_history

    def get_dataframe(self):
        return pd.DataFrame(self.results)