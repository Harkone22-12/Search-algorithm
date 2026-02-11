import random
import numpy as np
from typing import Any, Tuple, List, Dict
from Source.Problems.problem import SearchProblem
from Source.Search.Search import SearchAlgorithm
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem

class AntColonyOptimization(SearchAlgorithm):
    """
    Ant Colony Optimization (ACO_R) for Continuous Domains.
    
    Thuật toán mô phỏng cách kiến tìm đường dựa trên nồng độ pheromone.
    Trong không gian liên tục, pheromone được mô phỏng bằng một lưu trữ các 
    giải pháp tốt nhất (solution archive).
    """
    
    def __init__(
        self,
        population_size: int = 50,  # Số lượng kiến (mỗi vòng lặp)
        archive_size: int = 10,     # Số lượng giải pháp tốt nhất được giữ lại (pheromone)
        q: float = 0.1,             # Tham số điều chỉnh tốc độ hội tụ (càng nhỏ càng nhanh)
        xi: float = 0.85,           # Tham số tương tự tốc độ bay hơi pheromone
        max_iterations: int = 100,
        seed: int = None
    ):
        super().__init__()
        self.population_size = population_size
        self.archive_size = archive_size
        self.q = q
        self.xi = xi
        self.max_iterations = max_iterations
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def search(self, problem: SearchProblem) -> Dict[str, Any]:
        assert isinstance(problem, OptimizationProblem)
        
        # 1. Khởi tạo Archive (Lưu trữ các giải pháp ban đầu)
        archive = []
        for _ in range(self.archive_size):
            state = problem.generate_random_state()
            cost = problem.evaluate_state(state)
            archive.append({'state': np.array(state), 'cost': cost})
            self.expanded_nodes += 1
            
        # Sắp xếp archive theo giá trị cost tăng dần (minimization)
        archive.sort(key=lambda x: x['cost'])
        
        best_state = archive[0]['state']
        best_fitness = archive[0]['cost']
        iteration_history = [best_fitness]

        # 2. Vòng lặp chính
        for _ in range(self.max_iterations):
            # Tính trọng số (weights) cho mỗi giải pháp trong archive
            # Giải pháp tốt hơn (index nhỏ hơn) sẽ có trọng số cao hơn
            weights = np.zeros(self.archive_size)
            for i in range(self.archive_size):
                # Công thức Gaussian cho trọng số
                exponent = -((i)**2) / (2 * (self.q * self.archive_size)**2)
                weights[i] = (1.0 / (self.q * self.archive_size * np.sqrt(2 * np.pi))) * np.exp(exponent)
            
            # Tính xác suất chọn một "hàng" trong archive làm mẫu
            probabilities = weights / np.sum(weights)
            
            new_solutions = []
            
            # Mỗi con kiến trong đàn sẽ tạo ra một giải pháp mới
            for _ in range(self.population_size):
                # Bước chọn mẫu (Roulette wheel selection dựa trên pheromone)
                l = np.random.choice(range(self.archive_size), p=probabilities)
                
                # Tạo giải pháp mới dựa trên mẫu được chọn
                new_state = []
                for d in range(problem.dimensions):
                    # Tính độ lệch chuẩn trung bình (Standard deviation) để lấy mẫu
                    # Dựa trên khoảng cách giữa các giải pháp trong archive
                    sigma_sum = 0
                    for r in range(self.archive_size):
                        sigma_sum += abs(archive[r]['state'][d] - archive[l]['state'][d])
                    
                    sigma = self.xi * (sigma_sum / (self.archive_size - 1))
                    
                    # Lấy mẫu từ phân phối Gaussian xung quanh điểm được chọn
                    val = np.random.normal(archive[l]['state'][d], sigma)
                    
                    # Ràng buộc trong bounds
                    val = max(problem.bounds[d][0], min(problem.bounds[d][1], val))
                    new_state.append(val)
                
                new_state = tuple(new_state)
                new_cost = problem.evaluate_state(new_state)
                new_solutions.append({'state': np.array(new_state), 'cost': new_cost})
                self.expanded_nodes += 1

            # 3. Cập nhật Archive (Chỉ giữ lại k giải pháp tốt nhất tổng thể)
            archive.extend(new_solutions)
            archive.sort(key=lambda x: x['cost'])
            archive = archive[:self.archive_size] # Cắt bỏ những cái kém hơn
            
            # Cập nhật Best
            if archive[0]['cost'] < best_fitness:
                best_fitness = archive[0]['cost']
                best_state = archive[0]['state']
                
            iteration_history.append(best_fitness)

        return {
            "best_state": tuple(best_state),
            "cost": best_fitness,
            "history": iteration_history,
            "expanded_nodes": self.expanded_nodes
        }