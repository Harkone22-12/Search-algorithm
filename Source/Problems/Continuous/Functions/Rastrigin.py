import numpy as np
import math
from typing import List, Tuple
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem

class RastriginProblem(OptimizationProblem):
    """Rastrigin function optimization problem."""
    
    def __init__(self, dimensions: int = 5, bounds: Tuple[float, float] = (-5.12, 5.12)):
        """
        Khởi tạo bài toán Rastrigin.
        
        Args:
            dimensions: Số chiều (mặc định là 5)
            bounds: Khoảng tìm kiếm (mặc định [-5.12, 5.12])
        """
        self.dimensions = dimensions
        self.bounds = [bounds] * dimensions
        self.optimum_value = 0.0
        self.optimum_solution = np.zeros(dimensions)
    
    def get_start_state(self) -> List[float]:
        """Tạo trạng thái bắt đầu ngẫu nhiên."""
        return self.generate_random_state()
    
    def evaluate_state(self, state: List[float]) -> float:
        """
        Đánh giá hàm Rastrigin:
        f(x) = 10n + Σ(x_i^2 - 10*cos(2*π*x_i))
        """
        n = len(state)
        res = 10 * n
        for x in state:
            res += (x**2 - 10 * math.cos(2 * math.pi * x))
        return res
    
    def generate_random_state(self) -> List[float]:
        """Tạo lời giải ngẫu nhiên trong giới hạn bounds."""
        return [
            np.random.uniform(self.bounds[i][0], self.bounds[i][1])
            for i in range(self.dimensions)
        ]
    
    def is_goal(self, state: List[float]) -> bool:
        """Kiểm tra nếu đạt mục tiêu (thường không dùng trong tối ưu hóa liên tục)."""
        return False
    
    def get_successors(self, state: List[float]) -> List[Tuple[List[float], float]]:
        """Lấy các lời giải lân cận bằng cách thay đổi nhỏ ở từng chiều."""
        successors = []
        for i in range(self.dimensions):
            for delta in [-0.1, 0.1]:
                neighbor = state.copy()
                neighbor[i] += delta
                # Giới hạn giá trị trong bounds
                neighbor[i] = np.clip(neighbor[i], self.bounds[i][0], self.bounds[i][1])
                
                # Tính toán chi phí chênh lệch (cost)
                cost = self.evaluate_state(neighbor) - self.evaluate_state(state)
                successors.append((neighbor, cost))
        return successors