import numpy as np
from abc import ABC, abstractmethod


class ObjectiveFunction(ABC):
    """Lớp trừu tượng để định nghĩa hàm mục tiêu"""
    
    @abstractmethod
    def evaluate(self, x):
        """Tính giá trị hàm mục tiêu"""
        pass


class FireflyAlgorithm:
    """
    Lớp thực hiện thuật toán Firefly Algorithm (FA)
    
    Các tham số:
    -----------
    objective_func : ObjectiveFunction
        Hàm mục tiêu cần tối ưu hóa
    lb : array-like
        Cận dưới của từng biến số
    ub : array-like
        Cận trên của từng biến số
    population_size : int
        Số lượng đom đóm (mặc định: 30)
    alpha : float
        Tham số độ lệch ngẫu nhiên ban đầu (mặc định: 0.5)
    beta0 : float
        Độ hấp dẫn tối đa tại khoảng cách = 0 (mặc định: 1.0)
    gamma : float
        Hệ số hấp thụ ánh sáng (mặc định: 0.01)
    """
    
    def __init__(self, objective_func, lb, ub, population_size=30, 
                 alpha=0.5, beta0=1.0, gamma=0.01):
        self.objective_func = objective_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = len(lb)
        self.population_size = population_size
        self.alpha = alpha
        self.alpha_initial = alpha
        self.beta0 = beta0
        self.gamma = gamma
        
        # Khởi tạo các biến chính
        self.fireflies = None  # Ma trận vị trí các đom đóm
        self.intensity = None  # Mảng cường độ sáng (giá trị hàm mục tiêu)
        self.best_solution = None  # Giải pháp tốt nhất tìm được
        self.best_intensity = np.inf  # Cường độ sáng của giải pháp tốt nhất (tối ưu hóa Min)
        self.iteration_best_intensity = []  # Lưu cường độ sáng tốt nhất mỗi lần lặp
        
    def _initialize(self):
        """
        Bước 1: Khởi tạo (Initialization)
        Rải ngẫu nhiên các đom đóm vào không gian tìm kiếm
        """
        self.fireflies = np.random.uniform(
            self.lb, self.ub, (self.population_size, self.dim)
        )
        self.intensity = np.zeros(self.population_size)
        
        # Tính cường độ sáng ban đầu (giá trị hàm mục tiêu)
        for i in range(self.population_size):
            self.intensity[i] = self.objective_func.evaluate(self.fireflies[i])
        
        # Khởi tạo giải pháp tốt nhất (cái có cường độ sáng nhỏ nhất - tối ưu hóa Min)
        best_idx = np.argmin(self.intensity)
        self.best_solution = self.fireflies[best_idx].copy()
        self.best_intensity = self.intensity[best_idx]
        
        print(f"Khởi tạo hoàn tất. Cường độ sáng tốt nhất ban đầu: {self.best_intensity:.6f}")
        
    def _calculate_distance(self, firefly_i, firefly_j):
        """
        Tính khoảng cách Euclidean giữa hai đom đóm
        
        Công thức: r = ||x_i - x_j|| = sqrt(sum((x_i - x_j)^2))
        """
        return np.linalg.norm(firefly_i - firefly_j)
    
    def _calculate_attractiveness(self, r):
        """
        Tính độ hấp dẫn (Attractiveness)
        
        Công thức: beta = beta0 * exp(-gamma * r^2)
        
        Tham số:
        -------
        r : float
            Khoảng cách giữa hai đom đóm
            
        Returns:
        -------
        float : Độ hấp dẫn beta
        """
        return self.beta0 * np.exp(-self.gamma * r ** 2)
    
    def _update_firefly_position(self, i, j):
        """
        Cập nhật vị trí đom đóm i dựa trên đom đóm j (j sáng hơn i)
        
        Công thức: x_i = x_i + beta*(x_j - x_i) + alpha*(rand - 0.5)
        
        Tham số:
        -------
        i : int
            Chỉ số đom đóm cần cập nhật
        j : int
            Chỉ số đom đóm sáng hơn (hấp dẫn hơn)
        """
        # Tính khoảng cách
        r = self._calculate_distance(self.fireflies[i], self.fireflies[j])
        
        # Tính độ hấp dẫn
        beta = self._calculate_attractiveness(r)
        
        # Cập nhật vị trí theo công thức
        # x_i = x_i + beta*(x_j - x_i) + alpha*(rand - 0.5)
        random_perturbation = self.alpha * (np.random.rand(self.dim) - 0.5)
        self.fireflies[i] = (self.fireflies[i] + 
                            beta * (self.fireflies[j] - self.fireflies[i]) + 
                            random_perturbation)
        
        # Xử lý biên (Boundary Handling) - Giới hạn vị trí trong phạm vi
        self.fireflies[i] = np.clip(self.fireflies[i], self.lb, self.ub)
        
        # Tính lại cường độ sáng cho đom đóm i
        self.intensity[i] = self.objective_func.evaluate(self.fireflies[i])
    
    def _move_fireflies(self):
        """
        Bước chính: Di chuyển các đom đóm (So sánh từng cặp)
        
        Thuật toán:
        - Với mỗi đom đóm i
        - So sánh với mỗi đom đóm j
        - Nếu j sáng hơn i (intensity[j] < intensity[i])
        - Thì i di chuyển về phía j
        """
        for i in range(self.population_size):
            for j in range(self.population_size):
                # Nếu đom đóm j sáng hơn đom đóm i
                # (cường độ sáng nhỏ hơn - bài toán tối ưu hóa Min)
                if self.intensity[j] < self.intensity[i]:
                    self._update_firefly_position(i, j)
    
    def _decay_alpha(self, iteration, max_iterations):
        """
        Giảm dần tham số alpha (Alpha Decay - Tùy chọn)
        
        Khi số lần lặp tăng, alpha giảm để giảm độ ngẫu nhiên
        và tập trung hơn vào khai thác (exploitation)
        
        Công thức: alpha = alpha_initial * (1 - iteration / max_iterations)
        """
        self.alpha = self.alpha_initial * (1 - iteration / max_iterations)
    
    def _update_best_solution(self):
        """Cập nhật giải pháp tốt nhất"""
        best_idx = np.argmin(self.intensity)
        
        if self.intensity[best_idx] < self.best_intensity:
            self.best_intensity = self.intensity[best_idx]
            self.best_solution = self.fireflies[best_idx].copy()
        
        self.iteration_best_intensity.append(self.best_intensity)
    
    def optimize(self, max_iterations=100, verbose=True):
        """
        Thực hiện tối ưu hóa
        
        Tham số:
        --------
        max_iterations : int
            Số lần lặp tối đa
        verbose : bool
            In thông tin trong quá trình tối ưu
            
        Returns:
        --------
        dict : Chứa giải pháp tốt nhất và thông tin tối ưu
        """
        # Bước 1: Khởi tạo
        self._initialize()
        
        # Lặp các bước chính
        for iteration in range(max_iterations):
            # Bước chính: Di chuyển các đom đóm
            self._move_fireflies()
            
            # Cập nhật giải pháp tốt nhất
            self._update_best_solution()
            
            # Giảm dần tham số alpha
            self._decay_alpha(iteration, max_iterations)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Lần lặp {iteration + 1}/{max_iterations} - "
                      f"Best Intensity: {self.best_intensity:.6f}")
        
        return {
            'best_solution': self.best_solution,
            'best_intensity': self.best_intensity,
            'iteration_intensity': self.iteration_best_intensity,
            'final_fireflies': self.fireflies,
            'final_intensity': self.intensity
        }

