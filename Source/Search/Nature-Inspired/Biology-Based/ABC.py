import numpy as np
from abc import ABC, abstractmethod


class ObjectiveFunction(ABC):
    """Lớp trừu tượng để định nghĩa hàm mục tiêu"""
    
    @abstractmethod
    def evaluate(self, x):
        """Tính giá trị hàm mục tiêu"""
        pass


class ArtificialBeeColony:
    """
    Lớp thực hiện thuật toán Artificial Bee Colony (ABC)
    
    Các tham số:
    -----------
    objective_func : ObjectiveFunction
        Hàm mục tiêu cần tối ưu hóa
    lb : array-like
        Cận dưới của từng biến số
    ub : array-like
        Cận trên của từng biến số
    colony_size : int
        Số lượng con ong (mặc định: 30)
    limit : int
        Ngưỡng tối đa cho biến trial (mặc định: 100)
    """
    
    def __init__(self, objective_func, lb, ub, colony_size=30, limit=100):
        self.objective_func = objective_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = len(lb)
        self.colony_size = colony_size
        self.num_employed = colony_size // 2  # Số ong thợ
        self.limit = limit
        
        # Khởi tạo các biến chính
        self.foods = None  # Ma trận vị trí các nguồn hoa
        self.fitness = None  # Mảng độ nồng nàn của mật
        self.trial = None  # Bộ đếm số lần tìm kiếm không cải thiện
        self.best_solution = None  # Giải pháp tốt nhất tìm được
        self.best_fitness = -np.inf  # Fitness của giải pháp tốt nhất
        self.iteration_best_fitness = []  # Lưu fitness tốt nhất mỗi lần lặp
        
    def _initialize(self):
        """
        Bước 1: Khởi tạo (Initialization)
        Rải ngẫu nhiên các con ong vào không gian tìm kiếm
        """
        self.foods = np.random.uniform(
            self.lb, self.ub, (self.colony_size, self.dim)
        )
        self.fitness = np.zeros(self.colony_size)
        self.trial = np.zeros(self.colony_size)
        
        # Tính fitness ban đầu
        for i in range(self.colony_size):
            self.fitness[i] = self._calculate_fitness(self.foods[i])
            
        # Khởi tạo giải pháp tốt nhất
        best_idx = np.argmax(self.fitness)
        self.best_solution = self.foods[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        
        print(f"Khởi tạo hoàn tất. Fitness tốt nhất ban đầu: {self.best_fitness:.6f}")
        
    def _calculate_fitness(self, solution):
        """
        Tính độ nồng nàn của mật (Fitness)
        
        Công thức:
        - Nếu f(x) >= 0: fitness = 1 / (1 + f(x))
        - Nếu f(x) < 0: fitness = 1 + |f(x)|
        """
        f_value = self.objective_func.evaluate(solution)
        
        if f_value >= 0:
            fitness = 1.0 / (1.0 + f_value)
        else:
            fitness = 1.0 + np.abs(f_value)
            
        return fitness
    
    def _employed_bee_phase(self):
        """
        Bước 2: Giai đoạn Ong thợ (Employed Bee Phase)
        Mỗi ong thợ tìm vị trí mới gần vị trí hiện tại
        
        Công thức: v_ij = x_ij + phi_ij * (x_ij - x_kj)
        """
        for i in range(self.num_employed):
            # Chọn ngẫu nhiên một hàng xóm k (k != i)
            k = np.random.choice(
                [j for j in range(self.colony_size) if j != i]
            )
            
            # Tạo vị trí mới
            v = self.foods[i].copy()
            
            for j in range(self.dim):
                # phi là số ngẫu nhiên từ -1 đến 1
                phi = np.random.uniform(-1, 1)
                v[j] = self.foods[i][j] + phi * (self.foods[i][j] - self.foods[k][j])
                
                # Đảm bảo vị trí mới nằm trong giới hạn
                v[j] = np.clip(v[j], self.lb[j], self.ub[j])
            
            # Greedy Selection (Lựa chọn tham lam)
            v_fitness = self._calculate_fitness(v)
            
            if v_fitness > self.fitness[i]:
                self.foods[i] = v
                self.fitness[i] = v_fitness
                self.trial[i] = 0
            else:
                self.trial[i] += 1
    
    def _onlooker_bee_phase(self):
        """
        Bước 3: Giai đoạn Ong quan sát (Onlooker Bee Phase)
        Ong quan sát chọn nguồn hoa dựa trên xác suất fitness
        
        Công thức xác suất: P_i = fitness_i / sum(fitness)
        """
        # Tính xác suất chọn cho mỗi nguồn hoa
        probs = self.fitness / np.sum(self.fitness)
        
        # Mỗi con ong quan sát chọn một nguồn hoa dựa trên xác suất
        for _ in range(self.colony_size - self.num_employed):
            # Chọn nguồn hoa dựa trên xác suất
            i = np.random.choice(self.colony_size, p=probs)
            
            # Chọn hàng xóm ngẫu nhiên k
            k = np.random.choice(
                [j for j in range(self.colony_size) if j != i]
            )
            
            # Tìm vị trí mới
            v = self.foods[i].copy()
            
            for j in range(self.dim):
                phi = np.random.uniform(-1, 1)
                v[j] = self.foods[i][j] + phi * (self.foods[i][j] - self.foods[k][j])
                v[j] = np.clip(v[j], self.lb[j], self.ub[j])
            
            # Greedy Selection
            v_fitness = self._calculate_fitness(v)
            
            if v_fitness > self.fitness[i]:
                self.foods[i] = v
                self.fitness[i] = v_fitness
                self.trial[i] = 0
            else:
                self.trial[i] += 1
    
    def _scout_bee_phase(self):
        """
        Bước 4: Giai đoạn Ong trinh sát (Scout Bee Phase)
        Cơ chế thoát khỏi cực trị địa phương
        
        Nếu trial vượt quá limit, thay thế bằng giải pháp mới ngẫu nhiên
        """
        for i in range(self.colony_size):
            if self.trial[i] >= self.limit:
                # Tạo giải pháp mới hoàn toàn ngẫu nhiên
                self.foods[i] = np.random.uniform(
                    self.lb, self.ub, self.dim
                )
                self.fitness[i] = self._calculate_fitness(self.foods[i])
                self.trial[i] = 0
    
    def _update_best_solution(self):
        """Cập nhật giải pháp tốt nhất"""
        best_idx = np.argmax(self.fitness)
        
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_solution = self.foods[best_idx].copy()
            
        self.iteration_best_fitness.append(self.best_fitness)
    
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
        
        # Lặp các giai đoạn
        for iteration in range(max_iterations):
            # Bước 2: Giai đoạn Ong thợ
            self._employed_bee_phase()
            
            # Bước 3: Giai đoạn Ong quan sát
            self._onlooker_bee_phase()
            
            # Bước 4: Giai đoạn Ong trinh sát
            self._scout_bee_phase()
            
            # Cập nhật giải pháp tốt nhất
            self._update_best_solution()
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Lần lặp {iteration + 1}/{max_iterations} - "
                      f"Best Fitness: {self.best_fitness:.6f}")
        
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'iteration_fitness': self.iteration_best_fitness,
            'final_foods': self.foods,
            'final_fitness': self.fitness
        }



