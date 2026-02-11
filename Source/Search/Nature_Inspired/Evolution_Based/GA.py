import random
import numpy as np
from typing import Any, Tuple, List, Dict
from Source.Problems.problem import SearchProblem
from Source.Search.Search import SearchAlgorithm
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem

class GeneticAlgorithm(SearchAlgorithm):
    """
    Genetic Algorithm (GA) for Continuous Optimization.
    
    Quy trình tiến hóa: 
    Selection (Chọn lọc) -> Crossover (Lai ghép) -> Mutation (Đột biến).
    """
    
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_rate: float = 0.1,    # Tỉ lệ giữ lại các cá thể tốt nhất không qua biến đổi
        max_iterations: int = 100,
        seed: int = None
    ):
        super().__init__()
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.max_iterations = max_iterations
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def search(self, problem: SearchProblem) -> Dict[str, Any]:
        assert isinstance(problem, OptimizationProblem)
        
        # 1. Khởi tạo quần thể ban đầu
        population = [problem.generate_random_state() for _ in range(self.population_size)]
        self.expanded_nodes += self.population_size
        
        # Đánh giá độ thích nghi (fitness) - GA thường tối đa hóa, 
        # nhưng ở đây ta tối thiểu hóa cost nên fitness càng thấp càng tốt.
        fitness = [problem.evaluate_state(ind) for ind in population]
        
        best_idx = np.argmin(fitness)
        best_state = population[best_idx]
        best_fitness = fitness[best_idx]
        
        iteration_history = [best_fitness]

        for _ in range(self.max_iterations):
            new_population = []
            
            # --- ELITISM: Giữ lại những cá thể xuất sắc nhất ---
            n_elite = int(self.population_size * self.elitism_rate)
            # Sắp xếp index dựa trên fitness tăng dần
            sorted_indices = np.argsort(fitness)
            for i in range(n_elite):
                new_population.append(population[sorted_indices[i]])

            # --- TẠO THẾ HỆ MỚI ---
            while len(new_population) < self.population_size:
                # 2. SELECTION: Tournament Selection
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                
                # 3. CROSSOVER: Arithmetic Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._arithmetic_crossover(parent1, parent2)
                else:
                    child1, child2 = child1, child2 = parent1, parent2
                
                # 4. MUTATION: Gaussian Mutation
                child1 = self._gaussian_mutation(child1, problem)
                child2 = self._gaussian_mutation(child2, problem)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            # Đánh giá quần thể mới
            population = new_population
            fitness = [problem.evaluate_state(ind) for ind in population]
            self.expanded_nodes += len(population)
            
            # Cập nhật kỷ lục
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_state = population[current_best_idx]
                
            iteration_history.append(best_fitness)

        return {
            "best_state": best_state,
            "cost": best_fitness,
            "history": iteration_history,
            "expanded_nodes": self.expanded_nodes
        }

    def _tournament_selection(self, population: List, fitness: List, k: int = 3) -> tuple:
        """Chọn cá thể tốt nhất từ k cá thể ngẫu nhiên."""
        selected_indices = random.sample(range(len(population)), k)
        best_in_tournament = selected_indices[0]
        for idx in selected_indices:
            if fitness[idx] < fitness[best_in_tournament]:
                best_in_tournament = idx
        return population[best_in_tournament]

    def _arithmetic_crossover(self, p1: tuple, p2: tuple) -> Tuple[tuple, tuple]:
        """Lai ghép tổ hợp tuyến tính giữa 2 cha mẹ."""
        alpha = random.random()
        c1 = tuple(alpha * np.array(p1) + (1 - alpha) * np.array(p2))
        c2 = tuple(alpha * np.array(p2) + (1 - alpha) * np.array(p1))
        return c1, c2

    def _gaussian_mutation(self, individual: tuple, problem: OptimizationProblem) -> tuple:
        """Đột biến bằng cách cộng nhiễu Gaussian."""
        if random.random() > self.mutation_rate:
            return individual
        
        ind_list = list(individual)
        for i in range(len(ind_list)):
            # Độ lệch chuẩn dựa trên 10% dải giá trị của bound
            scale = (problem.bounds[i][1] - problem.bounds[i][0]) * 0.1
            ind_list[i] += random.gauss(0, scale)
            
            # Đảm bảo nằm trong giới hạn
            ind_list[i] = max(problem.bounds[i][0], min(problem.bounds[i][1], ind_list[i]))
            
        return tuple(ind_list)