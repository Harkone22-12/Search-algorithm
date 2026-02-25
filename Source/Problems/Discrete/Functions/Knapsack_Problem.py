import numpy as np
import random
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem

class KnapsackProblem(OptimizationProblem):
    def __init__(self, dimensions: int = 15, capacity: float = 50.0):
        super().__init__(dimensions=dimensions)
        self.weights = np.random.uniform(5, 25, self.dimensions)
        self.values = np.random.uniform(10, 100, self.dimensions)
        self.capacity = capacity
        self.bounds = [(0, 1)] * self.dimensions

    def generate_random_state(self):
        return [random.choice([0, 1]) for _ in range(self.dimensions)]

    def heuristic(self, state) -> float:
        return 0.0

    def evaluate_state(self, state) -> float:
        if isinstance(state, tuple) and len(state) == 2:
            bits = np.array(state[1])
        else:
            bits = np.array([int(round(float(i))) for i in state])
        
        w, v = np.sum(bits * self.weights), np.sum(bits * self.values)
        if w > self.capacity:
            return float((w - self.capacity) * 1000)
        return float(-v)

    def get_start_state(self): 
        # For Knapsack, we always start building from index 0
        return (0, tuple([0] * self.dimensions))

    def is_goal(self, state): 
        return state[0] == self.dimensions

    def get_successors(self, state):
        if isinstance(state, tuple) and len(state) == 2:
            idx, taken = state
            if idx >= self.dimensions: return []
            
            # ... (bên trong if của get_successors)
            res = []
            
            # 1. LEAVE (Đưa vào trước để nằm dưới đáy Stack của DFS)
            res.append(((idx + 1, taken), 0.0))

            # 2. TAKE (Đưa vào sau để DFS lấy ra ưu tiên duyệt trước)
            curr_w = sum(np.array(taken) * self.weights)
            if curr_w + self.weights[idx] <= self.capacity:
                new_t = list(taken)
                new_t[idx] = 1
                res.append(((idx + 1, tuple(new_t)), -self.values[idx]))
            
            return res
        else:
            # Local search neighbor
            b = list(state)
            i = random.randint(0, self.dimensions - 1)
            b[i] = 1 - b[i]
            return [(tuple(b), self.evaluate_state(b) - self.evaluate_state(state))]