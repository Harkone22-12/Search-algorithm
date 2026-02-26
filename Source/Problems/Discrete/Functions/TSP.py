import numpy as np
import random
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem

class TSPProblem(OptimizationProblem):
    def __init__(self, dimensions: int = 8):
        super().__init__(dimensions=dimensions)
        self.n = dimensions
        self.dist_matrix = np.random.uniform(10, 50, (self.n, self.n))
        np.fill_diagonal(self.dist_matrix, 0)
        self.bounds = [(0, self.n - 1)] * self.n

    def generate_random_state(self):
        """Used by GA, ACO, and ABC."""
        state = list(range(self.n))
        random.shuffle(state)
        return tuple(state)

    def heuristic(self, state) -> float:
        if isinstance(state, tuple) and len(state) == 2:
            return float((self.n - len(state[1])) * 10.0)
        return 0.0

    def evaluate_state(self, state) -> float:
        # Handle both tree-search tuples and local-search lists
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], tuple):
            tour = list(state[1])
        else:
            tour = [int(round(float(i))) for i in state]

        if len(set(tour)) != self.n or len(tour) != self.n:
            return 1e7 # Penalty for invalid tour
        
        d = 0.0
        for i in range(self.n):
            u, v = tour[i], tour[(i + 1) % self.n]
            d += self.dist_matrix[u][v]
        return float(d)

    def get_start_state(self): 
        """
        Dành cho Tree Search (A*, BFS, DFS). 
        Bắt đầu tại thành phố 0, tập các thành phố đã thăm là (0,)
        """
        return (0, (0,))

    def is_goal(self, state): 
        # Phải kiểm tra chắc chắn đây là định dạng của Tree Search (curr, visited)
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], tuple):
            return len(state[1]) == self.n
        # Dành cho Local Search (full tour)
        return True
    def get_successors(self, state):
        # DETECT: Is this a tree search state (tuple of 2) or a local search state (full tour)?
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], tuple):
            curr, visited = state
            res = []
            for nxt in range(self.n):
                if nxt not in visited:
                    cost = self.dist_matrix[curr][nxt]
                    if len(visited) == self.n - 1:
                        cost += self.dist_matrix[nxt][visited[0]]
                    res.append(((nxt, visited + (nxt,)), cost))
            return res
        else:
            # Local Search: Swap two cities in the full tour
            t = list(state)
            i, j = random.sample(range(self.n), 2)
            t[i], t[j] = t[j], t[i]
            return [(tuple(t), self.evaluate_state(t) - self.evaluate_state(state))]