import numpy as np
import math
from typing import List, Tuple
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem

class RastriginProblem(OptimizationProblem):
    """Rastrigin function optimization problem."""
    
    def __init__(self, dimensions: int = 5, bounds: Tuple[float, float] = (-5.12, 5.12)):
        """
        Initialize the Rastrigin problem.
        
        Args:
            dimensions: Number of dimensions (default 5)
            bounds: Search space bounds (default [-5.12, 5.12])
        """
        self.dimensions = dimensions
        self.bounds = [bounds] * dimensions
        self.optimum_value = 0.0
        self.optimum_solution = np.zeros(dimensions)
    
    def get_start_state(self) -> List[float]:
        """Get an initial random solution."""
        return self.generate_random_state()
    
    def evaluate_state(self, state: List[float]) -> float:
        """
        Evaluate the Rastrigin function:
        f(x) = 10n + Σ(x_i^2 - 10*cos(2*π*x_i))
        """
        x = np.array(state)
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    def generate_random_state(self) -> List[float]:
        """Generate a random solution within the bounds."""
        return [
            np.random.uniform(self.bounds[i][0], self.bounds[i][1])
            for i in range(self.dimensions)
        ]
    
    def is_goal(self, state: List[float]) -> bool:
        """Check if the goal is reached (never true for continuous optimization)."""
        return False
    
    def get_successors(self, state: List[float]) -> List[Tuple[List[float], float]]:
        """Get neighboring solutions by applying a small delta to each dimension."""
        successors = []
        for i in range(self.dimensions):
            for delta in [-0.1, 0.1]:
                neighbor = state.copy()
                neighbor[i] += delta
                # Clip the value to stay within bounds
                neighbor[i] = np.clip(neighbor[i], self.bounds[i][0], self.bounds[i][1])
                
                # Calculate the difference in cost
                cost = self.evaluate_state(neighbor) - self.evaluate_state(state)
                successors.append((neighbor, cost))
        return successors