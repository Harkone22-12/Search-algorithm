"""
Sphere Function - Test Problems for Continuous Optimization

Standard benchmark function for testing algorithms:
f(x) = Σ(x_i²)
Optimum: f(0, 0, ..., 0) = 0
"""

import numpy as np
from typing import List, Tuple
from Source.Problems.problem import SearchProblem
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem


class SphereProblem(OptimizationProblem):
    """Sphere function optimization problem."""
    
    def __init__(self, dimensions: int = 5, bounds: Tuple[float, float] = (-5.12, 5.12)):
        """
        Initialize Sphere problem.
        
        Args:
            dimensions: Number of dimensions (default 5)
            bounds: Search space bounds (default [-5.12, 5.12])
        """
        self.dimensions = dimensions
        self.bounds = [bounds] * dimensions
        self.optimum_value = 0.0
        self.optimum_solution = np.zeros(dimensions)
    
    def get_start_state(self) -> List[float]:
        """Get initial random solution."""
        return self.generate_random_state()
    
    def evaluate_state(self, state: List[float]) -> float:
        """Evaluate Sphere function: f(x) = Σ(x_i²)"""
        return sum(x**2 for x in state)
    
    def generate_random_state(self) -> List[float]:
        """Generate random solution within bounds."""
        return [
            np.random.uniform(self.bounds[i][0], self.bounds[i][1])
            for i in range(self.dimensions)
        ]
    
    def is_goal(self, state: List[float]) -> bool:
        """Check if goal is reached (never true for optimization)."""
        return False
    
    def get_successors(self, state: List[float]) -> List[Tuple[List[float], float]]:
        """Get neighboring solutions."""
        successors = []
        for i in range(self.dimensions):
            for delta in [-0.1, 0.1]:
                neighbor = state.copy()
                neighbor[i] += delta
                neighbor[i] = np.clip(neighbor[i], self.bounds[i][0], self.bounds[i][1])
                cost = self.evaluate_state(neighbor) - self.evaluate_state(state)
                successors.append((neighbor, cost))
        return successors
