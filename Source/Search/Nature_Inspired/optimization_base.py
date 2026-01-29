"""
Base classes for optimization algorithms.
Provides common interfaces for GA, DE, ACO, SA, PSO, and other metaheuristics.
"""

import random
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple
import numpy as np

from Source.Problems.problem import SearchProblem


class OptimizationProblem(SearchProblem):
    """
    Abstract base class for optimization problems.
    Adapts optimization problems to the SearchProblem interface.
    
    This class serves as a bridge between traditional optimization problems
    and the search framework, enabling unified benchmarking.
    """
    
    @abstractmethod
    def get_start_state(self):
        """Returns the initial solution/state."""
        pass
    
    def is_goal(self, state) -> bool:
        """
        For optimization, we usually don't have a specific goal state.
        Returns False by default.
        """
        return False
    
    @abstractmethod
    def get_successors(self, state):
        """
        Returns list of (neighbor_state, cost_difference).
        The cost is the difference from current state.
        """
        pass
    
    @abstractmethod
    def evaluate_state(self, state) -> float:
        """Get the cost/fitness value of a state."""
        pass
    
    def heuristic(self, state) -> float:
        """
        Heuristic function for optimization.
        Returns 0 by default (not used in most metaheuristics).
        """
        return 0


class FunctionOptimizationProblem(OptimizationProblem):
    """
    Concrete implementation for optimizing continuous mathematical functions.
    
    This class can be used by all optimization algorithms (SA, GA, DE, PSO, ACO, etc.)
    for benchmarking on standard test functions.
    
    Common test functions:
    - Rastrigin, Sphere, Rosenbrock, Ackley, Griewank
    - Schwefel, Levy, Michalewicz, Zakharov, etc.
    """
    
    def __init__(
        self,
        objective_function: Callable[[List[float]], float],
        bounds: List[Tuple[float, float]],
        x: Optional[List[float]] = None,
        step_size: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize a function optimization problem instance.
        
        Args:
            objective_function: Function to minimize f(x) -> float
            bounds: List of (min, max) tuples for each dimension
            x: Initial position (random if not provided)
            step_size: Size of perturbation for neighbor generation
            seed: Random seed for reproducibility
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.dimensions = len(bounds)
        self.step_size = step_size
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if x is None:
            self.start_state = tuple([
                random.uniform(bound[0], bound[1])
                for bound in bounds
            ])
        else:
            self.start_state = tuple(x)
    
    def get_start_state(self):
        """Returns the initial solution as a tuple."""
        return self.start_state
    
    def is_goal(self, state) -> bool:
        """
        For continuous optimization, there's no specific goal state.
        Returns False.
        """
        return False
    
    def evaluate_state(self, state) -> float:
        """Evaluate the objective function at the given state."""
        return self.objective_function(list(state))
    
    def get_successors(self, state):
        """
        Generate neighbor solutions by adding Gaussian noise.
        
        Args:
            state: Current state (position tuple)
            
        Returns:
            List of (neighbor_state, cost_difference) tuples
        """
        neighbors = []
        current_cost = self.evaluate_state(state)
        
        # Generate a single neighbor
        neighbor_state = list(state)
        
        for j in range(self.dimensions):
            neighbor_state[j] = state[j] + random.gauss(0, self.step_size)
            # Enforce bounds
            neighbor_state[j] = max(
                self.bounds[j][0],
                min(self.bounds[j][1], neighbor_state[j])
            )
        
        neighbor_state = tuple(neighbor_state)
        neighbor_cost = self.evaluate_state(neighbor_state)
        cost_diff = neighbor_cost - current_cost
        
        neighbors.append((neighbor_state, cost_diff))
        
        return neighbors
    
    def generate_random_state(self) -> tuple:
        """Generate a random state within bounds (useful for population-based algorithms)."""
        return tuple([
            random.uniform(bound[0], bound[1])
            for bound in self.bounds
        ])


# Common benchmark functions for testing optimization algorithms
class BenchmarkFunctions:
    """Collection of standard benchmark functions for optimization."""
    
    @staticmethod
    def rastrigin(x: List[float]) -> float:
        """
        Rastrigin function: highly multimodal, many local minima.
        Global minimum: f(0,...,0) = 0
        Search domain: [-5.12, 5.12]^n
        """
        A = 10
        n = len(x)
        return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)
    
    @staticmethod
    def sphere(x: List[float]) -> float:
        """
        Sphere function: simple unimodal function.
        Global minimum: f(0,...,0) = 0
        Search domain: [-100, 100]^n
        """
        return sum(xi**2 for xi in x)
    
    @staticmethod
    def rosenbrock(x: List[float]) -> float:
        """
        Rosenbrock function: narrow valley, difficult optimization.
        Global minimum: f(1,...,1) = 0
        Search domain: [-5, 10]^n
        """
        return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
                   for i in range(len(x) - 1))
    
    @staticmethod
    def ackley(x: List[float]) -> float:
        """
        Ackley function: many local minima.
        Global minimum: f(0,...,0) = 0
        Search domain: [-32, 32]^n
        """
        n = len(x)
        sum1 = sum(xi**2 for xi in x)
        sum2 = sum(np.cos(2 * np.pi * xi) for xi in x)
        return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e
    
    @staticmethod
    def griewank(x: List[float]) -> float:
        """
        Griewank function: many widespread local minima.
        Global minimum: f(0,...,0) = 0
        Search domain: [-600, 600]^n
        """
        sum_part = sum(xi**2 for xi in x) / 4000
        prod_part = np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])
        return sum_part - prod_part + 1
