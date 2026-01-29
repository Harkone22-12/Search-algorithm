import math
import random
from abc import ABC, abstractmethod
from typing import Callable, Tuple, List, Optional
import numpy as np

from Source.Problems.problem import SearchProblem
from Source.Search.Search import SearchAlgorithm
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem


class SimulatedAnnealing(SearchAlgorithm):
    """
    Simulated Annealing optimization algorithm implementation.
    Inherits from SearchAlgorithm following the project structure.
    
    Attributes:
        initial_temperature (float): Starting temperature
        cooling_rate (float): Rate at which temperature decreases
        min_temperature (float): Minimum temperature threshold
        max_iterations (int): Maximum iterations per temperature level
    """
    
    def __init__(
        self,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        max_iterations: int = 1000,
        seed: Optional[int] = None
    ):
        """
        Initialize the Simulated Annealing optimizer.
        
        Args:
            initial_temperature: Starting temperature (higher = more exploration)
            cooling_rate: Multiplier for temperature decrease (0 < rate < 1)
            min_temperature: Temperature threshold to stop optimization
            max_iterations: Maximum iterations at each temperature level
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.best_state = None
        self.best_cost = float('inf')
        self.iteration_history = []
        self.temperature_history = []
        self.path = []  # Path of states visited
    
    def _acceptance_probability(
        self,
        current_cost: float,
        neighbor_cost: float,
        temperature: float
    ) -> float:
        """
        Calculate acceptance probability for a neighbor solution.
        
        Uses Metropolis criterion: exp(-(delta_cost) / temperature)
        
        Args:
            current_cost: Cost of current solution
            neighbor_cost: Cost of neighbor solution
            temperature: Current temperature
            
        Returns:
            Probability of accepting the neighbor solution
        """
        delta_cost = neighbor_cost - current_cost
        
        if delta_cost < 0:  # Improvement found
            return 1.0
        
        if temperature == 0:
            return 0.0
        
        return math.exp(-delta_cost / temperature)
    
    def search(self, problem: OptimizationProblem):
        """
        Run the Simulated Annealing optimization algorithm.
        
        Args:
            problem: OptimizationProblem instance to optimize
            
        Returns:
            Dictionary containing:
                - path: List of states visited
                - cost: Total cost (best cost found)
                - best_state: Best solution found
                - expanded_nodes: Number of state evaluations
                - iteration_history: Cost history
                - stats: Additional statistics
        """
        current_state = problem.get_start_state()
        current_cost = problem.evaluate_state(current_state)
        
        self.best_state = current_state
        self.best_cost = current_cost
        
        temperature = self.initial_temperature
        iteration_count = 0
        accepted_count = 0
        
        self.iteration_history = []
        self.temperature_history = []
        self.path = [current_state]
        
        while temperature > self.min_temperature:
            for i in range(self.max_iterations):
                # Get successor (neighbor) states
                successors = problem.get_successors(current_state)
                
                if not successors:
                    break
                    
                # Get first neighbor (SA typically explores one neighbor per iteration)
                neighbor_state, _ = successors[0]
                neighbor_cost = problem.evaluate_state(neighbor_state)
                
                # Calculate acceptance probability
                acceptance_prob = self._acceptance_probability(
                    current_cost,
                    neighbor_cost,
                    temperature
                )
                
                # Accept or reject neighbor
                if random.random() < acceptance_prob:
                    current_state = neighbor_state
                    current_cost = neighbor_cost
                    self.path.append(current_state)
                    accepted_count += 1
                
                # Update best solution found
                if current_cost < self.best_cost:
                    self.best_state = current_state
                    self.best_cost = current_cost
                
                self.expanded_nodes += 1
                iteration_count += 1
                self.iteration_history.append(self.best_cost)
            
            # Cool down
            temperature *= self.cooling_rate
            self.temperature_history.append(temperature)
        
        # Compile statistics
        stats = {
            'total_iterations': iteration_count,
            'accepted_moves': accepted_count,
            'acceptance_rate': accepted_count / iteration_count if iteration_count > 0 else 0,
            'final_temperature': temperature,
            'expanded_nodes': self.expanded_nodes
        }
        
        return {
            'path': self.path,
            'cost': self.best_cost,
            'best_state': self.best_state,
            'expanded_nodes': self.expanded_nodes,
            'iteration_history': self.iteration_history,
            'stats': stats
        }
    
    def get_iteration_history(self) -> List[float]:
        """Returns the cost history during optimization."""
        return self.iteration_history
    
    def get_temperature_history(self) -> List[float]:
        """Returns the temperature history during optimization."""
        return self.temperature_history