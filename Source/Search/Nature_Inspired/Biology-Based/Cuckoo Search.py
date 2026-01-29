import random
import math
from typing import Tuple, List
from Source.Problems import problem
import numpy as np
from Source.Problems.problem import SearchProblem
from Source.Search.Search import SearchAlgorithm
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem

# filepath: e:\Git\Search-algorithm\Source\Search\Nature-Inspired\Biology-Based\Cuckoo Search.py
"""
Cuckoo Search (CS) Algorithm
A nature-inspired metaheuristic based on the parasitic breeding behavior of cuckoo birds.
"""




class CuckooSearch(SearchAlgorithm):
    """
    Cuckoo Search Algorithm for optimization.
    
    Based on:
    - Parasitic breeding behavior of cuckoo birds
    - Lévy flight for exploration
    - Nest abandonment and new nest construction
    """
    
    def __init__(
        self,
        population_size: int = 25,
        pa: float = 0.25,
        max_iterations: int = 1000,
        seed: int = None
    ):
        """
        Initialize Cuckoo Search.
        
        Args:
            population_size: Number of nests (solutions)
            pa: Probability of nest discovery (0.0 to 1.0)
            max_iterations: Maximum number of iterations
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.population_size = population_size
        self.pa = pa  # Discovery probability
        self.max_iterations = max_iterations
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def search(self, problem: SearchProblem):
        assert isinstance(problem, OptimizationProblem)
        """
        Execute Cuckoo Search algorithm.
        
        Returns:
            - path: history of best solutions found
            - cost: best fitness value
        """
        # Initialize nests with random solutions
        nests = [problem.generate_random_state() for _ in range(self.population_size)]
        fitness = [problem.evaluate_state(nest) for nest in nests]
        
        best_nest = nests[np.argmin(fitness)]
        best_fitness = min(fitness)
        
        iteration_history = [best_fitness]
        
        for iteration in range(self.max_iterations):
            # Generate new solutions via Lévy flight
            for i in range(self.population_size):
                # Generate cuckoo solution via Lévy flight
                new_nest = self._levy_flight(best_nest, problem)
                new_fitness = problem.evaluate_state(new_nest)
                
                self.expanded_nodes += 1
                
                # Greedy selection: replace if better
                if new_fitness < fitness[i]:
                    nests[i] = new_nest
                    fitness[i] = new_fitness
                
                # Update best solution
                if new_fitness < best_fitness:
                    best_nest = new_nest
                    best_fitness = new_fitness
            
            # Nest abandonment and discovery
            nests, fitness = self._abandon_nests(nests, fitness, problem)
            
            iteration_history.append(best_fitness)
        
        return {
            "best_state": best_nest,
            "cost": best_fitness,
            "history": iteration_history,
            "expanded_nodes": self.expanded_nodes
        }
    
    def _levy_flight(self, best_solution, problem, beta: float = 1.5) -> tuple:
        """
        Generate new solution using Lévy flight.
        
        Args:
            best_solution: Current best solution
            problem: Optimization problem
            beta: Lévy flight parameter
            
        Returns:
            New solution from Lévy flight
        """
        n = problem.dimensions
        
        # Calculate sigma
        numerator = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        denominator = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma = (numerator / denominator) ** (1 / beta)
        
        # Generate random numbers from normal distribution
        u = np.random.normal(0, sigma, n)
        v = np.random.normal(0, 1, n)
        
        # Lévy step
        step = u / (np.abs(v) ** (1 / beta))
        
        # New solution
        new_nest = list(best_solution)
        for j in range(n):
            new_nest[j] = new_nest[j] + step[j]
            # Enforce bounds
            new_nest[j] = max(
                problem.bounds[j][0],
                min(problem.bounds[j][1], new_nest[j])
            )
        
        return tuple(new_nest)
    
    def _abandon_nests(self, nests: List, fitness: List, problem: SearchProblem) -> Tuple[List, List]:
        """
        Abandon nests with probability pa and generate new ones.
        
        Args:
            nests: Current nest population
            fitness: Fitness values of nests
            problem: Optimization problem
            
        Returns:
            Updated nests and fitness values
        """
        new_nests = nests.copy()
        new_fitness = fitness.copy()
        
        for i in range(self.population_size):
            if random.random() < self.pa:
                # Abandon this nest and create a new random one
                new_nests[i] = problem.generate_random_state()
                new_fitness[i] = problem.evaluate_state(new_nests[i])
                self.expanded_nodes += 1
        
        return new_nests, new_fitness