import numpy as np
from typing import Dict, Any, List
from Source.Search.Search import SearchAlgorithm
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem
from Source.Problems.problem import SearchProblem


class DifferentialEvolution(SearchAlgorithm):
    """
    Differential Evolution (DE) Algorithm for continuous optimization.
    
    DE is an evolutionary algorithm that uses vector differences for mutation,
    making it particularly effective for continuous optimization problems.
    
    Key features:
    - Simple yet powerful optimization strategy
    - Uses difference vectors for mutation
    - Self-organizing search behavior
    - Good for multimodal optimization
    
    Attributes:
        population_size (int): Number of candidate solutions
        F (float): Differential weight/scaling factor (typically 0.5-1.0)
        CR (float): Crossover probability (typically 0.1-0.9)
        max_iterations (int): Maximum number of generations
        strategy (str): DE mutation strategy ('rand/1', 'best/1', 'current-to-best/1')
    """
    
    def __init__(
        self,
        population_size: int = 50,
        F: float = 0.8,
        CR: float = 0.9,
        max_iterations: int = 100,
        strategy: str = 'rand/1',
        seed: int | None = None
    ):
        """
        Initialize Differential Evolution optimizer.
        
        Args:
            population_size: Number of individuals in the population (NP)
            F: Differential weight for mutation (scale factor), range [0, 2]
            CR: Crossover probability, range [0, 1]
            max_iterations: Maximum number of generations
            strategy: Mutation strategy ('rand/1', 'best/1', 'current-to-best/1')
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.max_iterations = max_iterations
        self.strategy = strategy
        
        if seed is not None:
            np.random.seed(seed)
        
        self.population = None
        self.fitness = None
        self.best_state = None
        self.best_cost = np.inf
        self.history: List[float] = []
    
    def search(self, problem: SearchProblem) -> Dict[str, Any]:
        """
        Execute the Differential Evolution algorithm.
        
        Args:
            problem: OptimizationProblem instance to optimize
            
        Returns:
            Dictionary containing:
                - best_state: Best solution found
                - cost: Best fitness value
                - history: Convergence history
                - expanded_nodes: Number of fitness evaluations
                - stats: Additional statistics
        """
        assert isinstance(problem, OptimizationProblem)
        self.expanded_nodes = 0
        self.history.clear()
        
        dim = problem.dimensions
        bounds = problem.bounds
        
        # Initialize population randomly within bounds
        self.population = np.array([
            [np.random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
            for _ in range(self.population_size)
        ])
        
        # Evaluate initial population
        self.fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            self.fitness[i] = problem.evaluate_state(self.population[i])
            self.expanded_nodes += 1
        
        # Track best solution
        best_idx = np.argmin(self.fitness)
        self.best_state = self.population[best_idx].copy()
        self.best_cost = self.fitness[best_idx]
        self.history.append(self.best_cost)
        
        # Main evolution loop
        for generation in range(self.max_iterations):
            new_population = np.zeros_like(self.population)
            
            for i in range(self.population_size):
                # Mutation: Create mutant vector
                mutant = self._mutate(i, dim)
                
                # Ensure mutant is within bounds
                for d in range(dim):
                    mutant[d] = np.clip(mutant[d], bounds[d][0], bounds[d][1])
                
                # Crossover: Create trial vector
                trial = self._crossover(self.population[i], mutant, dim)
                
                # Ensure trial is within bounds
                for d in range(dim):
                    trial[d] = np.clip(trial[d], bounds[d][0], bounds[d][1])
                
                # Selection: Greedy selection between target and trial
                trial_fitness = problem.evaluate_state(trial)
                self.expanded_nodes += 1
                
                if trial_fitness < self.fitness[i]:
                    new_population[i] = trial
                    self.fitness[i] = trial_fitness
                    
                    # Update global best if needed
                    if trial_fitness < self.best_cost:
                        self.best_cost = trial_fitness
                        self.best_state = trial.copy()
                else:
                    new_population[i] = self.population[i]
            
            # Update population
            self.population = new_population
            
            # Record history
            self.history.append(self.best_cost)
        
        return {
            "best_state": self.best_state,
            "cost": self.best_cost,
            "history": self.history,
            "expanded_nodes": self.expanded_nodes,
            "stats": {
                "population_size": self.population_size,
                "F": self.F,
                "CR": self.CR,
                "strategy": self.strategy,
                "iterations": self.max_iterations
            }
        }
    
    def _mutate(self, target_idx: int, dim: int) -> np.ndarray:
        """
        Generate mutant vector using the specified strategy.
        
        Args:
            target_idx: Index of the target vector
            dim: Problem dimensionality
            
        Returns:
            Mutant vector
        """
        if self.strategy == 'rand/1':
            # Select three random distinct individuals
            indices = list(range(self.population_size))
            indices.remove(target_idx)
            r1, r2, r3 = np.random.choice(indices, size=3, replace=False)
            
            # Mutant = x_r1 + F * (x_r2 - x_r3)
            mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
            
        elif self.strategy == 'best/1':
            # Select two random distinct individuals
            indices = list(range(self.population_size))
            indices.remove(target_idx)
            r1, r2 = np.random.choice(indices, size=2, replace=False)
            
            # Mutant = x_best + F * (x_r1 - x_r2)
            best_idx = np.argmin(self.fitness)
            mutant = self.population[best_idx] + self.F * (self.population[r1] - self.population[r2])
            
        elif self.strategy == 'current-to-best/1':
            # Select two random distinct individuals
            indices = list(range(self.population_size))
            indices.remove(target_idx)
            r1, r2 = np.random.choice(indices, size=2, replace=False)
            
            # Mutant = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            best_idx = np.argmin(self.fitness)
            mutant = (self.population[target_idx] + 
                     self.F * (self.population[best_idx] - self.population[target_idx]) +
                     self.F * (self.population[r1] - self.population[r2]))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return mutant
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray, dim: int) -> np.ndarray:
        """
        Perform binomial crossover between target and mutant vectors.
        
        Args:
            target: Target vector (parent)
            mutant: Mutant vector
            dim: Problem dimensionality
            
        Returns:
            Trial vector
        """
        trial = np.zeros(dim)
        
        # Ensure at least one parameter from mutant is used
        j_rand = np.random.randint(0, dim)
        
        for j in range(dim):
            # Binomial crossover
            if np.random.random() < self.CR or j == j_rand:
                trial[j] = mutant[j]
            else:
                trial[j] = target[j]
        
        return trial
