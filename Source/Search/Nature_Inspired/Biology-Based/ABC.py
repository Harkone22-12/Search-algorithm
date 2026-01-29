import numpy as np
from typing import Optional, Dict, List, Any
from Source.Search.Search import SearchAlgorithm
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem


class ArtificialBeeColony(SearchAlgorithm):
    """
    Artificial Bee Colony (ABC) optimization algorithm implementation.
    Inherits from SearchAlgorithm following the project structure.
    
    Attributes:
        colony_size (int): Number of bees in the colony
        num_employed (int): Number of employed bees (half of colony_size)
        limit (int): Maximum number of failed attempts before abandoning a source
        foods (np.ndarray): Array of food source positions
        fitness (np.ndarray): Fitness values of food sources
        trial (np.ndarray): Counter for unsuccessful trials per food source
        best_solution (np.ndarray): Best solution found
        best_fitness (float): Best fitness value found
    """
    
    def __init__(
        self,
        colony_size: int = 30,
        limit: int = 100,
        seed: Optional[int] = None
    ):
        """
        Initialize the Artificial Bee Colony optimizer.
        
        Args:
            colony_size: Total number of bees in colony (default: 30)
            limit: Maximum unsuccessful attempts before abandoning source (default: 100)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.colony_size = colony_size
        self.num_employed = colony_size // 2
        self.limit = limit
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize main variables
        self.foods = None  # Food source positions
        self.fitness = None  # Fitness values (nectar quantity)
        self.trial = None  # Trial counters for each food source
        self.best_solution = None  # Best solution found
        self.best_fitness = -np.inf  # Best fitness value
        self.iteration_history: List[float] = []  # Fitness history
        self.path: List[Any] = []  # Path of states visited
        
    def _initialize(self, problem: OptimizationProblem) -> None:
        """
        Initialization phase: randomly distribute bees in search space.
        Initialize food sources, fitness values, and trial counters.
        """
        dim = problem.get_dimension()
        self.foods = np.random.uniform(0, 1, (self.colony_size, dim))
        self.fitness = np.zeros(self.colony_size)
        self.trial = np.zeros(self.colony_size)
        
        # Calculate initial fitness
        for i in range(self.colony_size):
            self.fitness[i] = self._calculate_fitness(problem, self.foods[i])
            
        # Initialize best solution
        best_idx = np.argmax(self.fitness)
        self.best_solution = self.foods[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        
    def _calculate_fitness(self, problem: OptimizationProblem, solution: np.ndarray) -> float:
        """
        Calculate fitness (nectar quantity).
        
        Formula:
            fitness = 1 / (1 + f(x)) if f(x) >= 0
            fitness = 1 + |f(x)| if f(x) < 0
            
        Args:
            problem: OptimizationProblem instance
            solution: Solution vector to evaluate
            
        Returns:
            Fitness value
        """
        f_value = problem.evaluate_state(solution)
        
        if f_value >= 0:
            return 1.0 / (1.0 + f_value)
        else:
            return 1.0 + np.abs(f_value)
    
    def _employed_bee_phase(self, problem: OptimizationProblem) -> None:
        """
        Employed bee phase: each employed bee searches near current position.
        
        Formula: v_ij = x_ij + phi_ij * (x_ij - x_kj)
        where phi is random in [-1, 1] and k is randomly selected neighbor.
        """
        for i in range(self.num_employed):
            # Select random neighbor k (k != i)
            k = np.random.choice(
                [j for j in range(self.colony_size) if j != i]
            )
            
            # Create new position
            v = self.foods[i].copy()
            dim = len(v)
            
            for j in range(dim):
                # phi is random in [-1, 1]
                phi = np.random.uniform(-1, 1)
                v[j] = self.foods[i][j] + phi * (self.foods[i][j] - self.foods[k][j])
                
                # Ensure position is within bounds [0, 1]
                v[j] = np.clip(v[j], 0, 1)
            
            # Greedy selection
            v_fitness = self._calculate_fitness(problem, v)
            
            if v_fitness > self.fitness[i]:
                self.foods[i] = v
                self.fitness[i] = v_fitness
                self.trial[i] = 0
            else:
                self.trial[i] += 1
    
    def _onlooker_bee_phase(self, problem: OptimizationProblem) -> None:
        """
        Onlooker bee phase: onlooker bees select food sources based on fitness probability.
        
        Formula: P_i = fitness_i / sum(fitness)
        Onlookers select sources with better fitness with higher probability.
        """
        # Calculate selection probabilities
        probs = self.fitness / np.sum(self.fitness)
        
        # Each onlooker bee selects a food source based on probability
        for _ in range(self.colony_size - self.num_employed):
            # Select food source based on probability
            i = np.random.choice(self.colony_size, p=probs)
            
            # Select random neighbor k
            k = np.random.choice(
                [j for j in range(self.colony_size) if j != i]
            )
            
            # Find new position
            v = self.foods[i].copy()
            dim = len(v)
            
            for j in range(dim):
                phi = np.random.uniform(-1, 1)
                v[j] = self.foods[i][j] + phi * (self.foods[i][j] - self.foods[k][j])
                v[j] = np.clip(v[j], 0, 1)
            
            # Greedy selection
            v_fitness = self._calculate_fitness(problem, v)
            
            if v_fitness > self.fitness[i]:
                self.foods[i] = v
                self.fitness[i] = v_fitness
                self.trial[i] = 0
            else:
                self.trial[i] += 1
    
    def _scout_bee_phase(self, problem: OptimizationProblem) -> None:
        """
        Scout bee phase: mechanism for escaping local optima.
        
        If trial exceeds limit, replace with new random solution.
        """
        for i in range(self.colony_size):
            if self.trial[i] >= self.limit:
                # Create completely new random solution
                self.foods[i] = np.random.uniform(0, 1, problem.get_dimension())
                self.fitness[i] = self._calculate_fitness(problem, self.foods[i])
                self.trial[i] = 0
    
    def _update_best_solution(self) -> None:
        """Update best solution found so far."""
        best_idx = np.argmax(self.fitness)
        
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_solution = self.foods[best_idx].copy()
        
        self.iteration_history.append(self.best_fitness)
    
    def search(self, problem: OptimizationProblem, max_iterations: int = 100) -> Dict[str, Any]:
        """
        Run the Artificial Bee Colony optimization algorithm.
        
        Args:
            problem: OptimizationProblem instance to optimize
            max_iterations: Maximum number of iterations (default: 100)
            
        Returns:
            Dictionary containing:
                - path: List of states visited
                - cost: Total cost (best cost found)
                - best_state: Best solution found
                - expanded_nodes: Number of state evaluations
                - iteration_history: Cost history
                - stats: Additional statistics
        """
        # Step 1: Initialize
        self._initialize(problem)
        
        # Main loop
        for iteration in range(max_iterations):
            # Step 2: Employed bee phase
            self._employed_bee_phase(problem)
            
            # Step 3: Onlooker bee phase
            self._onlooker_bee_phase(problem)
            
            # Step 4: Scout bee phase
            self._scout_bee_phase(problem)
            
            # Update best solution
            self._update_best_solution()
            
            self.expanded_nodes += 1
        
        # Compile statistics
        stats = {
            'total_iterations': max_iterations,
            'colony_size': self.colony_size,
            'num_employed': self.num_employed,
            'expanded_nodes': self.expanded_nodes
        }
        
        return {
            'path': self.path,
            'cost': -self.best_fitness,  # Convert back to minimization
            'best_state': self.best_solution,
            'expanded_nodes': self.expanded_nodes,
            'iteration_history': self.iteration_history,
            'stats': stats
        }
    
    def get_iteration_history(self) -> List[float]:
        """Returns the fitness history during optimization."""
        return self.iteration_history



