import numpy as np
from typing import Optional, Dict, List, Any
from Source.Search.Search import SearchAlgorithm
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem


class FireflyAlgorithm(SearchAlgorithm):
    """
    Firefly Algorithm (FA) optimization algorithm implementation.
    Inherits from SearchAlgorithm following the project structure.
    
    Attributes:
        population_size (int): Number of fireflies
        alpha (float): Randomization parameter (exploration)
        beta0 (float): Attractiveness at distance 0
        gamma (float): Light absorption coefficient
        fireflies (np.ndarray): Array of firefly positions
        intensity (np.ndarray): Intensity (cost) of each firefly
        best_solution (np.ndarray): Best solution found
        best_intensity (float): Best intensity value found
    """
    
    def __init__(
        self,
        population_size: int = 30,
        alpha: float = 0.5,
        beta0: float = 1.0,
        gamma: float = 0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize the Firefly Algorithm optimizer.
        
        Args:
            population_size: Number of fireflies (default: 30)
            alpha: Randomization parameter initial value (default: 0.5)
            beta0: Attractiveness at distance 0 (default: 1.0)
            gamma: Light absorption coefficient (default: 0.01)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.population_size = population_size
        self.alpha = alpha
        self.alpha_initial = alpha
        self.beta0 = beta0
        self.gamma = gamma
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize main variables
        self.fireflies = None  # Firefly positions
        self.intensity = None  # Intensity values (objective function)
        self.best_solution = None  # Best solution found
        self.best_intensity = np.inf  # Best intensity (minimization)
        self.iteration_history: List[float] = []  # Intensity history
        self.path: List[Any] = []  # Path of states visited
        
    def _initialize(self, problem: OptimizationProblem) -> None:
        """
        Initialization phase: randomly distribute fireflies in search space.
        Initialize firefly positions, intensity values.
        """
        dim = problem.get_dimension()
        self.fireflies = np.random.uniform(
            0, 1, (self.population_size, dim)
        )
        self.intensity = np.zeros(self.population_size)
        
        # Calculate initial intensity (objective function values)
        for i in range(self.population_size):
            self.intensity[i] = problem.evaluate_state(self.fireflies[i])
        
        # Initialize best solution (minimum intensity - minimization)
        best_idx = np.argmin(self.intensity)
        self.best_solution = self.fireflies[best_idx].copy()
        self.best_intensity = self.intensity[best_idx]
        
    def _calculate_distance(self, firefly_i: np.ndarray, firefly_j: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two fireflies.
        
        Formula: r = ||x_i - x_j|| = sqrt(sum((x_i - x_j)^2))
        
        Args:
            firefly_i: Position of firefly i
            firefly_j: Position of firefly j
            
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(firefly_i - firefly_j)
    
    def _calculate_attractiveness(self, r: float) -> float:
        """
        Calculate attractiveness between two fireflies.
        
        Formula: beta = beta0 * exp(-gamma * r^2)
        
        Args:
            r: Distance between two fireflies
            
        Returns:
            Attractiveness value
        """
        return self.beta0 * np.exp(-self.gamma * r ** 2)
    
    def _update_firefly_position(self, i: int, j: int, problem: OptimizationProblem) -> None:
        """
        Update position of firefly i based on brighter firefly j.
        
        Formula: x_i = x_i + beta*(x_j - x_i) + alpha*(rand - 0.5)
        
        Args:
            i: Index of firefly to update
            j: Index of brighter firefly (more attractive)
            problem: OptimizationProblem instance
        """
        # Calculate distance
        r = self._calculate_distance(self.fireflies[i], self.fireflies[j])
        
        # Calculate attractiveness
        beta = self._calculate_attractiveness(r)
        
        # Update position according to formula
        # x_i = x_i + beta*(x_j - x_i) + alpha*(rand - 0.5)
        random_perturbation = self.alpha * (np.random.rand(problem.get_dimension()) - 0.5)
        self.fireflies[i] = (self.fireflies[i] + 
                            beta * (self.fireflies[j] - self.fireflies[i]) + 
                            random_perturbation)
        
        # Boundary handling - constrain position to [0, 1]
        self.fireflies[i] = np.clip(self.fireflies[i], 0, 1)
        
        # Recalculate intensity for firefly i
        self.intensity[i] = problem.evaluate_state(self.fireflies[i])
    
    def _move_fireflies(self, problem: OptimizationProblem) -> None:
        """
        Main step: move fireflies (pair-wise comparisons).
        
        Algorithm:
            - For each firefly i
            - Compare with each firefly j
            - If j is brighter than i (intensity[j] < intensity[i])
            - Then i moves towards j
        """
        for i in range(self.population_size):
            for j in range(self.population_size):
                # If firefly j is brighter than firefly i
                # (lower intensity value - minimization problem)
                if self.intensity[j] < self.intensity[i]:
                    self._update_firefly_position(i, j, problem)
    
    def _decay_alpha(self, iteration: int, max_iterations: int) -> None:
        """
        Decrease alpha parameter (Alpha Decay).
        
        As iterations increase, alpha decreases to reduce randomness
        and focus more on exploitation phase.
        
        Formula: alpha = alpha_initial * (1 - iteration / max_iterations)
        
        Args:
            iteration: Current iteration number
            max_iterations: Total maximum iterations
        """
        self.alpha = self.alpha_initial * (1 - iteration / max_iterations)
    
    def _update_best_solution(self) -> None:
        """Update best solution found so far."""
        best_idx = np.argmin(self.intensity)
        
        if self.intensity[best_idx] < self.best_intensity:
            self.best_intensity = self.intensity[best_idx]
            self.best_solution = self.fireflies[best_idx].copy()
        
        self.iteration_history.append(self.best_intensity)
    
    def search(self, problem: OptimizationProblem, max_iterations: int = 100) -> Dict[str, Any]:
        """
        Run the Firefly Algorithm optimization algorithm.
        
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
            # Main step: Move fireflies
            self._move_fireflies(problem)
            
            # Update best solution
            self._update_best_solution()
            
            # Decrease alpha parameter
            self._decay_alpha(iteration, max_iterations)
            
            self.expanded_nodes += 1
        
        # Compile statistics
        stats = {
            'total_iterations': max_iterations,
            'population_size': self.population_size,
            'final_alpha': self.alpha,
            'expanded_nodes': self.expanded_nodes
        }
        
        return {
            'path': self.path,
            'cost': self.best_intensity,
            'best_state': self.best_solution,
            'expanded_nodes': self.expanded_nodes,
            'iteration_history': self.iteration_history,
            'stats': stats
        }
    
    def get_iteration_history(self) -> List[float]:
        """Returns the intensity history during optimization."""
        return self.iteration_history

