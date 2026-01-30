import numpy as np
from typing import Dict, Any, List
from Source.Search.Search import SearchAlgorithm
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem
from Source.Problems.problem import SearchProblem


class ParticleSwarmOptimization(SearchAlgorithm):
    """
    Particle Swarm Optimization (PSO) Algorithm for continuous optimization.
    
    PSO is a population-based stochastic optimization technique inspired by the
    social behavior of bird flocking or fish schooling. Each particle represents
    a potential solution and moves through the search space influenced by its
    own experience and that of its neighbors.
    
    Key features:
    - Simple concept and easy to implement
    - Few parameters to tune
    - Good for continuous optimization
    - Balances exploration and exploitation
    
    Attributes:
        population_size (int): Number of particles in the swarm
        w (float): Inertia weight (controls previous velocity influence)
        c1 (float): Cognitive parameter (personal best influence)
        c2 (float): Social parameter (global best influence)
        max_iterations (int): Maximum number of iterations
        v_max (float): Maximum velocity (optional constraint)
    """
    
    def __init__(
        self,
        population_size: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        max_iterations: int = 100,
        v_max: float | None = None,
        seed: int | None = None
    ):
        """
        Initialize Particle Swarm Optimization.
        
        Args:
            population_size: Number of particles in the swarm
            w: Inertia weight (typically 0.4-0.9), controls exploration vs exploitation
            c1: Cognitive/personal learning coefficient (typically 1.5-2.0)
            c2: Social/global learning coefficient (typically 1.5-2.0)
            max_iterations: Maximum number of iterations
            v_max: Maximum velocity (if None, set as fraction of search space)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.population_size = population_size
        self.w = w
        self.w_initial = w
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.v_max = v_max
        
        if seed is not None:
            np.random.seed(seed)
        
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_costs = None
        self.global_best_position = None
        self.global_best_cost = np.inf
        self.history: List[float] = []
    
    def search(self, problem: SearchProblem) -> Dict[str, Any]:
        """
        Execute the Particle Swarm Optimization algorithm.
        
        Args:
            problem: OptimizationProblem instance to optimize
            
        Returns:
            Dictionary containing:
                - best_state: Best solution found (global best position)
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
        
        # Set v_max if not provided (typically 10-20% of search space)
        if self.v_max is None:
            self.v_max = np.array([
                0.2 * (bounds[d][1] - bounds[d][0]) for d in range(dim)
            ])
        elif isinstance(self.v_max, (int, float)):
            self.v_max = np.array([self.v_max] * dim)
        
        # Initialize particle positions randomly within bounds
        self.particles = np.array([
            [np.random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
            for _ in range(self.population_size)
        ])
        
        # Initialize velocities randomly (typically small values)
        self.velocities = np.array([
            [np.random.uniform(-self.v_max[d], self.v_max[d]) for d in range(dim)]
            for _ in range(self.population_size)
        ])
        
        # Initialize personal best positions and costs
        self.personal_best_positions = self.particles.copy()
        self.personal_best_costs = np.zeros(self.population_size)
        
        for i in range(self.population_size):
            self.personal_best_costs[i] = problem.evaluate_state(self.particles[i])
            self.expanded_nodes += 1
        
        # Initialize global best
        best_idx = np.argmin(self.personal_best_costs)
        self.global_best_position = self.personal_best_positions[best_idx].copy()
        self.global_best_cost = self.personal_best_costs[best_idx]
        self.history.append(self.global_best_cost)
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                # Generate random coefficients for stochastic behavior
                r1 = np.random.random(dim)
                r2 = np.random.random(dim)
                
                # Update velocity
                # v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.particles[i])
                
                self.velocities[i] = (self.w * self.velocities[i] + 
                                     cognitive_component + 
                                     social_component)
                
                # Apply velocity constraints
                for d in range(dim):
                    self.velocities[i][d] = np.clip(
                        self.velocities[i][d], 
                        -self.v_max[d], 
                        self.v_max[d]
                    )
                
                # Update position
                self.particles[i] = self.particles[i] + self.velocities[i]
                
                # Apply boundary constraints
                for d in range(dim):
                    self.particles[i][d] = np.clip(
                        self.particles[i][d],
                        bounds[d][0],
                        bounds[d][1]
                    )
                
                # Evaluate new position
                current_cost = problem.evaluate_state(self.particles[i])
                self.expanded_nodes += 1
                
                # Update personal best
                if current_cost < self.personal_best_costs[i]:
                    self.personal_best_costs[i] = current_cost
                    self.personal_best_positions[i] = self.particles[i].copy()
                    
                    # Update global best
                    if current_cost < self.global_best_cost:
                        self.global_best_cost = current_cost
                        self.global_best_position = self.particles[i].copy()
            
            # Record history
            self.history.append(self.global_best_cost)
            
            # Optional: Linearly decreasing inertia weight
            # Helps transition from exploration (high w) to exploitation (low w)
            self.w = self.w_initial - (self.w_initial - 0.4) * (iteration / self.max_iterations)
        
        return {
            "best_state": self.global_best_position,
            "cost": self.global_best_cost,
            "history": self.history,
            "expanded_nodes": self.expanded_nodes,
            "stats": {
                "population_size": self.population_size,
                "initial_w": self.w_initial,
                "final_w": self.w,
                "c1": self.c1,
                "c2": self.c2,
                "iterations": self.max_iterations,
                "v_max": self.v_max if isinstance(self.v_max, float) else "dynamic"
            }
        }
    
    def get_swarm_diversity(self) -> float:
        """
        Calculate swarm diversity as average distance from center of mass.
        Useful for monitoring exploration vs exploitation.
        
        Returns:
            Average diversity measure
        """
        if self.particles is None:
            return 0.0
        
        center = np.mean(self.particles, axis=0)
        distances = [np.linalg.norm(p - center) for p in self.particles]
        return np.mean(distances)
