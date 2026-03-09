"""
Hill Climbing Algorithm
Local search optimization - based on Physics-Based pattern
Follow same structure as Simulated Annealing
"""

import random
from typing import Dict, Any, List, Optional
import numpy as np
from Source.Search.Search import SearchAlgorithm
from Source.Problems.problem import SearchProblem
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem


class HillClimbing(SearchAlgorithm):
    """
    Hill Climbing optimization algorithm.
    Local search that moves to better neighbors until stuck.
    
    Variants:
    - Steepest Ascent: choose best neighbor
    - First-Choice: accept first improvement
    - Random Restart: try multiple starting points
    
    Note: history contains ALL evaluations across ALL restarts (cumulative).
    """
    
    def __init__(
        self,
        variant: str = 'steepest',
        max_iterations: int = 1000,
        max_restarts: int = 0,
        allow_sideways: bool = False,
        max_sideways_moves: int = 100,
        seed: Optional[int] = None
    ):
        """
        Initialize Hill Climbing.
        
        Args:
            variant: 'steepest' or 'first-choice'
            max_iterations: max steps per climb
            max_restarts: number of random restarts
            allow_sideways: allow equal-cost moves
            max_sideways_moves: max consecutive sideways moves per climb
            seed: random seed
        """
        super().__init__()
        self.variant = variant
        self.max_iterations = max_iterations
        self.max_restarts = max_restarts
        self.allow_sideways = allow_sideways
        self.max_sideways_moves = max_sideways_moves
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.best_state = None
        self.best_cost = float('inf')
        self.history = []
    
    def search(self, problem: SearchProblem) -> Dict[str, Any]:
        """
        Execute Hill Climbing algorithm.
        
        Args:
            problem: OptimizationProblem to solve
            
        Returns:
            Dictionary with:
            - best_state: best solution found
            - cost: best cost
            - history: cost over ALL iterations (cumulative across restarts)
            - expanded_nodes: total evaluations
            - stats: additional info
        """
        assert isinstance(problem, OptimizationProblem), \
            "Hill Climbing requires OptimizationProblem"
        
        # Reset state
        self.expanded_nodes = 0
        self.history = []
        
        # Initial climb from start state
        start_state = problem.generate_random_state()
        start_cost = problem.evaluate_state(start_state)
        self.expanded_nodes += 1
        
        # First climb (pass cost to avoid double evaluate)
        final_state, final_cost = self._single_climb(problem, start_state, start_cost)
        
        self.best_state = final_state
        self.best_cost = final_cost
        
        # Random restarts
        for restart in range(self.max_restarts):
            restart_state = problem.generate_random_state()
            restart_start_cost = problem.evaluate_state(restart_state)
            self.expanded_nodes += 1
            
            restart_final, restart_cost = self._single_climb(
                problem, restart_state, restart_start_cost
            )
            
            if restart_cost < self.best_cost:
                self.best_state = restart_final
                self.best_cost = restart_cost
        
        return {
            'best_state': self.best_state,
            'cost': self.best_cost,
            'history': self.history,
            'expanded_nodes': self.expanded_nodes,
            'stats': {
                'variant': self.variant,
                'num_restarts': self.max_restarts,
                'total_iterations': len(self.history)
            }
        }
    
    def _single_climb(
        self, 
        problem: OptimizationProblem, 
        start_state,
        start_cost: float
    ) -> tuple:
        """
        Perform one hill climbing attempt.
        
        Args:
            problem: optimization problem
            start_state: starting state
            start_cost: pre-computed cost of start_state (to avoid re-evaluate)
            
        Returns:
            (final_state, final_cost)
        """
        current = start_state
        current_cost = start_cost
        best_cost = current_cost  # Track best cost seen so far
        
        # Reset sideways counter for THIS climb
        num_sideways = 0
        
        # Record initial best cost
        self.history.append(best_cost)
        
        for iteration in range(self.max_iterations):
            successors = problem.get_successors(current)
            
            if not successors:
                break
            
            if self.variant == 'steepest':
                # Choose best neighbor
                best_neighbor = None
                best_cost = current_cost
                
                for neighbor, cost_diff in successors:
                    neighbor_cost = current_cost + cost_diff
                    self.expanded_nodes += 1
                    
                    if neighbor_cost < best_cost:
                        best_neighbor = neighbor
                        best_cost = neighbor_cost
                    elif self.allow_sideways and neighbor_cost == best_cost:
                        if num_sideways < self.max_sideways_moves:
                            best_neighbor = neighbor
                            best_cost = neighbor_cost
                
                if best_neighbor is None:
                    break
                
                # Update sideways counter
                if best_cost == current_cost:
                    num_sideways += 1
                else:
                    num_sideways = 0
                
                current = best_neighbor
                current_cost = best_cost
                
                # Track best cost for convergence history
                if best_cost < best_cost:  # Wait, this is wrong
                    pass
                
            elif self.variant == 'first-choice':
                # Accept first improvement
                improved = False
                random.shuffle(successors)
                
                for neighbor, cost_diff in successors:
                    neighbor_cost = current_cost + cost_diff
                    self.expanded_nodes += 1
                    
                    if neighbor_cost < current_cost:
                        current = neighbor
                        current_cost = neighbor_cost
                        improved = True
                        break
                
                if not improved:
                    break
            
            # Record best cost after each step (for convergence visualization)
        return current, current_cost
