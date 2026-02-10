"""
A* Search Algorithm
Traditional informed search - for pathfinding problems
"""

from typing import Dict, Any, List
import heapq
from Source.Search.Search import SearchAlgorithm
from Source.Problems.problem import SearchProblem


class AStarSearch(SearchAlgorithm):
    """
    A* Search algorithm for pathfinding.
    Uses f(n) = g(n) + h(n) where:
    - g(n): actual cost from start to n
    - h(n): heuristic estimate from n to goal
    """
    
    def __init__(self):
        super().__init__()
        self.visited = set()
    
    def search(self, problem: SearchProblem) -> Dict[str, Any]:
        """
        Execute A* search.
        
        Returns:
            Dictionary with:
            - path: list of states from start to goal
            - cost: total path cost
            - expanded_nodes: number of nodes expanded
        """
        self.expanded_nodes = 0
        self.visited = set()
        
        start = problem.get_start_state()
        
        # Priority queue: (f_cost, g_cost, state, path)
        pq = []
        g = 0
        h = problem.heuristic(start)
        f = g + h
        heapq.heappush(pq, (f, g, start, [start]))
        
        # Track best g-cost to each state
        best_g = {start: 0}
        
        while pq:
            curr_f, curr_g, curr_state, curr_path = heapq.heappop(pq)
            
            if curr_state in self.visited:
                continue
            
            self.visited.add(curr_state)
            self.expanded_nodes += 1
            
            # Goal check
            if problem.is_goal(curr_state):
                return {
                    'path': curr_path,
                    'cost': curr_g,
                    'expanded_nodes': self.expanded_nodes
                }
            
            # Expand successors
            for next_state, step_cost in problem.get_successors(curr_state):
                if next_state not in self.visited:
                    new_g = curr_g + step_cost
                    
                    if next_state not in best_g or new_g < best_g[next_state]:
                        best_g[next_state] = new_g
                        h = problem.heuristic(next_state)
                        new_f = new_g + h
                        new_path = curr_path + [next_state]
                        heapq.heappush(pq, (new_f, new_g, next_state, new_path))
        
        # No solution found
        return {
            'path': [],
            'cost': float('inf'),
            'expanded_nodes': self.expanded_nodes
        }
