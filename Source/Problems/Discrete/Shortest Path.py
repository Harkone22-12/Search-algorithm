from typing import Dict, List, Tuple
import random
from Source.Problems.problem import SearchProblem
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem


class ShortestPathProblem(SearchProblem, OptimizationProblem):
    """
    Shortest Path Problem on a weighted graph.

    Supports:
    - Traditional search: BFS, DFS, UCS, A*
    - Optimization algorithms: SA, PSO, DE, CS, FA

    Objective (minimization):
        Minimize total path cost from start node to goal node.
    """

    def __init__(self, graph: Dict[int, List[Tuple[int, float]]], start: int, goal: int):
        """
        Args:
            graph: adjacency list {node: [(neighbor, weight), ...]}
            start: start node
            goal: goal node
        """
        self.graph = graph
        self.start = start
        self.goal = goal
        self.nodes = list(graph.keys())

        # For optimization algorithms (path as sequence of nodes)
        self.dimensions = len(self.nodes)
        self.bounds = [(0, len(self.nodes) - 1)] * self.dimensions

        self.initial_state = start

    # ------------------------------------------------------------------
    # Traditional Search (BFS / DFS / A*)
    # ------------------------------------------------------------------
    def get_initial_state(self):
        return self.initial_state

    def is_goal(self, state) -> bool:
        return state == self.goal

    def get_successors(self, state):
        return [(neighbor, cost) for neighbor, cost in self.graph[state]]

    def get_cost(self, cost_so_far: float, step_cost: float) -> float:
        return cost_so_far + step_cost

    def heuristic(self, state) -> float:
        """
        Default heuristic = 0 (Dijkstra / UCS behavior)
        Can be overridden for spatial graphs.
        """
        return 0

    # ------------------------------------------------------------------
    # Optimization Algorithms (SA / PSO / DE / FA / CS)
    # ------------------------------------------------------------------
    def generate_random_state(self) -> Tuple[int, ...]:
        """
        Generate a random path (node sequence).
        Path validity is handled by penalty.
        """
        path = [self.start]
        current = self.start
        visited = {current}

        while current != self.goal:
            neighbors = [n for n, _ in self.graph[current] if n not in visited]
            if not neighbors:
                break
            current = random.choice(neighbors)
            visited.add(current)
            path.append(current)

        return tuple(path)

    def evaluate_state(self, state: Tuple[int, ...]) -> float:
        """
        Objective function: total path cost with penalties for invalid paths.
        """
        if not state or state[0] != self.start or state[-1] != self.goal:
            return 1e9

        cost = 0
        for i in range(len(state) - 1):
            u, v = state[i], state[i + 1]
            edge_cost = self._edge_cost(u, v)
            if edge_cost is None:
                return 1e9
            cost += edge_cost

        return cost

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _edge_cost(self, u: int, v: int):
        for neighbor, weight in self.graph[u]:
            if neighbor == v:
                return weight
        return None
