from typing import List, Tuple, Dict
import random
from Source.Problems.problem import SearchProblem
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem


class GraphColoringProblem(SearchProblem, OptimizationProblem):
    """
    Graph Coloring Problem (Discrete Optimization)

    Goal:
        Assign colors to vertices such that no adjacent vertices
        share the same color, using at most K colors.

    This implementation supports:
        - Traditional search (DFS / BFS / A*) via incremental coloring
        - Optimization algorithms (SA / PSO / DE / FA / CS) via full coloring vector

    Cost formulation (minimization):
        cost = number_of_conflicts
    A solution with cost = 0 is a valid coloring.
    """

    def __init__(self, graph: Dict[int, List[int]], num_colors: int):
        """
        Args:
            graph: adjacency list {node: [neighbors]}
            num_colors: maximum number of colors allowed
        """
        self.graph = graph
        self.num_colors = num_colors
        self.nodes = list(graph.keys())
        self.n = len(self.nodes)

        # For optimization algorithms
        self.dimensions = self.n
        self.bounds = [(0, num_colors - 1)] * self.n

        # Initial state for classical search
        # (current_node_index, color_assignment)
        self.initial_state = (0, tuple([-1] * self.n))

    # ------------------------------------------------------------------
    # Traditional Search (DFS / BFS / A*)
    # ------------------------------------------------------------------
    def get_initial_state(self):
        return self.initial_state

    def is_goal(self, state) -> bool:
        index, coloring = state
        return index == self.n

    def get_successors(self, state):
        index, coloring = state
        successors = []

        if index >= self.n:
            return successors

        node = self.nodes[index]

        for color in range(self.num_colors):
            if self._is_valid_color(index, color, coloring):
                new_coloring = list(coloring)
                new_coloring[index] = color
                successors.append((index + 1, tuple(new_coloring)))

        return successors

    def get_cost(self, state) -> float:
        _, coloring = state
        return self._count_conflicts(coloring)

    def heuristic(self, state) -> float:
        """
        Simple admissible heuristic:
        number of uncolored nodes
        """
        index, _ = state
        return self.n - index

    # ------------------------------------------------------------------
    # Optimization Algorithms (SA / PSO / DE / FA / CS)
    # ------------------------------------------------------------------
    def generate_random_state(self) -> Tuple[int, ...]:
        """Generate a random coloring (may contain conflicts)."""
        return tuple(random.randint(0, self.num_colors - 1) for _ in range(self.n))

    def evaluate_state(self, state) -> float:
        """Objective function: minimize number of conflicts."""
        return self._count_conflicts(state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_valid_color(self, node_idx: int, color: int, coloring: Tuple[int, ...]) -> bool:
        node = self.nodes[node_idx]
        for neighbor in self.graph[node]:
            neighbor_idx = self.nodes.index(neighbor)
            if coloring[neighbor_idx] == color:
                return False
        return True

    def _count_conflicts(self, coloring: Tuple[int, ...]) -> int:
        conflicts = 0
        for i, node in enumerate(self.nodes):
            for neighbor in self.graph[node]:
                j = self.nodes.index(neighbor)
                if coloring[i] != -1 and coloring[i] == coloring[j]:
                    conflicts += 1
        return conflicts // 2  # each conflict counted twice
