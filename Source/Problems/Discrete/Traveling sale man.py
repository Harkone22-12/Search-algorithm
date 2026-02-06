from typing import List, Tuple
import math
import random
from Source.Problems.problem import SearchProblem
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem


class TSPProblem(SearchProblem, OptimizationProblem):
    """
    Traveling Salesman Problem (TSP) – Discrete version.

    This problem is designed to work with:
    - Traditional search algorithms (DFS, BFS, A*, etc.)
    - Nature-inspired optimization algorithms (SA, PSO*, DE*, CS, etc.)

    Representation:
        - State: a permutation (list) of city indices, e.g. [0, 2, 1, 3]
        - Objective: minimize total tour distance (round trip)

    NOTE:
        - For optimization algorithms, we directly evaluate full tours
        - For traditional search, successors generate partial tours

    *PSO is naturally continuous, but can still be adapted by decoding
     continuous vectors → permutations (outside this class)
    """

    def __init__(self, cities: List[Tuple[float, float]], start_city: int = 0):
        """
        Args:
            cities: List of (x, y) coordinates
            start_city: Fixed starting city index
        """
        self.cities = cities
        self.n = len(cities)
        self.start_city = start_city

        # Precompute distance matrix
        self.dist = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                self.dist[i][j] = self._euclidean(i, j)

    # ======================
    # Shared utility
    # ======================
    def _euclidean(self, i: int, j: int) -> float:
        x1, y1 = self.cities[i]
        x2, y2 = self.cities[j]
        return math.hypot(x1 - x2, y1 - y2)

    def tour_cost(self, tour: List[int]) -> float:
        """Compute total round-trip distance of a tour."""
        cost = 0.0
        for i in range(len(tour) - 1):
            cost += self.dist[tour[i]][tour[i + 1]]
        cost += self.dist[tour[-1]][tour[0]]  # return to start
        return cost

    # ======================
    # Traditional Search API
    # ======================
    def get_start_state(self) -> Tuple[int, Tuple[int, ...]]:
        """
        State = (current_city, visited_cities)
        visited_cities is a tuple for hashability
        """
        return self.start_city, (self.start_city,)

    def is_goal(self, state) -> bool:
        _, visited = state
        return len(visited) == self.n

    def get_successors(self, state):
        """
        Generate successors by visiting an unvisited city.
        Cost = distance from current city to next city
        """
        current, visited = state
        successors = []

        for city in range(self.n):
            if city not in visited:
                next_state = (city, visited + (city,))
                cost = self.dist[current][city]
                successors.append((next_state, cost))

        return successors

    def heuristic(self, state) -> float:
        """
        Simple admissible heuristic for A*:
        - Minimum distance to any unvisited city
        """
        current, visited = state
        unvisited = [c for c in range(self.n) if c not in visited]

        if not unvisited:
            return self.dist[current][self.start_city]

        return min(self.dist[current][c] for c in unvisited)

    # ======================
    # Optimization API
    # ======================
    def evaluate_state(self, state: List[int]) -> float:
        """
        Used by SA, DE, CS, etc.
        State is a permutation of city indices
        """
        return self.tour_cost(state)

    def get_dimension(self) -> int:
        """
        Dimension for optimization algorithms.
        Each dimension represents a city index.
        """
        return self.n

    def random_state(self) -> List[int]:
        """Generate a random tour."""
        tour = list(range(self.n))
        random.shuffle(tour)
        return tour
