from typing import List, Tuple
import random
from Source.Problems.problem import SearchProblem
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem


class KnapsackProblem(SearchProblem, OptimizationProblem):
    """
    0/1 Knapsack Problem (Discrete Optimization)

    - Traditional search (BFS, DFS, A*): state-based formulation
    - Nature-inspired algorithms (SA, GA, CS, DE, PSO*): binary vector formulation

    Each item can be either taken (1) or not taken (0).
    Objective: maximize total value without exceeding capacity.
    (Converted to minimization by negating value.)
    """

    def __init__(self, weights: List[int], values: List[int], capacity: int):
        assert len(weights) == len(values)
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.n_items = len(weights)

        # For optimization algorithms
        self.dimensions = self.n_items
        self.bounds = [(0, 1)] * self.n_items

        # Initial state for traditional search
        self.initial_state = (0, tuple([0] * self.n_items))

    # ------------------------------------------------------------------
    # Traditional Search (DFS / BFS / A*)
    # ------------------------------------------------------------------
    def get_initial_state(self):
        return self.initial_state

    def is_goal(self, state) -> bool:
        index, _ = state
        return index == self.n_items

    def get_successors(self, state):
        index, taken = state
        successors = []

        if index >= self.n_items:
            return successors

        # Option 1: do not take item
        successors.append((index + 1, taken))

        # Option 2: take item if capacity allows
        current_weight = self._total_weight(taken)
        if current_weight + self.weights[index] <= self.capacity:
            new_taken = list(taken)
            new_taken[index] = 1
            successors.append((index + 1, tuple(new_taken)))

        return successors

    def get_cost(self, state) -> float:
        _, taken = state
        return -self._total_value(taken)

    def heuristic(self, state) -> float:
        """
        Simple admissible heuristic:
        optimistic value of remaining items (fractional knapsack idea)
        """
        index, taken = state
        remaining_capacity = self.capacity - self._total_weight(taken)
        value = 0

        for i in range(index, self.n_items):
            if self.weights[i] <= remaining_capacity:
                remaining_capacity -= self.weights[i]
                value += self.values[i]

        return -value

    # ------------------------------------------------------------------
    # Optimization Algorithms (SA / CS / DE / FA / PSO)
    # ------------------------------------------------------------------
    def generate_random_state(self) -> Tuple[int, ...]:
        """Generate a random valid binary solution."""
        state = [0] * self.n_items
        indices = list(range(self.n_items))
        random.shuffle(indices)

        total_weight = 0
        for i in indices:
            if total_weight + self.weights[i] <= self.capacity:
                state[i] = 1
                total_weight += self.weights[i]

        return tuple(state)

    def evaluate_state(self, state) -> float:
        """
        Objective function (minimization):
        -total_value + penalty if overweight
        """
        weight = self._total_weight(state)
        value = self._total_value(state)

        if weight > self.capacity:
            penalty = (weight - self.capacity) * 1000
            return penalty

        return -value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _total_weight(self, state) -> int:
        return sum(w for w, s in zip(self.weights, state) if s == 1)

    def _total_value(self, state) -> int:
        return sum(v for v, s in zip(self.values, state) if s == 1)
