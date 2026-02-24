import numpy as np
from typing import Dict, Any, List
from Source.Search.Search import SearchAlgorithm
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem
from Source.Problems.problem import SearchProblem


class ArtificialBeeColony(SearchAlgorithm):
    def __init__(
        self,
        colony_size: int = 30,
        limit: int = 100,
        max_iterations: int = 100,
        seed: int | None = None
    ):
        super().__init__()
        self.colony_size = colony_size
        self.num_employed = colony_size // 2
        self.limit = limit
        self.max_iterations = max_iterations

        if seed is not None:
            np.random.seed(seed)

        self.foods = None
        self.costs = None
        self.trials = None
        self.best_state = None
        self.best_cost = np.inf
        self.history: List[float] = []

    def search(self, problem: SearchProblem) -> Dict[str, Any]:
        assert isinstance(problem, OptimizationProblem)
        self.expanded_nodes = 0
        self.history.clear()

        dim = problem.dimensions
        bounds = problem.bounds

        # Initialize food sources
        self.foods = np.array([
            [np.random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
            for _ in range(self.colony_size)
        ])

        self.costs = np.zeros(self.colony_size)
        self.trials = np.zeros(self.colony_size)

        for i in range(self.colony_size):
            self.costs[i] = problem.evaluate_state(self.foods[i])
            self.expanded_nodes += 1

        best_idx = np.argmin(self.costs)
        self.best_state = self.foods[best_idx].copy()
        self.best_cost = self.costs[best_idx]

        for _ in range(self.max_iterations):
            # Employed bees
            for i in range(self.num_employed):
                k = np.random.choice([j for j in range(self.colony_size) if j != i])
                phi = np.random.uniform(-1, 1, dim)

                candidate = self.foods[i] + phi * (self.foods[i] - self.foods[k])
                for d in range(dim):
                    candidate[d] = np.clip(candidate[d], bounds[d][0], bounds[d][1])

                cost = problem.evaluate_state(candidate)
                self.expanded_nodes += 1

                if cost < self.costs[i]:
                    self.foods[i] = candidate
                    self.costs[i] = cost
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1

            # Onlooker bees
            probs = (1 / (1 + self.costs)) / np.sum(1 / (1 + self.costs))
            for _ in range(self.colony_size - self.num_employed):
                i = np.random.choice(self.colony_size, p=probs)
                k = np.random.choice([j for j in range(self.colony_size) if j != i])

                phi = np.random.uniform(-1, 1, dim)
                candidate = self.foods[i] + phi * (self.foods[i] - self.foods[k])
                for d in range(dim):
                    candidate[d] = np.clip(candidate[d], bounds[d][0], bounds[d][1])

                cost = problem.evaluate_state(candidate)
                self.expanded_nodes += 1

                if cost < self.costs[i]:
                    self.foods[i] = candidate
                    self.costs[i] = cost
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1

            # Scout bees
            for i in range(self.colony_size):
                if self.trials[i] >= self.limit:
                    self.foods[i] = np.array([
                        np.random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)
                    ])
                    self.costs[i] = problem.evaluate_state(self.foods[i])
                    self.expanded_nodes += 1
                    self.trials[i] = 0

            best_idx = np.argmin(self.costs)
            if self.costs[best_idx] < self.best_cost:
                self.best_cost = self.costs[best_idx]
                self.best_state = self.foods[best_idx].copy()

            self.history.append(self.best_cost)

        return {
            "best_state": self.best_state,
            "cost": self.best_cost,
            "history": self.history,
            "expanded_nodes": self.expanded_nodes,
            "stats": {
                "colony_size": self.colony_size,
                "limit": self.limit,
                "iterations": self.max_iterations
            }
        }
