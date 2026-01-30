import numpy as np
from typing import Dict, Any, List
from Source.Search.Search import SearchAlgorithm
from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem
from Source.Problems.problem import SearchProblem


class FireflyAlgorithm(SearchAlgorithm):
    def __init__(
        self,
        population_size: int = 30,
        alpha: float = 0.5,
        beta0: float = 1.0,
        gamma: float = 0.01,
        max_iterations: int = 100,
        seed: int | None = None
    ):
        super().__init__()
        self.population_size = population_size
        self.alpha = alpha
        self.alpha_initial = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.max_iterations = max_iterations

        if seed is not None:
            np.random.seed(seed)

        self.fireflies = None
        self.intensity = None
        self.best_state = None
        self.best_cost = np.inf
        self.history: List[float] = []

    def search(self, problem: SearchProblem) -> Dict[str, Any]:
        assert isinstance(problem, OptimizationProblem)
        self.expanded_nodes = 0
        self.history.clear()

        dim = problem.dimensions
        bounds = problem.bounds

        # Initialize population
        self.fireflies = np.array([
            [np.random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
            for _ in range(self.population_size)
        ])

        self.intensity = np.zeros(self.population_size)
        for i in range(self.population_size):
            self.intensity[i] = problem.evaluate_state(self.fireflies[i])
            self.expanded_nodes += 1

        best_idx = np.argmin(self.intensity)
        self.best_state = self.fireflies[best_idx].copy()
        self.best_cost = self.intensity[best_idx]

        # Main loop
        for t in range(self.max_iterations):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if self.intensity[j] < self.intensity[i]:
                        r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                        beta = self.beta0 * np.exp(-self.gamma * r * r)

                        step = (
                            beta * (self.fireflies[j] - self.fireflies[i]) +
                            self.alpha * (np.random.rand(dim) - 0.5)
                        )

                        new_pos = self.fireflies[i] + step

                        # Bound handling
                        for d in range(dim):
                            new_pos[d] = np.clip(
                                new_pos[d], bounds[d][0], bounds[d][1]
                            )

                        new_cost = problem.evaluate_state(new_pos)
                        self.expanded_nodes += 1

                        if new_cost < self.intensity[i]:
                            self.fireflies[i] = new_pos
                            self.intensity[i] = new_cost

                            if new_cost < self.best_cost:
                                self.best_cost = new_cost
                                self.best_state = new_pos.copy()

            self.history.append(self.best_cost)
            self.alpha = self.alpha_initial * (1 - t / self.max_iterations)

        return {
            "best_state": self.best_state,
            "cost": self.best_cost,
            "history": self.history,
            "expanded_nodes": self.expanded_nodes,
            "stats": {
                "population_size": self.population_size,
                "final_alpha": self.alpha,
                "iterations": self.max_iterations
            }
        }
