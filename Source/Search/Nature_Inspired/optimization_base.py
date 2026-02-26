import random
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class OptimizationProblem(ABC):
    """The root base class for all optimization problems."""
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    @abstractmethod
    def evaluate_state(self, state) -> float:
        """The core cost/fitness function."""
        pass

    def evaluate(self, state):
        """Bridge for algorithms calling .evaluate()."""
        return self.evaluate_state(state)