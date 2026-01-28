from abc import ABC, abstractmethod
from Source.Problems.problem import SearchProblem

class SearchAlgorithm(ABC):
    def __init__(self):
        self.expanded_nodes = 0

    @abstractmethod
    def search(self, problem: SearchProblem):
        """
        Returns:
        - path: list of states
        - cost: total cost
        """
        pass
