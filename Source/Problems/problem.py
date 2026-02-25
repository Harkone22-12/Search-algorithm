from Source.Search.Nature_Inspired.optimization_base import OptimizationProblem
from abc import abstractmethod

class SearchProblem(OptimizationProblem):
    """The bridge between classical search and metaheuristics."""
    @abstractmethod
    def get_start_state(self): pass

    @abstractmethod
    def is_goal(self, state): pass

    @abstractmethod
    def get_successors(self, state): pass

    def heuristic(self, state):
        return 0 # Default for A*