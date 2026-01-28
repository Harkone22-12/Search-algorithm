from abc import ABC, abstractmethod

class SearchProblem(ABC):
    @abstractmethod
    def get_start_state(self):
        pass

    @abstractmethod
    def is_goal(self, state):
        pass

    @abstractmethod
    def get_successors(self, state):
        """
        Return list of (next_state, cost)
        """
        pass

    def heuristic(self, state):
        """
        Default heuristic = 0
        A* sẽ override hoặc dùng cái này
        """
        return 0
