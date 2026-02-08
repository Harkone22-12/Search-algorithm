from collections import deque
from Source.Search.Search import SearchAlgorithm
from Source.Problems.problem import SearchProblem

class BFS(SearchAlgorithm):
    def search(self, problem: SearchProblem):
        """
        Khám phá các node ở tầng nông nhất trước.
        """
        start_state = problem.get_start_state()
        if problem.is_goal(start_state):
            return [start_state], 0

        # Queue chứa: (state hiện tại, đường đi đến state đó, tổng chi phí)
        frontier = deque([(start_state, [start_state], 0)])
        explored = {start_state}
        self.expanded_nodes = 0

        while frontier:
            current_state, path, current_cost = frontier.popleft()
            self.expanded_nodes += 1

            for next_state, cost in problem.get_successors(current_state):
                if next_state not in explored:
                    new_path = path + [next_state]
                    new_cost = current_cost + cost
                    
                    if problem.is_goal(next_state):
                        return new_path, new_cost
                    
                    explored.add(next_state)
                    frontier.append((next_state, new_path, new_cost))
        
        return None, float('inf')