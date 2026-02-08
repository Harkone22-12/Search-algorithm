from Source.Search.Search import SearchAlgorithm
from Source.Problems.problem import SearchProblem

class DFS(SearchAlgorithm):
    def search(self, problem: SearchProblem):
        """
        Khám phá các node sâu nhất trong cây tìm kiếm trước.
        """
        start_state = problem.get_start_state()
        
        # Stack chứa: (state hiện tại, đường đi đến state đó, tổng chi phí)
        frontier = [(start_state, [start_state], 0)]
        explored = set()
        self.expanded_nodes = 0

        while frontier:
            current_state, path, current_cost = frontier.pop()
            
            if problem.is_goal(current_state):
                return path, current_cost
            
            if current_state not in explored:
                explored.add(current_state)
                self.expanded_nodes += 1
                
                for next_state, cost in problem.get_successors(current_state):
                    if next_state not in explored:
                        new_path = path + [next_state]
                        new_cost = current_cost + cost
                        frontier.append((next_state, new_path, new_cost))
                        
        return None, float('inf')