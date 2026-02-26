from collections import deque
from Source.Search.Search import SearchAlgorithm
from Source.Problems.problem import SearchProblem

class BFS(SearchAlgorithm):
    def search(self, problem: SearchProblem):
        start_state = problem.get_start_state()
        self.expanded_nodes = 0

        if problem.is_goal(start_state):
            return {
                'path': [start_state], 
                'cost': 0.0, 
                'nodes': self.expanded_nodes
            }

        frontier = deque([(start_state, [start_state], 0)])
        explored = {start_state}

        while frontier:
            current_state, path, current_cost = frontier.popleft()
            self.expanded_nodes += 1

            for next_state, cost in problem.get_successors(current_state):
                if next_state not in explored:
                    new_path = path + [next_state]
                    new_cost = current_cost + cost
                    
                    if problem.is_goal(next_state):
                        return {
                            'path': new_path, 
                            'cost': float(new_cost), 
                            'nodes': self.expanded_nodes
                        }
                    
                    explored.add(next_state)
                    frontier.append((next_state, new_path, new_cost))
        
        return {'path': None, 'cost': float('inf'), 'nodes': self.expanded_nodes}