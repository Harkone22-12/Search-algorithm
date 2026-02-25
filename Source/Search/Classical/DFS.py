from Source.Search.Search import SearchAlgorithm
from Source.Problems.problem import SearchProblem

class DFS(SearchAlgorithm):
    def search(self, problem: SearchProblem):
        start_state = problem.get_start_state()
        frontier = [(start_state, [start_state], 0)]
        explored = set()
        self.expanded_nodes = 0

        while frontier:
            current_state, path, current_cost = frontier.pop()
            
            if problem.is_goal(current_state):
                return {
                    'path': path, 
                    'cost': float(current_cost), 
                    'nodes': self.expanded_nodes
                }
            
            if current_state not in explored:
                explored.add(current_state)
                self.expanded_nodes += 1
                
                for next_state, cost in problem.get_successors(current_state):
                    if next_state not in explored:
                        new_path = path + [next_state]
                        new_cost = current_cost + cost
                        frontier.append((next_state, new_path, new_cost))
                        
        return {'path': None, 'cost': float('inf'), 'nodes': self.expanded_nodes}