from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by the search
             max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####
    # max_frontier_size = 0
    max_frontier_size = 1
    num_nodes_expanded = 0
    path = []

    frontier = deque()
    start_node = Node(parent=None, state=problem.init_state, action=None, path_cost=0)
    frontier.append(start_node)

    visited = set()
    visited.add(start_node.state)

    while frontier: # BFS loop
        curr_node = frontier.popleft() # remove first node from queue
        num_nodes_expanded = num_nodes_expanded + 1
        if problem.goal_test(curr_node.state): # Found goal
            path = problem.trace_path(curr_node)
            return path, num_nodes_expanded, max_frontier_size
        
        actions = problem.get_actions(curr_node.state)
        # expanding children
        for action in actions:
            # Using 'get_child_node' from SimpleSearchProblem to get the next state as a Node
            child_node = problem.get_child_node(curr_node, action)

            if child_node.state not in visited:
                visited.add(child_node.state)
                frontier.append(child_node)

        # Update max_frontier_size if needed
        if len(frontier) > max_frontier_size:
            max_frontier_size = len(frontier) # consider returning only len(frontier) instead of max frontier size

    # If we exhaust the queue without finding a goal, return failure (empty path)
    return [], num_nodes_expanded, max_frontier_size

if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    # E = np.loadtxt('../datasets/stanford_large_network_facebook_combined.txt', dtype=int)
    E = np.loadtxt('./stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)