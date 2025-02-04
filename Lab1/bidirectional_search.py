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

        if problem.goal_test(curr_node.state): # found goal
            path = problem.trace_path(curr_node)
            return path, num_nodes_expanded, max_frontier_size
        
        actions = problem.get_actions(curr_node.state)
        # expanding children
        for action in actions:
            child_node = problem.get_child_node(curr_node, action)

            if child_node.state not in visited:
                visited.add(child_node.state)
                frontier.append(child_node)

    return [], num_nodes_expanded, len(frontier) # did not find path

def reconstruct_bidirectional_path(meet_state, visitedF, visitedB):

    def reconstruct_forward(node):
        if node is None:
            return []
        return reconstruct_forward(node.parent) + [node.state]

    def reconstruct_backward(node):
        if node is None:
            return []
        return [node.state] + reconstruct_backward(node.parent)

    pathF = reconstruct_forward(visitedF[meet_state])
    pathB = reconstruct_backward(visitedB[meet_state])[1:]

    return pathF + pathB

def expand_frontier(frontier, visited, problem):
    """Expands the given frontier by processing all nodes in it."""
    next_frontier = deque()
    num_nodes_expanded = 0
    
    while frontier:
        current = frontier.popleft()
        num_nodes_expanded += 1
        for action in problem.get_actions(current.state):
            child = problem.get_child_node(current, action)
            if child.state not in visited:
                visited[child.state] = child
                next_frontier.append(child)
    
    return next_frontier, num_nodes_expanded

def check_intersection(visitedF, visitedB):
    """Checks if the frontiers intersect and returns the best meeting state if found."""
    meeting_states = set(visitedF.keys()).intersection(set(visitedB.keys()))
    
    if meeting_states:
        best_state = None
        best_cost = float('inf')
        for state in meeting_states:
            cost = visitedF[state].path_cost + visitedB[state].path_cost
            if cost < best_cost:
                best_cost = cost
                best_state = state
        return best_state
    
    return None

def bidirectional_search(problem):
    """Performs bidirectional search using forward and backward frontiers."""

    num_nodes_expanded = 0
    max_frontier_size = 2

    goal_state = problem.goal_states[0]

    visitedF = {}
    visitedB = {}

    start_node = Node(parent=None, state=problem.init_state, action=None, path_cost=0)
    goal_node = Node(parent=None, state=goal_state, action=None, path_cost=0)
    frontierF = deque()
    frontierF.append(start_node)
    frontierB = deque()
    frontierB.append(goal_node)
    visitedF[start_node.state] = start_node
    visitedB[goal_node.state] = goal_node

    while frontierF and frontierB:
        frontierF, expandedF = expand_frontier(frontierF, visitedF, problem)
        num_nodes_expanded += expandedF
        max_frontier_size = max(max_frontier_size, len(frontierF) + len(frontierB))

        best_state = check_intersection(visitedF, visitedB)
        if best_state:
            return reconstruct_bidirectional_path(best_state, visitedF, visitedB), num_nodes_expanded, max_frontier_size

        frontierB, expandedB = expand_frontier(frontierB, visitedB, problem)
        num_nodes_expanded += expandedB
        max_frontier_size = max(max_frontier_size, len(frontierF) + len(frontierB))

        best_state = check_intersection(visitedF, visitedB)
        if best_state:
            return reconstruct_bidirectional_path(best_state, visitedF, visitedB), num_nodes_expanded, max_frontier_size

    return [], num_nodes_expanded, max_frontier_size  # No solution found

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
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Be sure to compare with breadth_first_search!