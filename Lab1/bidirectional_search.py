from collections import deque
import numpy as np
from breadth_first_search import breadth_first_search # delete this line
from search_problems import Node, GraphSearchProblem

from collections import deque
from search_problems import Node, GraphSearchProblem

def bidirectional_search(problem):
    """
    Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state.

    :param problem: instance of SimpleSearchProblem
    :return:
        path: a list of states (ints) describing the path from problem.init_state to problem.goal_states[0]
        num_nodes_expanded: number of nodes expanded by the search
        max_frontier_size: maximum frontier size during search
    """

    # For this assignment, we'll assume there's exactly one goal state: problem.goal_states[0].
    goal_state = problem.goal_states[0]
    
    # 1. Edge case: if initial state is already the goal, we are done.
    if problem.init_state == goal_state:
        return [problem.init_state], 0, 1

    ####
    # Data structures
    ####
    # Frontiers (double-ended queues) for forward and backward searches
    frontierF = deque()
    frontierB = deque()

    # Visited dictionaries: state -> Node
    # We store the Node so we can reconstruct paths later
    visitedF = {}
    visitedB = {}

    # Initialize the forward search
    start_node = Node(parent=None, 
                      state=problem.init_state, 
                      action=None, 
                      path_cost=0)
    frontierF.append(start_node)
    visitedF[start_node.state] = start_node

    # Initialize the backward search
    goal_node = Node(parent=None, 
                     state=goal_state, 
                     action=None, 
                     path_cost=0)
    frontierB.append(goal_node)
    visitedB[goal_node.state] = goal_node

    # Stats tracking
    max_frontier_size = 2  # we start with 1 node in each frontier
    num_nodes_expanded = 0

    ####
    # 2. Bidirectional loop
    ####
    while frontierF and frontierB:
        
        # Update max frontier size
        current_frontier_size = len(frontierF) + len(frontierB)
        if current_frontier_size > max_frontier_size:
            max_frontier_size = current_frontier_size

        # --- Expand from the forward frontier ---
        num_nodes_expanded += 1
        currentF = frontierF.popleft()

        # Get actions (neighbors) from the forward side
        actionsF = problem.get_actions(currentF.state)
        for action in actionsF:
            childF = problem.get_child_node(currentF, action)
            if childF.state not in visitedF:
                visitedF[childF.state] = childF
                frontierF.append(childF)

                # Check if we've seen this state from backward search
                if childF.state in visitedB:
                    # We have a meeting point, reconstruct full path!
                    return _reconstruct_bidirectional_path(problem, 
                                                           childF.state,
                                                           visitedF, 
                                                           visitedB), \
                           num_nodes_expanded, \
                           max_frontier_size

        # --- Expand from the backward frontier ---
        if frontierB:
            num_nodes_expanded += 1
            currentB = frontierB.popleft()

            # Get actions (neighbors) from the backward side
            # But note: We need to treat them as "reverse" edges, or rely on the same method 
            # (since it's an undirected graph, the same get_actions will work).
            actionsB = problem.get_actions(currentB.state)
            for action in actionsB:
                childB = problem.get_child_node(currentB, action)
                if childB.state not in visitedB:
                    visitedB[childB.state] = childB
                    frontierB.append(childB)

                    # Check if we've seen this state from forward search
                    if childB.state in visitedF:
                        # Meeting point found
                        return _reconstruct_bidirectional_path(problem, 
                                                               childB.state,
                                                               visitedF, 
                                                               visitedB), \
                               num_nodes_expanded, \
                               max_frontier_size

    # If we exhaust either frontier without meeting, no solution
    return [], num_nodes_expanded, max_frontier_size

def _reconstruct_bidirectional_path(problem, meet_state, visitedF, visitedB):
    """
    Given the state where forward and backward searches meet, reconstruct
    the full path from the problem.init_state to problem.goal_states[0].
    """

    # Step 1: Reconstruct from 'meet_state' back to initial state
    pathF = []
    # We'll climb up parents in the forward dictionary
    nodeF = visitedF[meet_state]
    while nodeF is not None:
        pathF.append(nodeF.state)
        nodeF = nodeF.parent
    pathF.reverse()  # because we built it from child -> parent

    # Step 2: Reconstruct from 'meet_state' forward to goal in the backward dictionary
    # For the backward dictionary, the 'meet_state' node is presumably a child of the goal node 
    # if we traced from goal->...->meet_state. So we climb up parents in visitedB 
    # until we reach the actual goal_node (parent=None).
    pathB = []
    nodeB = visitedB[meet_state]
    while nodeB is not None:
        pathB.append(nodeB.state)
        nodeB = nodeB.parent
    # Notice this path goes [meet_state -> ... -> goal_state]
    # We actually want to go from meet_state to goal_state, so we do NOT reverse it
    # but be mindful that if your backward search "parents" point in a different direction,
    # you might have to adjust. In many setups, we simply reverse pathB except the first node
    # to seamlessly connect.

    # Because pathB currently starts at meet_state and ends at goal_state, 
    # but we want to avoid duplicating 'meet_state' in the join. 
    # So we'll skip the first element of pathB (which is meet_state),
    # then add the rest to pathF.
    return pathF + pathB[1:]


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