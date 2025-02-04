import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem

def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. 

    :param problem: an instance of GridSearchProblem
    :return: (path, num_nodes_expanded, max_frontier_size)
             path is a list of int states from init_state to the single goal,
             num_nodes_expanded is how many nodes were popped from the priority queue,
             max_frontier_size is the largest frontier size observed at any point.
    """
    num_nodes_expanded = 0
    max_frontier_size = 1

    if problem.goal_test(problem.init_state):
        return [problem.init_state], 0, 0

    order_idx = 0
    open_nodes = queue.PriorityQueue()
    start_node = Node(parent=None, state=problem.init_state, action=None, path_cost=0)
    escst_start = problem.heuristic(start_node.state)  # escst(n) = cst(n) + h(n)
    open_nodes.put((escst_start, order_idx, start_node)) # add to priority queue

    # best cost for each state
    best_cost = {start_node.state: 0}
    closed = set()

    def expand_node(current_node, best_cost, open_nodes, order_idx):
        """Expand current_nd, insert its children into open_nodes if relevant."""
        current_state = current_node.state
        actions = problem.get_actions(current_state)

        for action in actions:
            child_node = problem.get_child_node(current_node, action)

            if child_node.state in closed:
                continue

            # check if better than known path
            if child_node.state in best_cost and child_node.path_cost >= best_cost[child_node.state]:
                continue

            best_cost[child_node.state] = child_node.path_cost
            order_idx += 1
            escst_child = child_node.path_cost + problem.heuristic(child_node.state)
            open_nodes.put((escst_child, order_idx, child_node))

        return order_idx

    while not open_nodes.empty():
        _, _, current = open_nodes.get() # get node with lowest escst

        if current.state in closed:
            continue

        if problem.goal_test(current.state):  # found goal
            solution_path = problem.trace_path(current)
            return solution_path, num_nodes_expanded, max_frontier_size

        num_nodes_expanded += 1
        closed.add(current.state)

        order_idx = expand_node(current, best_cost, open_nodes, order_idx)

        max_frontier_size = max(max_frontier_size, open_nodes.qsize())

    return [], num_nodes_expanded, max_frontier_size


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. 
    You do NOT need to submit code that determines the values here: that should be computed on your own machine. 
    Simply fill in the values!
    
    :return: (transition_start_probability, transition_end_probability, peak_probability)
    """
    # TODO: replace these placeholder values once you have run your experiments
    transition_start_probability = -1
    transition_end_probability = -1
    peak_nodes_expanded_probability = -1
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Example test on a small 10x10 random grid
    p_occ = 0.25
    M, N = 10, 10
    problem = get_random_grid_problem(p_occ, M, N)

    # Run A*
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    correct = problem.check_solution(path)
    print("Solution is correct:", correct)
    print("Path found:", path)
    print("Number of nodes expanded:", num_nodes_expanded)
    print("Max frontier size:", max_frontier_size)

    # Plot the solution path (this pops up a matplotlib window)
    problem.plot_solution(path)