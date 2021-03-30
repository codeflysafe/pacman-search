# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    mystack = util.Stack()
    startNode = (problem.getStartState(), '', 0, [])
    mystack.push(startNode)
    visited = set()
    while mystack:
        node = mystack.pop()
        state, action, cost, path = node
        if state not in visited:
            visited.add(state)
            if problem.isGoalState(state):
                path = path + [(state, action)]
                break
            succNodes = problem.expand(state)
            for succNode in succNodes:
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost +
                           succCost, path + [(state, action)])
                mystack.push(newNode)
    actions = [action[1] for action in path]
    del actions[0]
    return actions


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    myqueue = util.Queue()
    startNode = (problem.getStartState(), '', 0, [])
    myqueue.push(startNode)
    visited = set()
    while myqueue:
        node = myqueue.pop()
        state, action, cost, path = node
        if state not in visited:
            visited.add(state)
            if problem.isGoalState(state):
                path = path + [(state, action)]
                break
            succNodes = problem.expand(state)
            for succNode in succNodes:
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost +
                           succCost, path + [(state, action)])
                myqueue.push(newNode)
    actions = [action[1] for action in path]
    del actions[0]
    return actions


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


# def aStarSearch(problem, heuristic=nullHeuristic):
#     # COMP90054 Task 1, Implement your A Star search algorithm here
#     """Search the node that has the lowest combined cost and heuristic first."""
#     "*** YOUR CODE HERE ***"
#     util.raiseNotDefined()


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    frontier = {}
    visited = set()
    start = problem.getStartState()
    # (state,action,fScore,gScore,path)
    frontier[start] = (start, "", 0, 0, [])
    miniState = start
    miniF = 0
    while len(frontier) > 0:
        current_state = miniState
        state, action, _, gScore, path = frontier.pop(current_state)
        if state not in visited:
            visited.add(state)
            if problem.isGoalState(state):
                path = path + [(state, action)]
                return [action[1] for action in path[1:]]

            succNodes = problem.expand(state)
            for succNode in succNodes:
                succState, succAction, succCost = succNode
                newFScore = gScore + succCost + heuristic(succState, problem)

                if succState in frontier:
                    if newFScore < frontier[succState][2]:
                        frontier[succState] = (
                            succState, succAction, newFScore, gScore + succCost, path + [(state, action)])
                else:
                    frontier[succState] = (
                        succState, succAction, newFScore, gScore + succCost, path + [(state, action)])

        miniState = None
        miniF = -1

        for k in frontier.keys():
            if miniState is None or (k != miniState and frontier[k][2] < miniF):
                miniState = k
                miniF = frontier[k][2]

    return None


infinity = float('inf')


class Node:
    def __init__(self, state, action, parent):
        self.state = state
        self.g = 0  # Distance to start node
        self.f = 0  # Total cost
        self.action = action
        self.parent = parent

    # Compare nodes

    def __eq__(self, other):
        return self.state == other.state

    # Sort nodes
    def __lt__(self, other):
        return self.f < other.f

    # Print node
    def __repr__(self):
        return ('({0},{1})'.format(self.state, self.f))

    def key(self):
        return self.parent + "_" + self.state

    def solution(self):
        solution = []
        solution.append(self.action)
        path = self
        while path.parent is not None:
            path = path.parent
            solution.append(path.action)
        solution = solution[:-1]
        solution.reverse()
        return solution


def rbfs(problem, node: Node, f_limit, heuristic, frontier):

    # (state, action, fScore, gScore, path) = node
    if problem.isGoalState(node.state):
        return node, None

    succNodes = problem.expand(node.state)
    successors = []
    for succNode in succNodes:
        succState, succAction, succCost = succNode

        # if frontier[]
        childNode = Node(succState, succAction, node)
        childNode.g = succCost + node.g
        childNode.f = childNode.g + heuristic(succState, problem)
        childNode.f = childNode.f
        successors.append(childNode)

    if len(successors) == 0:
        return None, infinity

    # for i in range(len(successors)):
    #     successors[i].f = max(successors[i].f, node.f)

    while True:
        successors.sort()
        best = successors[0]
        if best.f > f_limit:
            return None, best.f
        limit = f_limit
        if len(successors) > 1:
            limit = min(f_limit, successors[1].f)
        result, best.f = rbfs(problem, best, limit, heuristic)
        successors[0] = best
        if result is not None:
            return result, None


def recursivebfs(problem, heuristic=nullHeuristic):

    frontier = {}
    # COMP90054 Task 2, Implement your Recursive Best First Search algorithm here
    node = Node(problem.getStartState(), '', None)
    node.f = heuristic(node.state, problem)
    frontier[node.key()] = node
    result, _ = rbfs(problem, node, infinity, heuristic, frontier)
    return result.solution()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
rebfs = recursivebfs
