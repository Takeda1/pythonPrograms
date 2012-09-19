# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util

#===============================================================================
# NODE CLASS 
# with printState()
#===============================================================================
class Node:

    def __init__(self, problem, parent, action, state, priority):
        self.problem = problem
        self.parent = parent
        self.action = action
        self.state = state
        self.priority = priority
        
    def printState(self):
        print " "
        print "NODE STATE: "
        print "Parent: ", self.parent
        print "Action: ", self.action
        print "State: ", self.state
        print "Priority: ", self.priority
        print " "
            
        
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    
    You do not need to change anything in this class, ever.
    """
    
    def getStartState(self):
        """
        Returns the start state for the search problem 
        """
        util.raiseNotDefined()
        
    def isGoalState(self, state):
        """
            state: Search state
        
        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
            state: Search state
        
        For a given state, this should return a list of triples, 
        (successor, action, stepCost), where 'successor' is a 
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental 
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
        actions: A list of actions to take
 
        This method returns the total cost of a particular sequence of actions.    The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.    For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return    [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].
    
    Your search algorithm needs to return a list of actions that reaches
    the goal.    Make sure to implement a graph search algorithm [Fig. 3.7].
    
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

#===============================================================================
# expandNodeToFrontier(parent) pushes previously unseen nodes onto the stack
# isInFrontier(node) and isInExplored(node) use Sets of tuples (state, priority)
# getSolution(node) goes back from goal node to start node, generating solution
# The DFS solution implemented follows straight from GRAPH-Search, [p 77]
#===============================================================================
    def expandNodeToFrontier():
        successors = problem.getSuccessors(current.state)
        child = None
        for i in successors:
            child = Node(problem, current, i[1], i[0], 1)
            if isInFrontier(child)==False and isInExplored(child)==False:
                frontier.push(child)
                frontierSet.add((i[0], i[2]))
            
    def isInFrontier(node):
        for x in frontierSet:
            if node.state == x[0]:
                return True
        return False
            
    def isInExplored(node):
        for x in exploredSet:
            if node.state == x[0]:
                return True
        return False
    
    def getSolution(node):
        solution = [node.action]
        while node.parent.action != None:
            node = node.parent
            solution.append(node.action)
        solution.reverse()
        return solution
    
    
    #Initialize Problem
    head = Node(problem, None, None, problem.getStartState(), 1)
    current = head
    frontier = util.Stack()
    frontier.push(current)
    frontierSet = set()
    frontierSet.add((current.state, 1))
    exploredSet = set()
    expandNodeToFrontier()

    
    while True:
        if frontier.isEmpty():
            return None
        current = frontier.pop()
        frontierSet.remove((current.state, 1))
        if problem.isGoalState(current.state):
            solution = getSolution(current)
            return solution
        exploredSet.add((current.state, 1))
        expandNodeToFrontier()
    util.raiseNotDefined()





def breadthFirstSearch(problem):
    "Search the shallowest nodes in the search tree first. [p 81]"
    "*** YOUR CODE HERE ***"
    
    def isInFrontier(node):
        for x in frontierSet:
            if node.state == x[0]:
                return True
        return False
            
    def isInExplored(node):
        for x in exploredSet:
            if node.state == x[0]:    
                return True
        return False
    
    def getSolution(node):
        solution = [node.action]
        while node.parent.action != None:
            node = node.parent
            solution.append(node.action)
        solution.reverse()
        return solution
    
    
    #Initialize Problem
    head = Node(problem, None, None, problem.getStartState(), 1)
    current = head
    if problem.isGoalState(current.state):
        return getSolution(current)
    frontier = util.Queue()
    frontier.push(current)
    frontierSet = set()
    frontierSet.add((current.state, 1))
    exploredSet = set()
    
    while True:
        if frontier.isEmpty():
            return None
        current = frontier.pop()
        frontierSet.remove((current.state, 1))
        exploredSet.add((current.state, 1))
        successors = problem.getSuccessors(current.state)
        child = None
        for i in successors:
            child = Node(problem, current, i[1], i[0], 1)
            if isInFrontier(child)==False and isInExplored(child)==False:
                if problem.isGoalState(child.state):
                    return getSolution(child)
                frontier.push(child)
                frontierSet.add((i[0], i[2]))
                                
    util.raiseNotDefined()
            
def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
        
    #  TODO:
    #  ideally figure out how to replace frontier node with child
    #  (this is the last line of UNIFORM-COST-SEARCH, p. 84)
    #  It should be nearly as efficient (in small cases) to just add the new child
    #  into the frontier, without removing the old one.
    #  Note:
    #  the frontierSet can't add duplicate values to it, so it might try to 
    #  remove things that don't exist, that's why I use .discard() instead of .remove()
    
    def isInFrontier(node):
        for x in frontierSet:
            if node.state == x[0]:
                return True
        return False
    
    def hasHigherPathCost(node):
        for x in frontierSet:
            if node.state == x[0]:
                return True
        return False
            
    def isInExplored(node):
        for x in exploredSet:
            if node.state == x[0]:
                return True
        return False
    
    def getSolution(node):
        solution = [node.action]
        while node.parent.action != None:
            node = node.parent
            solution.append(node.action)
        solution.reverse()
        return solution
    
    #Initialize Problem
    head = Node(problem, None, None, problem.getStartState(), 0)
    current = head
    frontier = util.PriorityQueue()
    frontier.push(current, current.priority)
    frontierSet = set()
    frontierSet.add((current.state, current.priority))
    exploredSet = set()

    while True:
        child = None
        if frontier.isEmpty():
            return None
        current = frontier.pop()
        frontierSet.discard((current.state, current.priority))
        if problem.isGoalState(current.state):
            return getSolution(current)
        exploredSet.add((current.state, current.priority))
        
        successors = problem.getSuccessors(current.state)
        for i in successors:
            child = Node(problem, current, i[1], i[0], i[2])
            if isInFrontier(child)==False and isInExplored(child)==False:
                frontier.push(child, child.priority)
                frontierSet.add((i[0], i[2]))
            elif isInFrontier(child)==True and hasHigherPathCost(child):
                frontier.push(child, child.priority)
                frontierSet.add((i[0], i[2]))
        
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.    This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    
    def isInFrontier(node):
        for x in frontierSet:
            if node.state == x[0]:
                return True
        return False
    
    def hasHigherPathCost(node):
        for x in frontierSet:
            if node.state == x[0]:
                return True
        return False
            
    def isInExplored(node):
        for x in exploredSet:
            if node.state == x[0]:
                return True
        return False
    
    def getSolution(node):
        solution = [node.action]
        while node.parent.action != None:
            node = node.parent
            solution.append(node.action)
        solution.reverse()
        return solution
    
    def bestEstimate(state, priority):
        return (heuristic(state, problem)+priority)
        
    
    #Initialize Problem
    head = Node(problem, None, None, problem.getStartState(), 0)
    current = head
    frontier = util.PriorityQueue()
    frontier.push(current, current.priority)
    frontierSet = set()
    frontierSet.add((current.state, current.priority))
    exploredSet = set()

    while True:
        child = None
        if frontier.isEmpty():
            return None
        current = frontier.pop()
        frontierSet.discard((current.state, current.priority))
        if problem.isGoalState(current.state):
            return getSolution(current)
        exploredSet.add((current.state, current.priority))
        successors = problem.getSuccessors(current.state)
        for i in successors:
            child = Node(problem, current, i[1], i[0], bestEstimate(i[0], i[2]))
            if isInFrontier(child)==False and isInExplored(child)==False:
                frontier.push(child, child.priority)
                frontierSet.add((child.state, child.priority))
            elif isInFrontier(child)==True and hasHigherPathCost(child):
                frontier.push(child, child.priority)
                frontierSet.add((child.state, child.priority))
        
    util.raiseNotDefined()    
    
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch