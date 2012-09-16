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
# NODE CLASS:
# getState()
# getSuccessors()
# childState(direction) returns state of child
# Slightly more work is done in my node class than in the book's [p 79]
#===============================================================================
class Node:

    def __init__(self, problem, parent, action):
        self.problem = problem
        self.parent = parent
        self.action = action
        if parent is None:
            self.state = problem.getStartState()
        else:
            self.state = parent.childState(action)
        self.successors = self.problem.getSuccessors(self.state)
        
    def getState(self):
        return self.state
        
    def getSuccessors(self):
        return self.successors
    def childState(self, direction):
        from game import Directions
        n = Directions.NORTH
        e = Directions.EAST
        s = Directions.SOUTH
        w = Directions.WEST
        if direction == n:
            return (self.state[0], self.state[1]+1)
        elif direction == e:
            return (self.state[0]+1, self.state[1])
        elif direction == s:
            return (self.state[0], self.state[1]-1)
        elif direction == w:
            return (self.state[0]-1, self.state[1])
        else:
            return ('Error')
    def printState(self):
        print " "
        print "NODE STATE: "
        print "Parent: ", self.parent
        print "Action: ", self.action
        print "State: ", self.state
        print "Successors: ", self.successors
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
# isInFrontier(node) pops and pushes all nodes in frontier stack to find a match
# isInExplored(node) uses keys (state tuples) to see if node has been explored
# getSolution(node) goes back from goal node to start node, generating solution
# The DFS solution implemented follows straight from GRAPH-Search, [p 77]
#===============================================================================
    def expandNodeToFrontier(parent):
        successors = parent.getSuccessors()
        tempNode = None
        for i in successors:
            tempNode = Node(problem, parent, i[1])
            if isInFrontier(tempNode)==False and isInExplored(tempNode)==False:
                frontier.push(tempNode)
            
    def isInFrontier(node):
        tempStack = util.Stack()
        while frontier.isEmpty() == False:
            tempNode = frontier.pop()
            tempStack.push(tempNode)
            if(tempNode.getState == node.getState):
                while tempStack.isEmpty() == False:
                    tempNode = tempStack.pop()
                    frontier.push(tempNode)
                return True;
        while tempStack.isEmpty() == False:
            tempNode = tempStack.pop()
            frontier.push(tempNode)
        return False
            
    def isInExplored(node):
        state = node.getState()
        if state in exploredSet.keys():
            return True
        return False
    
    def getSolution(node):
        solution = [node.action]
        while node.parent.action != None:
            node = node.parent
            solution.append(node.action)
        solution.reverse()
        return solution
    
    head = Node(problem, None, None)
    current = head
    frontier = util.Stack()
    exploredSet = {}
    expandNodeToFrontier(head)
    
    while True:
        if frontier.isEmpty():
            return None
        current = frontier.pop()
        if problem.isGoalState(current.getState()):
            solution = getSolution(current)
            return solution
        exploredSet[current.getState()] = current
        expandNodeToFrontier(current)
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    "Search the shallowest nodes in the search tree first. [p 81]"
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
            
def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
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
    util.raiseNotDefined()    
    
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch