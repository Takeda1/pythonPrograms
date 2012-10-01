# searchAgents.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
This file contains all of the agents that can be selected to 
control Pacman.    To select an agent, use the '-p' option
when running pacman.py.    Arguments can be passed to your agent
using '-a'.    For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a searchFunction=depthFirstSearch

Commands to invoke other search strategies can be found in the 
project description.

Please only change the parts of the file you are asked to.
Look for the lines that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.    Follow the
project description for details.

Good luck and happy searching!
"""
from game import Directions
from game import Agent 
from game import Actions
import util
import time
import search
import searchAgents #@UnresolvedImport

class GoWestAgent(Agent):
    "An agent that goes West until it can't."
    
    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#             after you fill in parts of search.py    #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search algorithm for a
    supplied search problem, then returns actions to follow that path.
    
    As a default, this agent runs DFS on a PositionSearchProblem to find location (1,1)
    
    Options for fn include:
        depthFirstSearch or dfs
        breadthFirstSearch or bfs
        
    
    Note: You should NOT change any code in SearchAgent
    """
        
    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems
        
        # Get the search function from the name and heuristic
        if fn not in dir(search): 
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn) 
            self.searchFunction = func
        else:
            if heuristic in dir(searchAgents):
                heur = getattr(searchAgents, heuristic)
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic)) 
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)
            
        # Get the search problem type from the name
        if prob not in dir(searchAgents) or not prob.endswith('Problem'): 
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = getattr(searchAgents, prob)
        print('[SearchAgent] using problem type ' + prob) 
        
    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game board. Here, we
        choose a path to the goal.    In this phase, the agent should compute the path to the
        goal and store it in a local variable.    All of the work is done in this method!
        
        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions    = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)
        
    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in registerInitialState).    Return
        Directions.STOP if there is no further action to take.
        
        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]        
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test,
    successor function and cost function.    This search problem can be 
    used to find paths to a particular point on the pacman board.
    
    The state space consists of (x,y) positions in a pacman game.
    
    Note: this search problem is fully specified; you should NOT change it.
    """
    
    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True):
        """
        Stores the start and goal.    
        
        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal 
        
        # For display purposes only
        if isGoal:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable
            
        return isGoal     
    
    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.
        
         As noted in search.py:
                 For a given state, this should return a list of triples, 
         (successor, action, stepCost), where 'successor' is a 
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental 
         cost of expanding to that successor
        """
        
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )
                
        # Bookkeeping for display purposes
        self._expanded += 1 
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)
            
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.    If those actions
        include an illegal move, return 999999
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.    
    
    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
            self.searchFunction = search.uniformCostSearch
            costFn = lambda pos: .5 ** pos[0] 
            self.searchType = lambda state: PositionSearchProblem(state, costFn)
            
class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.    
    
    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
            self.searchFunction = search.uniformCostSearch
            costFn = lambda pos: 2 ** pos[0] 
            self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.    Time to write code!    #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """
    
    def __init__(self, startingGameState, costFn = lambda x: 1):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2 
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # Number of search nodes expanded
        
        "*** YOUR CODE HERE ***"
        self.costFn = costFn
        self.startState = (startingGameState.getPacmanPosition()[0], startingGameState.getPacmanPosition()[1], frozenset([]))
        
    def getStartState(self):
        "Returns the start state (in your state space, not the full Pacman state space)"
        "*** YOUR CODE HERE ***"
        return self.startState
        util.raiseNotDefined()
        
    def isGoalState(self, state):
        "Returns whether this search state is a goal state of the problem"
        "*** YOUR CODE HERE ***"
        x,y = state[0],state[1]
        if len(state[2])>=3:
            for corner in self.corners:
                if (x,y) == corner and (x,y) not in state[2]:
                    return True
        return False

        util.raiseNotDefined()
             
    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.
        
         As noted in search.py:
                 For a given state, this should return a list of triples, 
         (successor, action, stepCost), where 'successor' is a 
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental 
         cost of expanding to that successor
        """
        
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #     x,y = currentPosition
            #     dx, dy = Actions.directionToVector(action)
            #     nextx, nexty = int(x + dx), int(y + dy)
            #     hitsWall = self.walls[nextx][nexty]
            
            "*** YOUR CODE HERE ***"

            x,y = state[0],state[1]
            tempSet = set([])
            for i in state[2]:
                tempSet.add(i)
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                for corner in self.corners:
                    if (x,y) == corner and (x,y) not in tempSet:
                        tempSet.add((x, y))
                finalSet = frozenset(tempSet)
                nextState = (nextx, nexty, finalSet)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )
        self._expanded += 1
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.    If those actions
        include an illegal move, return 999999.    This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.
    
        state:     The current search state 
                         (a data structure you chose in your search problem)
        
        problem: The CornersProblem instance for this layout.    
        
    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem; i.e.
    it should be admissible.    (You need not worry about consistency for
    this heuristic to receive full credit.)
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)
    
    "*** YOUR CODE HERE ***"
    """
    This heuristic checks the distance to the closest unvisited corner,
    then checks the distance from this unvisited corner to the 
    next closest remaining unvisited corner, for all corners.
    The heuristic returns the sum of these distances.
    """
    
    def notVisited(corner):
        for i in state[2]:
            if corner == i:
                return False
        return True
    
    def closestDistance(xy1, points):
        minDistance = 999
        visitedPoint = None
        for xy2 in points:
            distance = abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
            if distance < minDistance:
                minDistance = distance
                visitedPoint = xy2
        if visitedPoint != None:
            points.remove(visitedPoint)
        return minDistance, points, visitedPoint
    
    def cornersNotVisited(corners):
        unVisited = []
        for corner in corners:
            if notVisited(corner):
                unVisited.append(corner)
        return unVisited
    
    xy1 = (state[0], state[1])
    unVisited = cornersNotVisited(corners)
    heuristic = 0
    nextClosestDistance = 0
    nextClosestDistance,unVisited,xy1 = closestDistance(xy1, unVisited)
    while len(unVisited) > 0:
        nextClosestDistance,unVisited,xy1 = closestDistance(xy1, unVisited)
        heuristic += nextClosestDistance
    return heuristic
    return 0

def cornersFastInconsistentHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.
    
        state:     The current search state 
                         (a data structure you chose in your search problem)
        
        problem: The CornersProblem instance for this layout.    
        
    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem; i.e.
    it should be admissible.    (You need not worry about consistency for
    this heuristic to receive full credit.)
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)
    
    "*** YOUR CODE HERE ***"
    """
    This heuristic checks the distance to the closest unvisited corner,
    then checks the distance from this unvisited corner to the 
    next closest remaining unvisited corner, for all corners.
    The heuristic returns the sum of these distances.
    """
    
    def notVisited(corner):
        for i in state[2]:
            if corner == i:
                return False
        return True
    
    def closestDistance(xy1, points):
        minDistance = 999
        visitedPoint = None
        for xy2 in points:
            #distance = round(( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5)
            distance = abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
            if distance < minDistance:
                minDistance = distance
                visitedPoint = xy2
        if visitedPoint != None:
            points.remove(visitedPoint)
        return minDistance, points, visitedPoint
    
    def cornersNotVisited(corners):
        unVisited = []
        for corner in corners:
            if notVisited(corner):
                unVisited.append(corner)
        return unVisited
    
    xy1 = (state[0], state[1])
    unVisited = cornersNotVisited(corners)
    heuristic = 0
    while len(unVisited) > 0:
        nextClosestDistance,unVisited,xy1 = closestDistance(xy1, unVisited)
        heuristic += nextClosestDistance
    return heuristic
    return 0

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your cornerHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the 
    food (dots) in a Pacman game.
    
    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
        pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
        foodGrid:             a Grid (see game.py) of either True or False, specifying remaining food 
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0
        self.heuristicInfo = {} # A dictionary for the heuristic to store information
            
    def getStartState(self):
        return self.start
    
    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.    If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.
    
    This heuristic must be consistent to ensure correctness.    First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.
    
    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!    On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    
    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a 
    Grid (see game.py) of either True or False. You can call foodGrid.asList()
    to get a list of food coordinates instead.
    
    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, problem.walls gives you a Grid of where the walls are.
    
    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use. For example,
    if you only want to count the walls once and store that value, try:
        problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    
    def accurateDistance(point1, point2, gameState):
        x1, y1 = point1
        x2, y2 = point2
        walls = gameState.getWalls()
        assert not walls[x1][y1], 'point1 is a wall: ' + point1
        assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
        prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False)
        return len(search.bfs(prob))
    
    def generateEdges(vertices):
        edges = set([])
        while len(vertices) > 0:
            xy1 = vertices.pop()
            for i in vertices:
                xy2 = i
                #weight = abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
                weight = accurateDistance(xy1, xy2, problem.startingGameState)
                edges.add((xy1, xy2, weight))
        return edges
    
    def minEdge():
        e = None
        u = None
        v = None
        w = 9999
        for i in edges:
            if i[2] < w:
                if i[0] in visited and i[1] in unVisited:
                    e = i
                    u = i[0]
                    v = i[1]
                    w = i[2]
                elif i[1] in visited and i[0] in unVisited:
                    e = i
                    u = i[0]
                    v = i[0]
                    w = i[2]
        return e, v, w

    food = set([])
    countX = -1
    countY = -1
    for i in foodGrid:
        countX+=1
        countY = -1
        for j in i:
            countY+=1
            if j == True:
                food.add((countX, countY))
    tempFoodSet = food
    food = frozenset(food)
    unVisited = set([])
    for i in food:
        unVisited.add(i)
    if 'edges' not in problem.heuristicInfo.keys():
        problem.heuristicInfo['edges'] = generateEdges(tempFoodSet)
    edges = problem.heuristicInfo['edges']
    
    heuristic = 0
    visited = set([])
    newEdges = set([])
    if len(unVisited) != 0:
        visited.add(unVisited.pop())
    while visited != food:
        e,v,w = minEdge()
        unVisited.remove(v)
        visited.add(v)
        newEdges.add(e)
        heuristic+=w
    return heuristic
    
class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0): 
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment: 
                legal = currentState.getLegalActions()
                if action not in legal: 
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)
        
    def findPathToClosestDot(self, gameState):
        "Returns a path (a list of actions) to the closest dot, starting from gameState"
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        foodGrid = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        return search.bfs(problem)
        util.raiseNotDefined()
    
class AnyFoodSearchProblem(PositionSearchProblem):
    """
        A search problem for finding a path to any food.
        
        This search problem is just like the PositionSearchProblem, but
        has a different goal test, which you need to fill in below.    The
        state space and successor function do not need to be changed.
        
        The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
        inherits the methods of the PositionSearchProblem.
        
        You can use this search problem to help you fill in 
        the findPathToClosestDot method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.    You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0
        
    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test
        that will complete the problem definition.
        """
        x,y = state
        
        "*** YOUR CODE HERE ***"
        foodPositions = set([])
        countX = -1
        countY = -1
        for i in self.food:
            countX+=1
            countY = -1
            for j in i:
                countY+=1
                if j == True:
                    foodPositions.add((countX, countY))
        if state in foodPositions:
            return True
        return False
        util.raiseNotDefined()

##################
# Mini-contest 1 #
##################
class Position:

    def __init__(self, state, numBranches, transition):
        self.state = state
        self.numBranches = numBranches
        self.transition = transition
        self.successorStates = self.getSuccessorStates()
        
    def getSuccessorStates(self):
        successors = []
        for i in self.transition.values():
            successors.append(i)
        return successors
class Box:
    def __init__(self, exits, interiors, fastPaths = None, fullPaths = None):
        self.exits = exits
        self.interiors = interiors
        self.fastPaths = fastPaths
        self.fullPaths = fullPaths
    def setFastPaths(self, newPaths):
        self.fastPaths = newPaths
    def setFullPaths(self, newPaths):
        self.fullPaths = newPaths
class ApproximateSearchAgent(Agent):
    "Implement your contest entry here.    Change anything but the class name."
    def __init__(self):
        self.solution = {}
        self.startState = None
        self.food = None
    
    def registerInitialState(self, state):
        "This method is called before any moves are made."
        "*** YOUR CODE HERE ***"
        self.solution[(0,0)] = 'West'
        self.startState = state.getPacmanPosition()
        foodGrid = state.getFood()
        self.food = set([])
        countX = -1
        countY = -1
        for i in foodGrid:
            countX+=1
            countY = -1
            for j in i:
                countY+=1
                if j == True:
                    self.food.add((countX, countY))
                    
        positionDictionary = {}
        expanded = []
        newGameState = state
        pacmanInitialPosition = state.data.agentStates[0].start.pos
        legal = newGameState.getLegalPacmanActions()
        if Directions.STOP in legal: legal.remove(Directions.STOP)
        successors = []
        transition = {}
        for action in legal:
            successor = (newGameState.generateSuccessor(0, action), action)
            successors.append(successor)
            transition[action] = successor[0].data.agentStates[0].configuration.pos
        newPosition = Position(self.startState, len(successors), transition)
        positionDictionary[self.startState] = newPosition
        edgePieceDictionary = {}
        vertexDictionary = {}
        deadEndDictionary = {}
        boxDictionary = {}
        edgeDictionary = {}
        
        
        for i in self.food:
            newGameState.data.agentStates[0].start.pos = i
            legal = newGameState.getLegalPacmanActions()
            if Directions.STOP in legal: legal.remove(Directions.STOP)
            successors = []
            transition = {}
            for action in legal:
                successor = (newGameState.generateSuccessor(0, action), action)
                successors.append(successor)
                transition[action] = successor[0].data.agentStates[0].configuration.pos
            newPosition = Position(i, len(successors), transition)
            positionDictionary[i] = newPosition
            if newPosition.numBranches == 1:
                deadEndDictionary[i] = newPosition
            elif newPosition.numBranches == 2:
                edgePieceDictionary[i] = newPosition
            else: 
                vertexDictionary[i] = newPosition
        
                
        
        def getSolution(node):
            solution = [node.action]
            while node.parent.action != None:
                node = node.parent
                solution.append(node.action)
            solution.reverse()
            return solution 
        
        def getSuccessors(node, dictionary):
            successors = {}
            for i in node.transition.keys():
                successors[i] = dictionary[node.transition[i]]
            return successors
        
        def isCorner(position):
            if position.numBranches == 2:
                directions = position.transition.keys()
                if directions[0] == 'North' and directions[1] == 'South':
                    return False
                elif directions[0] == 'East' and directions[1] == 'West':
                    return False
                elif directions[0] == 'South' and directions[1] == 'North':
                    return False
                elif directions[0] == 'West' and directions[1] == 'East':
                    return False
                return True
            return False
        
        def oppositeDirection(direction):
            if direction == 'North':
                return 'South'
            elif direction == 'East':
                return 'West'
            elif direction == 'South':
                return 'North'
            else:
                return 'East'
            
        def orthogonalDirections(direction):
            if direction == 'North' or direction == 'South':
                return 'East','West'
            else:
                return 'South','North'
        
        def getNewPosition(coordinates, action):
            x,y = coordinates
            if action == 'North':
                return (x, y+1)
            elif action == 'East':
                return (x+1, y)
            elif action == 'South':
                return (x, y-1)
            else:
                return (x-1, y)
            
        def reversePath(path):
            newPath = []
            tempPath = []
            for i in path:
                tempPath.append(i)
            while len(tempPath) > 0:
                newDirection = oppositeDirection(tempPath.pop())
                newPath.append(newDirection)
            return newPath
                
                
        def findFullPath(start, goal, box, newGameState, orientation):
            
            def isEaten():
                if len(unVisitedInteriors) == 0:
                    return True
                return False
            
            
            path = []
            position = start[0].state
            unVisitedInteriors = {}
            error = 0
            for i in box.exits:
                if start != i:
                    unVisitedInteriors[i[0].state] = i[0]      
            visitedInteriors = {position: start[0]}
            forward = goal[1]
            backward = oppositeDirection(forward)
            if orientation == 0:
                left, right = orthogonalDirections(forward)
            else:
                right, left = orthogonalDirections(forward)
            currentState = 0
            
            for i in box.interiors:
                unVisitedInteriors[i.state] = i

            while True:
                newBackward = getNewPosition(position, backward)
                newLeft = getNewPosition(position, left)
                newRight = getNewPosition(position, right)
                newForward = getNewPosition(position, forward)
                
                if currentState == 0:
                    while currentState == 0:
                        newBackward = getNewPosition(position, backward)
                        newLeft = getNewPosition(position, left)
                        if newLeft in unVisitedInteriors.keys():
                            path.append(left)
                            position = newLeft
                            visitedInteriors[position] = unVisitedInteriors[position]    
                            del unVisitedInteriors[position]
                        else: currentState =1
                        
                elif currentState == 1:
                    newBackward = getNewPosition(position, backward)
                    newLeft = getNewPosition(position, left)
                    if newBackward in unVisitedInteriors.keys():
                        while currentState == 1:
                            newBackward = getNewPosition(position, backward)
                            newLeft = getNewPosition(position, left)
                            if newLeft in unVisitedInteriors.keys():
                                path.append(left)
                                position = newLeft
                                visitedInteriors[position] = unVisitedInteriors[position]    
                                del unVisitedInteriors[position]
                            elif newBackward in unVisitedInteriors.keys():
                                path.append(backward)
                                position = newBackward
                                visitedInteriors[position] = unVisitedInteriors[position]
                                del unVisitedInteriors[position]
                                newBackward = getNewPosition(position, backward)
                            else: currentState = 2
                    else: 
                        if len(path) == 0:
                            currentState = 4
                        else:
                            currentState = 3
                    
                elif currentState == 2:
                    newRight = getNewPosition(position, right)
                    while newRight in unVisitedInteriors.keys() and isEaten() == False:
                        path.append(right)
                        position = newRight
                        visitedInteriors[position] = unVisitedInteriors[position]
                        del unVisitedInteriors[newRight]
                        newRight = getNewPosition(position, right)
                    if isEaten():
                        currentState = 5
                    newForward = getNewPosition(position, forward)
                    newLeft = getNewPosition(position, left)
                    while newForward not in unVisitedInteriors.keys() and isEaten() == False: 
                        if newLeft in unVisitedInteriors.keys():
                            path.append(left)
                            position = newLeft
                        newForward = getNewPosition(position, forward)
                        newLeft = getNewPosition(position, left)
                        
                    if isEaten():
                        currentState = 5
                    else:
                        currentState = 3
                        path.append(forward)
                        position = newForward
                        visitedInteriors[position] = unVisitedInteriors[position]
                        del unVisitedInteriors[position]
                    
                elif currentState == 3:
                    newLeft = getNewPosition(position, left)
                    while newLeft in unVisitedInteriors.keys() and isEaten() == False:
                        path.append(left)
                        position = newLeft
                        visitedInteriors[position] = unVisitedInteriors[position]
                        del unVisitedInteriors[newLeft]
                        newLeft = getNewPosition(position, left)
                    if isEaten():
                        currentState = 5
                    newForward = getNewPosition(position, forward)
                    newRight = getNewPosition(position, right)
                    while newForward not in unVisitedInteriors.keys() and isEaten() == False: 
                        if newRight in visitedInteriors.keys():
                            path.append(right)
                            position = newRight
                        else:
                            error +=1
                        if error == 2:
                            for i in range(1, 100):
                                path.append(i)
                            unVisitedInteriors.clear()
                        newForward = getNewPosition(position, forward)
                        newRight = getNewPosition(position, right)
                    if isEaten():
                        currentState = 5
                    else:
                        path.append(forward)
                        position = newForward
                        visitedInteriors[position] = unVisitedInteriors[position]
                        del unVisitedInteriors[position]
                        currentState = 4
                    
                elif currentState == 4:
                    newRight = getNewPosition(position, right)
                    while newRight in unVisitedInteriors.keys() and isEaten() == False:
                        path.append(right)
                        position = newRight
                        visitedInteriors[position] = unVisitedInteriors[position]
                        del unVisitedInteriors[newRight]
                        newRight = getNewPosition(position, right)
                    if isEaten():
                        currentState = 5
                    newForward = getNewPosition(position, forward)
                    newLeft = getNewPosition(position, right)
                    while newForward not in unVisitedInteriors.keys() and isEaten() == False: 
                        if newLeft in visitedInteriors.keys():
                            path.append(left)
                            position = newLeft
                        else:
                            error +=1
                        if error == 2:
                            for i in range(1, 100):
                                path.append(i)
                            unVisitedInteriors.clear()
                        newForward = getNewPosition(position, forward)
                        newLeft = getNewPosition(position, left)
                    if isEaten():
                        currentState = 5
                    else:
                        currentState = 3
                        path.append(forward)
                        position = newForward
                        visitedInteriors[position] = unVisitedInteriors[position]
                        del unVisitedInteriors[position]
                        
                elif currentState == 5:
                    if position == goal[0].state:
                        return path
                    else:
                        finalPath = findFastPath(position, goal[0].state, newGameState)
                        for i in finalPath:
                            path.append(i)
                        return path
                

        def findFastPath(point1, point2, gameState):
            x1, y1 = point1
            x2, y2 = point2
            walls = gameState.getWalls()
            assert not walls[x1][y1], 'point1 is a wall: ' + point1
            assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
            prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False)
            return search.bfs(prob)
                        
        def buildBox(interiors):
            
            def findBox():
                while True:
                    if frontier.isEmpty():
                        return box
                    current = frontier.pop()
                    box.interiors.append(current)
                    frontierSet.remove(current.state)
                    exploredSet.add(current.state)
                    successors = getSuccessors(current, positionDictionary)
                    for direction in successors.keys():
                        child = successors[direction]
                        if child.state not in frontierSet and child.state not in exploredSet:
                            if child.numBranches >= 3 or isCorner(child):
                                frontier.push(child)
                                frontierSet.add(child.state)
                            else:
                                if current not in box.interiors:
                                    print 'why?'
                                box.interiors.remove(current)
                                box.exits.append((current, direction))
            
            box = Box([], interiors)
            frontier = util.Queue()
            frontierSet = set()
            exploredSet = set()
            for i in interiors:
                frontier.push(i)
                frontierSet.add(i.state)
            box.interiors = []
            
            findBox()

            fullPaths = {}
            fastPaths  = {}
            tempExits = []
            for i in box.exits:
                tempExits.append(i)
            while len(tempExits) != 0:
                exit1 = tempExits.pop()
                state1 = exit1[0].state
                
                fp1 = findFullPath(exit1, exit1, box, newGameState, 0)
                fp2 = findFullPath(exit1, exit1, box, newGameState, 1)
                if len(fp1) < len(fp2):
                    fullPaths[(state1, state1)] = fp1
                else:
                    fullPaths[(state1, state1)] = fp2
                fastPaths[(state1, state1)] = []
                tempExits2 = []
                for i in tempExits:
                    tempExits2.append(i)
                count = 0
                while len(tempExits2) != 0:
                    count +=1
                    exit2 = tempExits2.pop()
                    state2 = exit2[0].state
                    fp1 = findFullPath(exit1, exit2, box, newGameState, 0)
                    count +=1
                    fp2 = findFullPath(exit1, exit2, box, newGameState, 1)
                    count +=1
                    if len(fp1) < len(fp2):
                        fullPaths[(state1, state2)] = fp1
                    else:
                        fullPaths[(state1, state2)] = fp2
                    count +=1
                    fastPaths[(state1, state2)] = findFastPath(state1, state2, newGameState)
                    fullPaths[(state2, state1)] = reversePath(fullPaths[(state1, state2)])
                    fastPaths[(state2, state1)] = reversePath(fastPaths[(state1, state2)])
            box.setFastPaths(fastPaths)
            box.setFullPaths(fullPaths)
            for i in box.interiors:
                if i.state in vertexDictionary.keys():
                    del vertexDictionary[i.state]
                elif i.state in edgePieceDictionary.keys():
                    del edgePieceDictionary[i.state]
            for i in box.exits:
                if i[0].state in vertexDictionary.keys():
                    del vertexDictionary[i[0].state]
                elif i[0].state in edgePieceDictionary.keys():
                    del edgePieceDictionary[i[0].state]
            return box
                     
        count = -1  
        
        newBox = True
        while newBox == True:
            newBox = False
            for key in vertexDictionary.keys():
                if newBox == False and key in vertexDictionary.keys() and vertexDictionary[key].numBranches > 3:
                    transitions = vertexDictionary[key].transition
                    for action in transitions.keys():
                        if newBox == False:
                            if transitions[action] in vertexDictionary.keys() or transitions[action] in edgePieceDictionary.keys():
                                if positionDictionary[transitions[action]].numBranches == 2 and isCorner(positionDictionary[transitions[action]]):
                                    count +=1
                                    boxDictionary[count] = buildBox([positionDictionary[key]])
                                    newBox = True
                                elif positionDictionary[transitions[action]].numBranches > 2:
                                    count +=1
                                    boxDictionary[count] = buildBox([positionDictionary[key]])
                                    newBox = True
        newBox = True    
        while newBox == True:
            newBox = False          
            for key in edgePieceDictionary.keys():
                if newBox == False and key in edgePieceDictionary.keys() and isCorner(edgePieceDictionary[key]):
                    corner1 = edgePieceDictionary[key]
                    directions = corner1.transition.keys()
                    direction1 = directions[0]
                    direction2 = directions[1]
    
                    if newBox == False and corner1.transition[direction1] in edgePieceDictionary.keys():
                        corner2 = edgePieceDictionary[corner1.transition[direction1]]
                        if isCorner(corner2):
                            if direction2 in corner2.transition.keys():
                                count +=1
                                boxDictionary[count] = buildBox([corner1, corner2])
                                newBox = True
                    elif newBox == False and corner1.transition[direction1] in vertexDictionary.keys():
                        side = vertexDictionary[corner1.transition[direction1]]
                        if side.numBranches == 3:
                            count +=1
                            boxDictionary[count] = buildBox([corner1])
                            newBox = True
                        
        for key in boxDictionary.keys():
            i = boxDictionary[key]
            print "Box Number",key
            print "Exits: "
            for j in i.exits:
                print j[0].state
            print "Interiors: "
            for j in i.interiors:
                print j.state
                
        for key in boxDictionary.keys():
            i = boxDictionary[key]
            for j in i.exits:
                if j[0].state not in vertexDictionary.keys():
                    vertexDictionary[j[0].state] = j[0]
        
        
        #Put pacman's start position in the vertexDictionary
        newGameState = state
        newGameState.data.agentStates[0].start.pos = pacmanInitialPosition
        legal = newGameState.getLegalPacmanActions()
        if Directions.STOP in legal: legal.remove(Directions.STOP)
        successors = []
        transition = {}
        for action in legal:
            successor = (newGameState.generateSuccessor(0, action), action)
            successors.append(successor)
            transition[action] = successor[0].data.agentStates[0].configuration.pos
        newPosition = Position(newGameState.data.agentStates[0].start.pos, len(successors), transition)
        vertexDictionary[newGameState.data.agentStates[0].start.pos] = newPosition    
        
        #Generate Edges 
        for vertex in vertexDictionary.values():
            for transitionKey in vertex.transition.keys():
                startState = vertex.state
                actionList = []
                delPieces = []
                newDirection = transitionKey
                newState = vertex.transition[newDirection]
                if newState in edgePieceDictionary.keys() or newState in deadEndDictionary.keys():
                    if newState in edgePieceDictionary.keys():
                        newPosition = edgePieceDictionary[newState]
                    else:
                        newPosition = deadEndDictionary[newState]
                    actionList.append(newDirection)
                    delPieces.append(newState)
                    newAdded = True
                    while newState in edgePieceDictionary.keys() and newAdded == True:
                        prevDirection = newDirection
                        prevPosition = newPosition
                        newAdded = False
                        for k in prevPosition.transition.keys():
                            if k != oppositeDirection(prevDirection) or len(prevPosition.transition.keys()) == 1:
                                if prevPosition.transition[k]in edgePieceDictionary.keys() or prevPosition.transition[k]in vertexDictionary.keys() or prevPosition.transition[k]in deadEndDictionary.keys():
                                    prevDirection = newDirection
                                    prevPosition = newPosition
                                    newDirection = k
                                    actionList.append(newDirection)
                                    newState = prevPosition.transition[newDirection]
                                    delPieces.append(newState)
                                    if newState in edgePieceDictionary.keys():
                                        newPosition = edgePieceDictionary[newState]
                                    elif newState in vertexDictionary.keys():
                                        newPosition = vertexDictionary[newState]
                                    elif newState in deadEndDictionary.keys():
                                        newPosition = deadEndDictionary[newState]    
                                    newAdded = True
                            
                                    
                if newState in vertexDictionary.keys() and len(actionList) > 1:
                    action1 = actionList[1]
                    edgeDictionary[startState, action1] = (newState, actionList)
                    reversedPath = reversePath(edgeDictionary[startState, action1][1])
                    action2 = reversedPath[0]
                    edgeDictionary[newState, action2] = (startState, reversedPath)
                elif newState in deadEndDictionary.keys() and len(actionList) > 1:
                    tempList1 = []
                    tempList2 = []
                    if len(actionList) > 1:
                        action1 = actionList[0]
                    for i in actionList:
                        tempList1.append(i)
                        tempList2.append(i)
                    tempList2 = reversePath(tempList2)
                    for i in tempList2:
                        tempList1.append(i)
                    actionList = tempList1
                    edgeDictionary[startState, action1] = (startState, actionList)
                for i in delPieces:
                    if i in deadEndDictionary.keys():
                        del deadEndDictionary[i]
                    elif i in edgePieceDictionary.keys():
                        del edgePieceDictionary[i]
       #Next generate Decisions:
       #For each decision, have a list of actions to add for the first visit
       #For each decision, have a list of actions to add for first visit to one of the nodes in that decision's box
       #Have a set of visited, frontier, and unvisited decisions
       #Put a decision back in the frontier if it still has unvisited paths connected
       #If at a decision with no unvisited paths, find shortest path to all unvisited decisions, 
       #and add them to the frontier if they path doesn't cross an unvisited decision            
                        
                              
        print '?'               
                

            

        
    def getAction(self, state):
        """
        From game.py: 
        The Agent will receive a GameState and must return an action from 
        Directions.{North, South, East, West, Stop}
        """ 
        "*** YOUR CODE HERE ***"
        return self.solution[0,0]
        util.raiseNotDefined()
        
def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built.    The gameState can be any game state -- Pacman's position
    in that state is ignored.
    
    Example usage: mazeDistance( (2,4), (5,6), gameState)
    
    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False)
    return len(search.bfs(prob))