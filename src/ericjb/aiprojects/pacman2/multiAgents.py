# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
from game import Actions
import time
import random, util
import search
from game import Agent
import searchAgents #@UnresolvedImport
import math

def findFastPath(point1, point2, gameState):
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False)
    solution = search.bfs(prob)
    return solution

def pathLengthToClosestFood(gameState):
    solver = ClosestDotSearchAgent()
    solution = solver.findPathToClosestDot(gameState)
    if solution == None:
        return 0
    return len(solution)
def pathToClosestFood(gameState):
    solver = ClosestDotSearchAgent()
    solution = solver.findPathToClosestDot(gameState)
    return solution

def getFoodPositions(foodGrid):
    foodPosSet = set([])
    countX = -1
    countY = -1
    for i in foodGrid:
        countX+=1
        countY = -1
        for j in i:
            countY+=1
            if j == True:
                foodPosSet.add((countX, countY))
    return foodPosSet

def getDistances(pacmanPos, newGhostStates, gameState):
    ghostDistSet = set([])
    ghostPosSet = set([])
    preyDistSet = set([])
    preyPosSet = set([])
    capPosSet = gameState.getCapsules()
    closestGhost = 999
    closestPrey = 999
    closestCapDistance = 999
    capToGhostDistance = 999
    closestCap = None
    
    for i in newGhostStates:
        if i.scaredTimer < 2:
            tempPos = []
            for j in i.configuration.pos:
                if isinstance(j, float):
                    tempPos.append(int(math.floor(j)))
                else:
                    tempPos.append(j)
            gPos = (tempPos[0], tempPos[1])
            ghostPosSet.add(gPos)
            if gPos != pacmanPos:
                if manhattanDistance(pacmanPos, gPos) < 4:
                    ghostDistSet.add(len(findFastPath(pacmanPos, gPos, gameState)))
                else:
                    ghostDistSet.add(10)
            else:
                ghostDistSet.add(0)
        else:
            tempPos = []
            for j in i.configuration.pos:
                if isinstance(j, float):
                    tempPos.append(int(math.floor(j)))
                else:
                    tempPos.append(j)
            gPos = (tempPos[0], tempPos[1])
            preyPosSet.add(gPos)
            if gPos != pacmanPos:
                if manhattanDistance(pacmanPos, gPos) < 4:
                    preyDistSet.add(len(findFastPath(pacmanPos, gPos, gameState)))
                else:
                    preyDistSet.add(manhattanDistance(pacmanPos, gPos))
            else:
                preyDistSet.add(0)
    for i in ghostDistSet:
        if i < closestGhost: closestGhost = i
    for i in preyDistSet:
        if i < closestPrey: closestPrey = i
    
    
    for i in capPosSet:
        testCapDistance = manhattanDistance(pacmanPos, i)
        if testCapDistance < closestCapDistance:
            closestCap = i
            closestCapDistance = testCapDistance
    
    if pacmanPos != closestCap:
        if closestCapDistance < 4:
            closestCapDistance = (len(findFastPath(pacmanPos, closestCap, gameState)))
        else:
            closestCapDistance = (closestCapDistance*2)
    else:
        closestCapDistance = 0
    
    for i in ghostPosSet:
        if i != closestCap and closestCap != None:
            tempDist = manhattanDistance(closestCap, i)
            if tempDist < 4:
                tempDist = len(findFastPath(closestCap, i, gameState))
            else: tempDist = (tempDist*2)
        else:
            tempDist = 0
        if tempDist < capToGhostDistance:
            capToGhostDistance = tempDist
    
                
    
    
    
    
    
        
    return ghostDistSet,preyDistSet,closestGhost,closestPrey,closestCap,closestCapDistance,capToGhostDistance

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    score = None
    foodLeft = None
    closestFood = None
    closestGhost = None
    foodLeftWeight = 1000
    closestFoodWeight = 100
    maxMazeDistance = (currentGameState.data.layout.height + currentGameState.data.layout.width)
    closestGhostWeight = 100/maxMazeDistance
    bestScore = 100000
    
    
    
    #print " "
    #print "Possible Successor:"
    #print "pPos:",newPos
    #print oldFood
    count = -1
    
    foodPosSet = getFoodPositions(oldFood)
    if len(foodPosSet) <=1 and newPos in foodPosSet:
        closestFood = 0
        foodLeft = 0
    elif newPos in foodPosSet:
        foodLeft = (len(foodPosSet)-1)
        closestFood = 1
    else:
        foodLeft = len(foodPosSet)
        closestFood = pathLengthToClosestFood(successorGameState)
    #print "Distance to Closest Food: ",closestFood
        
    ghostPosSet,ghostDistSet = getDistances(newPos, newGhostStates, successorGameState)
    closestGhost = 999
    for i in ghostDistSet:
        if i < closestGhost: closestGhost = i
    #print "Distance to Closest Ghost: ",closestGhost
    #for i in foodPosSet:
    #    print "Food at: ",i
    
    
    score = 0
    if closestGhost == 0:
        score -=bestScore
    elif closestGhost == 1:
        if closestFood == 0 and foodLeft == 0:
            score +=bestScore
        else:
            score -=(bestScore/2)
            score += ((foodLeftWeight/foodLeft)+(closestFoodWeight/closestFood)+(closestGhostWeight/closestGhost))
    elif closestGhost == 2:
        if closestFood == 0 and foodLeft == 0:
            score +=bestScore
        else:
            score -=(bestScore/4)
            score += ((foodLeftWeight/foodLeft)+(closestFoodWeight/closestFood)+(closestGhostWeight/closestGhost))
    elif closestGhost > 2:
        if closestFood == 0 and foodLeft == 0:
            score +=bestScore
        else:
            score += ((foodLeftWeight/foodLeft)+(closestFoodWeight/closestFood)+(closestGhostWeight/closestGhost))
    #print "Food Left: ",foodLeft
    #print "Closest Food: ",closestFood
    #print "Closest Ghost: ",closestGhost
    #print score
    return score

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    def maxValue(state, depth):
        actions = state.getLegalActions(0)
        #if Directions.STOP in actions: actions.remove(Directions.STOP)
        if terminalTest(state, depth):
            debugOut = ""
            for i in range(2, depth): debugOut +="        "
            debugOut +="Max:  "+str(utility(state))+"\n"
            debug.write(debugOut)
            return utility(state),state,Directions.STOP
        returnV = -float("inf")
        returnState = None
        returnAction = None
        parentState = state
        
        #For each action choose smallest ghost reaction
        #then choose best action
        for a in actions:
            successor = (parentState.generateSuccessor(0, a))
            agentV = float("inf")
            newAgentState = successor
            for i in range(1, len(state.data.agentStates)):
                newValue,newState,unused = minValue(newAgentState, depth, i)
                if newValue <= agentV:
                    agentV = newValue
                    newAgentState = newState
            if agentV >= returnV:
                returnState = successor
                returnV = agentV
                returnAction = a
        debugOut = ""
        for i in range(2, depth): debugOut +="        "
        debugOut +="Max:  "+str(utility(state))+"\n"
        debug.write(debugOut)
        return returnV,returnState,returnAction
        
    def minValue(state, depth, agent):
        actions = state.getLegalActions(agent)
        if Directions.STOP in actions: actions.remove(Directions.STOP)
        if terminalTest(state, depth):
            debugOut = "    "
            for i in range(2, depth): debugOut +="        "
            debugOut +="Min:  "+str(utility(state))+"\n"
            debug.write(debugOut)
            return utility(state),state,Directions.STOP
        returnV = float("inf")
        returnState = None
        returnAction = None
        for a in actions:
            successor = (state.generateSuccessor(agent, a))
            newValue,newState,unused = maxValue(successor, (depth+1))
            if newValue <= returnV:
                returnState = successor
                returnV = newValue
                returnAction = a
        debugOut = "    "
        for i in range(2, depth): debugOut +="        "
        debugOut +="Min:  "+str(utility(state))+"\n"
        debug.write(debugOut)
        return returnV,returnState,returnAction
    
    def utility(state):
        return self.evaluationFunction(state)
    
    def terminalTest(state, depth):
        if depth == self.depth or state.isLose() or state.isWin():
            return True
        return False
    
    debug = open('debug.txt', 'w')
    debug.write("Start: \n")
    debug.truncate()
    v,s,a = maxValue(gameState, 0)
    print "Minimax Value: ",v
    debug.close()
    return a 

    util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    
    def maxValue(state, depth, alpha, beta):
        actions = state.getLegalActions(0)
        if Directions.STOP in actions: actions.remove(Directions.STOP)
        if terminalTest(state, depth):
            return utility(state),state,Directions.STOP,alpha,beta
        returnV = -float("inf")
        returnState = None
        returnAction = None
        parentState = state
        
        #For each action run through all ghost reactions,
        #updating the best gamestate and value? each time
        for a in actions:
            successor = (parentState.generateSuccessor(0, a))
            agentV = float("inf")
            newAgentState = successor
            for i in range(1, len(state.data.agentStates)):
                newValue,newState,unused,newAlpha,newBeta = minValue(newAgentState, depth, i, alpha, beta)
                if newValue <= agentV:
                    agentV = newValue
                    newAgentState = newState
            if agentV >= beta: return agentV,state,Directions.STOP,alpha,beta
            if agentV > alpha: alpha = agentV
            if agentV >= returnV:
                returnState = successor
                returnV = agentV
                returnAction = a
        return returnV,returnState,returnAction,alpha,beta
        
    def minValue(state, depth, agent, alpha, beta):
        actions = state.getLegalActions(agent)
        if Directions.STOP in actions: actions.remove(Directions.STOP)
        if terminalTest(state, depth):
            return utility(state),state,Directions.STOP,alpha,beta
        returnV = float("inf")
        returnState = None
        returnAction = None
        for a in actions:
            successor = (state.generateSuccessor(agent, a))
            newValue,newState,unused,newAlpha,newBeta = maxValue(successor, (depth+1), alpha, beta)
            if newValue <= alpha: return newValue,state,Directions.STOP,alpha,beta
            if newValue < beta: beta = newValue
            if newValue <= returnV:
                returnState = successor
                returnV = newValue
                returnAction = a
        return returnV,returnState,returnAction,alpha,beta
    
    def utility(state):
        return self.evaluationFunction(state)
    
    def terminalTest(state, depth):
        if depth == self.depth or state.isLose() or state.isWin():
            return True
        return False
    
    v,s,a,alpha,beta = maxValue(gameState, 0, -float("inf"), float("inf"))
    print "Minimax Value: ",v
    return a


    util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    
    def maxValue(state, depth):
        actions = state.getLegalActions(0)
        #if Directions.STOP in actions: actions.remove(Directions.STOP)
        if terminalTest(state, depth):
            return utility(state),state,Directions.STOP
        returnV = -float("inf")
        returnState = None
        returnAction = None
        parentState = state
        
        #For each action choose smallest ghost expected value
        #then choose best action
        for a in actions:
            successor = (parentState.generateSuccessor(0, a))
            agentV = float("inf")
            newAgentState = successor
            for i in range(1, len(state.data.agentStates)):
                newValue,newState,unused = expectedValue(newAgentState, depth, i)
                if newValue <= agentV:
                    agentV = newValue
                    newAgentState = newState
            if agentV >= returnV:
                returnState = successor
                returnV = agentV
                returnAction = a
        return returnV,returnState,returnAction
    
    
    def expectedValue(state, depth, agent):
        actions = state.getLegalActions(agent)
        if Directions.STOP in actions: actions.remove(Directions.STOP)
        if terminalTest(state, depth):
            return utility(state),state,Directions.STOP
        returnState = None
        returnAction = None
        randMove = random.randint(1, len(actions))
        count = 0
        totalValue = 0
        for a in actions:
            count +=1
            successor = (state.generateSuccessor(agent, a))
            newValue,newState,unused = maxValue(successor, (depth+1))
            totalValue +=newValue
            if count == randMove:
                returnState = successor
                returnAction = a
        averageValue = (totalValue/len(actions))
        returnV = averageValue
        return returnV,state,returnAction
    
    def utility(state):
        return self.evaluationFunction(state)
    
    def terminalTest(state, depth):
        if depth == self.depth or state.isLose() or state.isWin():
            return True
        return False
    
    debug = open('debug.txt', 'w')
    debug.write("Start: \n")
    debug.truncate()
    v,s,a = maxValue(gameState, 0)
    #print "Minimax Value: ",v
    debug.close()
    return a 

    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
    
      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    score = 0
    foodLeft = None
    closestFood = None
    closestGhost = None
    justAteFoodWeight = 1000
    justAteGhostWeight = 5000
    closestFoodWeight = 100
    maxMazeDistance = (currentGameState.data.layout.height + currentGameState.data.layout.width)
    closestGhostWeight = 100/maxMazeDistance
    bestScore = 100000
    newPoints = currentGameState.data.scoreChange
    gameScore = round(currentGameState.data.score,-2)
    gameScoreWeight = bestScore
    
    if newPoints >20 and newPoints < 500:
        justAteGhost = 1
    else:
        justAteGhost = 0
    justAteCap = 0
    for i in newScaredTimes:
        if i > 38:
            justAteCap = 1
            
    foodPosSet = getFoodPositions(oldFood)
    foodLeft = len(foodPosSet)
    closestFood = pathLengthToClosestFood(currentGameState)    
    totalFood = len(getFoodPositions(currentGameState.data.layout.food))
    maxFoodWeight = (totalFood-1)*justAteFoodWeight
    currentFoodWeight = maxFoodWeight-((foodLeft-1)*justAteFoodWeight)
    ghostDistSet,preyDistSet,closestGhost,closestPrey,closestCap,closestCapDistance,capToGhostDistance = getDistances(newPos, newGhostStates, currentGameState)
    
    if len(ghostDistSet) == 0: ghostsExist = False
    else: ghostsExist = True
    if len(preyDistSet) == 0: preyExist = False
    else: preyExist = True
    
    score+=(gameScore*gameScoreWeight)
    if preyExist == False and closestCapDistance <= 5 and capToGhostDistance >= closestCapDistance:
        if closestGhost == 0:
            score -=bestScore
        else:
            score += currentFoodWeight+5000
            score += bestScore/closestCapDistance
        #print "CLOSE TO POWER PELLET!"
        #print "Position: ",newPos
        #print "Distance: ",closestCapDistance
        #print "GhostCap: ",capToGhostDistance
        #print "CloseGhst: ",closestGhost
        #print "Score: ",score
        score+=gameScore
                
    elif ghostsExist == False:
        if justAteCap == 1 and closestPrey < 4:
            score += bestScore*3
        score += (10-closestPrey)*2000
        score += currentFoodWeight
        #print "POWER PELLET!"
        #print "Position: ",newPos
        #print "Score: ",score
        #print "ClosePrey: ",closestPrey
        
    elif preyExist and ghostsExist:
        if justAteGhost == True and closestGhost > 1:
            score += bestScore*8
            #print "ATE GHOST!"
            #print "Position: ",newPos
            #print "Score: ",score
        elif closestGhost == 0:
            score -= bestScore
        elif closestGhost == 1:
            score -=(bestScore/2)
        elif closestGhost == 2:
            score -=(bestScore/4)
        elif closestGhost > 2:
            score += (30-closestPrey)*1000
            #print "Close Prey",closestPrey
            
    else:
        if justAteGhost == True and closestGhost > 1:
            score += bestScore*8
        if closestGhost == 0:
            score -=bestScore
        elif closestGhost == 1:
            if closestFood == 0 and foodLeft == 0:
                score +=bestScore
            else:
                score -=(bestScore/2)
                score += ((currentFoodWeight)+(closestFoodWeight/closestFood)+(closestGhostWeight/closestGhost)+(justAteGhostWeight*justAteGhost))
        elif closestGhost == 2:
            if closestFood == 0 and foodLeft == 0:
                score +=bestScore
            else:
                score -=(bestScore/4)
                score += ((currentFoodWeight)+(closestFoodWeight/closestFood)+(closestGhostWeight/closestGhost)+(justAteGhostWeight*justAteGhost))
        elif closestGhost > 2:
            if closestFood == 0 and foodLeft == 0:
                score +=bestScore
            else:
                score += ((currentFoodWeight)+(closestFoodWeight/closestFood)+(closestGhostWeight/closestGhost)+(justAteGhostWeight*justAteGhost))
    
    #print "prey?:",preyExist
    #print "cap: ",closestCapDistance
    #print "gho: ",capToGhostDistance
    #print "Food Left: ",foodLeftWeight
    #print "Closest Food: ",closestFood
    #print "Closest Ghost: ",closestGhost
    #print score
    return score
  
  
  
  
  
  
  
  
  
  
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()








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
            self.searchFunction = func
        else:
            if heuristic in dir(searchAgents):
                heur = getattr(searchAgents, heuristic)
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)
            
        # Get the search problem type from the name
        if prob not in dir(searchAgents) or not prob.endswith('Problem'): 
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = getattr(searchAgents, prob)
        
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