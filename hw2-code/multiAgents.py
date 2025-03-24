# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if action == Directions.STOP:
            return -math.inf
        
        ghostDist = manhattanDistance(newPos, newGhostStates[0].configuration.pos)
        newFoods = newFood.asList()
        # Eat all foods
        if len(newFoods) == 0:
            return math.inf

        # Find closest food
        closestFoodDist = math.inf
        for food in newFoods:
            closestFoodDist = min(closestFoodDist, manhattanDistance(newPos, food))
        return successorGameState.getScore() + ghostDist / closestFoodDist

def scoreEvaluationFunction(currentGameState: GameState):
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
    def getValue(self, gameState:GameState, agentIndex, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
         
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)[0]
        else:
            return self.minValue(gameState, agentIndex, depth)[0]

    def minValue(self, gameState: GameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        minVal = math.inf
        minAction = ""
        for action in actions:
            nextgameState = gameState.generateSuccessor(agentIndex, action)
            nextAgent = (agentIndex+1) % gameState.getNumAgents()
            
            nextVal = self.getValue(nextgameState, nextAgent, depth+1)
            if nextVal < minVal:
                minVal = nextVal
                minAction = action

        return minVal, minAction
           

    def maxValue(self, gameState: GameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        maxVal = -math.inf
        maxAction = ""
        for action in actions:
            nextgameState = gameState.generateSuccessor(agentIndex, action)
            nextAgent = (agentIndex+1) % gameState.getNumAgents()
            
            nextVal = self.getValue(nextgameState, nextAgent, depth+1)
            if nextVal > maxVal:
                maxVal = nextVal
                maxAction = action
            
        return maxVal, maxAction

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        _, optimalAction = self.maxValue(gameState, 0, 0)
        return optimalAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getValue(self, gameState:GameState, agentIndex, depth, alpha, beta):
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
         
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)[0]
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)[0]

    def minValue(self, gameState: GameState, agentIndex, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        minVal = math.inf
        minAction = ""
        for action in actions:
            nextgameState = gameState.generateSuccessor(agentIndex, action)
            nextAgent = (agentIndex+1) % gameState.getNumAgents()
            
            nextVal = self.getValue(nextgameState, nextAgent, depth+1, alpha, beta)
            if nextVal < minVal:
                minVal = nextVal
                minAction = action

            if nextVal < alpha:
                return minVal, minAction
            beta = min(beta, minVal)
                
        return minVal, minAction
           

    def maxValue(self, gameState: GameState, agentIndex, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        maxVal = -math.inf
        maxAction = ""
        for action in actions:
            nextgameState = gameState.generateSuccessor(agentIndex, action)
            nextAgent = (agentIndex+1) % gameState.getNumAgents()
            
            nextVal = self.getValue(nextgameState, nextAgent, depth+1, alpha, beta)
            if nextVal > maxVal:
                maxVal = nextVal
                maxAction = action
            
            if nextVal > beta:
                return maxVal, maxAction
            alpha = max(alpha, maxVal)

        return maxVal, maxAction

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        _, optimalAction = self.maxValue(gameState, 0, 0, -math.inf, math.inf)
        return optimalAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getValue(self, gameState:GameState, agentIndex, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
         
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)[0]
        else:
            return self.expValue(gameState, agentIndex, depth)[0]

    def expValue(self, gameState: GameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        expVal = 0
        for action in actions:
            nextgameState = gameState.generateSuccessor(agentIndex, action)
            nextAgent = (agentIndex+1) % gameState.getNumAgents()
            
            nextVal = self.getValue(nextgameState, nextAgent, depth+1)
            probablity = 1 / len(actions)
            expVal += nextVal * probablity

        return expVal, ""
           

    def maxValue(self, gameState: GameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        maxVal = -math.inf
        maxAction = ""
        for action in actions:
            nextgameState = gameState.generateSuccessor(agentIndex, action)
            nextAgent = (agentIndex+1) % gameState.getNumAgents()
            
            nextVal = self.getValue(nextgameState, nextAgent, depth+1)
            if nextVal > maxVal:
                maxVal = nextVal
                maxAction = action
            
        return maxVal, maxAction

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        _, optimalAction = self.maxValue(gameState, 0, 0)
        return optimalAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    legalActions = currentGameState.getLegalActions()
    for action in legalActions:
        successorGameState = currentGameState.generatePacmanSuccessor(action)
    
    currentPos = currentGameState.getPacmanPosition()
    currentFoods = currentGameState.getFood().asList()
    currentGhostStates = currentGameState.getGhostStates()
    currentCapsules = currentGameState.getCapsules()

    if currentGameState.isWin():
        return math.inf
    if currentGameState.isLose():
        return -math.inf

    # Find closest food
    closestFoodDist = math.inf
    for food in currentFoods:
        closestFoodDist = min(closestFoodDist, manhattanDistance(food, currentPos))
    
    # Find closest ghost
    closestGhostDist = math.inf
    for ghost in currentGhostStates:
        closestGhostDist = min(closestGhostDist, manhattanDistance(ghost.configuration.pos, currentPos))


    return 500000/len(currentFoods) + 5000/(len(currentCapsules)+1) + 500/(closestFoodDist) + closestGhostDist

# Abbreviation
better = betterEvaluationFunction
