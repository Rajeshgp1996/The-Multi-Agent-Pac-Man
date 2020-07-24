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
import random, util

from game import Agent

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
        currentFood = currentGameState.getFood()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #Array to store the scores
        scores = []

        #List to store all the positions that can result in pacman being pinned.
        dangerPos = []
        #Getting new position of the ghost
        ghostPos = newGhostStates[0].configuration.pos
        #List of "current" food items on the board
        currentFoodList = currentFood.asList()

        #List of food items after pacman has made a move
        foodlist = newFood.asList()

        #Manhattan distance of each food item from the pacmans new position
        for food in foodlist:
            scores.append(abs(newPos[0] - food[0]) + abs(newPos[1] - food[1]))

        #Saving all the danger positions for the pacman by enumerating all directions and checking if ghost
        #is encountered.
        dangerPos.append((ghostPos[0], ghostPos[1]))
        dangerPos.append((ghostPos[0] - 1, ghostPos[1]))
        dangerPos.append((ghostPos[0], ghostPos[1] + 1))
        dangerPos.append((ghostPos[0] + 1, ghostPos[1]))
        dangerPos.append((ghostPos[0], ghostPos[1] - 1))

        #Weighted parameters for ghosts and food items from pacmans position
        if newPos in dangerPos:
            return -100
        elif newPos in currentFoodList:
            return 1000
        elif len(scores):
            return 1000 / (max(scores) + min(scores))
        else:
            return 100

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

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        #Calling the generic function findRecursive to find the move based on the current game state according to
        # MIN-MAX Algo. Only first 4 parameters are relevant for simple min-max algorithm
        value, moveTaken = findRecusrive(0, gameState, self.depth, self.evaluationFunction, False, -sys.maxint, sys.maxint,False)
        return moveTaken

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Calling the generic function findRecursive to find the move based on the current game state according to
        # MIN-MAX Algorithm with alpha beta pruning.
        # This time we pass pruning parameter as True along with initial values of alpha and beta parameter as -inf
        # and positive inf.
        value, moveTaken = findRecusrive(0, gameState, self.depth, self.evaluationFunction,True, -sys.maxint, sys.maxint,False)
        return moveTaken

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
        # Calling the generic function findRecursive to find the move based on the current game state according to
        # expectimax algorithm.
        # This time we pass expectimax parameter as True without pruning enabled.
        value, moveTaken = findRecusrive(0, gameState, self.depth, self.evaluationFunction,False, None, None,True)
        return moveTaken

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

import sys
#Generic function to perform min-max algorithm with different variants like alpha-beta pruning and expectimax.
#Parameters
# agentId -> The agent number in the game. 0-> pacman, all other agents are ghosts.
# agentGameState ->  The current state of the game.
# depth -> The current depth in the tree. A depth is covered when all the agents have moved by one step.
# evaluationFunction -> Called at the leaf nodes to get values from the utility function
# pruning -> boolean parameter to enable alpha-beta pruning in the tree.
# alpha_value -> alpha parameter in alpha-beta pruning. Initial value is -inf.
# beta_value -> beta parameter in alpha-beta pruning. Initial value is +inf
# expectimax -> boolean parameter to enable expectimax search.
def findRecusrive(agentId, agentGameState, depth, evaluationFunction, pruning, alpha_value, beta_value, expectimax):

    #If lastAgent is already processed, update the agent number and reduce depth by 1.
    if (agentId == agentGameState.getNumAgents()):
        agentId = 0
        depth = depth - 1

    # Find all the legal moves for the current agent in the current state.
    legalMoves = agentGameState.getLegalActions(agentId)

    # Check if we reached at the leaf nodes or the provided depth is covered. Return the value from utility function
    # along with action as null.
    if(legalMoves is None or len(legalMoves) == 0 or depth == 0):
        value = evaluationFunction(agentGameState)
        return value, 0

    #Max-turn, Pacman's move.
    if(agentId == 0):
        prevMax = -sys.maxint
        moveTaken = None

        #Traverse through all the legal moves possible for this agent in the current state.
        for action in legalMoves:
            #Generate successor state.
            successorGameState = agentGameState.generateSuccessor(agentId, action)

            #Recursively call the function for giving turn to the next agent.
            current_value, rec_move = findRecusrive(agentId + 1, successorGameState, depth, evaluationFunction, pruning, alpha_value, beta_value, expectimax)

            #Update the maximum value and best move from all the child branches
            if(current_value > prevMax):
                moveTaken = action
                prevMax = current_value

            #If pruning is enabled and current_value is greater than beta value, we can return the action as the upper
            # min node will reject this value anyway. Hence can be pruned.
            if (pruning and current_value > beta_value):
                return current_value, action
            #Else we update the alpha value with the current maximum value.
            else:
                alpha_value = max(alpha_value, current_value)
        return prevMax, moveTaken

    #Min (Expecti) node turn
    else:
        expectisum = 0.0
        prevMin = sys.maxint
        moveTaken = None

        # Traverse through all the legal moves possible for this agent in the current state.
        for action in legalMoves:
            # Generate successor state.
            successorGameState = agentGameState.generateSuccessor(agentId, action)

            # Recursively call the function for giving turn to the next agent.
            current_value, rec_move = findRecusrive(agentId + 1, successorGameState, depth, evaluationFunction, pruning, alpha_value, beta_value, expectimax)

            # Update the minimum value and best move from all the child branches
            if (current_value < prevMin):
                moveTaken = action
                prevMin = current_value

            # If pruning is enabled and current_value is less than alpha value, we can return the action as the upper
            # max node will reject this value. Hence can be pruned.
            if (pruning and current_value < alpha_value):
                return current_value, action

            # Else we update the alpha value with the current maximum value.
            else:
                beta_value = min(beta_value, current_value)

            #Sum of all the values returned from child nodes returning in case of expectimax nodes.
            expectisum += current_value

        #If expectimax is enabled, we return the average of all the values from the child nodes.
        if(expectimax):
            return expectisum / len(legalMoves), None
        return prevMin, moveTaken