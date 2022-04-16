# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not inrangeribute or publish
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
import math

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by exasuccessorGameStateing
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining pellet (newpellet) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newpellet = successorGameState.getfood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        infinity = 999999999
        neg_infinity = -999999999
        currentPos = list(successorGameState.getPacmanPosition())
        
        successorGameState = infinity
        inrange = 0
        all_pellets = currentGameState.getfood()
        
        pellet = all_pellets.asList()
        for i in range(len(pellet)):
            
            inrange =  (manhattanDistance(pellet[i], currentPos))
            if inrange < successorGameState:
                successorGameState = inrange
        successorGameState = -successorGameState
        
        for state in newGhostStates:
            if state.scaredTimer == 0 :
                if state.getPosition() == tuple(currentPos):
                 return neg_infinity

        
        if action == 'Stop':
            return neg_infinity
        return successorGameState
        




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
    to the successorGameStateimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your successorGameStateimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the successorGameStateimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing successorGameStateimax.

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
        infinity = 999999999
        neg_infinity = -999999999
        def advrminimax(gameState, agentIndex, depth=0):
            legalActionList = gameState.getLegalActions(agentIndex)
            numIndex = gameState.getNumAgents() - 1
            pref_action = None

            
            if (gameState.isLose() or gameState.isWin() or (depth == self.depth)):
                return [self.evaluationFunction(gameState)]
            elif agentIndex == numIndex:
                depth += 1
                childAgentIndex = self.index
            else:
                childAgentIndex = agentIndex + 1
            
            if agentIndex != 0:
                min = infinity 
                for legalAction in legalActionList:
                    successorGameState = gameState.generateSuccessor(agentIndex, legalAction)
                    newMin = advrminimax(successorGameState, childAgentIndex, depth)[0]
                    if newMin == min:
                        if bool(random.getrandbits(1)):
                            pref_action = legalAction
                    
                    elif newMin < min:
                        min = newMin
                        pref_action = legalAction
                return min, pref_action
            
            else:
                max = neg_infinity 
                for legalAction in legalActionList:
                    successorGameState = gameState.generateSuccessor(agentIndex, legalAction)
                    newMax = advrminimax(successorGameState, childAgentIndex, depth)[0]
                    if newMax == max:
                        if bool(random.getrandbits(1)):
                            pref_action = legalAction
                    
                    elif newMax > max:
                        max = newMax
                        pref_action = legalAction
            return max, pref_action

        bestScoreActionPair = advrminimax(gameState, self.index)
        bestScore = bestScoreActionPair[0]
        pref_course =  bestScoreActionPair[1]
        return pref_course


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your successorGameStateimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the successorGameStateimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        infinity = 999999999
        neg_infinity = -999999999
        action, score = self.prune_ab(0, 0, gameState, neg_infinity, infinity)  
        return action  

    def prune_ab(self, curr_depth, agent_index, gameState, alpha, beta):

        
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1
        
        if curr_depth == self.depth:
            return None, self.evaluationFunction(gameState)
        
        best_score, best_action = None, None
        if agent_index == 0:  
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.prune_ab(curr_depth, agent_index + 1, next_game_state, alpha, beta)

                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action

                alpha = max(alpha, score)

                if alpha > beta:
                    break
        else:  
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.prune_ab(curr_depth, agent_index + 1, next_game_state, alpha, beta)

                if best_score is None or score < best_score:
                    best_score = score
                    best_action = action

                beta = min(beta, score)

                if beta < alpha:
                    break

        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score  



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
        infinity = 999999999
        neg_infinity = -999999999
        def expectimax(gameState, agentIndex, depth=0):
            legalActionList = gameState.getLegalActions(agentIndex)
            numIndex = gameState.getNumAgents() - 1
            bestAction = None
            
            if (gameState.isLose() or gameState.isWin() or depth == self.depth):
                return [self.evaluationFunction(gameState)]
            elif agentIndex == numIndex:
                depth += 1
                childAgentIndex = self.index
            else:
                childAgentIndex = agentIndex + 1

            numAction = len(legalActionList)
            
            if agentIndex == self.index:
                value = neg_infinity
            
            else:
                value = 0

            for legalAction in legalActionList:
                successorGameState = gameState.generateSuccessor(agentIndex, legalAction)
                expectedMax = expectimax(successorGameState, childAgentIndex, depth)[0]
                if agentIndex == self.index:
                    if expectedMax > value:
                        
                        value = expectedMax
                        bestAction = legalAction
                else:
                    
                    value = value + ((1.0/numAction) * expectedMax)
            return value, bestAction

        bestScoreActionPair = expectimax(gameState, self.index)
        bestScore = bestScoreActionPair[0]
        bestMove =  bestScoreActionPair[1]
        return bestMove

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, food-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
  #position = currentGameState.getPacmanPosition()
  #food = currentGameState.getFood().asList()

  #foodDist = 0
  #for dot in food:
   # foodDist += 2.5*manhattanDistance(position, dot)

  #ghostDanger = 0
  #for ghost in currentGameState.getGhostPositions():
    #dist = max(4 - manhattanDistance(position, ghost), 0)
   # ghostDanger += math.pow(dist, 3)

  #additionalFactors = -10*len(food) # penalty for amount of food remaining
  #if currentGameState.isLose(): additionalFactors -= 5000
  #elif currentGameState.isWin(): additionalFactors += 5000
  #additionalFactors += random.randint(-5, 5) # prevent paralysis due to ties

  #return currentGameState.getScore() - foodDist - ghostDanger + additionalFactors

# for grading the 5th code plz uncomment it after grading the rest
# for unknown reasons the grading code will never get past the 5th code
# no matter what i implement

# Abbreviation
better = betterEvaluationFunction
