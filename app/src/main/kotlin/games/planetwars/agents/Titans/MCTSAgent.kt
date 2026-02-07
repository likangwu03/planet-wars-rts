package games.planetwars.agents.strategic

import games.planetwars.agents.Action
import games.planetwars.agents.AbstractGameState
import games.planetwars.agents.GameStateWrapper
import games.planetwars.agents.PlanetWarsPlayer
import games.planetwars.core.*
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Monte Carlo Tree Search agent implementation for Planet Wars
 */
class MCTSAgent(
    private val simulationCount: Int = 50,
    private val explorationWeight: Double = 2.0,
    private val maxRolloutDepth: Int = 20,
    private val rolloutCount: Int = 3,
    private val timeLimit: Long = 200
) : PlanetWarsPlayer() {
    
    private var estimatedGamePhase: Double = 0.0
    
    private inner class MCTSNode(
        val state: AbstractGameState,
        val playerAtNode: Player,
        val parent: MCTSNode? = null
    ) {
        val children: MutableMap<Action, MCTSNode> = mutableMapOf()
        var visits: Int = 0
        var totalValue: Double = 0.0
        
        fun ucbScore(parentVisits: Int): Double {
            if (visits == 0) return Double.POSITIVE_INFINITY
            val exploitation = totalValue / visits
            val exploration = explorationWeight * sqrt(ln(parentVisits.toDouble()) / visits)
            return exploitation + exploration
        }
        
        fun isTerminal(): Boolean = state.isTerminal()
        
        fun selectBestChild(): MCTSNode? {
            if (children.isEmpty()) return null
            return children.values.maxByOrNull { it.ucbScore(visits) }
        }
        
        fun expand(): MCTSNode? {
            val possibleActions = state.getLegalActions(playerAtNode)
            val unexploredActions = possibleActions.filter { it !in children }
            
            if (unexploredActions.isEmpty()) return null

            val action = unexploredActions.random()
            val nextPlayer = if (playerAtNode == Player.Player1) Player.Player2 else Player.Player1
            val nextState = state.next(mapOf(playerAtNode to action))
            
            val childNode = MCTSNode(nextState, nextPlayer, this)
            children[action] = childNode
            return childNode
        }
    }
    
    override fun getAction(gameState: GameState): Action {
        updateGamePhase(gameState)
        
        val startTime = System.currentTimeMillis()
        val adaptableState = GameStateWrapper(gameState.deepCopy(), params)
        val rootNode = MCTSNode(adaptableState, player)
        
        var iterations = 0
        while (iterations < simulationCount && 
               (System.currentTimeMillis() - startTime) < timeLimit) {
            
            var node = rootNode
            
            // Selection
            while (!node.isTerminal() && node.children.isNotEmpty()) {
                node = node.selectBestChild() ?: break
            }
            
            // Expansion
            if (!node.isTerminal()) {
                val expandedNode = node.expand()
                if (expandedNode != null) node = expandedNode
            }
            
            // Simulation
            val value = simulation(node)
            
            // Backpropagation
            backpropagation(node, value)
            
            iterations++
        }
        
        return findBestAction(rootNode)
    }
    
    private fun findBestAction(rootNode: MCTSNode): Action {
        if (rootNode.children.isEmpty()) return Action.doNothing()
        return rootNode.children.entries.maxByOrNull { it.value.visits }?.key ?: Action.doNothing()
    }
    
    private fun simulation(node: MCTSNode): Double {
        var totalScore = 0.0
        repeat(rolloutCount) {
            totalScore += rollout(node.state.copy(), node.playerAtNode)
        }
        return totalScore / rolloutCount
    }
    
    private fun rollout(state: AbstractGameState, startingPlayer: Player): Double {
        var currentState = state
        var currentPlayer = startingPlayer
        var depth = 0
        
        while (!currentState.isTerminal() && depth < maxRolloutDepth) {
            val actions = currentState.getLegalActions(currentPlayer)
            if (actions.isEmpty()) break
            
            val action = if (Random.nextDouble() < 0.3 && actions.size > 1) {
                actions.random()
            } else {
                actions.random()
            }
            
            currentState = currentState.next(mapOf(currentPlayer to action))
            currentPlayer = if (currentPlayer == Player.Player1) Player.Player2 else Player.Player1
            depth++
        }
        
        return evaluateState(currentState)
    }
    
    private fun backpropagation(node: MCTSNode, value: Double) {
        var current: MCTSNode? = node
        while (current != null) {
            current.visits++
            current.totalValue += value
            current = current.parent
        }
    }
    
    private fun updateGamePhase(gameState: GameState) {
        val totalPlanets = gameState.planets.size
        val myPlanets = gameState.planets.count { it.owner == player }
        val oppPlanets = gameState.planets.count { it.owner == player.opponent() }
        val neutralPlanets = totalPlanets - myPlanets - oppPlanets
        
        estimatedGamePhase = 1.0 - (neutralPlanets.toDouble() / totalPlanets.coerceAtLeast(1))
    }
    
    private fun evaluateState(state: AbstractGameState): Double {
        if (state.isTerminal()) {
            val scores = state.getScore()
            val myScore = scores[player] ?: 0.0
            val oppScore = scores[player.opponent()] ?: 0.0
            
            return when {
                myScore > oppScore -> 1.0
                myScore < oppScore -> -1.0
                else -> 0.0
            }
        }
        
        val scores = state.getScore()
        val myShips = scores[player] ?: 0.0
        val oppShips = scores[player.opponent()] ?: 0.0
        
        return (myShips - oppShips) / (myShips + oppShips + 1.0)
    }
    
    override fun getAgentType(): String = "MCTS Agent"
    
    companion object {
        fun createFast(): MCTSAgent {
            return MCTSAgent(
                simulationCount = 30,
                explorationWeight = 1.5,
                maxRolloutDepth = 10,
                rolloutCount = 2
            )
        }
        
        fun createStrong(): MCTSAgent {
            return MCTSAgent(
                simulationCount = 100,
                explorationWeight = 2.0,
                maxRolloutDepth = 30,
                rolloutCount = 5
            )
        }
    }
}

fun main() {
    val gameParams = GameParams(numPlanets = 30, maxTicks = 1000)
    val gameState = GameStateFactory(gameParams).createGame()
    val mctsAgent = MCTSAgent.createFast()
    mctsAgent.prepareToPlayAs(Player.Player1, gameParams)
    println("Testing ${mctsAgent.getAgentType()}")
    val action = mctsAgent.getAction(gameState)
    println("Action: $action")
}
