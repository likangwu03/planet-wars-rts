package games.planetwars.agents.Titans

import games.planetwars.agents.Action
import games.planetwars.agents.AbstractGameState
import games.planetwars.agents.PlanetWarsPlayer
import games.planetwars.core.*
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Monte Carlo Tree Search agent implementation for Planet Wars
 */
class MCTSAgent(
    private val simulationCount: Int = 50,      // Number of iterations to run
    private val explorationWeight: Double = 2.0, // UCB exploration parameter 
    private val maxRolloutDepth: Int = 20,      // Max depth for random rollouts
    private val rolloutCount: Int = 3,          // Number of rollouts per leaf node
    private val timeLimit: Long = 200,          // Time limit in milliseconds
    private val progressiveBias: Double = 0.1,  // Weight for progressive bias heuristic
    private val adaptivePlayouts: Boolean = true // Adjust playout depth based on game phase
) : PlanetWarsPlayer() {
    
    // Keep track of game phase for adaptive playouts
    private var estimatedGamePhase: Double = 0.0 // 0.0 = early, 0.5 = mid, 1.0 = late
    
    /**
     * Node in the MCTS tree representing a game state and possible actions
     */
    private inner class MCTSNode(
        val state: AbstractGameState,
        val playerAtNode: Player,
        val parentAction: Action? = null,
        val parent: MCTSNode? = null
    ) {
        val children: MutableMap<Action, MCTSNode> = mutableMapOf()
        var visits: Int = 0
        var totalValue: Double = 0.0
        val heuristicValues: MutableMap<Action, Double> = mutableMapOf() // Cache for action heuristic values
        
        /**
         * Calculate the UCB (Upper Confidence Bound) score for this node
         */
        fun ucbScore(parentVisits: Int): Double {
            if (visits == 0) {
                return Double.POSITIVE_INFINITY
            }
            
            val exploitation = totalValue / visits
            val exploration = explorationWeight * sqrt(ln(parentVisits.toDouble()) / visits)
            
            return exploitation + exploration
        }
        
        /**
         * Calculate the UCB with progressive bias for an action
         */
        fun progressiveUcbScore(action: Action, parentVisits: Int): Double {
            val childNode = children[action] ?: return Double.POSITIVE_INFINITY
            
            val ucbScore = childNode.ucbScore(parentVisits)
            
            // Add progressive bias if enabled
            val heuristicValue = heuristicValues.getOrPut(action) {
                calculateActionHeuristic(state, action, playerAtNode)
            }
            
            // Progressive bias term decreases with more visits
            val progressiveBiasTerm = (progressiveBias * heuristicValue) / (1.0 + childNode.visits)
            
            return ucbScore + progressiveBiasTerm
        }
        
        /**
         * Check if this node has been fully expanded
         */
        fun isFullyExpanded(): Boolean {
            val possibleActions = state.getLegalActions(playerAtNode)
            return possibleActions.all { it in children }
        }
        
        /**
         * Check if this node is a terminal state
         */
        fun isTerminal(): Boolean {
            return state.isTerminal()
        }
        
        /**
         * Select the best child node according to UCB formula with progressive bias
         */
        fun selectBestChild(): MCTSNode? {
            if (children.isEmpty()) {
                return null
            }
            
            return children.entries.maxByOrNull { (action, _) -> 
                if (progressiveBias > 0.0) {
                    progressiveUcbScore(action, visits)
                } else {
                    children[action]?.ucbScore(visits) ?: Double.NEGATIVE_INFINITY
                }
            }?.value
        }
        
        /**
         * Expand this node by adding a new child node
         */
        fun expand(): MCTSNode? {
            val possibleActions = state.getLegalActions(playerAtNode)
            val unexploredActions = possibleActions.filter { it !in children }
            
            if (unexploredActions.isEmpty()) {
                return null
            }
            
            // Select action with highest heuristic value occasionally to guide search
            val action = if (Random.nextDouble() < 0.3) {
                unexploredActions.maxByOrNull { 
                    heuristicValues.getOrPut(it) { 
                        calculateActionHeuristic(state, it, playerAtNode) 
                    } 
                } ?: unexploredActions.random()
            } else {
                unexploredActions.random()
            }
            
            val nextPlayer = if (playerAtNode == Player.Player1) Player.Player2 else Player.Player1
            val nextState = state.next(mapOf(playerAtNode to action))
            
            val childNode = MCTSNode(nextState, nextPlayer, action, this)
            children[action] = childNode
            
            return childNode
        }
    }
    
    override fun getAction(gameState: GameState): Action {
        // Update estimated game phase based on number of planets controlled by each player
        updateGamePhase(gameState)
        
        val startTime = System.currentTimeMillis()
        val adaptableState: AbstractGameState = ForwardModelAdapter(gameState.deepCopy(), this.params)
        
        val rootNode = MCTSNode(adaptableState, player)
        
        var iterations = 0
        while (iterations < simulationCount && 
              (System.currentTimeMillis() - startTime) < timeLimit) {
            // 1. Selection
            val selectedNode = selection(rootNode)
            
            // 2. Expansion
            val expandedNode = if (!selectedNode.isTerminal()) {
                selectedNode.expand() ?: selectedNode
            } else {
                selectedNode
            }
            
            // 3. Simulation
            val simulationValue = simulation(expandedNode)
            
            // 4. Backpropagation
            backpropagation(expandedNode, simulationValue)
            
            iterations++
        }
        
        // Select best action with highest visit count
        val bestAction = findBestAction(rootNode)
        
        return bestAction
    }
    
    /**
     * Find the best action based on visit count or expected value
     */
    private fun findBestAction(rootNode: MCTSNode): Action {
        // Use visits (most robust) or expected value based on game phase
        return if (rootNode.children.isEmpty()) {
            Action.doNothing()
        } else if (estimatedGamePhase < 0.8) {
            // In early to mid game, prefer most explored actions
            rootNode.children.entries.maxByOrNull { 
                it.value.visits 
            }?.key ?: Action.doNothing()
        } else {
            // In endgame, prefer actions with highest expected value
            rootNode.children.entries.maxByOrNull { 
                if (it.value.visits > 0) it.value.totalValue / it.value.visits 
                else Double.NEGATIVE_INFINITY
            }?.key ?: Action.doNothing()
        }
    }
    
    /**
     * Update the estimated game phase based on game state
     */
    private fun updateGamePhase(gameState: GameState) {
        val totalPlanets = gameState.planets.size
        val myPlanets = gameState.planets.count { it.owner == player }
        val oppPlanets = gameState.planets.count { it.owner == player.opponent() }
        val neutralPlanets = totalPlanets - myPlanets - oppPlanets
        
        // Early game: lots of neutral planets
        // Mid game: few neutral planets
        // Late game: no neutral planets
        estimatedGamePhase = 1.0 - (neutralPlanets.toDouble() / totalPlanets.coerceAtLeast(1))
        
        // Adjust for extreme dominance/weakness
        val dominanceRatio = myPlanets.toDouble() / (oppPlanets.toDouble() + 1.0)
        if (dominanceRatio > 2.0 || dominanceRatio < 0.5) {
            estimatedGamePhase = (estimatedGamePhase + 1.0) / 2.0 // Bias toward end-game
        }
    }
    
    /**
     * Calculate a heuristic value for an action in the given state
     */
    private fun calculateActionHeuristic(state: AbstractGameState, action: Action, currentPlayer: Player): Double {
        if (action == Action.DO_NOTHING) {
            return 0.0 // Neutral value for doing nothing
        }
        
        val adapter = state as? ForwardModelAdapter
            ?: return 0.0 // Fallback if not the right type
            
        val gameState = adapter.getGameState()
        
        // Get source and destination planets
        val sourcePlanet = gameState.planets.find { it.id == action.sourcePlanetId }
            ?: return 0.0
            
        val destPlanet = gameState.planets.find { it.id == action.destinationPlanetId }
            ?: return 0.0
        
        // Heuristic factors
        val distance = sourcePlanet.position.distance(destPlanet.position)
        val shipRatio = action.numShips / (sourcePlanet.nShips + 1.0) // How much of our force are we committing
        
        // Different strategies based on target planet ownership
        return when (destPlanet.owner) {
            Player.Neutral -> {
                // Calculate if we can capture the neutral planet
                val shipAdvantage = action.numShips - destPlanet.nShips
                
                if (shipAdvantage > 0) {
                    // Value based on growth rate and inverse of distance
                    (10.0 * destPlanet.growthRate) / (1.0 + 0.1 * distance)
                } else {
                    // We cannot capture it, low value
                    0.1
                }
            }
            currentPlayer.opponent() -> {
                // Attacking enemy
                val shipAdvantage = action.numShips - destPlanet.nShips
                
                if (shipAdvantage > 0) {
                    // Value capturing enemy planets higher than neutrals
                    (15.0 * destPlanet.growthRate) / (1.0 + 0.1 * distance)
                } else {
                    // Harassing attacks can still be valuable
                    (3.0 * destPlanet.growthRate) / (1.0 + 0.1 * distance)
                }
            }
            currentPlayer -> {
                // Reinforcing own planet - usually low value
                if (destPlanet.nShips < 5.0) {
                    // Reinforcing a vulnerable planet
                    5.0 / (1.0 + 0.2 * distance)
                } else {
                    // Low value for reinforcing strong planets
                    1.0 / (1.0 + 0.5 * distance)
                }
            }
            else -> 0.0
        }
    }
    
    /**
     * Selection phase - recursively select nodes with highest UCB score until reaching 
     * a node that is not fully expanded or is terminal
     */
    private fun selection(node: MCTSNode): MCTSNode {
        var current = node
        
        while (!current.isTerminal() && current.isFullyExpanded()) {
            val nextNode = current.selectBestChild() ?: return current
            current = nextNode
        }
        
        return current
    }
    
    /**
     * Simulation phase - perform random playouts from the current state
     * to estimate the value of this position
     */
    private fun simulation(node: MCTSNode): Double {
        var totalScore = 0.0
        
        // Determine playout depth based on game phase if adaptive
        val actualPlayoutDepth = if (adaptivePlayouts) {
            // Use deeper playouts in late game, shallower in early game
            (5 + (maxRolloutDepth - 5) * estimatedGamePhase).toInt().coerceIn(5, maxRolloutDepth)
        } else {
            maxRolloutDepth
        }
        
        for (i in 0 until rolloutCount) {
            var currentState = node.state.copy()
            var depth = 0
            var currentPlayer = node.playerAtNode
            
            while (!currentState.isTerminal() && depth < actualPlayoutDepth) {
                val actions = currentState.getLegalActions(currentPlayer)
                
                // Occasionally use a heuristic-guided action instead of pure random
                val randomAction = if (Random.nextDouble() < 0.2 && actions.size > 1) {
                    // Get adapter from current state for heuristic calculation
                    val adapter = currentState as? ForwardModelAdapter
                    
                    if (adapter != null) {
                        actions.maxByOrNull { 
                            calculateActionHeuristic(currentState, it, currentPlayer) 
                        }
                    } else {
                        actions.randomOrNull()
                    }
                } else {
                    actions.randomOrNull()
                } ?: Action.doNothing()
                
                currentState = currentState.next(mapOf(currentPlayer to randomAction))
                
                currentPlayer = if (currentPlayer == Player.Player1) Player.Player2 else Player.Player1
                depth++
                
                // Early termination if one side has a clear advantage
                if (depth >= 5 && Random.nextDouble() < 0.1) {
                    val score = evaluateState(currentState)
                    if (Math.abs(score) > 0.8) {
                        break // Clear winner, no need to play out further
                    }
                }
            }
            
            // Evaluate the final state from our perspective
            val score = evaluateState(currentState)
            totalScore += score
        }
        
        return totalScore / rolloutCount
    }
    
    /**
     * Backpropagation phase - update value and visit counts up the tree
     */
    private fun backpropagation(node: MCTSNode, value: Double) {
        var current: MCTSNode? = node
        
        while (current != null) {
            current.visits++
            current.totalValue += value
            current = current.parent
        }
    }
    
    /**
     * Evaluate the game state from the perspective of our player
     */
    private fun evaluateState(state: AbstractGameState): Double {
        if (state.isTerminal()) {
            val scores = state.getScore()
            val myScore = scores[player] ?: 0.0
            val opponentScore = scores[player.opponent()] ?: 0.0
            
            return if (myScore > opponentScore) 1.0 
                  else if (myScore < opponentScore) -1.0 
                  else 0.0
        }
        
        val adapter = state as? ForwardModelAdapter
            ?: throw IllegalStateException("AbstractGameState is not a ForwardModelAdapter")
            
        val gameState = adapter.getGameState()
        return calculateComprehensiveScore(gameState)
    }
    
    /**
     * Calculate a comprehensive evaluation score for the current player
     */
    private fun calculateComprehensiveScore(gameState: GameState): Double {
        var myShipsOnPlanets = 0.0
        var myShipsInTransit = 0.0
        var myPlanetCount = 0
        var myTotalGrowthRate = 0.0
        var myPotentialTargets = 0.0  // Neutral/enemy planets close to my planets
        
        var oppShipsOnPlanets = 0.0
        var oppShipsInTransit = 0.0
        var oppPlanetCount = 0
        var oppTotalGrowthRate = 0.0
        
        val opponentPlayer = player.opponent()
        
        // Calculate distances between planets for strategic evaluation
        val distances = mutableMapOf<Pair<Int, Int>, Double>()
        for (p1 in gameState.planets) {
            for (p2 in gameState.planets) {
                if (p1.id != p2.id) {
                    distances[Pair(p1.id, p2.id)] = p1.position.distance(p2.position)
                }
            }
        }
        
        // Evaluate ship counts, planet counts, growth rates
        gameState.planets.forEach { planet ->
            when (planet.owner) {
                player -> {
                    myShipsOnPlanets += planet.nShips
                    myPlanetCount++
                    myTotalGrowthRate += planet.growthRate
                    
                    // Count nearby enemy/neutral planets as potential targets
                    gameState.planets.forEach { otherPlanet ->
                        if (otherPlanet.owner != player && otherPlanet.id != planet.id) {
                            val dist = distances[Pair(planet.id, otherPlanet.id)] ?: Double.MAX_VALUE
                            // Consider closer planets as higher value targets
                            myPotentialTargets += (1.0 / (1.0 + dist)) * 
                                (if (otherPlanet.owner == Player.Neutral) 1.0 else 2.0)
                        }
                    }
                }
                opponentPlayer -> {
                    oppShipsOnPlanets += planet.nShips
                    oppPlanetCount++
                    oppTotalGrowthRate += planet.growthRate
                }
                else -> { } // Neutral planet
            }
            
            // Count ships in transit
            planet.transporter?.let { transporter ->
                if (transporter.owner == player) {
                    myShipsInTransit += transporter.nShips
                } else if (transporter.owner == opponentPlayer) {
                    oppShipsInTransit += transporter.nShips
                }
            }
        }
        
        // Weighting factors
        val shipWeight = 1.0
        val planetCountWeight = 10.0
        val growthRateWeight = 20.0
        val shipsInTransitWeight = 0.8
        val potentialTargetsWeight = 5.0
        
        val myScore = (myShipsOnPlanets * shipWeight) +
                     (myShipsInTransit * shipsInTransitWeight) +
                     (myPlanetCount * planetCountWeight) +
                     (myTotalGrowthRate * growthRateWeight) +
                     (myPotentialTargets * potentialTargetsWeight)
                     
        val oppScore = (oppShipsOnPlanets * shipWeight) +
                      (oppShipsInTransit * shipsInTransitWeight) +
                      (oppPlanetCount * planetCountWeight) +
                      (oppTotalGrowthRate * growthRateWeight)
        
        return (myScore - oppScore) / (myScore + oppScore + 1.0) // Normalize to [-1, 1]
    }
    
    override fun getAgentType(): String {
        return "MCTSAgent"
    }

    /**
     * Configuration for different MCTS agent profiles
     */
    companion object {
        // Creates a fast but effective configuration for quick decision-making
        fun createFast(): MCTSAgent {
            return MCTSAgent(
                simulationCount = 30,
                maxRolloutDepth = 15,
                rolloutCount = 2,
                timeLimit = 100
            )
        }
        
        // Creates a more deliberate configuration for stronger play
        fun createStrong(): MCTSAgent {
            return MCTSAgent(
                simulationCount = 100,
                maxRolloutDepth = 30,
                rolloutCount = 5,
                timeLimit = 500
            )
        }
    }
}

// Extension function for random selection from a list or return null if empty
fun <T> List<T>.randomOrNull(): T? {
    if (isEmpty()) return null
    return random()
}

/**
 * Test the MCTS agent in a sample game
 */
fun main() {
    val gameParams = GameParams(numPlanets = 10)
    val gameState = GameStateFactory(gameParams).createGame()
    
    val mctsAgent = MCTSAgent.createFast()
    mctsAgent.prepareToPlayAs(Player.Player1, gameParams)
    
    println("Testing ${mctsAgent.getAgentType()}")
    val action = mctsAgent.getAction(gameState)
    println("Selected action: $action")
    
    // Optionally run a full game against another agent to test performance
    println("\nRunning test game...")
    val randomAgent = games.planetwars.agents.random.CarefulRandomAgent()
    val gameRunner = games.planetwars.runners.GameRunner(
        mctsAgent,
        randomAgent,
        gameParams
    )
    
    val result = gameRunner.runGame()
    println("Game completed in ${result.state.gameTick} ticks")
    println("Final score: Player1=${result.getShips(Player.Player1)}, Player2=${result.getShips(Player.Player2)}")
    println("Winner: ${result.getLeader()}")
} 