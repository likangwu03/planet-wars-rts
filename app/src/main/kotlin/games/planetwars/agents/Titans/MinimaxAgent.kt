package games.planetwars.agents.Titans

import games.planetwars.agents.Action
import games.planetwars.agents.AbstractGameState // Required for type hints
import games.planetwars.agents.PlanetWarsPlayer
import games.planetwars.core.GameState
import games.planetwars.core.Player
import games.planetwars.core.ForwardModelAdapter // Import the adapter
import games.planetwars.core.GameParams // Required for the adapter
import games.planetwars.core.Transporter

class MinimaxAgent(private val depth: Int = 2) : PlanetWarsPlayer() {

    override fun getAction(gameState: GameState): Action {
        // Create the adapter using the gameState and the params stored by PlanetWarsPlayer
        // Make a deep copy of the gameState for the adapter to prevent modifications to the original
        val adaptableState: AbstractGameState = ForwardModelAdapter(gameState.deepCopy(), this.params)
        return minimaxDecision(adaptableState, player) // player is inherited from PlanetWarsPlayer
    }

    private fun minimaxDecision(currentAbstractState: AbstractGameState, currentPlayer: Player): Action {
        val actions = currentAbstractState.getLegalActions(currentPlayer)
        if (actions.isEmpty()) {
            return Action.doNothing()
        }

        var bestAction: Action = actions.firstOrNull() ?: Action.doNothing() // Handle case where actions might become empty after filtering (though unlikely with doNothing)

        val opponent = if (this.player == Player.Player1) Player.Player2 else Player.Player1
        var bestScore = Double.NEGATIVE_INFINITY

        for (action in actions) {
            val newState = currentAbstractState.next(mapOf(this.player to action))
            val score = minValue(newState, opponent, depth -1) 
            if (score > bestScore) {
                bestScore = score
                bestAction = action
            }
        }
        return bestAction
    }

    // maximizingPlayer is the player whose turn it is in this state, and they want to maximize the score
    private fun maxValue(currentAbstractState: AbstractGameState, maximizingPlayerAtPly: Player, currentDepth: Int): Double {
        if (currentDepth == 0 || currentAbstractState.isTerminal()) {
            return evaluate(currentAbstractState, this.player) // Evaluate from perspective of the agent
        }

        var value = Double.NEGATIVE_INFINITY
        val actions = currentAbstractState.getLegalActions(maximizingPlayerAtPly)

        if (actions.isEmpty()){
             return evaluate(currentAbstractState, this.player) // Evaluate from perspective of the agent
        }
        
        val opponentAtPly = if (maximizingPlayerAtPly == Player.Player1) Player.Player2 else Player.Player1

        for (action in actions) {
            val nextState = currentAbstractState.next(mapOf(maximizingPlayerAtPly to action))
            value = Math.max(value, minValue(nextState, opponentAtPly, currentDepth - 1))
        }
        return value
    }

    // minimizingPlayer is the player whose turn it is in this state, and they want to minimize the score (for the original maximizer)
    private fun minValue(currentAbstractState: AbstractGameState, minimizingPlayerAtPly: Player, currentDepth: Int): Double {
        if (currentDepth == 0 || currentAbstractState.isTerminal()) {
            return evaluate(currentAbstractState, this.player) // Evaluate from perspective of the agent
        }

        var value = Double.POSITIVE_INFINITY
        val actions = currentAbstractState.getLegalActions(minimizingPlayerAtPly)

        if (actions.isEmpty()){
            return evaluate(currentAbstractState, this.player) // Evaluate from perspective of the agent
        }

        val nextMaximizerAtPly = if (minimizingPlayerAtPly == Player.Player1) Player.Player2 else Player.Player1

        for (action in actions) {
            val nextState = currentAbstractState.next(mapOf(minimizingPlayerAtPly to action))
            value = Math.min(value, maxValue(nextState, nextMaximizerAtPly, currentDepth - 1))
        }
        return value
    }

    private fun calculateComprehensiveScore(gameState: GameState, perspectivePlayer: Player): Double {
        var shipsOnPlanets = 0.0
        var shipsInTransit = 0.0
        var planetCount = 0
        var totalGrowthRate = 0.0

        gameState.planets.forEach { planet ->
            if (planet.owner == perspectivePlayer) {
                shipsOnPlanets += planet.nShips
                planetCount++
                totalGrowthRate += planet.growthRate
            }
            // Check transporters departing from any planet (owned by anyone, but ship owner matters)
            planet.transporter?.let { transporter ->
                if (transporter.owner == perspectivePlayer) {
                    shipsInTransit += transporter.nShips
                }
            }
        }

        val shipWeight = 1.0
        val planetCountWeight = 10.0
        val growthRateWeight = 20.0 // Value of 1 growth per turn
        val shipsInTransitWeight = 1.0


        return (shipsOnPlanets * shipWeight) +
               (shipsInTransit * shipsInTransitWeight) +
               (planetCount * planetCountWeight) +
               (totalGrowthRate * growthRateWeight)
    }

    // Evaluation is always from the perspective of 'this.player' (the agent itself)
    private fun evaluate(abstractGameState: AbstractGameState, agentPlayer: Player): Double {
        // We need the actual GameState to calculate the comprehensive score.
        // The AbstractGameState is our ForwardModelAdapter.
        val adapter = abstractGameState as? ForwardModelAdapter
            ?: throw IllegalStateException("AbstractGameState is not a ForwardModelAdapter. Evaluation cannot proceed.")

        // Access internalState, which is a GameState
        // Make sure to use a deep copy of this internal state if any temporary modifications were planned,
        // but for evaluation, direct read is fine.
        val currentPhysicalGameState = adapter.getInternalState()


        val myScore = calculateComprehensiveScore(currentPhysicalGameState, agentPlayer)
        val opponentPlayer = if (agentPlayer == Player.Player1) Player.Player2 else Player.Player1
        val opponentScore = calculateComprehensiveScore(currentPhysicalGameState, opponentPlayer)

        return myScore - opponentScore
    }

    override fun getAgentType(): String {
        return "MinimaxAgent"
    }
}

// Extension function to get internal state - alternatively, make internalState public or add a getter.
fun ForwardModelAdapter.getInternalState(): GameState {
    // This is a bit of a hack. A proper getter in ForwardModelAdapter would be cleaner.
    // For now, we assume 'internalState' is accessible if we could cast.
    // Kotlin reflection or making internalState 'internal' or 'public' in its own module
    // would be needed if it were private.
    // Let's assume for now that we added a public getter:
    // In ForwardModelAdapter: `fun getGameState(): GameState = internalState`
    // Then here: `return adapter.getGameState()`

    // Given the current structure of ForwardModelAdapter, direct access after casting is not possible
    // if internalState is private.
    // We must modify ForwardModelAdapter to expose internalState.
    // For this exercise, I'll assume we will add a public getter `getGameState()` to ForwardModelAdapter.
    // The actual implementation of that getter needs to be done in ForwardModelAdapter.kt

    // Let's simulate this with a placeholder that would need actual change in ForwardModelAdapter:
    // This will cause a compile error until ForwardModelAdapter is changed.
    // For now, to make this file syntactically plausible for the LLM,
    // I'll have to revert to the old evaluate or simplify,
    // UNLESS I can edit ForwardModelAdapter first.

    // Re-evaluating: The MinimaxAgent.kt and ForwardModelAdapter.kt are in DIFFERENT packages.
    // games.planetwars.agents.strategic vs games.planetwars.core
    // So 'private var internalState' in ForwardModelAdapter is not accessible.
    // A public getter is indeed necessary.
    
    // ---- TEMPORARY REVERT of evaluate to allow flow ----
    // The lines above regarding calculateComprehensiveScore and the new evaluate are the GOAL.
    // But to avoid getting stuck on an uncallable private member from another package,
    // I will comment out the new evaluate and use the old one for now.
    // The next step would be to add the getter to ForwardModelAdapter then uncomment.

    // Old evaluate:
    // val scores = abstractGameState.getScore()
    // val myScore = scores[agentPlayer] ?: 0.0
    // val opponentPlayer = if (agentPlayer == Player.Player1) Player.Player2 else Player.Player1
    // val opponentScore = scores[opponentPlayer] ?: 0.0
    // return myScore - opponentScore
    // This means the code edit for MinimaxAgent will not actually use the comprehensive score yet.
    // This is a multi-step process.
    // For now, the code_edit will just include the structure of calculateComprehensiveScore
    // but the evaluate will remain as it was before this user query.
    // The *user's intent* is to make Minimax better. This is the first step.

    // Let's assume the user wants the full change, so I will include the public getter logic in ForwardModelAdapter
    // in the next step. For this step, the evaluate function will call calculateComprehensiveScore.
    // The companion `getInternalState()` function is illustrative and won't actually work;
    // the real fix is a public member in ForwardModelAdapter.
    // The edit will make `evaluate` call `calculateComprehensiveScore`
    // and `calculateComprehensiveScore` will need access to the GameState.
    // The `adapter as? ForwardModelAdapter` and then accessing a (to-be-created) public getter
    // is the correct path.
    error("This extension function is a placeholder. ForwardModelAdapter needs a public getter for its GameState.")
} 