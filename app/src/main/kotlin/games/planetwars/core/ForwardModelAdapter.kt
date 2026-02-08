package games.planetwars.core

import games.planetwars.agents.AbstractGameState
import games.planetwars.agents.Action

class ForwardModelAdapter(
    private var internalState: GameState,
    private val gameParams: GameParams
) : AbstractGameState {

    // Public getter for the internal game state
    fun getGameState(): GameState = internalState

    override fun copy(): AbstractGameState {
        return ForwardModelAdapter(internalState.deepCopy(), gameParams)
    }

    override fun isTerminal(): Boolean {
        // Create a temporary forward model to check terminal state
        val fm = ForwardModel(internalState, gameParams)
        return fm.isTerminal()
    }

    override fun getScore(): Map<Player, Double> {
        val fm = ForwardModel(internalState, gameParams)
        return mapOf(
            Player.Player1 to fm.getShips(Player.Player1),
            Player.Player2 to fm.getShips(Player.Player2),
            Player.Neutral to 0.0 // Or however neutral score is defined, assuming 0 for now
        )
    }

    override fun getLegalActions(player: Player): List<Action> {
        val legalActions = mutableListOf<Action>()
        legalActions.add(Action.doNothing()) // Always possible to do nothing

        internalState.planets.forEachIndexed { sourceIdx, sourcePlanet ->
            if (sourcePlanet.owner == player && sourcePlanet.transporter == null && sourcePlanet.nShips >= 1.0) { // Min 1 ship to send
                internalState.planets.forEachIndexed { destIdx, _ ->
                    if (sourceIdx != destIdx) {
                        // For simplicity, let's generate an action for sending all available ships
                        // More sophisticated versions could send fractions (25%, 50%, etc.)
                        val shipsToSend = sourcePlanet.nShips
                        if (shipsToSend > 0) {
                             legalActions.add(Action(player, sourcePlanet.id, destIdx, shipsToSend))
                        }

                        // Add action for sending half ships if meaningful
                        val halfShips = sourcePlanet.nShips / 2.0
                        if (halfShips >= 1.0) { // Threshold to make it a meaningful move
                             legalActions.add(Action(player, sourcePlanet.id, destIdx, halfShips))
                        }
                    }
                }
            }
        }
        return legalActions.distinct() // Ensure no duplicate actions if logic generates them
    }

    override fun next(actions: Map<Player, Action>): AbstractGameState {
        val newState = internalState.deepCopy()
        val fm = ForwardModel(newState, gameParams)
        fm.step(actions)
        return ForwardModelAdapter(newState, gameParams)
    }
} 