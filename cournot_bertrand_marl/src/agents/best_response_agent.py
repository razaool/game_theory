"""
Best response agent for economic competition games.

This agent implements the best response strategy, choosing the optimal
action given the current state of the game.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent


class BestResponseAgent(BaseAgent):
    """
    Agent that plays best response strategy.
    
    This agent calculates the optimal action given the current state
    of the game, implementing the best response function.
    """
    
    def __init__(self, agent_id: int, game, name: Optional[str] = None):
        """
        Initialize the best response agent.
        
        Args:
            agent_id: Unique identifier for the agent
            game: Game instance (needed for best response calculation)
            name: Optional name for the agent
        """
        super().__init__(agent_id, name)
        self.game = game
        self.last_actions: List[float] = []
        
    def choose_action(self, state: Optional[Dict[str, Any]] = None) -> float:
        """
        Choose action using best response strategy.
        
        Args:
            state: Current game state (optional)
            
        Returns:
            Best response action
        """
        if not self.last_actions:
            # If no previous actions, choose a random action
            return np.random.uniform(0, self.game.max_quantity)
        
        # Calculate best response to other agents' last actions
        other_actions = [action for i, action in enumerate(self.last_actions) if i != self.agent_id]
        
        if len(other_actions) == 0:
            # If no other actions, choose a random action
            return np.random.uniform(0, self.game.max_quantity)
        
        # Get best response from the game
        best_response = self.game.get_best_response(self.agent_id, other_actions)
        
        # Add some exploration noise
        noise = np.random.normal(0, 0.1 * best_response)
        action = max(0, best_response + noise)
        
        return action
    
    def update(self, actions: List[float], payoffs: List[float], 
               game_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the agent based on game outcome.
        
        Args:
            actions: Actions taken by all agents
            payoffs: Payoffs received by all agents
            game_info: Additional game information
        """
        # Store the actions for next round
        self.last_actions = actions.copy()
        
        # Update history
        self.action_history.append(actions[self.agent_id])
        self.payoff_history.append(payoffs[self.agent_id])
        self.total_payoff += payoffs[self.agent_id]
    
    def reset(self) -> None:
        """Reset the agent's state."""
        super().reset()
        self.last_actions = []
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"BestResponseAgent(id={self.agent_id}, name='{self.name}')"
