"""
Q-learning agent for economic competition games.

This agent implements a simple Q-learning algorithm to learn
optimal strategies in economic games.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Agent that learns using Q-learning algorithm.
    
    This agent discretizes the action space and learns the
    value of state-action pairs using Q-learning.
    """
    
    def __init__(
        self, 
        agent_id: int, 
        action_space_size: int = 20,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.1,
        name: Optional[str] = None
    ):
        """
        Initialize the Q-learning agent.
        
        Args:
            agent_id: Unique identifier for the agent
            action_space_size: Number of discrete actions
            learning_rate: Learning rate for Q-learning
            discount_factor: Discount factor for future rewards
            exploration_rate: Probability of random action
            name: Optional name for the agent
        """
        super().__init__(agent_id, name)
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Q-table (state -> action -> value)
        self.q_table: Dict[str, np.ndarray] = {}
        self.last_state = None
        self.last_action = None
        
    def _get_state_key(self, actions: List[float]) -> str:
        """
        Convert state to string key for Q-table.
        
        Args:
            actions: Current actions of all agents
            
        Returns:
            State key string
        """
        # Discretize other agents' actions
        other_actions = [actions[i] for i in range(len(actions)) if i != self.agent_id]
        if not other_actions:
            return "empty"
        
        # Create state based on other agents' actions
        state_parts = []
        for action in other_actions:
            # Discretize action into bins
            bin_size = 10.0 / self.action_space_size
            bin_index = min(int(action / bin_size), self.action_space_size - 1)
            state_parts.append(str(bin_index))
        
        return "_".join(state_parts)
    
    def _discretize_action(self, action: float, max_action: float = 10.0) -> int:
        """
        Discretize continuous action to discrete action index.
        
        Args:
            action: Continuous action value
            max_action: Maximum action value
            
        Returns:
            Discrete action index
        """
        bin_size = max_action / self.action_space_size
        action_index = min(int(action / bin_size), self.action_space_size - 1)
        return max(0, action_index)
    
    def _continuous_action(self, action_index: int, max_action: float = 10.0) -> float:
        """
        Convert discrete action index to continuous action.
        
        Args:
            action_index: Discrete action index
            max_action: Maximum action value
            
        Returns:
            Continuous action value
        """
        bin_size = max_action / self.action_space_size
        return action_index * bin_size + bin_size / 2
    
    def choose_action(self, state: Optional[Dict[str, Any]] = None) -> float:
        """
        Choose action using Q-learning with epsilon-greedy exploration.
        
        Args:
            state: Current game state (optional)
            
        Returns:
            Action to take
        """
        # Get current state key
        if state and 'actions' in state:
            state_key = self._get_state_key(state['actions'])
        else:
            state_key = "empty"
        
        # Initialize Q-table for this state if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.exploration_rate:
            # Explore: choose random action
            action_index = np.random.randint(0, self.action_space_size)
        else:
            # Exploit: choose best action
            action_index = np.argmax(self.q_table[state_key])
        
        # Convert to continuous action
        action = self._continuous_action(action_index)
        
        # Store for update
        self.last_state = state_key
        self.last_action = action_index
        
        return action
    
    def update(self, actions: List[float], payoffs: List[float], 
               game_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Update Q-table based on game outcome.
        
        Args:
            actions: Actions taken by all agents
            payoffs: Payoffs received by all agents
            game_info: Additional game information
        """
        if self.last_state is None or self.last_action is None:
            return
        
        # Get current state
        current_state = self._get_state_key(actions)
        
        # Initialize Q-table for current state if needed
        if current_state not in self.q_table:
            self.q_table[current_state] = np.zeros(self.action_space_size)
        
        # Get reward (payoff for this agent)
        reward = payoffs[self.agent_id]
        
        # Q-learning update
        old_value = self.q_table[self.last_state][self.last_action]
        next_max = np.max(self.q_table[current_state])
        
        new_value = old_value + self.learning_rate * (
            reward + self.discount_factor * next_max - old_value
        )
        
        self.q_table[self.last_state][self.last_action] = new_value
        
        # Update history
        self.action_history.append(actions[self.agent_id])
        self.payoff_history.append(reward)
        self.total_payoff += reward
    
    def reset(self) -> None:
        """Reset the agent's state."""
        super().reset()
        self.q_table = {}
        self.last_state = None
        self.last_action = None
    
    def get_q_table_stats(self) -> Dict[str, Any]:
        """Get statistics about the Q-table."""
        if not self.q_table:
            return {}
        
        total_states = len(self.q_table)
        total_entries = sum(len(q_values) for q_values in self.q_table.values())
        avg_q_value = np.mean([np.mean(q_values) for q_values in self.q_table.values()])
        max_q_value = max(np.max(q_values) for q_values in self.q_table.values())
        
        return {
            'total_states': total_states,
            'total_entries': total_entries,
            'average_q_value': avg_q_value,
            'max_q_value': max_q_value
        }
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return (f"QLearningAgent(id={self.agent_id}, name='{self.name}', "
                f"action_space_size={self.action_space_size})")
