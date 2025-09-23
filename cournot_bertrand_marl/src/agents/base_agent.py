"""
Base agent class for economic competition games.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for learning agents in economic games.
    
    This class provides the common interface for all agents
    that can participate in Cournot and Bertrand competition.
    """
    
    def __init__(self, agent_id: int, name: Optional[str] = None):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Optional name for the agent
        """
        self.agent_id = agent_id
        self.name = name or f"Agent_{agent_id}"
        self.action_history: List[float] = []
        self.payoff_history: List[float] = []
        self.total_payoff = 0.0
        
    @abstractmethod
    def choose_action(self, state: Optional[Dict[str, Any]] = None) -> float:
        """
        Choose an action based on current state.
        
        Args:
            state: Current game state (optional)
            
        Returns:
            Action to take
        """
        pass
    
    @abstractmethod
    def update(self, actions: List[float], payoffs: List[float], 
               game_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the agent based on game outcome.
        
        Args:
            actions: Actions taken by all agents
            payoffs: Payoffs received by all agents
            game_info: Additional game information
        """
        pass
    
    def reset(self) -> None:
        """Reset the agent's state."""
        self.action_history = []
        self.payoff_history = []
        self.total_payoff = 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        if not self.action_history:
            return {}
        
        return {
            'name': self.name,
            'agent_id': self.agent_id,
            'total_rounds': len(self.action_history),
            'average_action': np.mean(self.action_history),
            'action_std': np.std(self.action_history),
            'average_payoff': np.mean(self.payoff_history),
            'payoff_std': np.std(self.payoff_history),
            'total_payoff': self.total_payoff
        }
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(id={self.agent_id}, name='{self.name}')"
