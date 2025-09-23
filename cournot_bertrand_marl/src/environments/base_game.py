"""
Base game class for economic competition models.

This module provides the foundational framework for implementing
Cournot and Bertrand competition games with MARL agents.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class BaseGame(ABC):
    """
    Abstract base class for economic competition games.
    
    This class provides the common interface and functionality
    for both Cournot and Bertrand competition models.
    """
    
    def __init__(
        self,
        n_firms: int,
        demand_params: Tuple[float, float],
        cost_params: float,
        max_quantity: float = 100.0,
        max_price: float = 100.0
    ):
        """
        Initialize the base game.
        
        Args:
            n_firms: Number of competing firms
            demand_params: (intercept, slope) for linear demand P = a - b*Q
            cost_params: Marginal cost parameter
            max_quantity: Maximum quantity any firm can produce
            max_price: Maximum price any firm can charge
        """
        self.n_firms = n_firms
        self.demand_params = demand_params
        self.cost_params = cost_params
        self.max_quantity = max_quantity
        self.max_price = max_price
        
        # Game state
        self.agents: List[Any] = []
        self.game_history: List[Dict[str, Any]] = []
        self.current_round = 0
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate game parameters."""
        if self.n_firms < 2:
            raise ValueError("Number of firms must be at least 2")
        if self.demand_params[1] <= 0:
            raise ValueError("Demand slope must be positive")
        if self.cost_params < 0:
            raise ValueError("Cost parameters must be non-negative")
    
    def add_agent(self, agent: Any) -> None:
        """Add an agent to the game."""
        if len(self.agents) >= self.n_firms:
            raise ValueError(f"Maximum number of agents ({self.n_firms}) reached")
        self.agents.append(agent)
    
    def get_demand_function(self, total_quantity: float) -> float:
        """
        Calculate market price based on total quantity.
        
        Args:
            total_quantity: Sum of all firms' quantities
            
        Returns:
            Market price
        """
        a, b = self.demand_params
        price = max(0, a - b * total_quantity)
        return price
    
    def get_cost_function(self, quantity: float) -> float:
        """
        Calculate total cost for a given quantity.
        
        Args:
            quantity: Production quantity
            
        Returns:
            Total cost
        """
        return self.cost_params * quantity
    
    @abstractmethod
    def calculate_payoffs(self, actions: List[float]) -> List[float]:
        """
        Calculate payoffs for all agents given their actions.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            List of payoffs for each agent
        """
        pass
    
    @abstractmethod
    def get_nash_equilibrium(self) -> Tuple[List[float], float, List[float]]:
        """
        Calculate the Nash equilibrium of the game.
        
        Returns:
            Tuple of (equilibrium_actions, equilibrium_price, equilibrium_payoffs)
        """
        pass
    
    def step(self, actions: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Execute one step of the game.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            Tuple of (payoffs, game_info)
        """
        if len(actions) != len(self.agents):
            raise ValueError("Number of actions must match number of agents")
        
        # Calculate payoffs
        payoffs = self.calculate_payoffs(actions)
        
        # Calculate market information
        market_price = self.get_demand_function(sum(actions))
        total_quantity = sum(actions)
        
        # Store game information
        game_info = {
            'actions': actions.copy(),
            'payoffs': payoffs.copy(),
            'market_price': market_price,
            'total_quantity': total_quantity,
            'round': self.current_round
        }
        
        # Update game state
        self.game_history.append(game_info)
        self.current_round += 1
        
        return payoffs, game_info
    
    def reset(self) -> None:
        """Reset the game state."""
        self.game_history = []
        self.current_round = 0
    
    def get_game_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the game."""
        if not self.game_history:
            return {}
        
        # Extract data
        actions = [info['actions'] for info in self.game_history]
        payoffs = [info['payoffs'] for info in self.game_history]
        prices = [info['market_price'] for info in self.game_history]
        quantities = [info['total_quantity'] for info in self.game_history]
        
        # Calculate statistics
        stats = {
            'total_rounds': len(self.game_history),
            'average_price': np.mean(prices),
            'price_std': np.std(prices),
            'average_quantity': np.mean(quantities),
            'quantity_std': np.std(quantities),
            'average_payoffs': [np.mean([p[i] for p in payoffs]) for i in range(self.n_firms)],
            'payoff_std': [np.std([p[i] for p in payoffs]) for i in range(self.n_firms)],
            'total_payoffs': [np.sum([p[i] for p in payoffs]) for i in range(self.n_firms)]
        }
        
        return stats
    
    def plot_strategy_evolution(self, save_path: Optional[str] = None) -> None:
        """Plot how strategies evolve over time."""
        if not self.game_history:
            print("No game history to plot")
            return
        
        # Extract data
        rounds = [info['round'] for info in self.game_history]
        actions = np.array([info['actions'] for info in self.game_history])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for i in range(self.n_firms):
            plt.plot(rounds, actions[:, i], label=f'Firm {i+1}', linewidth=2)
        
        # Add Nash equilibrium line if available
        try:
            nash_actions, _, _ = self.get_nash_equilibrium()
            for i, nash_action in enumerate(nash_actions):
                plt.axhline(y=nash_action, color=f'C{i}', linestyle='--', 
                           alpha=0.7, label=f'Firm {i+1} Nash')
        except:
            pass
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Action', fontsize=12)
        plt.title('Strategy Evolution Over Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_payoff_evolution(self, save_path: Optional[str] = None) -> None:
        """Plot how payoffs evolve over time."""
        if not self.game_history:
            print("No game history to plot")
            return
        
        # Extract data
        rounds = [info['round'] for info in self.game_history]
        payoffs = np.array([info['payoffs'] for info in self.game_history])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for i in range(self.n_firms):
            plt.plot(rounds, payoffs[:, i], label=f'Firm {i+1}', linewidth=2)
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Payoff', fontsize=12)
        plt.title('Payoff Evolution Over Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_market_dynamics(self, save_path: Optional[str] = None) -> None:
        """Plot market price and total quantity over time."""
        if not self.game_history:
            print("No game history to plot")
            return
        
        # Extract data
        rounds = [info['round'] for info in self.game_history]
        prices = [info['market_price'] for info in self.game_history]
        quantities = [info['total_quantity'] for info in self.game_history]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Price plot
        ax1.plot(rounds, prices, 'b-', linewidth=2, label='Market Price')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.set_title('Market Price Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Quantity plot
        ax2.plot(rounds, quantities, 'r-', linewidth=2, label='Total Quantity')
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Quantity', fontsize=12)
        ax2.set_title('Total Quantity Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def __repr__(self) -> str:
        """String representation of the game."""
        return (f"{self.__class__.__name__}(n_firms={self.n_firms}, "
                f"demand_params={self.demand_params}, "
                f"cost_params={self.cost_params})")
