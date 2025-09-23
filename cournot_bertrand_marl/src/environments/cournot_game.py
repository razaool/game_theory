"""
Cournot competition game implementation.

This module implements the classic Cournot model where firms compete
by choosing production quantities simultaneously.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .base_game import BaseGame


class CournotGame(BaseGame):
    """
    Cournot competition game where firms choose production quantities.
    
    In Cournot competition, firms simultaneously choose how much to produce.
    The market price is determined by the total quantity supplied according
    to the demand function P = a - b*Q_total.
    
    Each firm's profit is: π_i = P * q_i - c * q_i
    where P is the market price, q_i is firm i's quantity, and c is marginal cost.
    """
    
    def __init__(
        self,
        n_firms: int,
        demand_params: Tuple[float, float],
        cost_params: float,
        max_quantity: float = 100.0
    ):
        """
        Initialize Cournot competition game.
        
        Args:
            n_firms: Number of competing firms
            demand_params: (intercept, slope) for linear demand P = a - b*Q
            cost_params: Marginal cost parameter
            max_quantity: Maximum quantity any firm can produce
        """
        super().__init__(n_firms, demand_params, cost_params, max_quantity)
        
        # Cournot-specific parameters
        self.action_space = (0, max_quantity)  # Continuous action space
        
        # Validate Cournot-specific parameters
        self._validate_cournot_parameters()
    
    def _validate_cournot_parameters(self) -> None:
        """Validate Cournot-specific parameters."""
        a, b = self.demand_params
        if a <= self.cost_params:
            raise ValueError("Demand intercept must be greater than marginal cost")
    
    def calculate_payoffs(self, quantities: List[float]) -> List[float]:
        """
        Calculate payoffs for all firms given their production quantities.
        
        Args:
            quantities: List of production quantities for each firm
            
        Returns:
            List of profits for each firm
        """
        if len(quantities) != self.n_firms:
            raise ValueError("Number of quantities must match number of firms")
        
        # Calculate market price
        total_quantity = sum(quantities)
        market_price = self.get_demand_function(total_quantity)
        
        # Calculate individual profits
        payoffs = []
        for quantity in quantities:
            if quantity < 0:
                raise ValueError("Quantities must be non-negative")
            
            revenue = market_price * quantity
            cost = self.get_cost_function(quantity)
            profit = revenue - cost
            payoffs.append(profit)
        
        return payoffs
    
    def get_nash_equilibrium(self) -> Tuple[List[float], float, List[float]]:
        """
        Calculate the Nash equilibrium of the Cournot game.
        
        For symmetric firms with linear demand P = a - b*Q and constant
        marginal cost c, the Nash equilibrium is:
        - q* = (a - c) / (b * (n + 1))
        - P* = (a + n*c) / (n + 1)
        - π* = (a - c)² / (b * (n + 1)²)
        
        Returns:
            Tuple of (equilibrium_quantities, equilibrium_price, equilibrium_payoffs)
        """
        a, b = self.demand_params
        c = self.cost_params
        n = self.n_firms
        
        # Calculate equilibrium quantities (symmetric)
        equilibrium_quantity = (a - c) / (b * (n + 1))
        equilibrium_quantities = [equilibrium_quantity] * n
        
        # Calculate equilibrium price
        equilibrium_price = (a + n * c) / (n + 1)
        
        # Calculate equilibrium payoffs
        equilibrium_payoff = (a - c)**2 / (b * (n + 1)**2)
        equilibrium_payoffs = [equilibrium_payoff] * n
        
        return equilibrium_quantities, equilibrium_price, equilibrium_payoffs
    
    def get_best_response(self, firm_id: int, other_quantities: List[float]) -> float:
        """
        Calculate the best response quantity for a firm given other firms' quantities.
        
        Args:
            firm_id: ID of the firm (not used for symmetric case)
            other_quantities: Quantities chosen by other firms
            
        Returns:
            Best response quantity
        """
        if len(other_quantities) != self.n_firms - 1:
            raise ValueError("Number of other quantities must be n_firms - 1")
        
        a, b = self.demand_params
        c = self.cost_params
        
        # Sum of other firms' quantities
        other_total = sum(other_quantities)
        
        # Best response: q_i = (a - c - b * Q_{-i}) / (2 * b)
        best_response = (a - c - b * other_total) / (2 * b)
        
        # Ensure non-negative
        best_response = max(0, best_response)
        
        return best_response
    
    def get_reaction_function(self, firm_id: int, other_quantities: List[float]) -> float:
        """
        Alias for get_best_response for compatibility.
        """
        return self.get_best_response(firm_id, other_quantities)
    
    def calculate_consumer_surplus(self, total_quantity: float) -> float:
        """
        Calculate consumer surplus for given total quantity.
        
        Args:
            total_quantity: Total quantity supplied
            
        Returns:
            Consumer surplus
        """
        a, b = self.demand_params
        market_price = self.get_demand_function(total_quantity)
        
        # Consumer surplus = 0.5 * (a - P) * Q
        consumer_surplus = 0.5 * (a - market_price) * total_quantity
        return max(0, consumer_surplus)
    
    def calculate_producer_surplus(self, quantities: List[float]) -> float:
        """
        Calculate producer surplus (total profit) for given quantities.
        
        Args:
            quantities: List of production quantities
            
        Returns:
            Producer surplus
        """
        payoffs = self.calculate_payoffs(quantities)
        return sum(payoffs)
    
    def calculate_total_surplus(self, quantities: List[float]) -> float:
        """
        Calculate total surplus (consumer + producer) for given quantities.
        
        Args:
            quantities: List of production quantities
            
        Returns:
            Total surplus
        """
        total_quantity = sum(quantities)
        consumer_surplus = self.calculate_consumer_surplus(total_quantity)
        producer_surplus = self.calculate_producer_surplus(quantities)
        return consumer_surplus + producer_surplus
    
    def get_efficiency_metrics(self, quantities: List[float]) -> Dict[str, float]:
        """
        Calculate various efficiency metrics for given quantities.
        
        Args:
            quantities: List of production quantities
            
        Returns:
            Dictionary of efficiency metrics
        """
        # Calculate surpluses
        total_quantity = sum(quantities)
        consumer_surplus = self.calculate_consumer_surplus(total_quantity)
        producer_surplus = self.calculate_producer_surplus(quantities)
        total_surplus = consumer_surplus + producer_surplus
        
        # Calculate competitive benchmark (perfect competition)
        a, b = self.demand_params
        c = self.cost_params
        competitive_quantity = (a - c) / b
        competitive_price = c
        competitive_consumer_surplus = 0.5 * (a - c) * competitive_quantity
        competitive_total_surplus = competitive_consumer_surplus
        
        # Calculate efficiency ratios
        efficiency_ratio = total_surplus / competitive_total_surplus if competitive_total_surplus > 0 else 0
        deadweight_loss = competitive_total_surplus - total_surplus
        
        return {
            'consumer_surplus': consumer_surplus,
            'producer_surplus': producer_surplus,
            'total_surplus': total_surplus,
            'efficiency_ratio': efficiency_ratio,
            'deadweight_loss': deadweight_loss,
            'competitive_quantity': competitive_quantity,
            'competitive_price': competitive_price
        }
    
    def plot_reaction_functions(self, save_path: Optional[str] = None) -> None:
        """
        Plot the reaction functions for a 2-firm Cournot game.
        
        Note: This only works for 2-firm games.
        """
        if self.n_firms != 2:
            raise ValueError("Reaction function plot only available for 2-firm games")
        
        import matplotlib.pyplot as plt
        
        # Create quantity range
        q_range = np.linspace(0, self.max_quantity, 100)
        
        # Calculate reaction functions
        reaction_1 = []
        reaction_2 = []
        
        for q in q_range:
            # Firm 1's reaction to Firm 2's quantity
            br_1 = self.get_best_response(0, [q])
            reaction_1.append(br_1)
            
            # Firm 2's reaction to Firm 1's quantity
            br_2 = self.get_best_response(1, [q])
            reaction_2.append(br_2)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(q_range, reaction_1, 'b-', linewidth=2, label='Firm 1 Reaction Function')
        plt.plot(reaction_2, q_range, 'r-', linewidth=2, label='Firm 2 Reaction Function')
        
        # Add Nash equilibrium point
        nash_quantities, _, _ = self.get_nash_equilibrium()
        plt.plot(nash_quantities[1], nash_quantities[0], 'ko', markersize=10, 
                label='Nash Equilibrium')
        
        plt.xlabel('Firm 2 Quantity', fontsize=12)
        plt.ylabel('Firm 1 Quantity', fontsize=12)
        plt.title('Cournot Reaction Functions', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def simulate_best_response_dynamics(
        self, 
        initial_quantities: List[float], 
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> List[Dict[str, Any]]:
        """
        Simulate best response dynamics to find equilibrium.
        
        Args:
            initial_quantities: Starting quantities for each firm
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            List of iteration results
        """
        if len(initial_quantities) != self.n_firms:
            raise ValueError("Number of initial quantities must match number of firms")
        
        quantities = initial_quantities.copy()
        history = []
        
        for iteration in range(max_iterations):
            # Store current state
            payoffs = self.calculate_payoffs(quantities)
            market_price = self.get_demand_function(sum(quantities))
            
            history.append({
                'iteration': iteration,
                'quantities': quantities.copy(),
                'payoffs': payoffs.copy(),
                'market_price': market_price,
                'total_quantity': sum(quantities)
            })
            
            # Update quantities using best response
            new_quantities = []
            for i in range(self.n_firms):
                other_quantities = [q for j, q in enumerate(quantities) if j != i]
                best_response = self.get_best_response(i, other_quantities)
                new_quantities.append(best_response)
            
            # Check convergence
            max_change = max(abs(new_quantities[i] - quantities[i]) for i in range(self.n_firms))
            if max_change < tolerance:
                break
            
            quantities = new_quantities
        
        return history
    
    def __repr__(self) -> str:
        """String representation of the Cournot game."""
        return (f"CournotGame(n_firms={self.n_firms}, "
                f"demand_params={self.demand_params}, "
                f"cost_params={self.cost_params})")
