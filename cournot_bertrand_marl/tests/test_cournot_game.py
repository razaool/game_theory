"""
Test cases for Cournot game implementation.
"""

import sys
import os
sys.path.append(os.path.join('..', 'src'))

import pytest
import numpy as np
from environments.cournot_game import CournotGame


class TestCournotGame:
    """Test cases for CournotGame class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.game = CournotGame(
            n_firms=2,
            demand_params=(100, 1),  # P = 100 - Q
            cost_params=10,          # Marginal cost = 10
            max_quantity=50
        )
    
    def test_game_initialization(self):
        """Test game initialization."""
        assert self.game.n_firms == 2
        assert self.game.demand_params == (100, 1)
        assert self.game.cost_params == 10
        assert self.game.max_quantity == 50
    
    def test_demand_function(self):
        """Test demand function calculation."""
        # Test various quantities
        assert self.game.get_demand_function(0) == 100  # P = 100 - 0 = 100
        assert self.game.get_demand_function(50) == 50  # P = 100 - 50 = 50
        assert self.game.get_demand_function(100) == 0  # P = 100 - 100 = 0
        assert self.game.get_demand_function(150) == 0  # P = 100 - 150 = 0 (non-negative)
    
    def test_cost_function(self):
        """Test cost function calculation."""
        assert self.game.get_cost_function(0) == 0
        assert self.game.get_cost_function(10) == 100  # 10 * 10 = 100
        assert self.game.get_cost_function(20) == 200  # 10 * 20 = 200
    
    def test_payoff_calculation(self):
        """Test payoff calculation."""
        # Test with known quantities
        quantities = [30, 30]  # Both firms produce 30
        payoffs = self.game.calculate_payoffs(quantities)
        
        # Total quantity = 60, Price = 100 - 60 = 40
        # Each firm's profit = 40 * 30 - 10 * 30 = 1200 - 300 = 900
        expected_payoff = 900
        assert abs(payoffs[0] - expected_payoff) < 0.01
        assert abs(payoffs[1] - expected_payoff) < 0.01
    
    def test_nash_equilibrium(self):
        """Test Nash equilibrium calculation."""
        nash_quantities, nash_price, nash_payoffs = self.game.get_nash_equilibrium()
        
        # For 2 firms with P = 100 - Q and c = 10:
        # q* = (100 - 10) / (1 * (2 + 1)) = 90 / 3 = 30
        # P* = (100 + 2 * 10) / (2 + 1) = 120 / 3 = 40
        # π* = (100 - 10)² / (1 * (2 + 1)²) = 8100 / 9 = 900
        
        expected_quantity = 30.0
        expected_price = 40.0
        expected_payoff = 900.0
        
        assert abs(nash_quantities[0] - expected_quantity) < 0.01
        assert abs(nash_quantities[1] - expected_quantity) < 0.01
        assert abs(nash_price - expected_price) < 0.01
        assert abs(nash_payoffs[0] - expected_payoff) < 0.01
        assert abs(nash_payoffs[1] - expected_payoff) < 0.01
    
    def test_best_response(self):
        """Test best response calculation."""
        # Test best response for Firm 1 when Firm 2 produces 30
        best_response = self.game.get_best_response(0, [30])
        
        # BR_1 = (100 - 10 - 1 * 30) / (2 * 1) = 60 / 2 = 30
        expected_best_response = 30.0
        assert abs(best_response - expected_best_response) < 0.01
    
    def test_efficiency_metrics(self):
        """Test efficiency metrics calculation."""
        quantities = [30, 30]  # Nash equilibrium quantities
        metrics = self.game.get_efficiency_metrics(quantities)
        
        # Check that all required metrics are present
        required_metrics = [
            'consumer_surplus', 'producer_surplus', 'total_surplus',
            'efficiency_ratio', 'deadweight_loss', 'competitive_quantity', 'competitive_price'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_game_step(self):
        """Test game step execution."""
        actions = [30, 30]
        payoffs, game_info = self.game.step(actions)
        
        # Check payoffs
        assert len(payoffs) == 2
        assert all(isinstance(p, (int, float)) for p in payoffs)
        
        # Check game info
        assert 'actions' in game_info
        assert 'payoffs' in game_info
        assert 'market_price' in game_info
        assert 'total_quantity' in game_info
        assert 'round' in game_info
        
        # Check values
        assert game_info['actions'] == actions
        assert game_info['payoffs'] == payoffs
        assert game_info['total_quantity'] == 60
        assert game_info['market_price'] == 40
        assert game_info['round'] == 0
    
    def test_best_response_dynamics(self):
        """Test best response dynamics simulation."""
        initial_quantities = [0, 0]
        history = self.game.simulate_best_response_dynamics(initial_quantities, max_iterations=10)
        
        # Check that we get a history
        assert len(history) > 0
        
        # Check that quantities converge to Nash equilibrium
        final_quantities = history[-1]['quantities']
        nash_quantities, _, _ = self.game.get_nash_equilibrium()
        
        for i in range(len(final_quantities)):
            assert abs(final_quantities[i] - nash_quantities[i]) < 0.1


if __name__ == "__main__":
    pytest.main([__file__])
