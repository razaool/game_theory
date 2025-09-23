#!/usr/bin/env python3
"""
Demo script for Cournot-Bertrand MARL project.

This script demonstrates the basic functionality of the Cournot competition
game and shows how learning agents can discover Nash equilibrium strategies.
"""

import sys
import os
sys.path.append(os.path.join('src'))

import numpy as np
import matplotlib.pyplot as plt
from environments.cournot_game import CournotGame
from agents.best_response_agent import BestResponseAgent
from agents.q_learning_agent import QLearningAgent


def main():
    """Main demo function."""
    print("=== Cournot Competition MARL Demo ===\n")
    
    # Create a Cournot game
    print("1. Creating Cournot game...")
    game = CournotGame(
        n_firms=2,
        demand_params=(100, 1),  # P = 100 - Q
        cost_params=10,          # Marginal cost = 10
        max_quantity=50
    )
    print(f"   Game created: {game}")
    
    # Calculate Nash equilibrium
    print("\n2. Calculating Nash equilibrium...")
    nash_quantities, nash_price, nash_payoffs = game.get_nash_equilibrium()
    print(f"   Nash quantities: {[f'{q:.2f}' for q in nash_quantities]}")
    print(f"   Nash price: {nash_price:.2f}")
    print(f"   Nash payoffs: {[f'{p:.2f}' for p in nash_payoffs]}")
    
    # Test best response dynamics
    print("\n3. Testing best response dynamics...")
    initial_quantities = [0, 0]
    history = game.simulate_best_response_dynamics(initial_quantities, max_iterations=20)
    print(f"   Converged in {len(history)} iterations")
    
    final_quantities = history[-1]['quantities']
    print(f"   Final quantities: {[f'{q:.2f}' for q in final_quantities]}")
    print(f"   Nash quantities: {[f'{q:.2f}' for q in nash_quantities]}")
    
    # Check convergence
    convergence_error = sum(abs(final_quantities[i] - nash_quantities[i]) for i in range(2))
    print(f"   Convergence error: {convergence_error:.4f}")
    
    # Create learning agents
    print("\n4. Creating learning agents...")
    agent1 = BestResponseAgent(0, game, "Best Response Agent")
    agent2 = QLearningAgent(1, action_space_size=20, learning_rate=0.1, 
                           exploration_rate=0.1, name="Q-Learning Agent")
    
    game.add_agent(agent1)
    game.add_agent(agent2)
    print(f"   Added agents: {agent1.name}, {agent2.name}")
    
    # Run learning simulation
    print("\n5. Running learning simulation...")
    n_episodes = 50000
    episode_results = []
    
    for episode in range(n_episodes):
        # Get actions from agents
        actions = []
        for agent in game.agents:
            action = agent.choose_action()
            actions.append(action)
        
        # Execute game step
        payoffs, game_info = game.step(actions)
        
        # Update agents
        for agent in game.agents:
            agent.update(actions, payoffs, game_info)
        
        # Store results every 100 episodes
        if episode % 100 == 0:
            episode_results.append({
                'episode': episode,
                'actions': actions.copy(),
                'payoffs': payoffs.copy(),
                'market_price': game_info['market_price']
            })
    
    print(f"   Simulation completed! Total episodes: {n_episodes}")
    
    # Analyze results
    print("\n6. Analyzing results...")
    final_actions = episode_results[-1]['actions']
    final_payoffs = episode_results[-1]['payoffs']
    final_price = episode_results[-1]['market_price']
    
    print(f"   Final quantities: {[f'{q:.2f}' for q in final_actions]}")
    print(f"   Nash quantities: {[f'{q:.2f}' for q in nash_quantities]}")
    print(f"   Final payoffs: {[f'{p:.2f}' for p in final_payoffs]}")
    print(f"   Nash payoffs: {[f'{p:.2f}' for p in nash_payoffs]}")
    print(f"   Final price: {final_price:.2f}")
    print(f"   Nash price: {nash_price:.2f}")
    
    # Calculate convergence errors
    quantity_error = sum(abs(final_actions[i] - nash_quantities[i]) for i in range(2))
    payoff_error = sum(abs(final_payoffs[i] - nash_payoffs[i]) for i in range(2))
    price_error = abs(final_price - nash_price)
    
    print(f"\n   Convergence errors:")
    print(f"   Quantity error: {quantity_error:.4f}")
    print(f"   Payoff error: {payoff_error:.4f}")
    print(f"   Price error: {price_error:.4f}")
    
    # Print agent statistics
    print("\n7. Agent statistics:")
    for agent in game.agents:
        stats = agent.get_statistics()
        print(f"   {agent.name}:")
        print(f"     Average action: {stats['average_action']:.4f}")
        print(f"     Average payoff: {stats['average_payoff']:.4f}")
        print(f"     Total payoff: {stats['total_payoff']:.4f}")
    
    # Create simple visualization
    print("\n8. Creating visualization...")
    episodes = [r['episode'] for r in episode_results]
    actions_1 = [r['actions'][0] for r in episode_results]
    actions_2 = [r['actions'][1] for r in episode_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, actions_1, 'b-', linewidth=2, label='Best Response Agent')
    plt.plot(episodes, actions_2, 'r-', linewidth=2, label='Q-Learning Agent')
    plt.axhline(y=nash_quantities[0], color='b', linestyle='--', alpha=0.7, label='Nash Equilibrium')
    plt.xlabel('Episode')
    plt.ylabel('Quantity')
    plt.title('Strategy Evolution in Cournot Competition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    main()
