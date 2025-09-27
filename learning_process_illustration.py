import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns

# Set up the plotting style with Rose Pine Dawn theme
plt.style.use('rose-pine-dawn')
sns.set_palette("husl")

def create_learning_process_illustration():
    """Create an illustration showing the Q-learning process in Cournot competition."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Q-Learning Process in Cournot Competition', fontsize=20, fontweight='bold', y=0.95)
    
    # Panel 1: State Representation
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.set_title('State Representation', fontsize=16, fontweight='bold', pad=20)
    
    # Agent 1
    agent1_box = FancyBboxPatch((1, 5.5), 3, 2, boxstyle="round,pad=0.1", 
                                facecolor='#EBBCBA', edgecolor='#31748F', linewidth=2)
    ax1.add_patch(agent1_box)
    ax1.text(2.5, 6.5, 'Agent 1', ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(2.5, 6.0, 'Q-Learning', ha='center', va='center', fontsize=12)
    
    # Agent 2
    agent2_box = FancyBboxPatch((6, 5.5), 3, 2, boxstyle="round,pad=0.1", 
                                facecolor='#EBBCBA', edgecolor='#31748F', linewidth=2)
    ax1.add_patch(agent2_box)
    ax1.text(7.5, 6.5, 'Agent 2', ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(7.5, 6.0, 'Q-Learning', ha='center', va='center', fontsize=12)
    
    # State observation arrows
    arrow1 = ConnectionPatch((5, 6.5), (6, 6.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc='#9CCFD8', ec='#9CCFD8')
    ax1.add_patch(arrow1)
    ax1.text(5.5, 7.2, 'Observes', ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#F6C177', alpha=0.7))
    
    arrow2 = ConnectionPatch((6, 6.5), (5, 6.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc='#9CCFD8', ec='#9CCFD8')
    ax1.add_patch(arrow2)
    
    # State information
    state_box = FancyBboxPatch((2, 2), 6, 2.5, boxstyle="round,pad=0.1", 
                               facecolor='#E0DEF4', edgecolor='#9CCFD8', linewidth=2)
    ax1.add_patch(state_box)
    ax1.text(5, 3.8, 'State: Opponent\'s Previous Action', ha='center', va='center', 
             fontsize=14, fontweight='bold')
    ax1.text(5, 3.3, 's_t = q_{-i, t-1}', ha='center', va='center', fontsize=12, 
             fontfamily='monospace')
    ax1.text(5, 2.8, 'Example: s_t = 25 (opponent chose q=25 last period)', ha='center', va='center', 
             fontsize=11, style='italic')
    ax1.text(5, 2.3, 'Discretized: s_t ∈ {low, medium, high}', ha='center', va='center', 
             fontsize=11, style='italic')
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('Agents observe each other\'s previous actions to form state representation', 
                   fontsize=12, style='italic')
    
    # Panel 2: Action Space Evolution
    ax2.set_title('Action Space Evolution', fontsize=16, fontweight='bold', pad=20)
    
    # Baseline action space
    baseline_actions = list(range(0, 51, 5))
    baseline_colors = ['#EBBCBA' if i != 30 else '#9CCFD8' for i in baseline_actions]
    baseline_colors[6] = '#F6C177'  # Highlight Nash equilibrium
    
    ax2.bar(range(len(baseline_actions)), [1]*len(baseline_actions), 
            color=baseline_colors, alpha=0.7, width=0.8)
    ax2.set_xticks(range(len(baseline_actions)))
    ax2.set_xticklabels([f'{q}' for q in baseline_actions], rotation=45)
    ax2.set_ylabel('Available Actions', fontsize=12)
    ax2.set_title('Baseline: Wide Action Space (0-50)', fontsize=14, fontweight='bold')
    ax2.text(6, 0.5, 'Nash q*=30', ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#F6C177', alpha=0.8))
    
    # Add optimization arrow
    ax2.annotate('', xy=(10.5, 0.5), xytext=(10, 0.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='#31748F'))
    ax2.text(10.25, 0.8, 'Optimize', ha='center', va='center', fontsize=10, 
             color='#31748F', fontweight='bold')
    
    # Ultra-optimized action space (below)
    optimized_actions = [20, 25, 28, 30, 32, 35, 40]
    optimized_colors = ['#EBBCBA' if i != 30 else '#9CCFD8' for i in optimized_actions]
    optimized_colors[3] = '#F6C177'  # Highlight Nash equilibrium
    
    y_pos = -0.5
    ax2.bar(range(len(optimized_actions)), [0.5]*len(optimized_actions), 
            bottom=[y_pos]*len(optimized_actions), color=optimized_colors, alpha=0.7, width=0.8)
    ax2.set_xticks(range(len(optimized_actions)))
    ax2.set_xticklabels([f'{q}' for q in optimized_actions], rotation=45)
    ax2.text(3, y_pos+0.25, 'Ultra-Optimized: Aligned Action Space (20-40)', 
             ha='center', va='center', fontsize=14, fontweight='bold')
    ax2.text(3, y_pos-0.1, 'Nash q*=30', ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#F6C177', alpha=0.8))
    
    ax2.set_ylim(-1, 1.2)
    ax2.set_xlabel('Quantity Choices', fontsize=12)
    
    # Panel 3: Reward Function
    ax3.set_title('Reward Function Design', fontsize=16, fontweight='bold', pad=20)
    
    # Market price simulation
    quantities = np.linspace(20, 40, 100)
    a, b, c = 100, 1, 10
    market_prices = a - b * (quantities * 2)  # Total quantity for 2 firms
    profits = (market_prices - c) * quantities
    
    ax3.plot(quantities, profits, 'o-', color='#9CCFD8', linewidth=3, markersize=4, 
             label='Base Profit: π = (P - c) × q')
    
    # Nash equilibrium point
    nash_q = 30
    nash_p = a - b * (nash_q * 2)
    nash_profit = (nash_p - c) * nash_q
    ax3.plot(nash_q, nash_profit, 'o', color='#F6C177', markersize=10, 
             label=f'Nash Equilibrium (q={nash_q}, π={nash_profit:.1f})')
    
    # Reward shaping
    nash_distance = np.abs(quantities - nash_q)
    shaped_rewards = profits - nash_distance * 5  # Penalty for distance from Nash
    ax3.plot(quantities, shaped_rewards, 's-', color='#EBBCBA', linewidth=3, markersize=4,
             label='Shaped Reward: π - α|q - q*|')
    
    ax3.set_xlabel('Quantity Choice (q)', fontsize=12)
    ax3.set_ylabel('Reward', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(20, 40)
    
    # Add reward shaping explanation
    ax3.text(0.02, 0.98, 'Reward Shaping Benefits:', transform=ax3.transAxes, 
             fontsize=12, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#F6C177', alpha=0.7))
    ax3.text(0.02, 0.88, '• Guides agents toward Nash equilibrium', transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top')
    ax3.text(0.02, 0.82, '• Reduces exploration time', transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top')
    ax3.text(0.02, 0.76, '• Improves convergence stability', transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top')
    
    # Panel 4: Exploration vs Exploitation
    ax4.set_title('Exploration vs Exploitation Strategy', fontsize=16, fontweight='bold', pad=20)
    
    # Epsilon decay over episodes
    episodes = np.linspace(0, 15000, 1000)
    epsilon_init = 0.4
    epsilon_decay = 0.9998
    epsilon_min = 0.02
    epsilon_values = np.maximum(epsilon_min, epsilon_init * (epsilon_decay ** episodes))
    
    ax4.plot(episodes, epsilon_values, '-', color='#9CCFD8', linewidth=3, 
             label='ε-greedy parameter')
    
    # Add phase annotations
    ax4.axvspan(0, 3000, alpha=0.2, color='#EBBCBA', label='Exploration Phase')
    ax4.axvspan(3000, 10000, alpha=0.2, color='#F6C177', label='Learning Phase')
    ax4.axvspan(10000, 15000, alpha=0.2, color='#9CCFD8', label='Exploitation Phase')
    
    ax4.set_xlabel('Training Episodes', fontsize=12)
    ax4.set_ylabel('Exploration Rate (ε)', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 0.45)
    
    # Add strategy explanation
    ax4.text(0.02, 0.98, 'ε-greedy Strategy:', transform=ax4.transAxes, 
             fontsize=12, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#E0DEF4', alpha=0.7))
    ax4.text(0.02, 0.88, '• ε = 0.4: 40% random exploration', transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top')
    ax4.text(0.02, 0.82, '• ε → 0.02: 2% random exploration', transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top')
    ax4.text(0.02, 0.76, '• Balance: Learn vs exploit', transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    return fig

# Create and save the illustration
if __name__ == "__main__":
    fig = create_learning_process_illustration()
    plt.savefig('/Users/razaool/game_theory/learning_process_illustration.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close the figure instead of showing it
    
    print("Learning process illustration saved as 'learning_process_illustration.png'")
