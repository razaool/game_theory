# Game Theory Project

A comprehensive exploration of game theory concepts, from classical models to modern multi-agent reinforcement learning implementations.

## Project Overview

This repository demonstrates the fundamental principles of game theory through interactive Jupyter notebooks, combining theoretical analysis with computational simulations and visualizations.

## Notebooks Implemented

### 1. **Prisoner's Dilemma** (`prisoners_dilemma.ipynb`)
**Introduction to Game Theory**

- **Concept**: The classic two-player simultaneous game
- **Scenario**: Two suspects must choose to cooperate or defect
- **Key Learning**: Nash equilibrium, dominant strategies, and Pareto efficiency
- **Features**:
  - Interactive payoff matrix visualization
  - Nash equilibrium analysis
  - Strategic implications explanation
  - Interactive simulation
  - Real-world applications

### 2. **Cournot Oligopoly** (`cournot_oligopoly_prices.ipynb`)
**Quantity Competition in Oligopoly**

- **Concept**: Firms compete by choosing quantities simultaneously
- **Economic Insight**: How prices decrease as the number of firms increases
- **Key Features**:
  - Mathematical derivation of Nash equilibrium
  - Price-quantity relationships
  - Welfare analysis (consumer surplus, producer surplus, deadweight loss)
  - Convergence to competitive markets as n → ∞
  - Four-panel visualization showing market dynamics

### 3. **Bertrand Competition** (`bertrand_competition.ipynb`)
**Price Competition in Oligopoly**

- **Concept**: Firms compete by choosing prices simultaneously
- **Economic Insight**: The famous Bertrand Paradox - even 2 firms achieve competitive outcomes
- **Key Features**:
  - Nash equilibrium derivation (price = marginal cost)
  - Comparison with Cournot competition
  - Four-panel visualization of the Bertrand Paradox
  - Best response functions and profit landscapes
  - Price competition dynamics simulation
  - Comprehensive welfare analysis

### 4. **Learning Game Theory** (`learning_game_theory.ipynb`)
**Multi-Agent Reinforcement Learning**

- **Concept**: AI agents learn to play game-theoretic equilibria through trial and error
- **Innovation**: Bridge between artificial intelligence and economic theory
- **Key Features**:
  - Q-Learning agent implementation
  - Cournot competition learning simulation
  - Bertrand competition learning simulation
  - Convergence analysis and visualizations
  - Performance comparison with theoretical predictions
  - Six-panel visualization showing learning dynamics

## Visual Design

All notebooks feature the **Rose Pine Dawn** matplotlib theme, providing:
- Light, warm backgrounds for excellent readability
- Professional academic appearance
- Consistent aesthetic across all visualizations
- High contrast for printed materials

## Key Research Questions Explored

1. **Can AI agents discover Nash equilibria without knowing the theory?**
2. **How do learning algorithms compare to theoretical predictions?**
3. **What are the convergence properties of different learning approaches?**
4. **How does the number of firms affect market outcomes?**
5. **Why do Cournot and Bertrand models produce different results?**

## Technical Implementation

### **Mathematical Foundations**
- Linear demand functions: Q = a - bP
- Constant marginal cost: C = cq
- Nash equilibrium derivations
- Welfare economics calculations

### **Learning Algorithms**
- Q-Learning with epsilon-greedy exploration
- State-action value tables
- Reward-based strategy updates
- Convergence analysis

### **Visualization Techniques**
- Moving averages for smooth convergence plots
- Theoretical benchmark comparisons
- Multi-panel layouts for comprehensive analysis
- Interactive elements for exploration

## Getting Started

### **Prerequisites**
```bash
pip install numpy matplotlib seaborn pandas
```

### **Running the Notebooks**
1. Clone the repository
2. Install required packages
3. Open any notebook in Jupyter Lab/Notebook
4. Run all cells to see the complete analysis

### **Installation of Rose Pine Theme**
The notebooks automatically install the Rose Pine matplotlib theme for beautiful visualizations.

## Results and Insights

### **Theoretical Validation**
- Learning agents successfully discover Nash equilibria
- Convergence to theoretical predictions within acceptable error margins
- Demonstration of fundamental game-theoretic principles

### **Learning Dynamics**
- **Early episodes**: High exploration, random strategies
- **Middle episodes**: Gradual convergence as agents learn
- **Later episodes**: Stable strategies near Nash equilibrium

### **Economic Implications**
- **Cournot**: Moderate competition, positive profits
- **Bertrand**: Intense competition, zero economic profits
- **Learning**: Agents can discover optimal strategies without prior knowledge

## Future Extensions

### **Potential Additions**
- **Advanced MARL**: Deep Q-Networks, Actor-Critic methods
- **Continuous Action Spaces**: More realistic price/quantity choices
- **Repeated Games**: Cooperation and collusion emergence
- **Market Dynamics**: Entry/exit decisions, market structure evolution
- **Human vs AI**: Interactive gameplay against learning agents

### **Research Applications**
- Algorithmic trading strategies
- Auction design and bidding behavior
- Market making and pricing optimization
- Multi-agent resource allocation

## Educational Value

This project serves as:
- **Undergraduate/Graduate Course Material**: Comprehensive game theory education
- **Research Reference**: MARL applications in economics
- **Interactive Learning**: Hands-on exploration of economic concepts
- **Visual Learning**: Beautiful plots enhance understanding

## Contributing

Contributions are welcome! Areas for improvement:
- Additional game types (Stackelberg, Hotelling, etc.)
- More sophisticated learning algorithms
- Interactive visualizations
- Performance optimizations
- Documentation enhancements

## License

This project is open source and available under the MIT License.
