# Cournot-Bertrand Multi-Agent Reinforcement Learning

A comprehensive implementation of Multi-Agent Reinforcement Learning (MARL) for Cournot and Bertrand competition models, focusing on game theory, Markov games, and Nash equilibrium analysis.

## 🎯 Project Overview

This project implements classic economic game theory models using modern MARL techniques to study:

- **Cournot Competition**: Quantity competition between firms
- **Bertrand Competition**: Price competition between firms
- **Nash Equilibrium Analysis**: Theoretical vs. learned equilibria
- **Learning Dynamics**: How agents discover optimal strategies
- **Strategic Interaction**: Multi-agent decision making

## 🏗️ Project Structure

```
cournot_bertrand_marl/
├── src/
│   ├── environments/          # Game environments
│   ├── agents/               # Learning agents
│   ├── analysis/             # Game theory analysis
│   └── utils/                # Utility functions
├── notebooks/                # Jupyter notebooks
├── tests/                    # Unit tests
└── requirements.txt          # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd cournot_bertrand_marl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from src.environments.cournot_game import CournotGame
from src.agents.q_learning_agent import QLearningAgent

# Create a Cournot game with 3 firms
game = CournotGame(n_firms=3, demand_params=(100, 1), cost_params=10)

# Add learning agents
for i in range(3):
    agent = QLearningAgent(agent_id=i)
    game.add_agent(agent)

# Run simulation
results = game.simulate(episodes=1000)
```

## 📊 Key Features

### Game Theory Analysis
- Nash equilibrium calculation
- Best response analysis
- Comparative statics
- Efficiency analysis

### MARL Algorithms
- Q-Learning
- Deep Q-Networks (DQN)
- Policy Gradient Methods
- Best Response Dynamics

### Visualization
- Strategy evolution plots
- Equilibrium convergence analysis
- Profit distribution analysis
- Interactive dashboards

## 🎮 Game Models

### Cournot Competition
- **Agents**: Competing firms
- **Actions**: Production quantities
- **Payoffs**: Profit functions
- **Equilibrium**: Quantity-based Nash equilibrium

### Bertrand Competition
- **Agents**: Competing firms
- **Actions**: Prices
- **Payoffs**: Profit functions
- **Equilibrium**: Price-based Nash equilibrium

## 📈 Analysis Tools

- **Equilibrium Analysis**: Compare theoretical vs. learned equilibria
- **Convergence Analysis**: Track strategy evolution over time
- **Efficiency Analysis**: Measure market efficiency and welfare
- **Sensitivity Analysis**: Study parameter effects

## 🔬 Research Applications

- Market competition analysis
- Antitrust policy evaluation
- Strategic behavior in oligopolies
- Learning in economic games

## 📚 Dependencies

- **NumPy/SciPy**: Numerical computations
- **PyTorch**: Deep learning
- **Ray RLlib**: Multi-agent RL
- **Matplotlib/Seaborn**: Visualization
- **Nashpy**: Game theory tools

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 📖 References

- Cournot, A. (1838). *Recherches sur les principes mathématiques de la théorie des richesses*
- Bertrand, J. (1883). *Théorie mathématique de la richesse sociale*
- Fudenberg, D. & Tirole, J. (1991). *Game Theory*
- Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction*
