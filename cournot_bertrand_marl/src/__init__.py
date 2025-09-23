"""
Cournot-Bertrand Multi-Agent Reinforcement Learning Package

A comprehensive implementation of MARL for economic game theory models.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .environments import CournotGame
from .agents import QLearningAgent, BestResponseAgent

__all__ = [
    "CournotGame",
    "QLearningAgent",
    "BestResponseAgent",
]
