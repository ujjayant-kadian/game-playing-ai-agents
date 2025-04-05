"""
AI Game Playing Agents Assignment (CS7IS2)

This package contains:
- Game environments (Tic Tac Toe, Connect 4)
- AI agents (Minimax, Q-learning)
- Semi-intelligent opponents
- Experiment utilities
""" 
from games import TicTacToe, Connect4, TicTacToeUI, Connect4UI
from agents import MinimaxTicTacToe, MinimaxConnect4, QLearningTicTacToe, QLearningConnect4
from opponents import DefaultOpponentTTT, DefaultOpponentC4
from utils import MetricsManager


__all__ = [
    "TicTacToe",
    "Connect4",
    "TicTacToeUI",
    "Connect4UI",
    "MinimaxTicTacToe",
    "MinimaxConnect4",
    "QLearningTicTacToe",
    "QLearningConnect4",
    "DefaultOpponentTTT",
    "DefaultOpponentC4",
    "MetricsManager"
]