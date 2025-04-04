"""
Game environments for AI agent training and testing.

This package contains implementations of games (Tic Tac Toe and Connect 4)
with both the game logic and UI components.
"""

from games.tic_tac_toe import TicTacToe, TicTacToeUI, PlayerType, GameMode
from games.connect4 import Connect4, Connect4UI 
__all__ = ['TicTacToe', 'TicTacToeUI', 'PlayerType', 'GameMode', 'Connect4', 'Connect4UI'] 