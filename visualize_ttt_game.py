#!/usr/bin/env python3
import sys
from games import TicTacToe, TicTacToeUI, PlayerType, GameMode

# Create UI instance
ui = TicTacToeUI()

# Set game mode
ui.set_game_mode(GameMode.HUMAN_VS_HUMAN)

# Run the game
ui.run()
