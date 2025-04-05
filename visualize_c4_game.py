#!/usr/bin/env python3
import sys
from games import Connect4, Connect4UI, PlayerType, GameMode

# Create UI instance
ui = Connect4UI()

# Set game mode
ui.set_game_mode(GameMode.HUMAN_VS_HUMAN)

# Run the game
ui.run()
