# Implementation Details

## Connect4 Game

### Overview
The Connect4 game is implemented using Python with the help of the `numpy` and `pygame` libraries. The game logic is encapsulated in the `Connect4` class, while the user interface is managed by the `Connect4UI` class. The game supports multiple gameplay modes including human vs human, human vs AI, human vs semi-intelligent agent, AI vs semi-intelligent agent, and AI vs AI.

### Connect4 Class
- **Attributes**:
  - `rows` and `cols`: Define the dimensions of the game board (6x7).
  - `board`: A 2D numpy array representing the game board, initialized with zeros.
  - `current_player`: Tracks the current player (1 or 2).
  - `game_over`: A boolean indicating if the game has ended.
  - `winner`: Stores the winner of the game (1, 2, or 0 for a draw).
  - `last_move`: Stores the last move made as a tuple (row, col).

- **Methods**:
  - `reset()`: Resets the game to its initial state.
  - `get_state()`: Returns a copy of the current board state.
  - `get_legal_moves()`: Returns a list of columns where a move can be made.
  - `make_move(col)`: Places a piece in the specified column if the move is valid.
  - `_check_game_over()`: Checks if the game has ended due to a win or draw.
  - `is_game_over()`: Returns whether the game is over.
  - `get_winner()`: Returns the winner of the game.

### Player Types and Game Modes
- **PlayerType (Enum)**:
  - `HUMAN`: Human player controlled by user input
  - `AI`: AI player with sophisticated decision-making
  - `SEMI_INTELLIGENT`: Semi-intelligent player with basic decision-making

- **GameMode (Enum)**:
  - `HUMAN_VS_HUMAN`: Two human players
  - `HUMAN_VS_AI`: Human player against AI
  - `HUMAN_VS_SEMI`: Human player against semi-intelligent agent
  - `AI_VS_SEMI`: AI player against semi-intelligent agent
  - `AI_VS_AI`: Two AI players competing

### Connect4UI Class
- **Attributes**:
  - `cell_size`, `width`, `height`: Define the size of the game window and cells.
  - `screen`: The pygame display surface.
  - `bg_color`, `board_color`, `player1_color`, `player2_color`, `text_color`, `highlight_color`: Define the colors used in the UI.
  - `font`, `big_font`: Fonts used for rendering text.
  - `game_mode`: Current game mode (from GameMode enum).
  - `player1_type`, `player2_type`: Types of players (from PlayerType enum).
  - `player1_agent`, `player2_agent`: AI agent objects for players 1 and 2.
  - `player_move`: Boolean to track if current turn is for human input.
  - `ai_move_delay`: Delay between AI moves for better visualization.
  - `anim_active`, `anim_col`, `anim_row`, `anim_y`, `anim_speed`, `anim_player`: Variables for handling piece drop animation.

- **Methods**:
  - `set_game_mode(mode)`: Sets the game mode and initializes appropriate player types.
  - `set_player1_agent(agent)`, `set_player2_agent(agent)`: Set AI agents for players.
  - `run()`: Main game loop handling events, AI moves, and rendering.
  - `_start_animation(col)`: Initiates the drop animation for a piece.
  - `_update_animation()`: Updates the animation state.
  - `_handle_click(pos)`: Handles mouse click events to make moves.
  - `_draw()`: Renders the game board and UI elements.

### Reasoning
The Connect4 implementation separates game logic from UI, allowing for easier maintenance and potential reuse of the game logic in different interfaces. The use of numpy arrays for the board allows efficient manipulation and checking of game states. Pygame provides a simple way to create a graphical interface, making the game interactive and visually appealing.

The design includes enums for player types and game modes, making it easy to configure different gameplay scenarios. The UI can dynamically switch between these modes, allowing for a versatile gaming experience. The code supports both human and AI players, with mechanisms to delegate moves to appropriate agents based on the current player type.

## TicTacToe Game

### Overview
The TicTacToe game is also implemented using Python with `numpy` and `pygame`. Similar to Connect4, it separates game logic from UI and supports multiple game modes.

### TicTacToe Class
- **Attributes**:
  - `board`: A 3x3 numpy array representing the game board, initialized with zeros.
  - `current_player`: Tracks the current player (1 for X, 2 for O).
  - `game_over`: A boolean indicating if the game has ended.
  - `winner`: Stores the winner of the game (1, 2, or 0 for a draw).

- **Methods**:
  - `reset()`: Resets the game to its initial state.
  - `get_state()`: Returns a copy of the current board state.
  - `get_legal_moves()`: Returns a list of available moves as (row, col) tuples.
  - `make_move(move)`: Places a piece on the board if the move is valid.
  - `_check_game_over()`: Checks if the game has ended due to a win or draw.
  - `is_game_over()`: Returns whether the game is over.
  - `get_winner()`: Returns the winner of the game.

### Player Types and Game Modes
The TicTacToe game uses the same PlayerType and GameMode enums as Connect4, providing consistent behavior across both games.

### TicTacToeUI Class
- **Attributes**:
  - `width`, `height`: Define the size of the game window.
  - `screen`: The pygame display surface.
  - `bg_color`, `line_color`, `x_color`, `o_color`, `text_color`, `highlight_color`: Define the colors used in the UI.
  - `board_size`, `cell_size`, `board_margin`, `line_width`: Define the size and position of the board and cells.
  - `font`, `big_font`: Fonts used for rendering text.
  - `game_mode`: Current game mode (from GameMode enum).
  - `player1_type`, `player2_type`: Types of players (from PlayerType enum).
  - `player1_agent`, `player2_agent`: AI agent objects for players 1 and 2.
  - `player_move`: Boolean to track if current turn is for human input.
  - `ai_move_delay`: Delay between AI moves for better visualization.

- **Methods**:
  - `set_game_mode(mode)`: Sets the game mode and initializes appropriate player types.
  - `set_player1_agent(agent)`, `set_player2_agent(agent)`: Set AI agents for players.
  - `run()`: Main game loop handling events, AI moves, and rendering.
  - `_handle_click(pos)`: Handles mouse click events to make moves.
  - `_draw()`: Renders the game board and UI elements.

### Reasoning
The TicTacToe implementation follows a similar structure to Connect4, with a clear separation between game logic and UI. This design choice enhances modularity and allows for easy updates or changes to either component. 

Both games share a consistent approach to handling different player types and game modes, making the codebase more maintainable and providing a unified experience. The UI for both games includes keyboard shortcuts (1-5) to easily switch between game modes during gameplay, allowing for flexible testing and demonstration of different scenarios.

## Game Controls

### Tic Tac Toe
- Click on a cell to place your mark
- Press 'R' to restart the game
- Press '1-5' to switch between game modes (see Game Modes section)

### Connect 4
- Click on a column to drop your piece
- Hover over a column to see a preview
- Press 'R' to restart the game
- Press '1-5' to switch between game modes (see Game Modes section)

## Game Modes

Both games support the following modes that can be changed during gameplay by pressing the corresponding number key:

1. **Human vs Human**: Two human players take turns making moves
2. **Human vs AI**: Human player against an AI opponent
3. **Human vs Semi-Intelligent**: Human player against a semi-intelligent opponent
4. **AI vs Semi-Intelligent**: AI player against semi-intelligent opponent (fully automated gameplay)
5. **AI vs AI**: Two AI players competing against each other (fully automated gameplay)

## Using the Game Logic with AI Agents

Both game classes (TicTacToe and Connect4) provide methods for AI agents to interact with the game:

- `get_state()`: Get the current game board state
- `get_legal_moves()`: Get available legal moves
- `make_move(move)`: Make a move and update game state
- `is_game_over()`: Check if the game is over
- `get_winner()`: Get the winner (if any)

### Example of AI Integration

```python
# For Tic Tac Toe
from tic_tac_toe import TicTacToeUI, PlayerType, GameMode

# Create custom AI agent
class MyTicTacToeAgent:
    def get_move(self, state):
        # Your AI logic here
        return (row, col)  # Return move as (row, col)

# Create custom semi-intelligent agent
class MySemiIntelligentAgent:
    def get_move(self, state):
        # Simpler decision logic here
        return (row, col)  # Return move as (row, col)

# Set up the game with AI
game_ui = TicTacToeUI()

# Set player agents
game_ui.set_player1_agent(MyTicTacToeAgent())  # For Player 1 (X)
game_ui.set_player2_agent(MySemiIntelligentAgent())  # For Player 2 (O)

# Set game mode (e.g., AI vs Semi-Intelligent)
game_ui.set_game_mode(GameMode.AI_VS_SEMI)

# Run the game
game_ui.run()
```

Similar approach works for Connect 4 integration.

## Player Types

Both games support three types of players:
- **Human**: Controlled by user input
- **AI**: Controlled by an AI agent with sophisticated decision-making
- **Semi-Intelligent**: Controlled by a simpler agent with basic decision-making

For AI and Semi-Intelligent players, you need to provide agent implementations with a `get_move(state)` method that returns a valid move for the current state. 