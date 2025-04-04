# Implementation Details for Default Opponents

## DefaultOpponentC4

The `DefaultOpponentC4` class implements a semi-intelligent agent for Connect 4. It follows a simple strategy:
- **Winning Move**: Plays a winning move if available.
- **Blocking Move**: Blocks the opponent's winning move if possible.
- **Center Preference**: Prefers center columns for strategic advantage.
- **Random Move**: Plays randomly if no strategic move is found.

### Implementation Details
- **Initialization**: The agent is initialized with a `player_number` (1 for Red, 2 for Yellow).
- **get_move(state)**: Determines the next move based on the current board state.
  - Checks for winning moves and blocking moves by simulating potential moves.
  - Uses weighted random choice to prefer center columns.

## DefaultOpponentTTT

The `DefaultOpponentTTT` class implements a semi-intelligent agent for Tic Tac Toe. It follows a simple strategy:
- **Winning Move**: Plays a winning move if available.
- **Blocking Move**: Blocks the opponent's winning move if possible.
- **Center and Corner Preference**: Prefers the center square, then corners for strategic advantage.
- **Random Move**: Plays randomly if no strategic move is found.

### Implementation Details
- **Initialization**: The agent is initialized with a `player_number` (1 for X, 2 for O).
- **get_move(state)**: Determines the next move based on the current board state.
  - Checks for winning moves and blocking moves by simulating potential moves.
  - Prefers the center square and corners before playing randomly.

These default opponents provide a basic level of challenge and can be used to test AI agents or as opponents in human vs. AI games. 