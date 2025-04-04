# Minimax Implementation for Game-Playing Agents

This document details the implementation of the Minimax algorithm for adversarial search in game-playing scenarios. The implementation includes a base abstract class and specific implementations for Tic-Tac-Toe and Connect-4 games.

## Overview

The implementation follows an object-oriented approach with:

1. `MinimaxBase`: An abstract base class that implements the core Minimax algorithm with and without alpha-beta pruning
2. `MinimaxTicTacToe`: A concrete implementation for the Tic-Tac-Toe game
3. `MinimaxConnect4`: A concrete implementation for the Connect-4 game

## MinimaxBase Class

The `MinimaxBase` class serves as an abstract foundation for game-specific Minimax implementations. It provides:

### Key Features

- **Configurable Search Depth**: Limits the depth of the search tree to manage computational resources.
- **Alpha-Beta Pruning**: Optional pruning to eliminate branches that won't affect the final decision.
- **Performance Metrics**: Tracks states explored for performance analysis.
- **Abstract Methods**: Defines the interface that game-specific implementations must fulfill.

### Core Algorithm

The implementation includes two main approaches:
1. Standard Minimax (without pruning)
2. Alpha-Beta Pruning Minimax (more efficient)

The pruning version significantly reduces the number of states explored by eliminating branches that cannot influence the final decision, while guaranteeing the same result as the standard algorithm.

### Abstract Methods

Game-specific implementations must provide:

- `_get_current_player(state)`: Identifies the current player from the state
- `_get_legal_moves(state)`: Returns all valid moves in the current state
- `_make_move(state, move, player)`: Applies a move to the state
- `_check_winner(state)`: Determines if the game has ended and who won
- `_evaluate_board(state)`: Heuristic evaluation of non-terminal states
- `_evaluate_terminal(winner)`: Evaluation of terminal states

## Tic-Tac-Toe Implementation

The `MinimaxTicTacToe` class implements Minimax for the 3x3 Tic-Tac-Toe game.

### Key Implementation Details

- **Default Search Depth**: 9 (maximum possible moves in Tic-Tac-Toe)
- **State Representation**: 3x3 NumPy array with 0 (empty), 1 (player 1), and 2 (player 2)
- **Move Representation**: (row, col) tuples for cell positions
- **Current Player Determination**: By counting pieces (player with fewer pieces moves next)

### Evaluation Function

The evaluation function assesses board positions through:

1. **Terminal Evaluation**: 
   - Win: +100 points
   - Loss: -100 points
   - Draw: 0 points

2. **Non-Terminal Evaluation**:
   - Scores each line (row, column, diagonal) based on winning potential
   - Two own pieces with one empty: +10 (near win)
   - One own piece with two empty: +1 (potential future win)
   - Two opponent pieces with one empty: -10 (blocking needed)
   - One opponent piece with two empty: -1 (potential future threat)

## Connect-4 Implementation

The `MinimaxConnect4` class implements Minimax for the 6x7 Connect-4 game.

### Key Implementation Details

- **Default Search Depth**: 4 (limited due to higher branching factor)
- **State Representation**: 6x7 NumPy array with 0 (empty), 1 (player 1), and 2 (player 2)
- **Move Representation**: Column indices (pieces fall to lowest empty position)
- **Current Player Determination**: By counting pieces (player with fewer pieces moves next)

### Evaluation Function

The Connect-4 evaluation is more sophisticated due to the game's complexity:

1. **Window-Based Evaluation**:
   - Examines all possible 4-cell windows (horizontal, vertical, and both diagonals)
   - Scores based on piece configuration in each window
   
2. **Immediate Threat Detection**:
   - Identifies positions where one move creates a winning line
   - Prioritizes blocking opponent threats or creating own threats

3. **Position Weighting**:
   - Center columns are weighted more favorably (better strategic positions)
   - Controls more of the board and enables more potential winning lines

## Reasoning Behind Implementation Choices

### 1. Abstract Base Class Design

Using an abstract base class:
- **Promotes Code Reuse**: Core algorithm implemented once
- **Ensures Consistency**: All game implementations follow the same structure
- **Simplifies Extension**: Adding new games requires implementing only game-specific logic

### 2. Alpha-Beta Pruning

Implementing optional alpha-beta pruning:
- **Performance Optimization**: Substantially reduces the number of explored states
- **Same Outcome**: Produces identical results to standard minimax
- **Configurable**: Can be disabled for educational purposes or comparison

### 3. Different Evaluation Functions

Game-specific evaluation functions:
- **Connect-4**: More complex evaluation due to higher game complexity
- **Tic-Tac-Toe**: Simpler evaluation suitable for the smaller game space

### 4. Different Default Depths

- **Tic-Tac-Toe**: 9 (entire game tree can be explored)
- **Connect-4**: 4 (limited due to exponentially larger game tree)

### 5. Performance Considerations

- **State Tracking**: Counts explored states for performance analysis
- **Efficient Winner Checking**: Connect-4 implements optimized winner checking that only evaluates the area affected by the last move

## Conclusion

This implementation provides a flexible, extensible framework for Minimax-based game-playing agents. The object-oriented design separates the algorithm from game-specific details, allowing for clean extension to other games. The optional alpha-beta pruning significantly improves performance without sacrificing decision quality. 