# Q-Learning Implementation for Game Playing Agents

This document describes the implementation of Q-learning agents for both Tic-Tac-Toe and Connect 4 games. The implementation follows an object-oriented approach with a base class that provides core Q-learning functionality and game-specific classes that extend this functionality.

## Architecture Overview

The implementation consists of the following key components:

1. `QLearningAgent` (Abstract Base Class): Provides the foundational Q-learning algorithm implementation
2. `QLearningTicTacToe`: Implementation specific to Tic-Tac-Toe
3. `QLearningConnect4`: Implementation specific to Connect 4

## Core Q-Learning Algorithm

### QLearningAgent Base Class

The `QLearningAgent` abstract base class implements the core Q-learning algorithm with the following key features:

#### Key Parameters

- `alpha`: Learning rate that controls how much of the new Q-value is incorporated into the existing Q-value
- `gamma`: Discount factor that determines the importance of future rewards
- `epsilon`: Exploration rate that controls the probability of taking a random action
- `epsilon_decay`: Rate at which exploration decreases over time
- `epsilon_min`: Minimum exploration rate to ensure some exploration continues

#### Q-Table

The agent maintains a Q-table as a nested dictionary where:
- The outer key is a state representation
- The inner keys are actions
- The values are Q-values representing the expected future reward for each state-action pair

#### Action Selection Strategy

The agent uses an epsilon-greedy policy for action selection:
- With probability `epsilon`, a random action is selected (exploration)
- With probability `1-epsilon`, the action with the highest Q-value is selected (exploitation)

#### Q-Value Update Rule

The Q-values are updated using the standard Q-learning update rule:

```
Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
```

Where:
- `Q(s,a)` is the current Q-value for state s and action a
- `alpha` is the learning rate
- `reward` is the immediate reward received
- `gamma` is the discount factor
- `max(Q(s',a'))` is the maximum Q-value for the next state across all possible actions

#### Training Procedure

The training process involves:
1. Initializing a new game
2. Selecting and taking actions using the epsilon-greedy policy
3. Observing rewards and next states
4. Updating Q-values using the Q-learning update rule
5. Decaying the exploration rate
6. Repeating until the game terminates
7. Running multiple episodes to build a comprehensive Q-table

#### Evaluation

The agent can be evaluated against opponents (or random play) to measure performance metrics like win rate.

## Tic-Tac-Toe Implementation

### QLearningTicTacToe Class

The Tic-Tac-Toe implementation extends the base agent with the following game-specific features:

#### State Representation

States are represented as tuples of tuples to create a hashable key for the Q-table, with each cell containing:
- `0` for empty cells
- `1` for player 1's marks
- `2` for player 2's marks

#### Game Symmetry Exploitation

A key optimization in the Tic-Tac-Toe implementation is the use of board symmetries to improve learning efficiency. The implementation:

1. Identifies 8 symmetric board positions for each state-action pair:
   - Original board
   - 90°, 180°, and 270° rotations
   - Horizontal flip
   - Vertical flip
   - Diagonal flip
   - Anti-diagonal flip

2. Updates Q-values for all symmetric states simultaneously, which:
   - Speeds up learning by effectively multiplying the training data by 8
   - Ensures the agent learns equivalent strategies for symmetric board positions
   - Reduces the size of the state space that needs to be explored

#### Reward Structure

The reward structure for Tic-Tac-Toe is designed with:
- `reward_win = 1.0`: High positive reward for winning
- `reward_loss = -1.0`: High negative reward for losing
- `reward_draw = 0.2`: Small positive reward for draws (better than losing)
- `reward_move = -0.05`: Small negative reward per move to encourage shorter paths to victory

## Connect 4 Implementation

### QLearningConnect4 Class

The Connect 4 implementation is more complex due to the larger state space and requires additional optimizations:

#### State Representation

Like Tic-Tac-Toe, states are represented as tuples of tuples for the Q-table, but with a 6x7 board.

#### Advanced Heuristic Features

To handle the larger state space, the Connect 4 implementation incorporates domain-specific heuristics:

1. **Precomputed Window Patterns**:
   - Horizontal, vertical, and diagonal windows are precomputed for efficient pattern detection
   - This enables fast evaluation of potential winning lines and threats

2. **Threat Detection**:
   - `_detect_threats()`: Identifies immediate threats (three in a row with an open space)
   - `_detect_double_threats()`: Finds positions that create multiple threats simultaneously

3. **Heuristic Features**:
   The agent extracts various features from the board state:
   - Count of potential winning lines with 1, 2, or 3 pieces
   - Center column control (strategically important)
   - Vertical threat recognition
   - Multiple threats detection
   - Blocking opponent wins

#### Enhanced Action Selection

The Connect 4 implementation incorporates strategic knowledge into the action selection process:
- Prioritizes immediate winning moves
- Blocks opponent's winning moves
- Identifies and creates double threats when possible
- Falls back to Q-table values when no obvious strategic moves are available

#### Reward Structure

The reward structure is carefully designed:
- `reward_win = 1.0`: Reward for winning
- `reward_loss = -1.0`: Penalty for losing
- `reward_draw = 0.0`: Neutral for draws
- `reward_move = -0.01`: Small negative reward per move to encourage efficiency

## Training and Learning Efficiency

Both implementations include optimizations to improve learning efficiency:

1. **Epsilon Decay**: Exploration rate decays over time, transitioning from exploration to exploitation
2. **Periodic Evaluation**: Agents are evaluated at regular intervals to track progress
3. **Q-Table Management**: The implementation efficiently stores and retrieves Q-values using hashable state representations

## Conclusion

The implemented Q-learning agents demonstrate the application of reinforcement learning to two classic board games. The Tic-Tac-Toe implementation leverages board symmetry for efficient learning, while the Connect 4 implementation incorporates domain-specific heuristics to handle a larger state space.

Both implementations follow core Q-learning principles while adapting to the specific challenges of each game environment. The optimization techniques employed show how domain knowledge can be incorporated into reinforcement learning algorithms to improve learning efficiency and performance. 