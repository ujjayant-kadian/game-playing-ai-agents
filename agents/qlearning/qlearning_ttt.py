import numpy as np
from games.tic_tac_toe import TicTacToe
from .qlearning_base import QLearningAgent

class QLearningTicTacToe(QLearningAgent):
    def __init__(self, player_number=1, alpha=0.3, gamma=0.9, epsilon=0.3, 
                 epsilon_decay=0.9999, epsilon_min=0.01, metrics_manager=None):
        """
        Initialize the Q-learning agent for Tic-Tac-Toe.
        
        Args:
            player_number: The player number (1 or 2)
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum exploration rate
            metrics_manager: Metrics manager for tracking stats
        """
        super().__init__(player_number, alpha, gamma, epsilon, 
                         epsilon_decay, epsilon_min, metrics_manager)
        
        # Higher rewards for Tic-Tac-Toe due to shorter game length
        self.reward_win = 1.0
        self.reward_loss = -1.0
        self.reward_draw = 0.2  # Draws are better than losses
        self.reward_move = -0.05  # Small penalty for each move
    
    def state_to_key(self, state):
        """
        Convert the game state to a hashable key for the Q-table.
        
        Args:
            state: 3x3 numpy array representing the game board
            
        Returns:
            tuple: A hashable representation of the state
        """
        # Convert the board state to a tuple of tuples
        # This ensures the state is hashable and can be used as a dictionary key
        return tuple(map(tuple, state))
    
    def get_legal_moves(self, state):
        """
        Get all legal moves for the current state.
        
        Args:
            state: The game state
            
        Returns:
            list: List of legal moves as (row, col) tuples
        """
        return [(i, j) for i in range(3) for j in range(3) if state[i, j] == 0]
    
    def create_game(self):
        """Create a new Tic-Tac-Toe game instance."""
        game = TicTacToe()
        # If player 2, make a random first move as player 1
        if self.player_number == 2:
            # Choose a random move for player 1
            moves = self.get_legal_moves(game.get_state())
            if moves:
                move = moves[np.random.randint(len(moves))]
                game.make_move(move)
        return game
    
    def get_symmetries(self, state, action):
        """
        Get all symmetric states and corresponding actions.
        This helps the agent learn faster by exploiting the symmetry of the game.
        
        Args:
            state: The game state
            action: The action taken in that state
            
        Returns:
            list: List of (state_key, action) pairs for all symmetries
        """
        state_array = np.array(state).reshape(3, 3)
        row, col = action
        
        symmetries = []
        
        # Original
        symmetries.append((tuple(map(tuple, state_array)), (row, col)))
        
        # Rotate 90 degrees
        rot90 = np.rot90(state_array)
        new_row, new_col = 2 - col, row
        symmetries.append((tuple(map(tuple, rot90)), (new_row, new_col)))
        
        # Rotate 180 degrees
        rot180 = np.rot90(rot90)
        new_row, new_col = 2 - row, 2 - col
        symmetries.append((tuple(map(tuple, rot180)), (new_row, new_col)))
        
        # Rotate 270 degrees
        rot270 = np.rot90(rot180)
        new_row, new_col = col, 2 - row
        symmetries.append((tuple(map(tuple, rot270)), (new_row, new_col)))
        
        # Flip horizontally
        flip_h = np.fliplr(state_array)
        new_row, new_col = row, 2 - col
        symmetries.append((tuple(map(tuple, flip_h)), (new_row, new_col)))
        
        # Flip vertically
        flip_v = np.flipud(state_array)
        new_row, new_col = 2 - row, col
        symmetries.append((tuple(map(tuple, flip_v)), (new_row, new_col)))
        
        # Flip along main diagonal
        flip_diag = np.transpose(state_array)
        new_row, new_col = col, row
        symmetries.append((tuple(map(tuple, flip_diag)), (new_row, new_col)))
        
        # Flip along other diagonal
        flip_diag2 = np.rot90(np.transpose(rot90))
        new_row, new_col = 2 - col, 2 - row
        symmetries.append((tuple(map(tuple, flip_diag2)), (new_row, new_col)))
        
        return symmetries
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value for state-action pair and all its symmetries.
        
        Args:
            state: The game state
            action: The action taken
            reward: The reward received
            next_state: The resulting state
        """
        # Get all symmetric states and actions
        symmetries = self.get_symmetries(state, action)
        
        for sym_state, sym_action in symmetries:
            state_key = sym_state
            
            # Initialize q_table entry if it doesn't exist
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            # Get current Q-value
            if sym_action in self.q_table[state_key]:
                current_q = self.q_table[state_key][sym_action]
            else:
                current_q = 0.0
            
            # Get max Q-value for next state
            next_state_key = self.state_to_key(next_state)
            next_max_q = 0.0
            if next_state_key in self.q_table:
                next_q_values = self.q_table[next_state_key].values()
                if next_q_values:
                    next_max_q = max(next_q_values)
            
            # Update rule: Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
            new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
            self.q_table[state_key][sym_action] = new_q
