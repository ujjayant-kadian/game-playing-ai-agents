import random
import numpy as np
from games.connect4 import Connect4
from .qlearning_base import QLearningAgent

class QLearningConnect4(QLearningAgent):
    def __init__(self, player_number=1, alpha=0.2, gamma=0.95, epsilon=0.3, 
                 epsilon_decay=0.9995, epsilon_min=0.01, metrics_manager=None):
        """
        Initialize the Q-learning agent for Connect-4.
        
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
        
        # Adjust rewards for Connect-4
        self.reward_win = 1.0
        self.reward_loss = -1.0
        self.reward_draw = 0.0
        self.reward_move = -0.01  # Small penalty for each move
        
        # Caching for move detection
        self._horizontal_window_indices = self._precompute_horizontal_windows()
        self._vertical_window_indices = self._precompute_vertical_windows()
        self._diagonal_window_indices = self._precompute_diagonal_windows()
    
    def state_to_key(self, state):
        """
        Convert the game state to a hashable key for the Q-table.
        For Connect-4, we use a tuple of tuples representation.
        
        Args:
            state: 6x7 numpy array representing the game board
            
        Returns:
            tuple: A hashable representation of the state
        """
        return tuple(map(tuple, state))
    
    def get_legal_moves(self, state):
        """
        Get all legal moves for the current state in Connect-4.
        Legal moves are columns where a piece can be dropped.
        
        Args:
            state: The game state
            
        Returns:
            list: List of legal moves (column indices)
        """
        return [col for col in range(7) if state[0][col] == 0]
    
    def create_game(self):
        """Create a new Connect-4 game instance."""
        game = Connect4()
        # If player 2, make a random first move as player 1
        if self.player_number == 2:
            # Choose a random move for player 1
            moves = self.get_legal_moves(game.get_state())
            if moves:
                move = moves[np.random.randint(len(moves))]
                game.make_move(move)
        return game
    
    def _precompute_horizontal_windows(self):
        """Precompute all possible horizontal 4-in-a-row window indices."""
        windows = []
        for row in range(6):
            for col in range(4):
                window = [(row, col + i) for i in range(4)]
                windows.append(window)
        return windows
    
    def _precompute_vertical_windows(self):
        """Precompute all possible vertical 4-in-a-row window indices."""
        windows = []
        for row in range(3):
            for col in range(7):
                window = [(row + i, col) for i in range(4)]
                windows.append(window)
        return windows
    
    def _precompute_diagonal_windows(self):
        """Precompute all possible diagonal 4-in-a-row window indices."""
        windows = []
        # Positive slope diagonals
        for row in range(3, 6):
            for col in range(4):
                window = [(row - i, col + i) for i in range(4)]
                windows.append(window)
        # Negative slope diagonals
        for row in range(3):
            for col in range(4):
                window = [(row + i, col + i) for i in range(4)]
                windows.append(window)
        return windows
    
    def _count_window(self, state, window, player):
        """
        Count pieces in a window for a given player.
        Returns the count of player's pieces if the window doesn't contain opponent pieces,
        otherwise returns 0.
        """
        count = 0
        opponent = 3 - player
        for row, col in window:
            if state[row][col] == opponent:
                return 0
            if state[row][col] == player:
                count += 1
        return count
    
    def _detect_threats(self, state, player):
        """
        Detect immediate threats (3-in-a-row with an empty space) for the given player.
        
        Args:
            state: The game state
            player: The player to check threats for
            
        Returns:
            list: List of column indices where there are immediate threats
        """
        threats = []
        opponent = 3 - player
        
        # Check horizontal threats
        for window in self._horizontal_window_indices:
            # Count player and empty spaces in window
            player_count = 0
            empty_pos = None
            
            for row, col in window:
                if state[row][col] == player:
                    player_count += 1
                elif state[row][col] == 0:
                    empty_pos = (row, col)
                    
            # If 3 player pieces and 1 empty, it's a threat
            if player_count == 3 and empty_pos:
                # Make sure the empty position is valid (either at bottom or has support below)
                empty_row, empty_col = empty_pos
                if empty_row == 5 or state[empty_row + 1][empty_col] != 0:
                    if empty_col not in threats:
                        threats.append(empty_col)
        
        # Check vertical threats
        for window in self._vertical_window_indices:
            # For vertical windows, we need 3 player pieces and the top is empty
            cells = [(r, c) for r, c in window]
            player_count = sum(1 for r, c in cells if state[r][c] == player)
            bottom_cell = max(cells, key=lambda x: x[0])  # Cell with largest row index
            top_cell = min(cells, key=lambda x: x[0])     # Cell with smallest row index
            
            if player_count == 3 and state[top_cell[0]][top_cell[1]] == 0:
                if top_cell[1] not in threats:
                    threats.append(top_cell[1])
        
        # Check diagonal threats (both directions)
        for window in self._diagonal_window_indices:
            player_count = 0
            empty_pos = None
            
            for row, col in window:
                if state[row][col] == player:
                    player_count += 1
                elif state[row][col] == 0:
                    empty_pos = (row, col)
                    
            # If 3 player pieces and 1 empty, check if it's a valid move
            if player_count == 3 and empty_pos:
                empty_row, empty_col = empty_pos
                # Check if the move is valid (either at bottom or has support)
                if empty_row == 5 or state[empty_row + 1][empty_col] != 0:
                    if empty_col not in threats:
                        threats.append(empty_col)
        
        return threats
    
    def _detect_double_threats(self, state, player):
        """
        Detect positions that would create multiple threats simultaneously.
        These are usually winning moves.
        
        Args:
            state: The game state
            player: The player to check threats for
            
        Returns:
            list: List of column indices that create multiple threats
        """
        double_threats = []
        
        # For each legal move, simulate it and count resulting threats
        for col in range(7):
            # Skip if column is full
            if state[0][col] != 0:
                continue
                
            # Find the row where the piece would land
            for row in range(5, -1, -1):
                if state[row][col] == 0:
                    # Simulate the move
                    temp_state = [list(row) for row in state]
                    temp_state[row][col] = player
                    
                    # Count threats after this move
                    threats = self._detect_threats(temp_state, player)
                    if len(threats) >= 2:
                        double_threats.append(col)
                    break
        
        return double_threats

    def get_heuristic_features(self, state):
        """
        Extract heuristic features from the state to augment the Q-learning.
        Enhanced with strategies from minimax evaluation function.
        
        Features returned:
        - Number of potential winning lines with 1, 2, or 3 pieces
        - Center column control with positional weighting
        - Vertical threat recognition
        - Multiple threats detection
        
        Args:
            state: The game state
            
        Returns:
            dict: Dictionary of features
        """
        features = {
            'one_piece': 0,
            'two_pieces': 0,
            'three_pieces': 0,
            'center_control': 0,
            'vertical_threats': 0,
            'has_immediate_threat': 0,
            'has_double_threat': 0,
            'blocking_opponent_win': 0
        }
        
        player = self.player_number
        opponent = 3 - player
        
        # Count pieces in horizontal windows
        for window in self._horizontal_window_indices:
            count = self._count_window(state, window, player)
            if count == 1:
                features['one_piece'] += 1
            elif count == 2:
                features['two_pieces'] += 1
            elif count == 3:
                features['three_pieces'] += 1
                # Check if this is a valid threat (can be played immediately)
                for row, col in window:
                    if state[row][col] == 0:
                        # If this empty cell is at bottom or has support
                        if row == 5 or state[row+1][col] != 0:
                            features['has_immediate_threat'] = 1
        
        # Count pieces in vertical windows
        for window in self._vertical_window_indices:
            count = self._count_window(state, window, player)
            if count == 1:
                features['one_piece'] += 1
            elif count == 2:
                features['two_pieces'] += 1
            elif count == 3:
                features['three_pieces'] += 1
                features['vertical_threats'] += 1
                features['has_immediate_threat'] = 1
        
        # Count pieces in diagonal windows
        for window in self._diagonal_window_indices:
            count = self._count_window(state, window, player)
            if count == 1:
                features['one_piece'] += 1
            elif count == 2:
                features['two_pieces'] += 1
            elif count == 3:
                features['three_pieces'] += 1
                # Check if this is a valid threat (can be played immediately)
                for row, col in window:
                    if state[row][col] == 0:
                        # If this empty cell is at bottom or has support
                        if row == 5 or state[row+1][col] != 0:
                            features['has_immediate_threat'] = 1
        
        # Center column control with positional weighting
        center_col = 3
        for row in range(6):
            if state[row][center_col] == player:
                # More weight to lower positions (like in minimax)
                features['center_control'] += (6 - row)
        
        # Check for multiple threats
        threats = self._detect_threats(state, player)
        if len(threats) >= 2:
            features['has_double_threat'] = 1
        
        # Check if we need to block opponent's immediate win
        opponent_threats = self._detect_threats(state, opponent)
        if opponent_threats:
            features['blocking_opponent_win'] = 1
        
        return features
    
    def choose_action(self, state, legal_moves, training=False):
        """
        Choose an action using epsilon-greedy policy combined with enhanced heuristic knowledge.
        This implementation gives a sophisticated evaluation of potential moves.
        
        Args:
            state: The game state
            legal_moves: List of legal moves
            training: Whether we're in training mode (exploration enabled)
            
        Returns:
            int: Column to drop the piece
        """
        if not legal_moves:
            return None
            
        # During training, use epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # Check if we have an immediate winning move
        player_threats = self._detect_threats(state, self.player_number)
        if player_threats:
            # Prioritize the winning move
            return player_threats[0]
        
        # Check if we need to block opponent's immediate win
        opponent = 3 - self.player_number
        opponent_threats = self._detect_threats(state, opponent)
        if opponent_threats:
            # Need to block
            return opponent_threats[0]
        
        # Check for moves that create multiple threats (usually winning)
        double_threats = self._detect_double_threats(state, self.player_number)
        if double_threats:
            return double_threats[0]
        
        # Exploitation: choose the best action based on Q-values and enhanced heuristics
        best_action = None
        best_value = float('-inf')
        
        # Weight of heuristic values vs Q-values
        heuristic_weight = 0.3
        
        for action in legal_moves:
            # Get Q-value
            q_value = self.get_q_value(state, action)
            
            # Simulate the move to get heuristic value
            temp_state = np.array(state)
            for row in range(5, -1, -1):
                if temp_state[row][action] == 0:
                    temp_state[row][action] = self.player_number
                    break
                    
            temp_features = self.get_heuristic_features(temp_state)
            
            # Calculate heuristic value with improved weights
            heuristic_value = (
                0.1 * temp_features['one_piece'] + 
                0.3 * temp_features['two_pieces'] + 
                0.8 * temp_features['three_pieces'] +
                0.4 * temp_features['center_control'] +
                0.5 * temp_features['vertical_threats'] +
                2.0 * temp_features['has_immediate_threat'] +
                3.0 * temp_features['has_double_threat'] +
                1.0 * temp_features['blocking_opponent_win']
            )
            
            # Combine Q-value and heuristic
            total_value = (1 - heuristic_weight) * q_value + heuristic_weight * heuristic_value
            
            if total_value > best_value:
                best_value = total_value
                best_action = action
        
        return best_action
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value for state-action pair.
        Connect-4 has too many states for symmetry-based optimization to be practical.
        
        Args:
            state: The game state
            action: The action taken
            reward: The reward received
            next_state: The resulting state
        """
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        # Initialize q_table entry if it doesn't exist
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # Get current Q-value
        if action in self.q_table[state_key]:
            current_q = self.q_table[state_key][action]
        else:
            current_q = 0.0
        
        # Get max Q-value for next state
        next_max_q = 0.0
        if next_state_key in self.q_table:
            next_q_values = self.q_table[next_state_key].values()
            if next_q_values:
                next_max_q = max(next_q_values)
        
        # Update rule: Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action] = new_q
