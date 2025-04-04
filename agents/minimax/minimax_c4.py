import numpy as np
from .minimax_base import MinimaxBase

class MinimaxConnect4(MinimaxBase):
    """
    Minimax implementation for Connect-4.
    """
    
    def __init__(self, max_depth=4, metrics_manager=None, use_pruning=True):
        """
        Initialize the MinimaxConnect4 with configurable parameters.
        Default max_depth is 4 for Connect-4 due to its branching factor.
        
        Args:
            max_depth (int): Maximum depth for the minimax algorithm
            metrics_manager: Optional metrics manager for tracking performance
            use_pruning (bool): Whether to use alpha-beta pruning
        """
        super().__init__(max_depth, metrics_manager, use_pruning)
        self.rows = 6
        self.cols = 7
    
    def _get_current_player(self, state):
        """
        Get the current player from the state.
        In Connect-4, we determine current player by counting pieces.
        
        Args:
            state: Current state of the game (numpy array)
            
        Returns:
            The current player ID (1 or 2)
        """
        # Count number of each player's pieces
        p1_count = np.count_nonzero(state == 1)
        p2_count = np.count_nonzero(state == 2)
        
        # Player 1 goes first, so if counts are equal, it's player 1's turn
        return 1 if p1_count <= p2_count else 2
    
    def _get_legal_moves(self, state):
        """
        Get legal moves for the current state.
        In Connect-4, legal moves are columns that aren't filled.
        
        Args:
            state: Current state of the game
            
        Returns:
            List of legal moves (column indices)
        """
        return [col for col in range(self.cols) if state[0, col] == 0]
    
    def _make_move(self, state, move, player):
        """
        Make a move on the board.
        
        Args:
            state: Current state of the game
            move: Move to make (column index)
            player: Player making the move
            
        Returns:
            New state after making the move
        """
        col = move
        
        # Find the lowest empty row in the selected column
        for row in range(self.rows - 1, -1, -1):
            if state[row, col] == 0:
                state[row, col] = player
                break
        
        return state
    
    def _check_winner(self, state):
        """
        Check if there's a winner in the current state.
        
        Args:
            state: Current state of the game
            
        Returns:
            The winner (1 or 2), 0 for draw, None if game is not over
        """
        # Check horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if state[row, col] != 0 and state[row, col] == state[row, col+1] == state[row, col+2] == state[row, col+3]:
                    return state[row, col]
        
        # Check vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if state[row, col] != 0 and state[row, col] == state[row+1, col] == state[row+2, col] == state[row+3, col]:
                    return state[row, col]
        
        # Check diagonal (positive slope)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if state[row, col] != 0 and state[row, col] == state[row+1, col+1] == state[row+2, col+2] == state[row+3, col+3]:
                    return state[row, col]
        
        # Check diagonal (negative slope)
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if state[row, col] != 0 and state[row, col] == state[row-1, col+1] == state[row-2, col+2] == state[row-3, col+3]:
                    return state[row, col]
        
        # Check for draw (top row filled)
        if np.all(state[0, :] != 0):
            return 0  # Draw
        
        # Game not over yet
        return None
    def _count_immediate_threats(self, state):
        """
        Count the number of immediate winning threats for the current player.
        An immediate threat is an empty position that would result in a win.
        
        Args:
            state: Current state of the game
            
        Returns:
            Number of immediate threats
        """
        threats = 0
        
        # Try each possible move
        for col in range(self.cols):
            if state[0, col] != 0:  # Column is full
                continue
                
            # Find where piece would land
            for row in range(self.rows-1, -1, -1):
                if state[row, col] == 0:
                    # Try the move
                    test_state = state.copy()
                    test_state[row, col] = self.player
                    
                    # Check if this creates a win
                    if self._is_winning_move(test_state, row, col):
                        threats += 1
                    break
        
        return threats

    def _is_winning_move(self, state, row, col):
        """
        Check if the last move at (row, col) creates a win.
        More efficient than checking the entire board.
        
        Args:
            state: Current state of the game
            row: Row of last move
            col: Column of last move
        
        Returns:
            True if the move creates a win
        """
        player = state[row, col]
        
        # Check horizontal
        count = 0
        for c in range(max(0, col-3), min(self.cols, col+4)):
            if state[row, c] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0
                
        # Check vertical
        count = 0
        for r in range(max(0, row-3), min(self.rows, row+4)):
            if state[r, col] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0
                
        # Check diagonal (positive slope)
        count = 0
        for i in range(-3, 4):
            r = row + i
            c = col + i
            if 0 <= r < self.rows and 0 <= c < self.cols:
                if state[r, c] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
                    
        # Check diagonal (negative slope)
        count = 0
        for i in range(-3, 4):
            r = row - i
            c = col + i
            if 0 <= r < self.rows and 0 <= c < self.cols:
                if state[r, c] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
                    
        return False
    
    def _evaluate_board(self, state):
        """
        Evaluate the current board state using a heuristic.
        For Connect-4, we'll use a weighted scoring system based on potential connections.
        
        Args:
            state: Current state of the game
            
        Returns:
            Numerical score for the board state
        """
        score = 0
        
        # Check all possible four-in-a-row windows and score them
        
        # Horizontal windows
        for row in range(self.rows):
            for col in range(self.cols - 3):
                window = state[row, col:col+4]
                score += self._evaluate_window(window)
        
        # Vertical windows
        for row in range(self.rows - 3):
            for col in range(self.cols):
                window = state[row:row+4, col]
                score += self._evaluate_window(window)
        
        # Positive diagonal windows
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                window = np.array([state[row+i, col+i] for i in range(4)])
                score += self._evaluate_window(window)
        
        # Negative diagonal windows
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                window = np.array([state[row-i, col+i] for i in range(4)])
                score += self._evaluate_window(window)
        
        # Prefer center columns (better positions strategically)
        center_col = self.cols // 2
        for row in range(self.rows):
            if state[row, center_col] == self.player:
                # More weight to lower positions
                score += 3 * (self.rows - row)
        
        # Vertical threat bonus (they're harder to block)
        for col in range(self.cols):
            for row in range(self.rows - 3):
                window = state[row:row+4, col]
                our_pieces = np.count_nonzero(window == self.player)
                empty_slots = np.count_nonzero(window == 0)
                if our_pieces == 3 and empty_slots == 1:
                    score += 2  # Additional bonus for vertical threats
        
        # Detect forced moves (immediate threats)
        immediate_threats = self._count_immediate_threats(state)
        if immediate_threats > 1:
            score += 50  # Multiple threats usually lead to forced win
        
        return score
    
    def _evaluate_window(self, window):
        """
        Evaluate a window of 4 slots for potential connections.
        
        Args:
            window: Array of 4 positions
            
        Returns:
            Score for this window
        """
        our_pieces = np.count_nonzero(window == self.player)
        opponent_pieces = np.count_nonzero(window == (3 - self.player))
        empty_slots = np.count_nonzero(window == 0)
        
        if our_pieces == 4:
            return 100  # We win
        elif our_pieces == 3 and empty_slots == 1:
            return 5  # Potential win next move
        elif our_pieces == 2 and empty_slots == 2:
            return 2  # Potential future win
        
        if opponent_pieces == 3 and empty_slots == 1:
            return -4  # Block opponent win
        
        return 0
    
    def _evaluate_terminal(self, winner):
        """
        Evaluate a terminal state.
        
        Args:
            winner: The winner (1 or 2), 0 for draw
            
        Returns:
            Numerical score for the terminal state
        """
        if winner == 0:  # Draw
            return 0
        elif winner == self.player:  # We win
            return 1000000  # Very high value
        else:  # Opponent wins
            return -1000000  # Very low value
