import numpy as np
from .minimax_base import MinimaxBase

class MinimaxTicTacToe(MinimaxBase):
    """
    Minimax implementation for Tic-Tac-Toe.
    """
    
    def __init__(self, max_depth=9, metrics_manager=None, use_pruning=True):
        """
        Initialize the MinimaxTicTacToe with configurable parameters.
        Default max_depth is 9 since Tic-Tac-Toe has at most 9 moves.
        
        Args:
            max_depth (int): Maximum depth for the minimax algorithm
            metrics_manager: Optional metrics manager for tracking performance
            use_pruning (bool): Whether to use alpha-beta pruning
        """
        super().__init__(max_depth, metrics_manager, use_pruning)
    
    def _get_current_player(self, state):
        """
        Get the current player from the state.
        In Tic-Tac-Toe, we determine current player by counting pieces.
        
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
        In Tic-Tac-Toe, legal moves are empty cells.
        
        Args:
            state: Current state of the game
            
        Returns:
            List of legal moves as (row, col) tuples
        """
        return [(i, j) for i in range(3) for j in range(3) if state[i, j] == 0]
    
    def _make_move(self, state, move, player):
        """
        Make a move on the board.
        
        Args:
            state: Current state of the game
            move: Move to make as (row, col)
            player: Player making the move
            
        Returns:
            New state after making the move
        """
        row, col = move
        state[row, col] = player
        return state
    
    def _check_winner(self, state):
        """
        Check if there's a winner in the current state.
        
        Args:
            state: Current state of the game
            
        Returns:
            The winner (1 or 2), 0 for draw, None if game is not over
        """
        # Check rows
        for row in range(3):
            if state[row, 0] != 0 and state[row, 0] == state[row, 1] == state[row, 2]:
                return state[row, 0]
        
        # Check columns
        for col in range(3):
            if state[0, col] != 0 and state[0, col] == state[1, col] == state[2, col]:
                return state[0, col]
        
        # Check diagonals
        if state[0, 0] != 0 and state[0, 0] == state[1, 1] == state[2, 2]:
            return state[0, 0]
        
        if state[0, 2] != 0 and state[0, 2] == state[1, 1] == state[2, 0]:
            return state[0, 2]
        
        # Check for draw (all cells filled)
        if np.all(state != 0):
            return 0  # Draw
        
        # Game not over yet
        return None
    
    def _evaluate_board(self, state):
        """
        Evaluate the current board state using a heuristic.
        For Tic-Tac-Toe, we'll use a simple scoring system based on potential wins.
        
        Args:
            state: Current state of the game
            
        Returns:
            Numerical score for the board state
        """
        score = 0
        
        # Check rows, columns, diagonals for potential wins
        # Add points for our potential wins, subtract for opponent's
        
        # Check rows
        for row in range(3):
            score += self._evaluate_line(state[row, :])
        
        # Check columns
        for col in range(3):
            score += self._evaluate_line(state[:, col])
        
        # Check diagonals
        score += self._evaluate_line(np.array([state[0, 0], state[1, 1], state[2, 2]]))
        score += self._evaluate_line(np.array([state[0, 2], state[1, 1], state[2, 0]]))
        
        return score
    
    def _evaluate_line(self, line):
        """
        Evaluate a line (row, column, or diagonal) for potential wins.
        
        Args:
            line: Array representing a line on the board
            
        Returns:
            Score for this line
        """
        our_pieces = np.count_nonzero(line == self.player)
        opponent_pieces = np.count_nonzero(line == (3 - self.player))
        empty_cells = np.count_nonzero(line == 0)
        
        # If we have a potential win (all our pieces or empty)
        if opponent_pieces == 0:
            if our_pieces == 2 and empty_cells == 1:  # Almost win
                return 10
            elif our_pieces == 1 and empty_cells == 2:  # Potential future win
                return 1
        
        # If opponent has a potential win (all their pieces or empty)
        if our_pieces == 0:
            if opponent_pieces == 2 and empty_cells == 1:  # Opponent almost win
                return -10
            elif opponent_pieces == 1 and empty_cells == 2:  # Opponent potential future win
                return -1
        
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
            return 100
        else:  # Opponent wins
            return -100
