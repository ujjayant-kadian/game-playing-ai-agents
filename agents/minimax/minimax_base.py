import time
import numpy as np
from abc import ABC, abstractmethod

class MinimaxBase(ABC):
    """
    Abstract base class for minimax algorithm implementations.
    Game-specific implementations should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, max_depth=float('inf'), metrics_manager=None, use_pruning=True):
        """
        Initialize the MinimaxBase with configurable parameters.
        
        Args:
            max_depth (int): Maximum depth for the minimax algorithm (default: infinity)
            metrics_manager: Optional metrics manager for tracking performance
            use_pruning (bool): Whether to use alpha-beta pruning (default: True)
        """
        self.max_depth = max_depth
        self.metrics_manager = metrics_manager
        self.player = None  # Will be set when get_move is called
        self.use_pruning = use_pruning
        self.states_explored = 0  # Counter for states explored
    
    def get_move(self, state):
        """
        Get the best move for the current state using minimax algorithm.
        
        Args:
            state: Current state of the game
            
        Returns:
            The best move according to minimax algorithm
        """
        self.player = self._get_current_player(state)
        self.states_explored = 0  # Reset counter
        
        if self.use_pruning:
            best_move = self._find_best_move_with_pruning(state)
        else:
            best_move = self._find_best_move_without_pruning(state)
            
        # Record states explored if metrics manager is available
        if self.metrics_manager:
            self.metrics_manager.record_states_explored(self.states_explored)
        
        return best_move
    
    def _find_best_move_with_pruning(self, state):
        """
        Find the best move using minimax algorithm with alpha-beta pruning.
        
        Args:
            state: Current state of the game
            
        Returns:
            The best move according to minimax algorithm
        """
        best_val = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        for move in self._get_legal_moves(state):
            # Make the move
            new_state = self._make_move(state.copy(), move, self.player)
            
            # Calculate value for this move
            move_val = self._minimax_with_pruning(new_state, self.max_depth - 1, False, alpha, beta)
            
            # Update best move if this is better
            if move_val > best_val:
                best_val = move_val
                best_move = move
            
            # Update alpha
            alpha = max(alpha, best_val)
        
        return best_move
    
    def _find_best_move_without_pruning(self, state):
        """
        Find the best move using minimax algorithm without alpha-beta pruning.
        
        Args:
            state: Current state of the game
            
        Returns:
            The best move according to minimax algorithm
        """
        best_val = float('-inf')
        best_move = None
        
        for move in self._get_legal_moves(state):
            # Make the move
            new_state = self._make_move(state.copy(), move, self.player)
            
            # Calculate value for this move
            move_val = self._minimax_without_pruning(new_state, self.max_depth - 1, False)
            
            # Update best move if this is better
            if move_val > best_val:
                best_val = move_val
                best_move = move
        
        return best_move
    
    def _minimax_with_pruning(self, state, depth, is_maximizing, alpha, beta):
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            state: Current state of the game
            depth (int): Current depth in the search tree
            is_maximizing (bool): True if current player is maximizing, False otherwise
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            
        Returns:
            The best score for the current state
        """
        # Increment state exploration counter
        self.states_explored += 1
        
        # Check if we've reached a terminal state
        winner = self._check_winner(state)
        if winner is not None:
            return self._evaluate_terminal(winner)
        
        # Check if we've reached maximum depth
        if depth == 0:
            return self._evaluate_board(state)
        
        # Get current player
        current_player = self._get_player(is_maximizing)
        
        # Maximizing player
        if is_maximizing:
            max_eval = float('-inf')
            for move in self._get_legal_moves(state):
                new_state = self._make_move(state.copy(), move, current_player)
                eval = self._minimax_with_pruning(new_state, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        
        # Minimizing player
        else:
            min_eval = float('inf')
            for move in self._get_legal_moves(state):
                new_state = self._make_move(state.copy(), move, current_player)
                eval = self._minimax_with_pruning(new_state, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval
    
    def _minimax_without_pruning(self, state, depth, is_maximizing):
        """
        Minimax algorithm without alpha-beta pruning.
        
        Args:
            state: Current state of the game
            depth (int): Current depth in the search tree
            is_maximizing (bool): True if current player is maximizing, False otherwise
            
        Returns:
            The best score for the current state
        """
        # Increment state exploration counter
        self.states_explored += 1
        
        # Check if we've reached a terminal state
        winner = self._check_winner(state)
        if winner is not None:
            return self._evaluate_terminal(winner)
        
        # Check if we've reached maximum depth
        if depth == 0:
            return self._evaluate_board(state)
        
        # Get current player
        current_player = self._get_player(is_maximizing)
        
        # Get legal moves
        legal_moves = self._get_legal_moves(state)
        
        # If no legal moves, it's a draw
        if not legal_moves:
            return 0
        
        # Maximizing player
        if is_maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                new_state = self._make_move(state.copy(), move, current_player)
                eval = self._minimax_without_pruning(new_state, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        
        # Minimizing player
        else:
            min_eval = float('inf')
            for move in legal_moves:
                new_state = self._make_move(state.copy(), move, current_player)
                eval = self._minimax_without_pruning(new_state, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval
    
    def _get_player(self, is_maximizing):
        """
        Get the player ID based on whether it's a maximizing or minimizing move.
        
        Args:
            is_maximizing (bool): True if it's a maximizing move
            
        Returns:
            The player ID
        """
        if is_maximizing:
            return self.player
        else:
            return 3 - self.player  # Switch between 1 and 2
    
    @abstractmethod
    def _get_current_player(self, state):
        """
        Get the current player from the state.
        
        Args:
            state: Current state of the game
            
        Returns:
            The current player ID
        """
        pass
    
    @abstractmethod
    def _get_legal_moves(self, state):
        """
        Get legal moves for the current state.
        
        Args:
            state: Current state of the game
            
        Returns:
            List of legal moves
        """
        pass
    
    @abstractmethod
    def _make_move(self, state, move, player):
        """
        Make a move on the board.
        
        Args:
            state: Current state of the game
            move: Move to make
            player: Player making the move
            
        Returns:
            New state after making the move
        """
        pass
    
    @abstractmethod
    def _check_winner(self, state):
        """
        Check if there's a winner in the current state.
        
        Args:
            state: Current state of the game
            
        Returns:
            The winner (1 or 2), 0 for draw, None if game is not over
        """
        pass
    
    @abstractmethod
    def _evaluate_board(self, state):
        """
        Evaluate the current board state.
        
        Args:
            state: Current state of the game
            
        Returns:
            Numerical score for the board state
        """
        pass
    
    @abstractmethod
    def _evaluate_terminal(self, winner):
        """
        Evaluate a terminal state.
        
        Args:
            winner: The winner (1 or 2), 0 for draw
            
        Returns:
            Numerical score for the terminal state
        """
        pass
