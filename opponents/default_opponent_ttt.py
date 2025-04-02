import random
import numpy as np

from games.tic_tac_toe import TicTacToe

class DefaultOpponentTTT:
    """A semi-intelligent agent for Tic Tac Toe that:
    - Plays a winning move if available
    - Blocks opponent's winning move if possible
    - Otherwise plays randomly
    """
    def __init__(self, player_number=2):
        """Initialize the agent.
        
        Args:
            player_number: 1 for X, 2 for O (default is 2)
        """
        self.player_number = player_number
    
    def get_move(self, state):
        """Return the next move based on the current state.
        
        Args:
            state: Current game board state as numpy array
            
        Returns:
            Move as (row, col) tuple
        """
        game = TicTacToe()
        game.board = state.copy()
        game.current_player = self.player_number
        
        # Get all legal moves
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None
        
        # Check for winning move
        for move in legal_moves:
            test_game = TicTacToe()
            test_game.board = state.copy()
            test_game.current_player = self.player_number
            test_game.make_move(move)
            if test_game.is_game_over() and test_game.get_winner() == self.player_number:
                return move
        
        # Check for blocking opponent's winning move
        opponent = 3 - self.player_number
        for move in legal_moves:
            test_board = state.copy()
            test_board[move[0], move[1]] = opponent
            
            # Check if this move would make opponent win
            test_game = TicTacToe()
            test_game.board = test_board
            test_game._check_game_over()
            if test_game.is_game_over() and test_game.get_winner() == opponent:
                return move
        
        # If middle square is available, take it (strategic advantage)
        if (1, 1) in legal_moves:
            return (1, 1)
        
        # Prefer corners over sides (strategic advantage)
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        available_corners = [move for move in corners if move in legal_moves]
        if available_corners:
            return random.choice(available_corners)
        
        # Otherwise, play randomly
        return random.choice(legal_moves)


if __name__ == "__main__":
    # Example of using the default opponent
    from games.tic_tac_toe import TicTacToeUI, PlayerType, GameMode
    
    # Create the game UI
    game_ui = TicTacToeUI()
    
    # Create the default opponent and set as player 2
    default_opponent = DefaultOpponentTTT(player_number=2)
    game_ui.set_player2_agent(default_opponent)
    
    # Set the game mode to Human vs Semi-Intelligent
    game_ui.game_mode = GameMode.HUMAN_VS_SEMI
    game_ui.player1_type = PlayerType.HUMAN
    game_ui.player2_type = PlayerType.SEMI_INTELLIGENT
    
    # Run the game
    game_ui.run()
