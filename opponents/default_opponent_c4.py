import random
import numpy as np

from games.connect4 import Connect4

class DefaultOpponentC4:
    """A semi-intelligent agent for Connect 4 that:
    - Plays a winning move if available
    - Blocks opponent's winning move if possible
    - Prefers center columns (strategic advantage)
    - Otherwise plays randomly
    """
    def __init__(self, player_number=2):
        """Initialize the agent.
        
        Args:
            player_number: 1 for Red, 2 for Yellow (default is 2)
        """
        self.player_number = player_number
    
    def get_move(self, state):
        """Return the next move based on the current state.
        
        Args:
            state: Current game board state as numpy array
            
        Returns:
            Move as column index (0-6)
        """
        game = Connect4()
        game.board = state.copy()
        game.current_player = self.player_number
        
        # Get all legal moves
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None
        
        # Check for winning move
        for col in legal_moves:
            test_game = Connect4()
            test_game.board = state.copy()
            test_game.current_player = self.player_number
            test_game.make_move(col)
            if test_game.is_game_over() and test_game.get_winner() == self.player_number:
                return col
        
        # Check for blocking opponent's winning move
        opponent = 3 - self.player_number
        for col in legal_moves:
            test_game = Connect4()
            test_game.board = state.copy()
            test_game.current_player = opponent
            test_game.make_move(col)
            if test_game.is_game_over() and test_game.get_winner() == opponent:
                return col
        
        # Prefer middle columns
        # The closer to the middle, the higher the probability of being chosen
        weights = []
        middle = 3  # For a 7-column board, the middle is index 3
        for col in legal_moves:
            # Calculate weight based on distance from middle
            distance = abs(col - middle)
            weight = 7 - distance  # Higher weight for columns closer to middle
            weights.append(weight)
        
        # Normalize weights to probabilities
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Choose column based on weights
        return random.choices(legal_moves, weights=probabilities, k=1)[0]


if __name__ == "__main__":
    # Example of using the default opponent
    from games.connect4 import Connect4UI, PlayerType, GameMode
    
    # Create the game UI
    game_ui = Connect4UI()
    
    # Create the default opponent and set as player 2
    default_opponent = DefaultOpponentC4(player_number=2)
    game_ui.set_player2_agent(default_opponent)
    
    # Set the game mode to Human vs Semi-Intelligent
    game_ui.game_mode = GameMode.HUMAN_VS_SEMI
    game_ui.player1_type = PlayerType.HUMAN
    game_ui.player2_type = PlayerType.SEMI_INTELLIGENT
    
    # Run the game
    game_ui.run()
