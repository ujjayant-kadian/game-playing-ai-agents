import numpy as np
import pygame
import time
from enum import Enum

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0 for empty, 1 for X, 2 for O
        self.current_player = 1  # Player 1 (X) starts
        self.game_over = False
        self.winner = None

    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        """Return current state of the game."""
        return self.board.copy()
    
    def get_legal_moves(self):
        """Return list of legal moves as (row, col) tuples."""
        if self.game_over:
            return []
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]
    
    def make_move(self, move):
        """Make a move on the board.
        
        Args:
            move: tuple (row, col)
        
        Returns:
            bool: True if the move was valid, False otherwise
        """
        row, col = move
        if self.game_over or row < 0 or row > 2 or col < 0 or col > 2 or self.board[row, col] != 0:
            return False
        
        self.board[row, col] = self.current_player
        self._check_game_over()
        self.current_player = 3 - self.current_player  # Switch players (1 -> 2, 2 -> 1)
        return True
    
    def _check_game_over(self):
        """Check if the game is over (win or draw)."""
        # Check rows
        for row in range(3):
            if self.board[row, 0] != 0 and self.board[row, 0] == self.board[row, 1] == self.board[row, 2]:
                self.game_over = True
                self.winner = self.board[row, 0]
                return
        
        # Check columns
        for col in range(3):
            if self.board[0, col] != 0 and self.board[0, col] == self.board[1, col] == self.board[2, col]:
                self.game_over = True
                self.winner = self.board[0, col]
                return
        
        # Check diagonals
        if self.board[0, 0] != 0 and self.board[0, 0] == self.board[1, 1] == self.board[2, 2]:
            self.game_over = True
            self.winner = self.board[0, 0]
            return
        
        if self.board[0, 2] != 0 and self.board[0, 2] == self.board[1, 1] == self.board[2, 0]:
            self.game_over = True
            self.winner = self.board[0, 2]
            return
        
        # Check for draw
        if np.all(self.board != 0):
            self.game_over = True
            self.winner = 0  # Draw
            return
    
    def is_game_over(self):
        """Return whether the game is over."""
        return self.game_over
    
    def get_winner(self):
        """Return the winner (1 for X, 2 for O, 0 for draw, None if game not over)."""
        return self.winner

class PlayerType(Enum):
    HUMAN = 0
    AI = 1
    SEMI_INTELLIGENT = 2

class GameMode(Enum):
    HUMAN_VS_HUMAN = 0
    HUMAN_VS_AI = 1
    HUMAN_VS_SEMI = 2
    AI_VS_SEMI = 3
    AI_VS_AI = 4

class TicTacToeUI:
    def __init__(self):
        pygame.init()
        self.game = TicTacToe()
        self.width, self.height = 950, 950
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tic Tac Toe")
        
        # Colors
        self.bg_color = (240, 240, 240)
        self.line_color = (80, 80, 80)
        self.x_color = (66, 134, 244)
        self.o_color = (255, 87, 87)
        self.text_color = (50, 50, 50)
        self.highlight_color = (180, 180, 180)
        
        # Size and positions
        self.board_size = 450
        self.cell_size = self.board_size // 3
        self.board_margin = (self.width - self.board_size) // 2
        self.line_width = 4
        
        # Fonts
        self.font = pygame.font.SysFont('Arial', 30)
        self.big_font = pygame.font.SysFont('Arial', 50)
        
        # Game mode and players
        self.game_mode = GameMode.HUMAN_VS_HUMAN
        self.player1_type = PlayerType.HUMAN
        self.player2_type = PlayerType.HUMAN
        self.player1_agent = None  # AI agent for player 1
        self.player2_agent = None  # AI agent for player 2
        self.player_move = True  # True if current turn is for human input
        self.ai_move_delay = 0.5  # Delay between AI moves in seconds
        self.last_ai_move_time = 0  # Last time AI made a move
        
    def set_game_mode(self, mode):
        """Set the game mode and initialize appropriate players."""
        self.game_mode = mode
        
        if mode == GameMode.HUMAN_VS_HUMAN:
            self.player1_type = PlayerType.HUMAN
            self.player2_type = PlayerType.HUMAN
            self.player_move = True
            
        elif mode == GameMode.HUMAN_VS_AI:
            self.player1_type = PlayerType.HUMAN
            self.player2_type = PlayerType.AI
            self.player_move = True
            
        elif mode == GameMode.HUMAN_VS_SEMI:
            self.player1_type = PlayerType.HUMAN
            self.player2_type = PlayerType.SEMI_INTELLIGENT
            self.player_move = True
            
        elif mode == GameMode.AI_VS_SEMI:
            self.player1_type = PlayerType.AI
            self.player2_type = PlayerType.SEMI_INTELLIGENT
            self.player_move = False
            
        elif mode == GameMode.AI_VS_AI:
            # Both players are AI agents
            self.player1_type = PlayerType.AI
            self.player2_type = PlayerType.AI
            self.player_move = False
            
        # Reset the game
        self.game.reset()
        
    def set_player1_agent(self, agent):
        """Set AI agent for player 1."""
        self.player1_agent = agent
        
    def set_player2_agent(self, agent):
        """Set AI agent for player 2."""
        self.player2_agent = agent
        
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            current_time = time.time()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Human move handling
                    if self.player_move and not self.game.is_game_over():
                        self._handle_click(event.pos)
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset game
                        self.game.reset()
                        self.player_move = (self.player1_type == PlayerType.HUMAN)
                    
                    # Game mode selection keys
                    if event.key == pygame.K_1:
                        self.set_game_mode(GameMode.HUMAN_VS_HUMAN)
                    elif event.key == pygame.K_2:
                        self.set_game_mode(GameMode.HUMAN_VS_AI)
                    elif event.key == pygame.K_3:
                        self.set_game_mode(GameMode.HUMAN_VS_SEMI)
                    elif event.key == pygame.K_4:
                        self.set_game_mode(GameMode.AI_VS_SEMI)
                    elif event.key == pygame.K_5:
                        self.set_game_mode(GameMode.AI_VS_AI)
            
            # Handle AI agent moves
            if not self.game.is_game_over():
                current_player_num = self.game.current_player
                
                # Player 1's turn (X)
                if current_player_num == 1:
                    if self.player1_type in [PlayerType.AI, PlayerType.SEMI_INTELLIGENT] and (current_time - self.last_ai_move_time >= self.ai_move_delay):
                        agent = self.player1_agent
                        if agent and hasattr(agent, 'get_move'):
                            move = agent.get_move(self.game.get_state())
                            if move:
                                self.game.make_move(move)
                                self.last_ai_move_time = current_time
                                # After AI move, it could be human's turn
                                self.player_move = (self.player2_type == PlayerType.HUMAN)
                
                # Player 2's turn (O)
                else:
                    if self.player2_type in [PlayerType.AI, PlayerType.SEMI_INTELLIGENT] and (current_time - self.last_ai_move_time >= self.ai_move_delay):
                        agent = self.player2_agent
                        if agent and hasattr(agent, 'get_move'):
                            move = agent.get_move(self.game.get_state())
                            if move:
                                self.game.make_move(move)
                                self.last_ai_move_time = current_time
                                # After AI move, it could be human's turn
                                self.player_move = (self.player1_type == PlayerType.HUMAN)
            
            # Draw everything
            self._draw()
            pygame.display.flip()
            
            # Small delay to avoid high CPU usage
            time.sleep(0.01)
            
        pygame.quit()
        
    def _handle_click(self, pos):
        """Handle mouse click to make a move."""
        x, y = pos
        
        # Check if click is within the board
        if (self.board_margin <= x <= self.board_margin + self.board_size and 
            self.board_margin <= y <= self.board_margin + self.board_size):
            
            # Convert click position to grid indices
            row = (y - self.board_margin) // self.cell_size
            col = (x - self.board_margin) // self.cell_size
            
            # Make move if valid
            if self.game.make_move((row, col)):
                # After human move, determine who plays next
                current_player_num = self.game.current_player
                if current_player_num == 1:
                    self.player_move = (self.player1_type == PlayerType.HUMAN)
                else:
                    self.player_move = (self.player2_type == PlayerType.HUMAN)
        
    def _draw(self):
        """Draw the game board and UI."""
        # Fill background
        self.screen.fill(self.bg_color)
        
        # Draw title and status
        title = self.big_font.render("Tic Tac Toe", True, self.text_color)
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 20))
        
        # Display current game mode
        mode_names = {
            GameMode.HUMAN_VS_HUMAN: "Human vs Human",
            GameMode.HUMAN_VS_AI: "Human vs AI",
            GameMode.HUMAN_VS_SEMI: "Human vs Semi-Intelligent",
            GameMode.AI_VS_SEMI: "AI vs Semi-Intelligent",
            GameMode.AI_VS_AI: "AI vs AI"
        }
        mode_text = self.font.render(f"Mode: {mode_names[self.game_mode]}", True, self.text_color)
        self.screen.blit(mode_text, (self.width // 2 - mode_text.get_width() // 2, 80))
        
        # Draw instructions for changing modes
        mode_instructions = self.font.render("Press 1-5 to change mode", True, self.text_color)
        self.screen.blit(mode_instructions, (self.width // 2 - mode_instructions.get_width() // 2, 120))
        
        # Game status
        if self.game.is_game_over():
            if self.game.get_winner() == 1:
                status = self.font.render("X wins!", True, self.x_color)
            elif self.game.get_winner() == 2:
                status = self.font.render("O wins!", True, self.o_color)
            else:
                status = self.font.render("Draw!", True, self.text_color)
            
            restart = self.font.render("Press R to restart", True, self.text_color)
            self.screen.blit(restart, (self.width // 2 - restart.get_width() // 2, self.height - 50))
        else:
            current_player_text = "X's turn" if self.game.current_player == 1 else "O's turn"
            status = self.font.render(current_player_text, True, self.x_color if self.game.current_player == 1 else self.o_color)
        
        self.screen.blit(status, (self.width // 2 - status.get_width() // 2, self.height - 100))
        
        # Draw board
        for i in range(4):
            # Horizontal lines
            pygame.draw.line(
                self.screen, 
                self.line_color,
                (self.board_margin, self.board_margin + i * self.cell_size),
                (self.board_margin + self.board_size, self.board_margin + i * self.cell_size),
                self.line_width
            )
            
            # Vertical lines
            pygame.draw.line(
                self.screen,
                self.line_color,
                (self.board_margin + i * self.cell_size, self.board_margin),
                (self.board_margin + i * self.cell_size, self.board_margin + self.board_size),
                self.line_width
            )
        
        # Draw X's and O's
        for row in range(3):
            for col in range(3):
                center_x = self.board_margin + col * self.cell_size + self.cell_size // 2
                center_y = self.board_margin + row * self.cell_size + self.cell_size // 2
                
                if self.game.board[row, col] == 1:  # X
                    size = self.cell_size // 2 - 20
                    pygame.draw.line(
                        self.screen,
                        self.x_color,
                        (center_x - size, center_y - size),
                        (center_x + size, center_y + size),
                        self.line_width + 3
                    )
                    pygame.draw.line(
                        self.screen,
                        self.x_color,
                        (center_x - size, center_y + size),
                        (center_x + size, center_y - size),
                        self.line_width + 3
                    )
                    
                elif self.game.board[row, col] == 2:  # O
                    size = self.cell_size // 2 - 15
                    pygame.draw.circle(
                        self.screen,
                        self.o_color,
                        (center_x, center_y),
                        size,
                        self.line_width + 3
                    )

