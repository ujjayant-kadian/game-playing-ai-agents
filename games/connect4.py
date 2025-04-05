import numpy as np
import pygame
import time
from enum import Enum
from games.tic_tac_toe import PlayerType, GameMode

class Connect4:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)  # 0 for empty, 1 and 2 for players
        self.current_player = 1  # Player 1 starts
        self.game_over = False
        self.winner = None
        self.last_move = None
    
    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.last_move = None
        return self.get_state()
    
    def get_state(self):
        """Return current state of the game."""
        return self.board.copy()
    
    def get_legal_moves(self):
        """Return list of legal moves (columns where a piece can be dropped)."""
        if self.game_over:
            return []
        return [col for col in range(self.cols) if self.board[0, col] == 0]
    
    def make_move(self, col):
        """Make a move by dropping a piece in the specified column.
        
        Args:
            col: Column to drop the piece
        
        Returns:
            bool: True if the move was valid, False otherwise
        """
        if self.game_over or col < 0 or col >= self.cols or self.board[0, col] != 0:
            return False
        
        # Find the lowest empty row in the selected column
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.last_move = (row, col)
                break
        
        self._check_game_over()
        self.current_player = 3 - self.current_player  # Switch players (1 -> 2, 2 -> 1)
        return True
    
    def _check_game_over(self):
        """Check if the game is over (win or draw)."""
        if self.last_move is None:
            return
        
        row, col = self.last_move
        player = self.board[row, col]
        
        # Check horizontal
        for c in range(max(0, col - 3), min(col + 1, self.cols - 3)):
            if self.board[row, c] == player and self.board[row, c+1] == player \
               and self.board[row, c+2] == player and self.board[row, c+3] == player:
                self.game_over = True
                self.winner = player
                return
        
        # Check vertical
        for r in range(max(0, row - 3), min(row + 1, self.rows - 3)):
            if self.board[r, col] == player and self.board[r+1, col] == player \
               and self.board[r+2, col] == player and self.board[r+3, col] == player:
                self.game_over = True
                self.winner = player
                return
        
        # Check diagonal (positive slope)
        for r, c in zip(range(row, max(row-4, -1), -1), range(col, max(col-4, -1), -1)):
            if r+3 < self.rows and c+3 < self.cols:
                if self.board[r, c] == player and self.board[r+1, c+1] == player \
                   and self.board[r+2, c+2] == player and self.board[r+3, c+3] == player:
                    self.game_over = True
                    self.winner = player
                    return
        
        # Check diagonal (negative slope)
        for r, c in zip(range(row, max(row-4, -1), -1), range(col, min(col+4, self.cols))):
            if r+3 < self.rows and c-3 >= 0:
                if self.board[r, c] == player and self.board[r+1, c-1] == player \
                   and self.board[r+2, c-2] == player and self.board[r+3, c-3] == player:
                    self.game_over = True
                    self.winner = player
                    return
        
        # Check for draw
        if np.all(self.board[0, :] != 0):
            self.game_over = True
            self.winner = 0  # Draw
            return
    
    def is_game_over(self):
        """Return whether the game is over."""
        return self.game_over
    
    def get_winner(self):
        """Return the winner (1 or 2 for players, 0 for draw, None if game not over)."""
        return self.winner

class Connect4UI:
    def __init__(self):
        pygame.init()
        self.game = Connect4()
        self.cell_size = 80
        self.width = self.game.cols * self.cell_size
        self.height = (self.game.rows + 1) * self.cell_size + 200  # Extra space for UI
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Connect 4")
        
        # Colors
        self.bg_color = (0, 105, 148)  # Blue background
        self.board_color = (0, 65, 118)
        self.player1_color = (255, 51, 51)  # Red
        self.player2_color = (255, 236, 51)  # Yellow
        self.text_color = (0, 0, 0)
        self.highlight_color = (0, 180, 215)
        
        # Fonts
        self.font = pygame.font.SysFont('Arial', 30)
        self.big_font = pygame.font.SysFont('Arial', 40)
        
        # Game mode and players
        self.game_mode = GameMode.HUMAN_VS_HUMAN
        self.player1_type = PlayerType.HUMAN
        self.player2_type = PlayerType.HUMAN
        self.player1_agent = None  # AI agent for player 1
        self.player2_agent = None  # AI agent for player 2
        self.player_move = True  # True if current turn is for human input
        self.ai_move_delay = 0.5  # Delay between AI moves in seconds
        self.last_ai_move_time = 0  # Last time AI made a move
        
        # Mode names dictionary for UI
        self.mode_names = {
            GameMode.HUMAN_VS_HUMAN: "Human vs Human",
            GameMode.HUMAN_VS_AI: "Human vs AI",
            GameMode.HUMAN_VS_SEMI: "Human vs Semi-Intelligent",
            GameMode.AI_VS_SEMI: "AI vs Semi-Intelligent",
            GameMode.AI_VS_AI: "AI vs AI"
        }
        
        # Animation
        self.anim_active = False
        self.anim_col = 0
        self.anim_row = 0
        self.anim_y = 0
        self.anim_speed = 15
        self.anim_player = 0
    
    def set_game_mode(self, mode):
        """Set the game mode and initialize appropriate players."""
        if isinstance(mode, int):
            mode = GameMode(mode)  # Convert int to GameMode Enum
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
        self.anim_active = False
        
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
                    if self.player_move and not self.game.is_game_over() and not self.anim_active:
                        self._handle_click(event.pos)
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset game
                        self.game.reset()
                        self.player_move = (self.player1_type == PlayerType.HUMAN)
                        self.anim_active = False
            
            # Handle AI and semi-intelligent agent moves
            if not self.game.is_game_over() and not self.anim_active:
                current_player_num = self.game.current_player
                
                # Player 1's turn (Red)
                if current_player_num == 1:
                    if self.player1_type in [PlayerType.AI, PlayerType.SEMI_INTELLIGENT] and (current_time - self.last_ai_move_time >= self.ai_move_delay):
                        agent = self.player1_agent
                        if agent and hasattr(agent, 'get_move'):
                            move = agent.get_move(self.game.get_state())
                            if move is not None:
                                self._start_animation(move)
                                self.last_ai_move_time = current_time
                
                # Player 2's turn (Yellow)
                else:
                    if self.player2_type in [PlayerType.AI, PlayerType.SEMI_INTELLIGENT] and (current_time - self.last_ai_move_time >= self.ai_move_delay):
                        agent = self.player2_agent
                        if agent and hasattr(agent, 'get_move'):
                            move = agent.get_move(self.game.get_state())
                            if move is not None:
                                self._start_animation(move)
                                self.last_ai_move_time = current_time
            
            # Update animation
            if self.anim_active:
                self._update_animation()
            
            # Draw everything
            self._draw()
            pygame.display.flip()
            
            # Small delay to avoid high CPU usage
            time.sleep(0.01)
            
        pygame.quit()
    
    def _start_animation(self, col):
        """Start animation for dropping a piece."""
        if col in self.game.get_legal_moves():
            self.anim_active = True
            self.anim_col = col
            self.anim_row = 0
            for row in range(self.game.rows - 1, -1, -1):
                if self.game.board[row, col] == 0:
                    self.anim_row = row
                    break
            self.anim_y = self.cell_size
            self.anim_player = self.game.current_player
    
    def _update_animation(self):
        """Update dropping animation."""
        target_y = (self.anim_row + 1) * self.cell_size
        
        if self.anim_y < target_y:
            self.anim_y += self.anim_speed
        else:
            self.anim_active = False
            self.game.make_move(self.anim_col)
            
            # After a piece is dropped, determine who plays next
            current_player = self.game.current_player
            if current_player == 1:
                self.player_move = (self.player1_type == PlayerType.HUMAN)
            else:
                self.player_move = (self.player2_type == PlayerType.HUMAN)
        
    def _handle_click(self, pos):
        """Handle mouse click to make a move."""
        x, y = pos
        
        # Check if click is within the valid area (above the board)
        if y < self.cell_size and 0 <= x < self.width:
            col = x // self.cell_size
            if col in self.game.get_legal_moves():
                self._start_animation(col)
        
    def _draw(self):
        """Draw the game board and UI."""
        # Fill background
        self.screen.fill(self.bg_color)
        
        # Draw title and game mode info
        if self.game.is_game_over():
            if self.game.get_winner() == 1:
                status = self.big_font.render("Red wins!", True, self.player1_color)
            elif self.game.get_winner() == 2:
                status = self.big_font.render("Yellow wins!", True, self.player2_color)
            else:
                status = self.big_font.render("Draw!", True, self.text_color)
            
            restart = self.font.render("Press R to restart", True, self.text_color)
            self.screen.blit(restart, (self.width // 2 - restart.get_width() // 2, 20))
        else:
            if self.game.current_player == 1:
                status = self.big_font.render("Red's turn", True, self.player1_color)
            else:
                status = self.big_font.render("Yellow's turn", True, self.player2_color)
        
        self.screen.blit(status, (self.width // 2 - status.get_width() // 2, self.height - 160))
        
        # Display current game mode
        mode_text = self.font.render(f"Mode: {self.mode_names[self.game_mode]}", True, self.text_color)
        self.screen.blit(mode_text, (30, self.height - 50))
        
        # Draw board background
        pygame.draw.rect(
            self.screen, 
            self.board_color, 
            (0, self.cell_size, self.width, self.game.rows * self.cell_size)
        )
        
        # Draw empty slots and pieces
        for row in range(self.game.rows):
            for col in range(self.game.cols):
                center_x = col * self.cell_size + self.cell_size // 2
                center_y = (row + 1) * self.cell_size + self.cell_size // 2
                
                # Draw empty slot
                pygame.draw.circle(
                    self.screen,
                    self.bg_color,
                    (center_x, center_y),
                    self.cell_size // 2 - 5
                )
                
                # Draw piece
                if self.game.board[row, col] == 1:
                    pygame.draw.circle(
                        self.screen,
                        self.player1_color,
                        (center_x, center_y),
                        self.cell_size // 2 - 5
                    )
                elif self.game.board[row, col] == 2:
                    pygame.draw.circle(
                        self.screen,
                        self.player2_color,
                        (center_x, center_y),
                        self.cell_size // 2 - 5
                    )
        
        # Draw animation
        if self.anim_active:
            center_x = self.anim_col * self.cell_size + self.cell_size // 2
            center_y = self.anim_y
            
            color = self.player1_color if self.anim_player == 1 else self.player2_color
            pygame.draw.circle(
                self.screen,
                color,
                (center_x, center_y),
                self.cell_size // 2 - 5
            )
        
        # Draw column highlight on hover (only when it's human's turn)
        if not self.game.is_game_over() and not self.anim_active and self.player_move:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if mouse_y < self.cell_size:
                col = mouse_x // self.cell_size
                if 0 <= col < self.game.cols and col in self.game.get_legal_moves():
                    pygame.draw.rect(
                        self.screen,
                        self.highlight_color,
                        (col * self.cell_size, 0, self.cell_size, self.cell_size),
                        3
                    )
                    
                    # Draw preview piece
                    color = self.player1_color if self.game.current_player == 1 else self.player2_color
                    pygame.draw.circle(
                        self.screen,
                        color,
                        (col * self.cell_size + self.cell_size // 2, self.cell_size // 2),
                        self.cell_size // 2 - 5
                    )

