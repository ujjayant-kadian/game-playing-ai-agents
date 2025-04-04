import numpy as np
import random
import pickle
import os
from abc import ABC, abstractmethod

class QLearningAgent(ABC):
    def __init__(self, player_number=1, alpha=0.1, gamma=0.9, epsilon=0.1, 
                 epsilon_decay=0.999, epsilon_min=0.01, metrics_manager=None):
        """
        Initialize the Q-learning agent.
        
        Args:
            player_number: The player number (1 or 2)
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum exploration rate
            metrics_manager: Metrics manager for tracking stats
        """
        self.player_number = player_number
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.metrics_manager = metrics_manager
        
        # Initialize Q-table
        self.q_table = {}
        
        # Set rewards
        self.reward_win = 1.0
        self.reward_loss = -1.0
        self.reward_draw = 0.0
        self.reward_move = -0.01  # Small negative reward for each move to encourage faster winning
        
        # Training history
        self.training_stats = {
            'episode_rewards': [],
            'win_rate': [],
            'episode_lengths': []
        }
    
    def decay_epsilon(self):
        """Decay the exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair. Return 0 if not visited before."""
        state_key = self.state_to_key(state)
        if state_key in self.q_table and action in self.q_table[state_key]:
            return self.q_table[state_key][action]
        return 0.0
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value for state-action pair."""
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        # Initialize q_table entry if it doesn't exist
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # Get current Q-value
        current_q = self.get_q_value(state, action)
        
        # Get max Q-value for next state
        next_max_q = 0.0
        if next_state_key in self.q_table:
            next_q_values = self.q_table[next_state_key].values()
            if next_q_values:
                next_max_q = max(next_q_values)
        
        # Update rule: Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def choose_action(self, state, legal_moves, training=False):
        """Choose an action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            # Exploration: choose a random action
            return random.choice(legal_moves) if legal_moves else None
        
        # Exploitation: choose the best action
        best_action = None
        best_value = float('-inf')
        
        # Shuffle the legal moves to break ties randomly
        random.shuffle(legal_moves)
        
        for action in legal_moves:
            action_value = self.get_q_value(state, action)
            if action_value > best_value:
                best_value = action_value
                best_action = action
        
        return best_action
    
    def get_move(self, state):
        """Get a move for the current state (used during gameplay)."""
        legal_moves = self.get_legal_moves(state)
        if not legal_moves:
            return None
        return self.choose_action(state, legal_moves, training=False)
    
    def train(self, num_episodes=10000, eval_interval=500, eval_games=50, opponent=None):
        """Train the agent through self-play or against an opponent."""
        total_rewards = []
        
        # Check if metrics manager is available
        if self.metrics_manager:
            self.metrics_manager.set_q_table(self.q_table)
        
        for episode in range(1, num_episodes + 1):
            # Reset the game
            game = self.create_game()
            state = game.get_state()
            done = False
            episode_reward = 0
            moves = 0
            
            while not done:
                # Get legal moves
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    break
                
                # Choose action
                action = self.choose_action(state, legal_moves, training=True)
                
                # Make the move
                game.make_move(action)
                next_state = game.get_state()
                moves += 1
                
                # Check if game is over
                if game.is_game_over():
                    winner = game.get_winner()
                    if winner == self.player_number:
                        reward = self.reward_win
                    elif winner == 0:  # Draw
                        reward = self.reward_draw
                    else:  # Loss
                        reward = self.reward_loss
                    done = True
                else:
                    reward = self.reward_move
                    
                    # If playing against an opponent, let opponent make a move
                    if opponent and not done:
                        opponent_move = opponent.get_move(game.get_state())
                        if opponent_move:
                            game.make_move(opponent_move)
                            # Check if opponent won
                            if game.is_game_over():
                                winner = game.get_winner()
                                if winner == 3 - self.player_number:  # Opponent won
                                    reward = self.reward_loss
                                elif winner == 0:  # Draw
                                    reward = self.reward_draw
                                done = True
                            next_state = game.get_state()
                        else:
                            done = True
                
                # Update Q-values
                self.update_q_value(state, action, reward, next_state)
                
                # Update state
                state = next_state
                episode_reward += reward
            
            # Decay exploration rate
            self.decay_epsilon()
            
            # Record stats
            total_rewards.append(episode_reward)
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(moves)
            
            # Log with metrics manager
            if self.metrics_manager:
                self.metrics_manager.record_q_learning_reward(episode, episode_reward)
            
            # Periodically evaluate the agent
            if episode % eval_interval == 0:
                win_rate = self.evaluate(eval_games, opponent)
                self.training_stats['win_rate'].append((episode, win_rate))
                
                if self.metrics_manager:
                    self.metrics_manager.print_q_table_memory()
                
                print(f"Episode {episode}/{num_episodes}: Win rate: {win_rate:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}, "
                      f"Q-table size: {len(self.q_table)}")
        
        return self.training_stats
    
    def evaluate(self, num_games=100, opponent=None):
        """Evaluate the agent by playing against an opponent or randomly."""
        results = []
        
        for _ in range(num_games):
            game = self.create_game()
            done = False
            
            while not done:
                # Agent's turn
                if game.current_player == self.player_number:
                    action = self.get_move(game.get_state())
                    if action is None:
                        break
                    game.make_move(action)
                # Opponent's turn
                else:
                    if opponent:
                        opp_action = opponent.get_move(game.get_state())
                    else:
                        # Random opponent
                        legal_moves = game.get_legal_moves()
                        opp_action = random.choice(legal_moves) if legal_moves else None
                    
                    if opp_action is None:
                        break
                    game.make_move(opp_action)
                
                # Check if game is over
                if game.is_game_over():
                    winner = game.get_winner()
                    if winner == self.player_number:
                        results.append('win')
                    elif winner == 0:
                        results.append('draw')
                    else:
                        results.append('loss')
                    done = True
        
        # Record evaluation results with metrics manager
        if self.metrics_manager:
            self.metrics_manager.record_q_learning_evaluation(len(self.training_stats['episode_rewards']), results)
        
        win_rate = results.count('win') / len(results) if results else 0
        return win_rate
    
    def save(self, filepath):
        """Save the Q-table to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        
        # Also save training stats
        stats_filepath = f"{os.path.splitext(filepath)[0]}_stats.pkl"
        with open(stats_filepath, 'wb') as f:
            pickle.dump(self.training_stats, f)
    
    def load(self, filepath):
        """Load the Q-table from a file."""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        
        # Also try to load training stats
        stats_filepath = f"{os.path.splitext(filepath)[0]}_stats.pkl"
        if os.path.exists(stats_filepath):
            with open(stats_filepath, 'rb') as f:
                self.training_stats = pickle.load(f)
        
        # Update metrics manager
        if self.metrics_manager:
            self.metrics_manager.set_q_table(self.q_table)
    
    @abstractmethod
    def state_to_key(self, state):
        """Convert state to a key that can be used in the Q-table.
        Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_legal_moves(self, state):
        """Get legal moves for the given state.
        Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def create_game(self):
        """Create a new game instance.
        Must be implemented by subclasses."""
        pass
