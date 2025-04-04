import time
import csv
from collections import defaultdict
from pympler import asizeof

class MetricsManager:
    def __init__(self):
        self.reset_game_metrics()
        self.game_results = []
        self.q_learning_rewards = []
        self.q_learning_evals = []
        self.q_table = None

    # ───────────────────────────────
    # Game-level Metrics
    # ───────────────────────────────

    def reset_game_metrics(self):
        self.current_game = {
            'start_time': None,
            'end_time': None,
            'move_times': [],
            'moves': 0,
            'states_explored': 0,
            'outcome': None
        }

    def start_game(self):
        self.reset_game_metrics()
        self.current_game['start_time'] = time.time()

    def record_move(self, start_time, end_time):
        self.current_game['move_times'].append(end_time - start_time)
        self.current_game['moves'] += 1

    def record_state_explored(self):
        self.record_states_explored(1)
        
    def record_states_explored(self, count):
        self.current_game['states_explored'] += count

    def end_game(self, outcome):
        self.current_game['end_time'] = time.time()
        self.current_game['game_duration'] = self.current_game['end_time'] - self.current_game['start_time']
        self.current_game['outcome'] = outcome
        self.game_results.append(self.current_game.copy())

    def summarize_game_results(self):
        """
        Summarize game results and return metrics.
        
        Returns:
            dict: Dictionary containing all the metrics
        """
        total_games = len(self.game_results)
        if total_games == 0:
            print("No game data to summarize.")
            return {}

        wins = sum(1 for r in self.game_results if r['outcome'] == 'win')
        losses = sum(1 for r in self.game_results if r['outcome'] == 'loss')
        draws = total_games - wins - losses
        avg_moves = sum(r['moves'] for r in self.game_results) / total_games
        avg_time_per_move = sum(sum(r['move_times']) for r in self.game_results) / sum(r['moves'] for r in self.game_results) if sum(r['moves'] for r in self.game_results) > 0 else 0
        avg_game_duration = sum(r['game_duration'] for r in self.game_results) / total_games
        avg_states_explored = sum(r['states_explored'] for r in self.game_results) / total_games
        
        # Calculate max values
        max_moves = max(r['moves'] for r in self.game_results) if self.game_results else 0
        max_time_per_move = max(max(r['move_times']) if r['move_times'] else 0 for r in self.game_results) if self.game_results else 0
        max_game_duration = max(r['game_duration'] for r in self.game_results) if self.game_results else 0
        max_states_explored = max(r['states_explored'] for r in self.game_results) if self.game_results else 0
        
        # Calculate min values
        min_moves = min(r['moves'] for r in self.game_results) if self.game_results else 0
        min_time_per_move = min(min(r['move_times']) if r['move_times'] else float('inf') for r in self.game_results) if self.game_results else 0
        min_game_duration = min(r['game_duration'] for r in self.game_results) if self.game_results else 0
        min_states_explored = min(r['states_explored'] for r in self.game_results) if self.game_results else 0
        
        # Calculate total values
        total_moves = sum(r['moves'] for r in self.game_results)
        total_time = sum(sum(r['move_times']) for r in self.game_results)
        total_states_explored = sum(r['states_explored'] for r in self.game_results)

        # Create metrics dictionary
        metrics = {
            'total_games': total_games,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / total_games if total_games > 0 else 0,
            'loss_rate': losses / total_games if total_games > 0 else 0,
            'draw_rate': draws / total_games if total_games > 0 else 0,
            
            'avg_moves': avg_moves,
            'min_moves': min_moves,
            'max_moves': max_moves,
            'total_moves': total_moves,
            
            'avg_time_per_move': avg_time_per_move,
            'min_time_per_move': min_time_per_move,
            'max_time_per_move': max_time_per_move,
            'total_time': total_time,
            
            'avg_game_duration': avg_game_duration,
            'min_game_duration': min_game_duration,
            'max_game_duration': max_game_duration,
            
            'avg_states_explored': avg_states_explored,
            'min_states_explored': min_states_explored,
            'max_states_explored': max_states_explored,
            'total_states_explored': total_states_explored
        }

        # Print summary
        print("──── Game Summary ────")
        print(f"Total Games: {total_games}")
        print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
        print(f"Avg. Moves/Game: {avg_moves:.2f}")
        print(f"Avg. Time/Move: {avg_time_per_move:.4f} sec")
        print(f"Avg. Game Duration: {avg_game_duration:.2f} sec")
        print(f"Avg. States Explored (Minimax): {avg_states_explored:.0f}")
        print("──────────────────────")
        
        return metrics

    def save_game_log(self, filepath):
        if not self.game_results:
            return
        keys = self.game_results[0].keys()
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.game_results)

    # ───────────────────────────────
    # Q-Learning Metrics
    # ───────────────────────────────

    def set_q_table(self, q_table):
        self.q_table = q_table

    def get_q_table_memory(self):
        if self.q_table is None:
            return 0
        return asizeof.asizeof(self.q_table)

    def record_q_learning_reward(self, episode, total_reward):
        self.q_learning_rewards.append({'episode': episode, 'total_reward': total_reward})

    def record_q_learning_evaluation(self, episode, results):
        # results = ['win', 'loss', 'draw', ...]
        wins = results.count('win')
        losses = results.count('loss')
        draws = results.count('draw')
        self.q_learning_evals.append({
            'episode': episode,
            'wins': wins,
            'losses': losses,
            'draws': draws
        })

    def save_q_learning_logs(self, prefix="experiments/data/qlearning"):
        # Reward log
        if self.q_learning_rewards:
            with open(f"{prefix}_reward_log.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['episode', 'total_reward'])
                writer.writeheader()
                writer.writerows(self.q_learning_rewards)

        # Evaluation log
        if self.q_learning_evals:
            with open(f"{prefix}_eval_log.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['episode', 'wins', 'losses', 'draws'])
                writer.writeheader()
                writer.writerows(self.q_learning_evals)

    def print_q_table_memory(self):
        mem = self.get_q_table_memory()
        print(f"Q-table estimated memory: {mem / 1024:.2f} KB")

