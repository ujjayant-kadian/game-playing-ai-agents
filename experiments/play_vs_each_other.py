import argparse
import time
import numpy as np
import os
import signal
import json
from tqdm import tqdm

# Import games
from games import TicTacToe, Connect4

# Import agents
from agents import MinimaxTicTacToe, MinimaxConnect4
from agents.qlearning import QLearningTicTacToe, QLearningConnect4

# Import metrics
from utils import MetricsManager

# Define a timeout exception
class TimeoutException(Exception):
    pass

# Define timeout handler
def timeout_handler(signum, frame):
    raise TimeoutException("Move took too long")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate AI agents playing against each other')
    
    # Game selection
    parser.add_argument('--game', type=str, choices=['ttt', 'c4'], default='ttt',
                        help='Game to play: ttt (Tic-Tac-Toe) or c4 (Connect-4)')
    
    # Agent 1 configuration
    parser.add_argument('--agent1-type', type=str, choices=['minimax', 'qlearning'], default='minimax',
                        help='Type of agent 1 to use: minimax or qlearning (default: minimax)')
    
    # Agent 1 - Minimax specific configuration
    parser.add_argument('--agent1-max-depth', type=int, default=float('inf'),
                        help='Maximum depth for agent 1 minimax search (default: unlimited)')
    parser.add_argument('--agent1-use-pruning', action='store_true',
                        help='Use alpha-beta pruning for agent 1 minimax (default: False)')
    
    # Agent 1 - Q-learning specific configuration
    parser.add_argument('--agent1-q-table', type=str, default=None,
                        help='Path to a saved Q-table file for agent 1 Q-learning')
    parser.add_argument('--agent1-train', action='store_true',
                        help='Train agent 1 Q-learning before evaluation')
    parser.add_argument('--agent1-train-episodes', type=int, default=10000,
                        help='Number of episodes to train agent 1 (default: 10000)')
    parser.add_argument('--agent1-eval-interval', type=int, default=1000,
                        help='Evaluation interval during training for agent 1 (default: 1000)')
    parser.add_argument('--agent1-save-q-table', type=str, default=None,
                        help='Path to save agent 1 trained Q-table')
    
    # Agent 2 configuration
    parser.add_argument('--agent2-type', type=str, choices=['minimax', 'qlearning'], default='qlearning',
                        help='Type of agent 2 to use: minimax or qlearning (default: qlearning)')
    
    # Agent 2 - Minimax specific configuration
    parser.add_argument('--agent2-max-depth', type=int, default=float('inf'),
                        help='Maximum depth for agent 2 minimax search (default: unlimited)')
    parser.add_argument('--agent2-use-pruning', action='store_true',
                        help='Use alpha-beta pruning for agent 2 minimax (default: False)')
    
    # Agent 2 - Q-learning specific configuration
    parser.add_argument('--agent2-q-table', type=str, default=None,
                        help='Path to a saved Q-table file for agent 2 Q-learning')
    parser.add_argument('--agent2-train', action='store_true',
                        help='Train agent 2 Q-learning before evaluation')
    parser.add_argument('--agent2-train-episodes', type=int, default=10000,
                        help='Number of episodes to train agent 2 (default: 10000)')
    parser.add_argument('--agent2-eval-interval', type=int, default=1000,
                        help='Evaluation interval during training for agent 2 (default: 1000)')
    parser.add_argument('--agent2-save-q-table', type=str, default=None,
                        help='Path to save agent 2 trained Q-table')
    
    # Experiment configuration
    parser.add_argument('--num-games', type=int, default=10,
                        help='Number of games to play (default: 10)')
    parser.add_argument('--output-dir', type=str, default='experiments/against-each-other-results',
                        help='Directory to save experiment results (default: experiments/against-each-other-results)')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name (default: auto-generated from parameters)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show verbose output')
    parser.add_argument('--move-timeout', type=int, default=300,
                        help='Maximum seconds allowed for a single move (default: 300)')
    parser.add_argument('--force', action='store_true',
                        help='Force run experiment even if it might be very slow')
    
    args = parser.parse_args()
    
    # Auto-generate experiment name if not provided
    if args.experiment_name is None:
        agent1_str = f"{args.agent1_type}"
        agent2_str = f"{args.agent2_type}"
        
        if args.agent1_type == 'minimax':
            pruning_str = 'with_pruning' if args.agent1_use_pruning else 'no_pruning'
            depth_str = f'depth_{args.agent1_max_depth}' if args.agent1_max_depth != float('inf') else 'unlimited_depth'
            agent1_str = f"{agent1_str}_{pruning_str}_{depth_str}"
        else:  # qlearning
            q_table_str = 'with_q_table' if args.agent1_q_table else 'no_q_table'
            train_str = f'trained_{args.agent1_train_episodes}ep' if args.agent1_train else 'untrained'
            agent1_str = f"{agent1_str}_{q_table_str}_{train_str}"
            
        if args.agent2_type == 'minimax':
            pruning_str = 'with_pruning' if args.agent2_use_pruning else 'no_pruning'
            depth_str = f'depth_{args.agent2_max_depth}' if args.agent2_max_depth != float('inf') else 'unlimited_depth'
            agent2_str = f"{agent2_str}_{pruning_str}_{depth_str}"
        else:  # qlearning
            q_table_str = 'with_q_table' if args.agent2_q_table else 'no_q_table'
            train_str = f'trained_{args.agent2_train_episodes}ep' if args.agent2_train else 'untrained'
            agent2_str = f"{agent2_str}_{q_table_str}_{train_str}"
            
        args.experiment_name = f"{args.game}_agent1_{agent1_str}_vs_agent2_{agent2_str}"
    
    return args

def train_qlearning_agent(agent, agent_num, args, metrics, train_opponent=None):
    """Train a Q-learning agent."""
    # Extract appropriate args based on agent number
    train_episodes = getattr(args, f'agent{agent_num}_train_episodes')
    eval_interval = getattr(args, f'agent{agent_num}_eval_interval')
    save_q_table = getattr(args, f'agent{agent_num}_save_q_table')
    
    # Create output directory if it doesn't exist (for saving Q-table)
    if save_q_table:
        os.makedirs(os.path.dirname(os.path.abspath(save_q_table)), exist_ok=True)
    
    print(f"\n===== Training Q-Learning Agent {agent_num} =====")
    print(f"Game: {args.game}")
    print(f"Episodes: {train_episodes}")
    print(f"Agent plays as Player {agent.player_number}")
    print(f"Evaluation Interval: {eval_interval}")
    
    # Start timing for training
    train_start_time = time.time()
    
    # Train the agent
    training_stats = agent.train(
        num_episodes=train_episodes,
        eval_interval=eval_interval,
        eval_games=50,  # Fixed number of evaluation games during training
        opponent=train_opponent
    )
    
    train_end_time = time.time()
    train_duration = train_end_time - train_start_time
    
    print(f"\n===== Training Complete for Agent {agent_num} =====")
    print(f"Training duration: {train_duration:.1f} seconds")
    
    # Save the trained Q-table if requested
    if save_q_table:
        agent.save(save_q_table)
        print(f"Saved trained Q-table to {save_q_table}")
    
    # Print final training evaluation
    last_win_rate = training_stats['win_rate'][-1][1] if training_stats['win_rate'] else 0
    print(f"Final evaluation win rate: {last_win_rate:.2f}")
    
    # Update metrics with Q-table
    metrics.set_q_table(agent.q_table)
    
    # Print Q-table memory usage
    metrics.print_q_table_memory()
    
    print("\n")  # Add a blank line before evaluation
    
    return training_stats

def create_agent(game, agent_type, agent_args, metrics, player_number):
    """Create an agent of the specified type with the given arguments."""
    if game == 'ttt':
        if agent_type == 'minimax':
            agent = MinimaxTicTacToe(
                max_depth=agent_args.get('max_depth', float('inf')),
                use_pruning=agent_args.get('use_pruning', False),
                metrics_manager=metrics
            )
        else:  # qlearning
            agent = QLearningTicTacToe(metrics_manager=metrics)
            q_table_path = agent_args.get('q_table')
            if q_table_path:
                try:
                    agent.load(q_table_path)
                    print(f"Loaded Q-table from {q_table_path}")
                except Exception as e:
                    print(f"Error loading Q-table: {e}")
    else:  # Connect-4
        if agent_type == 'minimax':
            agent = MinimaxConnect4(
                max_depth=agent_args.get('max_depth', float('inf')),
                use_pruning=agent_args.get('use_pruning', False),
                metrics_manager=metrics
            )
        else:  # qlearning
            agent = QLearningConnect4(metrics_manager=metrics)
            q_table_path = agent_args.get('q_table')
            if q_table_path:
                try:
                    agent.load(q_table_path)
                    print(f"Loaded Q-table from {q_table_path}")
                except Exception as e:
                    print(f"Error loading Q-table: {e}")
    
    # Set player number
    agent.player_number = player_number
    
    return agent

def run_experiment(args):
    """Run the experiment with the given arguments."""
    # Safety check for potentially very slow experiments
    if (args.agent1_type == 'minimax' and args.game == 'c4' and 
            args.agent1_max_depth == float('inf') and not args.agent1_use_pruning and not args.force):
        print("WARNING: Running Connect-4 with Agent 1 as Minimax with unlimited depth and no pruning will be EXTREMELY slow.")
        print("Consider using a limited depth (--agent1-max-depth) or enabling pruning (--agent1-use-pruning).")
        print("To run anyway, use the --force flag.")
        return
    
    if (args.agent2_type == 'minimax' and args.game == 'c4' and 
            args.agent2_max_depth == float('inf') and not args.agent2_use_pruning and not args.force):
        print("WARNING: Running Connect-4 with Agent 2 as Minimax with unlimited depth and no pruning will be EXTREMELY slow.")
        print("Consider using a limited depth (--agent2-max-depth) or enabling pruning (--agent2-use-pruning).")
        print("To run anyway, use the --force flag.")
        return
    
    if (args.agent1_type == 'minimax' and args.game == 'c4' and 
            args.agent1_max_depth > 6 and not args.agent1_use_pruning and not args.force):
        print(f"WARNING: Running Connect-4 with Agent 1 as Minimax with depth {args.agent1_max_depth} and no pruning will be very slow.")
        print("Consider using a smaller depth or enabling pruning.")
        print("To run anyway, use the --force flag.")
        return
    
    if (args.agent2_type == 'minimax' and args.game == 'c4' and 
            args.agent2_max_depth > 6 and not args.agent2_use_pruning and not args.force):
        print(f"WARNING: Running Connect-4 with Agent 2 as Minimax with depth {args.agent2_max_depth} and no pruning will be very slow.")
        print("Consider using a smaller depth or enabling pruning.")
        print("To run anyway, use the --force flag.")
        return
    
    # Setup separate metrics managers for each agent
    metrics1 = MetricsManager()
    metrics2 = MetricsManager()
    
    # Create agent 1
    agent1_args = {
        'max_depth': args.agent1_max_depth,
        'use_pruning': args.agent1_use_pruning,
        'q_table': args.agent1_q_table
    }
    agent1 = create_agent(args.game, args.agent1_type, agent1_args, metrics1, player_number=1)
    
    # Create agent 2
    agent2_args = {
        'max_depth': args.agent2_max_depth,
        'use_pruning': args.agent2_use_pruning,
        'q_table': args.agent2_q_table
    }
    agent2 = create_agent(args.game, args.agent2_type, agent2_args, metrics2, player_number=2)
    
    # Train Q-learning agents if requested
    agent1_training_stats = None
    agent2_training_stats = None
    
    if args.agent1_type == 'qlearning' and args.agent1_train:
        agent1_training_stats = train_qlearning_agent(agent1, 1, args, metrics1, train_opponent=agent2)
    
    if args.agent2_type == 'qlearning' and args.agent2_train:
        agent2_training_stats = train_qlearning_agent(agent2, 2, args, metrics2, train_opponent=agent1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Game metrics for tracking overall game statistics
    game_metrics = MetricsManager()
    
    # Run games
    results = []
    timeouts = 0
    print(f"===== Evaluating Agents Against Each Other =====")
    print(f"Running {args.num_games} evaluation games...")
    for game_num in tqdm(range(args.num_games), desc="Playing games"):
        try:
            result = play_single_game(
                args.game, agent1, agent2, game_metrics, args.verbose, args.move_timeout
            )
            results.append(result)
            
            # Print result if verbose
            if args.verbose:
                print(f"Game {game_num+1} result: {result}")
                
        except TimeoutException:
            print(f"Game {game_num+1} timed out after {args.move_timeout} seconds")
            timeouts += 1
            # Record as a loss for the last player who was supposed to move
            # This is handled in the play_single_game function
            game_metrics.end_game('timeout')
    
    # Summarize results
    agent1_wins = results.count(1)
    agent2_wins = results.count(2)
    draws = results.count(0)
    
    print("\n===== Experiment Results =====")
    print(f"Game: {args.game}")
    
    # Display agent 1 configuration
    print(f"\nAgent 1 (Player 1): {args.agent1_type}")
    if args.agent1_type == 'minimax':
        print(f"  Max Depth: {args.agent1_max_depth}")
        print(f"  Alpha-Beta Pruning: {'Enabled' if args.agent1_use_pruning else 'Disabled'}")
    else:  # qlearning
        if args.agent1_train:
            q_table_info = f"Trained for {args.agent1_train_episodes} episodes"
            if args.agent1_save_q_table:
                q_table_info += f", saved to {args.agent1_save_q_table}"
        elif args.agent1_q_table:
            q_table_info = f"Loaded from {args.agent1_q_table}"
        else:
            q_table_info = "Default (untrained)"
        print(f"  Q-Table: {q_table_info}")
    
    # Display agent 2 configuration
    print(f"\nAgent 2 (Player 2): {args.agent2_type}")
    if args.agent2_type == 'minimax':
        print(f"  Max Depth: {args.agent2_max_depth}")
        print(f"  Alpha-Beta Pruning: {'Enabled' if args.agent2_use_pruning else 'Disabled'}")
    else:  # qlearning
        if args.agent2_train:
            q_table_info = f"Trained for {args.agent2_train_episodes} episodes"
            if args.agent2_save_q_table:
                q_table_info += f", saved to {args.agent2_save_q_table}"
        elif args.agent2_q_table:
            q_table_info = f"Loaded from {args.agent2_q_table}"
        else:
            q_table_info = "Default (untrained)"
        print(f"  Q-Table: {q_table_info}")
    
    print(f"\nGames Played: {args.num_games}")
    print(f"Agent 1 Wins: {agent1_wins} ({agent1_wins/args.num_games*100:.1f}%)")
    print(f"Agent 2 Wins: {agent2_wins} ({agent2_wins/args.num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/args.num_games*100:.1f}%)")
    if timeouts > 0:
        print(f"Timeouts: {timeouts} ({timeouts/args.num_games*100:.1f}%)")
    
    # Get detailed metrics
    overall_metrics = game_metrics.summarize_game_results()
    
    # Add experiment-specific information to metrics
    experiment_info = {
        'game': args.game,
        'agent1_type': args.agent1_type,
        'agent2_type': args.agent2_type,
        'num_games': args.num_games,
        'timeouts': timeouts
    }
    
    # Add agent-specific configuration to metrics
    if args.agent1_type == 'minimax':
        experiment_info.update({
            'agent1_max_depth': str(args.agent1_max_depth),
            'agent1_use_pruning': args.agent1_use_pruning
        })
    else:  # qlearning
        experiment_info.update({
            'agent1_q_table_path': args.agent1_q_table,
            'agent1_trained': args.agent1_train,
            'agent1_train_episodes': args.agent1_train_episodes if args.agent1_train else 0,
            'agent1_save_q_table_path': args.agent1_save_q_table
        })
    
    if args.agent2_type == 'minimax':
        experiment_info.update({
            'agent2_max_depth': str(args.agent2_max_depth),
            'agent2_use_pruning': args.agent2_use_pruning
        })
    else:  # qlearning
        experiment_info.update({
            'agent2_q_table_path': args.agent2_q_table,
            'agent2_trained': args.agent2_train,
            'agent2_train_episodes': args.agent2_train_episodes if args.agent2_train else 0,
            'agent2_save_q_table_path': args.agent2_save_q_table
        })
    
    # Combine all metrics
    all_metrics = {
        'experiment_info': experiment_info,
        'results': {
            'agent1_wins': agent1_wins,
            'agent2_wins': agent2_wins,
            'draws': draws,
            'agent1_win_rate': agent1_wins/args.num_games,
            'agent2_win_rate': agent2_wins/args.num_games,
            'draw_rate': draws/args.num_games,
        },
        'game_metrics': overall_metrics
    }
    
    # Add agent-specific metrics
    agent1_metrics = metrics1.summarize_game_results()
    agent2_metrics = metrics2.summarize_game_results()
    
    all_metrics['agent1_metrics'] = agent1_metrics
    all_metrics['agent2_metrics'] = agent2_metrics
    
    # Add Q-learning specific metrics if applicable
    if args.agent1_type == 'qlearning' and metrics1.q_table is not None:
        all_metrics['agent1_q_metrics'] = {
            'q_table_size': len(metrics1.q_table),
            'q_table_memory_kb': metrics1.get_q_table_memory() / 1024
        }
    
    if args.agent2_type == 'qlearning' and metrics2.q_table is not None:
        all_metrics['agent2_q_metrics'] = {
            'q_table_size': len(metrics2.q_table),
            'q_table_memory_kb': metrics2.get_q_table_memory() / 1024
        }
    
    # Add training stats if Q-learning was trained
    if args.agent1_type == 'qlearning' and args.agent1_train and agent1_training_stats:
        # Save only important training data points to avoid huge JSON files
        training_summary = {
            'episode_rewards': [agent1_training_stats['episode_rewards'][i] 
                               for i in range(0, len(agent1_training_stats['episode_rewards']), 
                                            max(1, args.agent1_train_episodes // 100))],  # Sample about 100 points
            'win_rate': agent1_training_stats['win_rate'],
            'episode_lengths': [agent1_training_stats['episode_lengths'][i]
                              for i in range(0, len(agent1_training_stats['episode_lengths']),
                                          max(1, args.agent1_train_episodes // 100))]  # Sample about 100 points
        }
        all_metrics['agent1_training_stats'] = training_summary
    
    if args.agent2_type == 'qlearning' and args.agent2_train and agent2_training_stats:
        # Save only important training data points to avoid huge JSON files
        training_summary = {
            'episode_rewards': [agent2_training_stats['episode_rewards'][i] 
                               for i in range(0, len(agent2_training_stats['episode_rewards']), 
                                            max(1, args.agent2_train_episodes // 100))],  # Sample about 100 points
            'win_rate': agent2_training_stats['win_rate'],
            'episode_lengths': [agent2_training_stats['episode_lengths'][i]
                              for i in range(0, len(agent2_training_stats['episode_lengths']),
                                          max(1, args.agent2_train_episodes // 100))]  # Sample about 100 points
        }
        all_metrics['agent2_training_stats'] = training_summary
    
    # Save raw metrics data as CSV
    metrics_file = os.path.join(args.output_dir, f"{args.experiment_name}_metrics.csv")
    game_metrics.save_game_log(metrics_file)
    print(f"Overall game metrics saved to {metrics_file}")
    
    # Save agent-specific metrics
    agent1_metrics_file = os.path.join(args.output_dir, f"{args.experiment_name}_agent1_metrics.csv")
    agent2_metrics_file = os.path.join(args.output_dir, f"{args.experiment_name}_agent2_metrics.csv")
    metrics1.save_game_log(agent1_metrics_file)
    metrics2.save_game_log(agent2_metrics_file)
    print(f"Agent 1 metrics saved to {agent1_metrics_file}")
    print(f"Agent 2 metrics saved to {agent2_metrics_file}")
    
    # Save Q-learning specific metrics if applicable
    if args.agent1_type == 'qlearning':
        qlearn_metrics_file = os.path.join(args.output_dir, f"{args.experiment_name}_agent1_qlearn_metrics")
        metrics1.save_q_learning_logs(qlearn_metrics_file)
        print(f"Agent 1 Q-learning metrics saved to {qlearn_metrics_file}_*.csv")
    
    if args.agent2_type == 'qlearning':
        qlearn_metrics_file = os.path.join(args.output_dir, f"{args.experiment_name}_agent2_qlearn_metrics")
        metrics2.save_q_learning_logs(qlearn_metrics_file)
        print(f"Agent 2 Q-learning metrics saved to {qlearn_metrics_file}_*.csv")
    
    # Save metrics data as JSON
    metrics_json_file = os.path.join(args.output_dir, f"{args.experiment_name}_metrics.json")
    with open(metrics_json_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Combined metrics JSON saved to {metrics_json_file}")
    
    # Save human-readable summary
    summary_file = os.path.join(args.output_dir, f"{args.experiment_name}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"===== Experiment Results =====\n")
        f.write(f"Game: {args.game}\n\n")
        
        # Write agent 1 configuration
        f.write(f"Agent 1 (Player 1): {args.agent1_type}\n")
        if args.agent1_type == 'minimax':
            f.write(f"  Max Depth: {args.agent1_max_depth}\n")
            f.write(f"  Alpha-Beta Pruning: {'Enabled' if args.agent1_use_pruning else 'Disabled'}\n")
        else:  # qlearning
            if args.agent1_train:
                q_table_info = f"Trained for {args.agent1_train_episodes} episodes"
                if args.agent1_save_q_table:
                    q_table_info += f", saved to {args.agent1_save_q_table}"
            elif args.agent1_q_table:
                q_table_info = f"Loaded from {args.agent1_q_table}"
            else:
                q_table_info = "Default (untrained)"
            f.write(f"  Q-Table: {q_table_info}\n")
        
        # Write agent 2 configuration
        f.write(f"\nAgent 2 (Player 2): {args.agent2_type}\n")
        if args.agent2_type == 'minimax':
            f.write(f"  Max Depth: {args.agent2_max_depth}\n")
            f.write(f"  Alpha-Beta Pruning: {'Enabled' if args.agent2_use_pruning else 'Disabled'}\n")
        else:  # qlearning
            if args.agent2_train:
                q_table_info = f"Trained for {args.agent2_train_episodes} episodes"
                if args.agent2_save_q_table:
                    q_table_info += f", saved to {args.agent2_save_q_table}"
            elif args.agent2_q_table:
                q_table_info = f"Loaded from {args.agent2_q_table}"
            else:
                q_table_info = "Default (untrained)"
            f.write(f"  Q-Table: {q_table_info}\n")
        
        f.write(f"\nGames Played: {args.num_games}\n")
        f.write(f"Agent 1 Wins: {agent1_wins} ({agent1_wins/args.num_games*100:.1f}%)\n")
        f.write(f"Agent 2 Wins: {agent2_wins} ({agent2_wins/args.num_games*100:.1f}%)\n")
        f.write(f"Draws: {draws} ({draws/args.num_games*100:.1f}%)\n")
        if timeouts > 0:
            f.write(f"Timeouts: {timeouts} ({timeouts/args.num_games*100:.1f}%)\n")
        
        # Add detailed metrics
        f.write("\n===== Detailed Game Metrics =====\n")
        
        # Agent 1 metrics
        f.write("\nAgent 1 Metrics:\n")
        f.write(f"Total Moves: {agent1_metrics['total_moves']}\n")
        f.write(f"Average Moves Per Game: {agent1_metrics['avg_moves']:.2f}\n")
        f.write(f"Min/Max Moves in a Game: {agent1_metrics['min_moves']}/{agent1_metrics['max_moves']}\n")
        f.write(f"Total Time: {agent1_metrics['total_time']:.2f} seconds\n")
        f.write(f"Average Time Per Move: {agent1_metrics['avg_time_per_move']:.4f} seconds\n")
        f.write(f"Min/Max Time for a Move: {agent1_metrics['min_time_per_move']:.4f}/{agent1_metrics['max_time_per_move']:.4f} seconds\n")
        
        # Add minimax-specific metrics for Agent 1 if applicable
        if args.agent1_type == 'minimax' and 'total_states_explored' in agent1_metrics:
            f.write(f"Total States Explored: {agent1_metrics['total_states_explored']}\n")
            f.write(f"Average States Explored Per Game: {agent1_metrics['avg_states_explored']:.0f}\n")
            f.write(f"Min/Max States Explored in a Game: {agent1_metrics['min_states_explored']}/{agent1_metrics['max_states_explored']}\n")
        f.write("\n")
        
        # Agent 2 metrics
        f.write("Agent 2 Metrics:\n")
        f.write(f"Total Moves: {agent2_metrics['total_moves']}\n")
        f.write(f"Average Moves Per Game: {agent2_metrics['avg_moves']:.2f}\n")
        f.write(f"Min/Max Moves in a Game: {agent2_metrics['min_moves']}/{agent2_metrics['max_moves']}\n")
        f.write(f"Total Time: {agent2_metrics['total_time']:.2f} seconds\n")
        f.write(f"Average Time Per Move: {agent2_metrics['avg_time_per_move']:.4f} seconds\n")
        f.write(f"Min/Max Time for a Move: {agent2_metrics['min_time_per_move']:.4f}/{agent2_metrics['max_time_per_move']:.4f} seconds\n")
        
        # Add minimax-specific metrics for Agent 2 if applicable
        if args.agent2_type == 'minimax' and 'total_states_explored' in agent2_metrics:
            f.write(f"Total States Explored: {agent2_metrics['total_states_explored']}\n")
            f.write(f"Average States Explored Per Game: {agent2_metrics['avg_states_explored']:.0f}\n")
            f.write(f"Min/Max States Explored in a Game: {agent2_metrics['min_states_explored']}/{agent2_metrics['max_states_explored']}\n")
        f.write("\n")
        
        # Overall game information
        f.write("Overall Game Information:\n")
        f.write(f"Average Game Duration: {overall_metrics['avg_game_duration']:.2f} seconds\n")
        f.write(f"Min/Max Game Duration: {overall_metrics['min_game_duration']:.2f}/{overall_metrics['max_game_duration']:.2f} seconds\n")
        
        # Add Q-learning specific metrics if applicable
        if args.agent1_type == 'qlearning' and args.agent1_train:
            f.write("\n===== Agent 1 Q-Learning Training Summary =====\n")
            f.write(f"Training Episodes: {args.agent1_train_episodes}\n")
            if agent1_training_stats and agent1_training_stats['win_rate']:
                final_win_rate = agent1_training_stats['win_rate'][-1][1]
                f.write(f"Final Evaluation Win Rate: {final_win_rate:.2f}\n")
            
            if metrics1.q_table:
                f.write(f"Q-table size: {len(metrics1.q_table)} states\n")
                mem_usage = metrics1.get_q_table_memory() / 1024
                f.write(f"Q-table memory usage: {mem_usage:.2f} KB\n")
                
        if args.agent2_type == 'qlearning' and args.agent2_train:
            f.write("\n===== Agent 2 Q-Learning Training Summary =====\n")
            f.write(f"Training Episodes: {args.agent2_train_episodes}\n")
            if agent2_training_stats and agent2_training_stats['win_rate']:
                final_win_rate = agent2_training_stats['win_rate'][-1][1]
                f.write(f"Final Evaluation Win Rate: {final_win_rate:.2f}\n")
            
            if metrics2.q_table:
                f.write(f"Q-table size: {len(metrics2.q_table)} states\n")
                mem_usage = metrics2.get_q_table_memory() / 1024
                f.write(f"Q-table memory usage: {mem_usage:.2f} KB\n")
        
    print(f"Summary saved to {summary_file}")

def play_single_game(game_name, agent1, agent2, game_metrics, verbose=False, move_timeout=300):
    """Play a single game between two agents and return the result."""
    # Create game instance based on name
    if game_name == 'ttt':
        from games import TicTacToe
        game = TicTacToe()
    else:  # Connect-4
        from games import Connect4
        game = Connect4()
    
    game_over = False
    
    # Start metrics for this game
    game_metrics.start_game()
    
    # Get the agent-specific metrics from the agents
    metrics1 = agent1.metrics_manager
    metrics2 = agent2.metrics_manager
    
    # Start game for agent-specific metrics
    metrics1.start_game()
    metrics2.start_game()
    
    # Track agent types for proper metrics logging
    agent1_type = type(agent1).__name__
    agent2_type = type(agent2).__name__
    
    move_count = 0
    while not game_over:
        move_count += 1
        state = game.get_state()
        current_player = game.current_player
        
        # Determine which agent to use
        current_agent = agent1 if current_player == 1 else agent2
        current_metrics = metrics1 if current_player == 1 else metrics2
        
        # Set up timeout
        if move_timeout > 0:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(move_timeout)
        
        try:
            # Get move with timing
            start_time = time.time()
            move = current_agent.get_move(state)
            end_time = time.time()
            
            # Disable alarm
            if move_timeout > 0:
                signal.alarm(0)
                
            # Record move metrics for both overall game and agent-specific
            game_metrics.record_move(start_time, end_time)
            current_metrics.record_move(start_time, end_time)
            
            if verbose:
                print(f"Player {current_player} ({type(current_agent).__name__}) took {end_time - start_time:.4f} seconds to move")
            
            # Make the move
            if move is not None:
                game.make_move(move)
            else:
                # No valid moves (should not happen in a well-formed game)
                if verbose:
                    print("Warning: Agent returned None as move")
                break
            
            # Check if game is over
            if game.is_game_over():
                winner = game.get_winner()
                if verbose:
                    if winner == 0:
                        print("Game ended in a draw")
                    else:
                        print(f"Player {winner} won")
                
                # Record outcome for overall game metrics
                if winner == 0:
                    outcome = 'draw'
                elif winner == 1:
                    outcome = 'win_agent1'
                else:
                    outcome = 'win_agent2'
                
                game_metrics.end_game(outcome)
                
                # Record outcome for agent-specific metrics
                if winner == 0:
                    metrics1.end_game('draw')
                    metrics2.end_game('draw')
                elif winner == 1:
                    metrics1.end_game('win')
                    metrics2.end_game('loss')
                else:  # winner == 2
                    metrics1.end_game('loss')
                    metrics2.end_game('win')
                
                return winner
                
        except TimeoutException:
            # Disable alarm
            if move_timeout > 0:
                signal.alarm(0)
            
            # Determine which agent timed out
            if current_player == 1:
                outcome = 'timeout_agent1'
                # Agent 1 timed out, so agent 2 wins
                winner = 2
                metrics1.end_game('timeout')
                metrics2.end_game('win')
            else:
                outcome = 'timeout_agent2'
                # Agent 2 timed out, so agent 1 wins
                winner = 1
                metrics1.end_game('win')
                metrics2.end_game('timeout')
                
            game_metrics.end_game(outcome)
            return winner
    
    # If we get here without a winner, it's a draw
    game_metrics.end_game('draw')
    metrics1.end_game('draw')
    metrics2.end_game('draw')
    return 0

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
