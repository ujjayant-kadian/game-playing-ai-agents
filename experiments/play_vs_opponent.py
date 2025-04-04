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
from opponents import DefaultOpponentTTT, DefaultOpponentC4

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
    parser = argparse.ArgumentParser(description='Evaluate AI agents against default opponents')
    
    # Game selection
    parser.add_argument('--game', type=str, choices=['ttt', 'c4'], default='ttt',
                        help='Game to play: ttt (Tic-Tac-Toe) or c4 (Connect-4)')
    
    # Agent type selection
    parser.add_argument('--agent-type', type=str, choices=['minimax', 'qlearning'], default='minimax',
                        help='Type of agent to use: minimax or qlearning (default: minimax)')
    
    # Agent configuration - common
    parser.add_argument('--agent-first', action='store_true',
                        help='AI agent plays first (player 1)')
    
    # Minimax specific configuration
    parser.add_argument('--max-depth', type=int, default=float('inf'),
                        help='Maximum depth for minimax search (default: unlimited)')
    parser.add_argument('--use-pruning', action='store_true',
                        help='Use alpha-beta pruning for minimax (default: False)')
    
    # Q-learning specific configuration
    parser.add_argument('--q-table', type=str, default=None,
                        help='Path to a saved Q-table file for the Q-learning agent')
    parser.add_argument('--train', action='store_true',
                        help='Train the Q-learning agent before evaluation')
    parser.add_argument('--train-episodes', type=int, default=10000,
                        help='Number of episodes to train (default: 10000)')
    parser.add_argument('--eval-interval', type=int, default=1000,
                        help='Evaluation interval during training (default: 1000)')
    parser.add_argument('--save-q-table', type=str, default=None,
                        help='Path to save the trained Q-table')
    
    # Experiment configuration
    parser.add_argument('--num-games', type=int, default=10,
                        help='Number of games to play (default: 10)')
    parser.add_argument('--output-dir', type=str, default='experiments/results',
                        help='Directory to save experiment results (default: experiments/results)')
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
        if args.agent_type == 'minimax':
            pruning_str = 'with_pruning' if args.use_pruning else 'no_pruning'
            depth_str = f'depth_{args.max_depth}' if args.max_depth != float('inf') else 'unlimited_depth'
            order_str = 'agent_first' if args.agent_first else 'agent_second'
            args.experiment_name = f"{args.game}_{args.agent_type}_{pruning_str}_{depth_str}_{order_str}"
        else:  # qlearning
            q_table_str = 'with_q_table' if args.q_table else 'no_q_table'
            train_str = f'trained_{args.train_episodes}ep' if args.train else 'untrained'
            order_str = 'agent_first' if args.agent_first else 'agent_second'
            args.experiment_name = f"{args.game}_{args.agent_type}_{q_table_str}_{train_str}_{order_str}"
    
    return args

def run_experiment(args):
    """Run the experiment with the given arguments."""
    # Safety check for potentially very slow experiments
    if args.agent_type == 'minimax' and args.game == 'c4' and args.max_depth == float('inf') and not args.use_pruning and not args.force:
        print("WARNING: Running Connect-4 with unlimited depth and no pruning will be EXTREMELY slow.")
        print("Consider using a limited depth (--max-depth) or enabling pruning (--use-pruning).")
        print("To run anyway, use the --force flag.")
        return
    
    if args.agent_type == 'minimax' and args.game == 'c4' and args.max_depth > 6 and not args.use_pruning and not args.force:
        print(f"WARNING: Running Connect-4 with depth {args.max_depth} and no pruning will be very slow.")
        print("Consider using a smaller depth or enabling pruning.")
        print("To run anyway, use the --force flag.")
        return
    
    # Setup metrics manager
    metrics = MetricsManager()
    
    # Create game instance and agents based on agent type
    if args.game == 'ttt':
        game_class = TicTacToe
        if args.agent_type == 'minimax':
            ai_agent = MinimaxTicTacToe(max_depth=args.max_depth, use_pruning=args.use_pruning, metrics_manager=metrics)
        else:  # qlearning
            ai_agent = QLearningTicTacToe(metrics_manager=metrics)
            if args.q_table:
                try:
                    ai_agent.load(args.q_table)
                    print(f"Loaded Q-table from {args.q_table}")
                except Exception as e:
                    print(f"Error loading Q-table: {e}")
        default_agent = DefaultOpponentTTT()
    else:  # Connect-4
        game_class = Connect4
        if args.agent_type == 'minimax':
            ai_agent = MinimaxConnect4(max_depth=args.max_depth, use_pruning=args.use_pruning, metrics_manager=metrics)
        else:  # qlearning
            ai_agent = QLearningConnect4(metrics_manager=metrics)
            if args.q_table:
                try:
                    ai_agent.load(args.q_table)
                    print(f"Loaded Q-table from {args.q_table}")
                except Exception as e:
                    print(f"Error loading Q-table: {e}")
        default_agent = DefaultOpponentC4()
    
    # Set the player numbers based on who goes first
    if args.agent_first:
        ai_agent.player_number = 1
        default_agent.player_number = 2
    else:
        ai_agent.player_number = 2
        default_agent.player_number = 1
    
    # Train Q-learning agent if requested
    if args.agent_type == 'qlearning' and args.train:
        # Create output directory if it doesn't exist (for saving Q-table)
        if args.save_q_table:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_q_table)), exist_ok=True)
        
        print(f"\n===== Training Q-Learning Agent =====")
        print(f"Game: {args.game}")
        print(f"Episodes: {args.train_episodes}")
        print(f"Agent plays as Player {ai_agent.player_number}")
        print(f"Evaluation Interval: {args.eval_interval}")
        
        # Set up opponent for training
        train_opponent = default_agent  # Use the default opponent for training
        
        # Start timing for training
        train_start_time = time.time()
        
        # Train the agent
        training_stats = ai_agent.train(
            num_episodes=args.train_episodes,
            eval_interval=args.eval_interval,
            eval_games=50,  # Fixed number of evaluation games during training
            opponent=train_opponent
        )
        
        train_end_time = time.time()
        train_duration = train_end_time - train_start_time
        
        print(f"\n===== Training Complete =====")
        print(f"Training duration: {train_duration:.1f} seconds")
        
        # Save the trained Q-table if requested
        if args.save_q_table:
            ai_agent.save(args.save_q_table)
            print(f"Saved trained Q-table to {args.save_q_table}")
        
        # Print final training evaluation
        last_win_rate = training_stats['win_rate'][-1][1] if training_stats['win_rate'] else 0
        print(f"Final evaluation win rate: {last_win_rate:.2f}")
        
        # Update metrics with Q-table
        metrics.set_q_table(ai_agent.q_table)
        
        # Print Q-table memory usage
        metrics.print_q_table_memory()
        
        print("\n")  # Add a blank line before evaluation
    
    # Set up agents for evaluation
    if args.agent_first:
        agent1 = ai_agent
        agent2 = default_agent
    else:
        agent1 = default_agent
        agent2 = ai_agent
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run games
    results = []
    timeouts = 0
    print(f"===== Evaluating Agent =====")
    print(f"Running {args.num_games} evaluation games...")
    for game_num in tqdm(range(args.num_games), desc="Playing games"):
        try:
            result = play_single_game(game_class, agent1, agent2, metrics, args.verbose, args.move_timeout)
            results.append(result)
            
            # Print result if verbose
            if args.verbose:
                print(f"Game {game_num+1} result: {result}")
                
        except TimeoutException:
            print(f"Game {game_num+1} timed out after {args.move_timeout} seconds")
            timeouts += 1
            # Record as a loss for the player who timed out
            # We assume it was the AI agent that timed out
            results.append(3 - (1 if args.agent_first else 2))
            metrics.end_game('timeout')
    
    # Summarize results
    ai_player = 1 if args.agent_first else 2
    wins = results.count(ai_player)
    losses = results.count(3 - ai_player)
    draws = results.count(0)
    
    print("\n===== Experiment Results =====")
    print(f"Game: {args.game}")
    
    # Display appropriate configuration details based on agent type
    if args.agent_type == 'minimax':
        print(f"Agent: Minimax, {'First' if args.agent_first else 'Second'} Player, "
              f"Max Depth: {args.max_depth}, "
              f"{'With' if args.use_pruning else 'Without'} Alpha-Beta Pruning")
    else:  # qlearning
        # Determine Q-table source for display
        if args.train:
            q_table_info = f"Trained for {args.train_episodes} episodes"
            if args.save_q_table:
                q_table_info += f", saved to {args.save_q_table}"
        elif args.q_table:
            q_table_info = f"Loaded from {args.q_table}"
        else:
            q_table_info = "Default (untrained)"
            
        print(f"Agent: Q-Learning, {'First' if args.agent_first else 'Second'} Player, "
              f"Q-Table: {q_table_info}")
        
        # If we have a Q-table, print its size
        if metrics.q_table is not None:
            metrics.print_q_table_memory()
    
    print(f"Games Played: {args.num_games}")
    print(f"Agent Wins: {wins} ({wins/args.num_games*100:.1f}%)")
    print(f"Agent Losses: {losses} ({losses/args.num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/args.num_games*100:.1f}%)")
    if timeouts > 0:
        print(f"Timeouts: {timeouts} ({timeouts/args.num_games*100:.1f}%)")
    
    # Get detailed metrics
    game_metrics = metrics.summarize_game_results()
    
    # Add experiment-specific information to metrics
    experiment_info = {
        'game': args.game,
        'agent_type': args.agent_type,
        'agent_player': 'first' if args.agent_first else 'second',
        'num_games': args.num_games,
        'timeouts': timeouts
    }
    
    # Add agent-specific configuration to metrics
    if args.agent_type == 'minimax':
        experiment_info.update({
            'max_depth': str(args.max_depth),
            'use_pruning': args.use_pruning
        })
    else:  # qlearning
        experiment_info.update({
            'q_table_path': args.q_table,
            'trained': args.train,
            'train_episodes': args.train_episodes if args.train else 0,
            'save_q_table_path': args.save_q_table
        })
    
    # Combine all metrics
    all_metrics = {
        'experiment_info': experiment_info,
        'agent_results': {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins/args.num_games,
            'loss_rate': losses/args.num_games,
            'draw_rate': draws/args.num_games,
        }
    }
    
    # Add appropriate metrics based on agent type
    if args.agent_type == 'minimax':
        all_metrics['game_metrics'] = game_metrics
    elif args.agent_type == 'qlearning':
        # For Q-learning, we focus on different metrics
        q_metrics = {
            'total_moves': game_metrics['total_moves'],
            'avg_moves': game_metrics['avg_moves'],
            'min_moves': game_metrics['min_moves'],
            'max_moves': game_metrics['max_moves'],
            'total_time': game_metrics['total_time'],
            'avg_time_per_move': game_metrics['avg_time_per_move'],
            'min_time_per_move': game_metrics['min_time_per_move'],
            'max_time_per_move': game_metrics['max_time_per_move'],
            'avg_game_duration': game_metrics['avg_game_duration'],
            'min_game_duration': game_metrics['min_game_duration'],
            'max_game_duration': game_metrics['max_game_duration']
        }
        
        # Add Q-table metrics if available
        if metrics.q_table is not None:
            q_metrics['q_table_size'] = len(metrics.q_table)
            q_metrics['q_table_memory_kb'] = metrics.get_q_table_memory() / 1024
        
        all_metrics['game_metrics'] = q_metrics
    
    # Add training stats if Q-learning was trained
    if args.agent_type == 'qlearning' and args.train and hasattr(ai_agent, 'training_stats'):
        # Save only important training data points to avoid huge JSON files
        training_summary = {
            'episode_rewards': [ai_agent.training_stats['episode_rewards'][i] 
                               for i in range(0, len(ai_agent.training_stats['episode_rewards']), 
                                            max(1, args.train_episodes // 100))],  # Sample about 100 points
            'win_rate': ai_agent.training_stats['win_rate'],
            'episode_lengths': [ai_agent.training_stats['episode_lengths'][i]
                              for i in range(0, len(ai_agent.training_stats['episode_lengths']),
                                          max(1, args.train_episodes // 100))]  # Sample about 100 points
        }
        all_metrics['training_stats'] = training_summary
    
    # Save raw metrics data as CSV
    metrics_file = os.path.join(args.output_dir, f"{args.experiment_name}_metrics.csv")
    metrics.save_game_log(metrics_file)
    print(f"Metrics saved to {metrics_file}")
    
    # Save Q-learning specific metrics if applicable
    if args.agent_type == 'qlearning':
        qlearn_metrics_file = os.path.join(args.output_dir, f"{args.experiment_name}_qlearn_metrics")
        metrics.save_q_learning_logs(qlearn_metrics_file)
        print(f"Q-learning metrics saved to {qlearn_metrics_file}_*.csv")
    
    # Save metrics data as JSON
    metrics_json_file = os.path.join(args.output_dir, f"{args.experiment_name}_metrics.json")
    with open(metrics_json_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics JSON saved to {metrics_json_file}")
    
    # Save human-readable summary
    summary_file = os.path.join(args.output_dir, f"{args.experiment_name}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"===== Experiment Results =====\n")
        f.write(f"Game: {args.game}\n")
        
        # Write agent-specific details
        if args.agent_type == 'minimax':
            f.write(f"Agent: Minimax, {'First' if args.agent_first else 'Second'} Player, "
                    f"Max Depth: {args.max_depth}, "
                    f"{'With' if args.use_pruning else 'Without'} Alpha-Beta Pruning\n")
        else:  # qlearning
            if args.train:
                q_table_info = f"Trained for {args.train_episodes} episodes"
                if args.save_q_table:
                    q_table_info += f", saved to {args.save_q_table}"
            elif args.q_table:
                q_table_info = f"Loaded from {args.q_table}"
            else:
                q_table_info = "Default (untrained)"
                
            f.write(f"Agent: Q-Learning, {'First' if args.agent_first else 'Second'} Player, "
                    f"Q-Table: {q_table_info}\n")
        
        f.write(f"Games Played: {args.num_games}\n")
        f.write(f"Agent Wins: {wins} ({wins/args.num_games*100:.1f}%)\n")
        f.write(f"Agent Losses: {losses} ({losses/args.num_games*100:.1f}%)\n")
        f.write(f"Draws: {draws} ({draws/args.num_games*100:.1f}%)\n")
        if timeouts > 0:
            f.write(f"Timeouts: {timeouts} ({timeouts/args.num_games*100:.1f}%)\n")
        
        # Add detailed metrics
        f.write("\n===== Detailed Game Metrics =====\n")
        f.write(f"Total Moves: {game_metrics['total_moves']}\n")
        f.write(f"Average Moves Per Game: {game_metrics['avg_moves']:.2f}\n")
        f.write(f"Min/Max Moves in a Game: {game_metrics['min_moves']}/{game_metrics['max_moves']}\n\n")
        
        f.write(f"Total Time: {game_metrics['total_time']:.2f} seconds\n")
        f.write(f"Average Time Per Move: {game_metrics['avg_time_per_move']:.4f} seconds\n")
        f.write(f"Min/Max Time for a Move: {game_metrics['min_time_per_move']:.4f}/{game_metrics['max_time_per_move']:.4f} seconds\n\n")
        
        f.write(f"Average Game Duration: {game_metrics['avg_game_duration']:.2f} seconds\n")
        f.write(f"Min/Max Game Duration: {game_metrics['min_game_duration']:.2f}/{game_metrics['max_game_duration']:.2f} seconds\n\n")
        
        if args.agent_type == 'minimax':
            f.write(f"Total States Explored: {game_metrics['total_states_explored']}\n")
            f.write(f"Average States Explored Per Game: {game_metrics['avg_states_explored']:.0f}\n")
            f.write(f"Min/Max States Explored in a Game: {game_metrics['min_states_explored']}/{game_metrics['max_states_explored']}\n")
        
        # Add Q-learning specific metrics if applicable
        if args.agent_type == 'qlearning' and args.train:
            f.write("\n===== Q-Learning Training Summary =====\n")
            f.write(f"Training Episodes: {args.train_episodes}\n")
            if ai_agent.training_stats['win_rate']:
                final_win_rate = ai_agent.training_stats['win_rate'][-1][1]
                f.write(f"Final Evaluation Win Rate: {final_win_rate:.2f}\n")
            
            if metrics.q_table:
                f.write(f"Q-table size: {len(metrics.q_table)} states\n")
                mem_usage = metrics.get_q_table_memory() / 1024
                f.write(f"Q-table memory usage: {mem_usage:.2f} KB\n")
        
    print(f"Summary saved to {summary_file}")

def play_single_game(game_class, agent1, agent2, metrics, verbose=False, move_timeout=300):
    """Play a single game between two agents and return the result."""
    game = game_class()
    game_over = False
    
    # Start metrics for this game
    metrics.start_game()
    
    # Track agent types for proper metrics logging
    agent1_is_minimax = isinstance(agent1, (MinimaxTicTacToe, MinimaxConnect4))
    agent2_is_minimax = isinstance(agent2, (MinimaxTicTacToe, MinimaxConnect4))
    agent1_is_qlearning = isinstance(agent1, (QLearningTicTacToe, QLearningConnect4))
    agent2_is_qlearning = isinstance(agent2, (QLearningTicTacToe, QLearningConnect4))
    
    move_count = 0
    while not game_over:
        move_count += 1
        state = game.get_state()
        current_player = game.current_player
        
        # Determine which agent to use
        current_agent = agent1 if current_player == 1 else agent2
        
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
                
            # Record move metrics
            metrics.record_move(start_time, end_time)
            
            if verbose:
                print(f"Player {current_player} took {end_time - start_time:.4f} seconds to move")
            
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
                
                # Record outcome
                if winner == 0:
                    outcome = 'draw'
                # Check if AI agent won (player 1 is agent1, player 2 is agent2)
                elif (winner == 1 and (agent1_is_minimax or agent1_is_qlearning)) or \
                     (winner == 2 and (agent2_is_minimax or agent2_is_qlearning)):
                    outcome = 'win'
                else:
                    outcome = 'loss'
                
                metrics.end_game(outcome)
                return winner
                
        except TimeoutException:
            # Disable alarm
            if move_timeout > 0:
                signal.alarm(0)
            raise
    
    # If we get here without a winner, it's a draw
    metrics.end_game('draw')
    return 0

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
