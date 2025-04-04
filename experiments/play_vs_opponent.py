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
    parser = argparse.ArgumentParser(description='Evaluate minimax agents against default opponents')
    
    # Game selection
    parser.add_argument('--game', type=str, choices=['ttt', 'c4'], default='ttt',
                        help='Game to play: ttt (Tic-Tac-Toe) or c4 (Connect-4)')
    
    # Agent configuration
    parser.add_argument('--minimax-first', action='store_true',
                        help='Minimax agent plays first (player 1)')
    parser.add_argument('--max-depth', type=int, default=float('inf'),
                        help='Maximum depth for minimax search (default: unlimited)')
    parser.add_argument('--use-pruning', action='store_true',
                        help='Use alpha-beta pruning (default: False)')
    
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
        pruning_str = 'with_pruning' if args.use_pruning else 'no_pruning'
        depth_str = f'depth_{args.max_depth}' if args.max_depth != float('inf') else 'unlimited_depth'
        order_str = 'minimax_first' if args.minimax_first else 'minimax_second'
        args.experiment_name = f"{args.game}_{pruning_str}_{depth_str}_{order_str}"
    
    return args

def run_experiment(args):
    """Run the experiment with the given arguments."""
    # Safety check for potentially very slow experiments
    if args.game == 'c4' and args.max_depth == float('inf') and not args.use_pruning and not args.force:
        print("WARNING: Running Connect-4 with unlimited depth and no pruning will be EXTREMELY slow.")
        print("Consider using a limited depth (--max-depth) or enabling pruning (--use-pruning).")
        print("To run anyway, use the --force flag.")
        return
    
    if args.game == 'c4' and args.max_depth > 6 and not args.use_pruning and not args.force:
        print(f"WARNING: Running Connect-4 with depth {args.max_depth} and no pruning will be very slow.")
        print("Consider using a smaller depth or enabling pruning.")
        print("To run anyway, use the --force flag.")
        return
    
    # Setup metrics manager
    metrics = MetricsManager()
    
    # Create game instance
    if args.game == 'ttt':
        game_class = TicTacToe
        minimax_agent = MinimaxTicTacToe(max_depth=args.max_depth, use_pruning=args.use_pruning, metrics_manager=metrics)
        default_agent = DefaultOpponentTTT()
    else:  # Connect-4
        game_class = Connect4
        minimax_agent = MinimaxConnect4(max_depth=args.max_depth, use_pruning=args.use_pruning, metrics_manager=metrics)
        default_agent = DefaultOpponentC4()
    
    # Determine player numbers
    if args.minimax_first:
        agent1 = minimax_agent
        agent2 = default_agent
        default_agent.player_number = 2  # Ensure correct player number
    else:
        agent1 = default_agent
        agent2 = minimax_agent
        default_agent.player_number = 1  # Ensure correct player number
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run games
    results = []
    timeouts = 0
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
            # We assume it was the minimax agent that timed out
            results.append(3 - (1 if args.minimax_first else 2))
            metrics.end_game('timeout')
    
    # Summarize results
    minimax_player = 1 if args.minimax_first else 2
    wins = results.count(minimax_player)
    losses = results.count(3 - minimax_player)
    draws = results.count(0)
    
    print("\n===== Experiment Results =====")
    print(f"Game: {args.game}")
    print(f"Minimax Config: {'First' if args.minimax_first else 'Second'} Player, "
          f"Max Depth: {args.max_depth}, "
          f"{'With' if args.use_pruning else 'Without'} Alpha-Beta Pruning")
    print(f"Games Played: {args.num_games}")
    print(f"Minimax Wins: {wins} ({wins/args.num_games*100:.1f}%)")
    print(f"Minimax Losses: {losses} ({losses/args.num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/args.num_games*100:.1f}%)")
    if timeouts > 0:
        print(f"Timeouts: {timeouts} ({timeouts/args.num_games*100:.1f}%)")
    
    # Get detailed metrics
    game_metrics = metrics.summarize_game_results()
    
    # Add experiment-specific information to metrics
    experiment_info = {
        'game': args.game,
        'minimax_player': 'first' if args.minimax_first else 'second',
        'max_depth': str(args.max_depth),
        'use_pruning': args.use_pruning,
        'num_games': args.num_games,
        'timeouts': timeouts
    }
    
    # Combine all metrics
    all_metrics = {
        'experiment_info': experiment_info,
        'minimax_results': {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins/args.num_games,
            'loss_rate': losses/args.num_games,
            'draw_rate': draws/args.num_games,
        },
        'game_metrics': game_metrics
    }
    
    # Save raw metrics data as JSON
    metrics_file = os.path.join(args.output_dir, f"{args.experiment_name}_metrics.csv")
    metrics.save_game_log(metrics_file)
    print(f"Metrics saved to {metrics_file}")
    
    metrics_json_file = os.path.join(args.output_dir, f"{args.experiment_name}_metrics.json")
    with open(metrics_json_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics JSON saved to {metrics_json_file}")
    
    # Save human-readable summary
    summary_file = os.path.join(args.output_dir, f"{args.experiment_name}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"===== Experiment Results =====\n")
        f.write(f"Game: {args.game}\n")
        f.write(f"Minimax Config: {'First' if args.minimax_first else 'Second'} Player, "
                f"Max Depth: {args.max_depth}, "
                f"{'With' if args.use_pruning else 'Without'} Alpha-Beta Pruning\n")
        f.write(f"Games Played: {args.num_games}\n")
        f.write(f"Minimax Wins: {wins} ({wins/args.num_games*100:.1f}%)\n")
        f.write(f"Minimax Losses: {losses} ({losses/args.num_games*100:.1f}%)\n")
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
        
        f.write(f"Total States Explored: {game_metrics['total_states_explored']}\n")
        f.write(f"Average States Explored Per Game: {game_metrics['avg_states_explored']:.0f}\n")
        f.write(f"Min/Max States Explored in a Game: {game_metrics['min_states_explored']}/{game_metrics['max_states_explored']}\n")
        
    print(f"Summary saved to {summary_file}")

def play_single_game(game_class, agent1, agent2, metrics, verbose=False, move_timeout=300):
    """Play a single game between two agents and return the result."""
    game = game_class()
    game_over = False
    
    # Start metrics for this game
    metrics.start_game()
    
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
                elif (winner == 1 and isinstance(agent1, (MinimaxTicTacToe, MinimaxConnect4))) or \
                     (winner == 2 and isinstance(agent2, (MinimaxTicTacToe, MinimaxConnect4))):
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
