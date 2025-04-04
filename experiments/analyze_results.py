import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set plot style
plt.style.use('ggplot')
sns.set_context("talk")

# Directory containing results
RESULTS_DIR = 'experiments/results'
OUTPUT_DIR = 'experiments/analysis/against-default-opponent'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_qlearning_training_rewards():
    """Plot episode vs reward for Q-learning training."""
    # Find all reward log files
    reward_files = glob.glob(os.path.join(RESULTS_DIR, '*qlearn_metrics_reward_log.csv'))
    
    if not reward_files:
        print("No Q-learning reward log files found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Group files by game type
    ttt_files = [f for f in reward_files if 'ttt_' in os.path.basename(f)]
    c4_files = [f for f in reward_files if 'c4_' in os.path.basename(f)]
    
    # Plot TTT training curves
    ax = axes[0]
    for file in ttt_files:
        try:
            base_name = os.path.basename(file).replace('_qlearn_metrics_reward_log.csv', '')
            df = pd.read_csv(file)
            # Check if the required columns exist
            if 'episode' in df.columns and 'total_reward' in df.columns:
                ax.plot(df['episode'], df['total_reward'], label=base_name)
            else:
                print(f"Warning: Required columns not found in {file}")
                print(f"Available columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    ax.set_title('Tic-Tac-Toe Q-Learning Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend(loc='best', fontsize='small')
    
    # Plot C4 training curves
    ax = axes[1]
    for file in c4_files:
        try:
            base_name = os.path.basename(file).replace('_qlearn_metrics_reward_log.csv', '')
            df = pd.read_csv(file)
            # Check if the required columns exist
            if 'episode' in df.columns and 'total_reward' in df.columns:
                ax.plot(df['episode'], df['total_reward'], label=base_name)
            else:
                print(f"Warning: Required columns not found in {file}")
                print(f"Available columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    ax.set_title('Connect-4 Q-Learning Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend(loc='best', fontsize='small')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'qlearning_training_rewards.png'), dpi=300)
    plt.close()
    
    print(f"Q-learning reward plots saved to {os.path.join(OUTPUT_DIR, 'qlearning_training_rewards.png')}")

def plot_qlearning_win_rates():
    """Plot win rate progression for Q-learning training."""
    # Find all eval log files
    eval_files = glob.glob(os.path.join(RESULTS_DIR, '*qlearn_metrics_eval_log.csv'))
    
    if not eval_files:
        print("No Q-learning evaluation log files found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Group files by game type
    ttt_files = [f for f in eval_files if 'ttt_' in os.path.basename(f)]
    c4_files = [f for f in eval_files if 'c4_' in os.path.basename(f)]
    
    # Plot TTT win rates
    ax = axes[0]
    for file in ttt_files:
        try:
            base_name = os.path.basename(file).replace('_qlearn_metrics_eval_log.csv', '')
            df = pd.read_csv(file)
            
            # Check if required columns exist
            if 'episode' in df.columns and 'wins' in df.columns and 'losses' in df.columns and 'draws' in df.columns:
                # Calculate win rate from wins, losses, draws
                total_games = df['wins'] + df['losses'] + df['draws']
                win_rate = df['wins'] / total_games
                ax.plot(df['episode'], win_rate, label=base_name)
            else:
                print(f"Warning: Required columns not found in {file}")
                print(f"Available columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    ax.set_title('Tic-Tac-Toe Q-Learning Win Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate')
    ax.legend(loc='best', fontsize='small')
    
    # Plot C4 win rates
    ax = axes[1]
    for file in c4_files:
        try:
            base_name = os.path.basename(file).replace('_qlearn_metrics_eval_log.csv', '')
            df = pd.read_csv(file)
            
            # Check if required columns exist
            if 'episode' in df.columns and 'wins' in df.columns and 'losses' in df.columns and 'draws' in df.columns:
                # Calculate win rate from wins, losses, draws
                total_games = df['wins'] + df['losses'] + df['draws']
                win_rate = df['wins'] / total_games
                ax.plot(df['episode'], win_rate, label=base_name)
            else:
                print(f"Warning: Required columns not found in {file}")
                print(f"Available columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    ax.set_title('Connect-4 Q-Learning Win Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate')
    ax.legend(loc='best', fontsize='small')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'qlearning_win_rates.png'), dpi=300)
    plt.close()
    
    print(f"Q-learning win rate plots saved to {os.path.join(OUTPUT_DIR, 'qlearning_win_rates.png')}")

def parse_summary_file(file_path):
    """Parse a summary.txt file to extract metrics."""
    data = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Extract agent type and configuration from filename
            filename = os.path.basename(file_path)
            data['filename'] = filename.replace('_summary.txt', '')
            
            # Game type
            data['game'] = 'Tic-Tac-Toe' if 'ttt_' in filename else 'Connect-4'
            
            # Extract agent type
            if 'minimax' in filename:
                data['agent_type'] = 'Minimax'
                if 'with_pruning' in filename:
                    data['pruning'] = 'With Pruning'
                else:
                    data['pruning'] = 'No Pruning'
                    
                if 'unlimited_depth' in filename:
                    data['depth'] = 'Unlimited'
                else:
                    depth_match = re.search(r'depth_(\d+)', filename)
                    if depth_match:
                        data['depth'] = depth_match.group(1)
            else:
                data['agent_type'] = 'Q-Learning'
                if 'with_q_table' in filename:
                    data['q_table'] = 'With Q-Table'
                else:
                    data['q_table'] = 'No Q-Table'
                    
                if 'trained' in filename:
                    training_match = re.search(r'trained_(\d+)ep', filename)
                    if training_match:
                        data['training'] = f"{training_match.group(1)} Episodes"
                else:
                    data['training'] = 'Untrained'
            
            # Player ordering
            if 'agent_first' in filename:
                data['player_order'] = 'Agent First'
            else:
                data['player_order'] = 'Agent Second'
            
            # Extract metrics from content
            avg_moves_match = re.search(r'Average Moves Per Game: ([\d\.]+)', content)
            if avg_moves_match:
                data['avg_moves'] = float(avg_moves_match.group(1))
                
            min_max_moves_match = re.search(r'Min/Max Moves in a Game: (\d+)/(\d+)', content)
            if min_max_moves_match:
                data['min_moves'] = int(min_max_moves_match.group(1))
                data['max_moves'] = int(min_max_moves_match.group(2))
                
            avg_time_match = re.search(r'Average Time Per Move: ([\d\.]+)', content)
            if avg_time_match:
                data['avg_time'] = float(avg_time_match.group(1))
                
            min_max_time_match = re.search(r'Min/Max Time for a Move: ([\d\.]+)/([\d\.]+)', content)
            if min_max_time_match:
                data['min_time'] = float(min_max_time_match.group(1))
                data['max_time'] = float(min_max_time_match.group(2))
                
            # For minimax, extract states explored
            if 'minimax' in filename:
                avg_states_match = re.search(r'Average States Explored Per Game: ([\d\.]+)', content)
                if avg_states_match:
                    data['avg_states'] = float(avg_states_match.group(1))
                    
                min_max_states_match = re.search(r'Min/Max States Explored in a Game: (\d+)/(\d+)', content)
                if min_max_states_match:
                    data['min_states'] = int(min_max_states_match.group(1))
                    data['max_states'] = int(min_max_states_match.group(2))
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None
    
    return data

def plot_moves_comparison():
    """Create bar plots comparing moves for different agent configurations."""
    # Find all summary files
    summary_files = glob.glob(os.path.join(RESULTS_DIR, '*_summary.txt'))
    
    if not summary_files:
        print("No summary files found")
        return
    
    # Parse all summary files
    data = [parse_summary_file(file) for file in summary_files]
    # Remove None values that might be from failed parsing
    data = [d for d in data if d is not None]
    
    if not data:
        print("No valid data parsed from summary files")
        return
    
    # Create dataframes for each game
    ttt_data = [d for d in data if d['game'] == 'Tic-Tac-Toe']
    c4_data = [d for d in data if d['game'] == 'Connect-4']
    
    # Plot for Tic-Tac-Toe
    if ttt_data:
        # Create a more informative label for each configuration
        for d in ttt_data:
            if d['agent_type'] == 'Minimax':
                d['config'] = f"Minimax\n{d.get('pruning', 'Unknown')}\nDepth {d.get('depth', 'Unknown')}"
            else:
                d['config'] = f"Q-Learning\n{d.get('q_table', 'Unknown')}\n{d.get('training', 'Unknown')}"
            
            # Include player order in config
            d['config'] += f"\n{d.get('player_order', 'Unknown')}"
        
        # Sort data for better visualization
        ttt_data.sort(key=lambda x: (x['agent_type'], x.get('config', '')))
        
        # Create plot
        plt.figure(figsize=(15, 10))
        
        # X positions
        configs = [d['config'] for d in ttt_data]
        x = np.arange(len(configs))
        width = 0.25
        
        # Plot bars
        plt.bar(x - width, [d.get('avg_moves', 0) for d in ttt_data], width, label='Avg Moves')
        plt.bar(x, [d.get('min_moves', 0) for d in ttt_data], width, label='Min Moves')
        plt.bar(x + width, [d.get('max_moves', 0) for d in ttt_data], width, label='Max Moves')
        
        # Add details
        plt.xlabel('Agent Configuration')
        plt.ylabel('Number of Moves')
        plt.title('Tic-Tac-Toe: Moves Comparison')
        plt.xticks(x, configs, rotation=90)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(OUTPUT_DIR, 'ttt_moves_comparison.png'), dpi=300)
        plt.close()
    
    # Plot for Connect-4
    if c4_data:
        # Create a more informative label for each configuration
        for d in c4_data:
            if d['agent_type'] == 'Minimax':
                d['config'] = f"Minimax\n{d.get('pruning', 'Unknown')}\nDepth {d.get('depth', 'Unknown')}"
            else:
                d['config'] = f"Q-Learning\n{d.get('q_table', 'Unknown')}\n{d.get('training', 'Unknown')}"
            
            # Include player order in config
            d['config'] += f"\n{d.get('player_order', 'Unknown')}"
        
        # Sort data for better visualization
        c4_data.sort(key=lambda x: (x['agent_type'], x.get('config', '')))
        
        # Create plot
        plt.figure(figsize=(15, 10))
        
        # X positions
        configs = [d['config'] for d in c4_data]
        x = np.arange(len(configs))
        width = 0.25
        
        # Plot bars
        plt.bar(x - width, [d.get('avg_moves', 0) for d in c4_data], width, label='Avg Moves')
        plt.bar(x, [d.get('min_moves', 0) for d in c4_data], width, label='Min Moves')
        plt.bar(x + width, [d.get('max_moves', 0) for d in c4_data], width, label='Max Moves')
        
        # Add details
        plt.xlabel('Agent Configuration')
        plt.ylabel('Number of Moves')
        plt.title('Connect-4: Moves Comparison')
        plt.xticks(x, configs, rotation=90)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(OUTPUT_DIR, 'c4_moves_comparison.png'), dpi=300)
        plt.close()
    
    print(f"Moves comparison plots saved to {OUTPUT_DIR}")

def plot_time_comparison():
    """Create bar plots comparing time for different agent configurations."""
    # Find all summary files
    summary_files = glob.glob(os.path.join(RESULTS_DIR, '*_summary.txt'))
    
    if not summary_files:
        print("No summary files found")
        return
    
    # Parse all summary files
    data = [parse_summary_file(file) for file in summary_files]
    # Remove None values that might be from failed parsing
    data = [d for d in data if d is not None]
    
    if not data:
        print("No valid data parsed from summary files")
        return
    
    # Create dataframes for each game
    ttt_data = [d for d in data if d['game'] == 'Tic-Tac-Toe']
    c4_data = [d for d in data if d['game'] == 'Connect-4']
    
    # Plot for Tic-Tac-Toe
    if ttt_data:
        # Create a more informative label for each configuration
        for d in ttt_data:
            if d['agent_type'] == 'Minimax':
                d['config'] = f"Minimax\n{d.get('pruning', 'Unknown')}\nDepth {d.get('depth', 'Unknown')}"
            else:
                d['config'] = f"Q-Learning\n{d.get('q_table', 'Unknown')}\n{d.get('training', 'Unknown')}"
            
            # Include player order in config
            d['config'] += f"\n{d.get('player_order', 'Unknown')}"
        
        # Sort data for better visualization
        ttt_data.sort(key=lambda x: (x['agent_type'], x.get('config', '')))
        
        # Create plot
        plt.figure(figsize=(15, 10))
        
        # X positions
        configs = [d['config'] for d in ttt_data]
        x = np.arange(len(configs))
        width = 0.25
        
        # Plot bars
        plt.bar(x - width, [d.get('avg_time', 0) for d in ttt_data], width, label='Avg Time (s)')
        plt.bar(x, [d.get('min_time', 0) for d in ttt_data], width, label='Min Time (s)')
        plt.bar(x + width, [d.get('max_time', 0) for d in ttt_data], width, label='Max Time (s)')
        
        # Add details
        plt.xlabel('Agent Configuration')
        plt.ylabel('Time (seconds)')
        plt.title('Tic-Tac-Toe: Time Comparison')
        plt.xticks(x, configs, rotation=90)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(OUTPUT_DIR, 'ttt_time_comparison.png'), dpi=300)
        plt.close()
    
    # Plot for Connect-4
    if c4_data:
        # Create a more informative label for each configuration
        for d in c4_data:
            if d['agent_type'] == 'Minimax':
                d['config'] = f"Minimax\n{d.get('pruning', 'Unknown')}\nDepth {d.get('depth', 'Unknown')}"
            else:
                d['config'] = f"Q-Learning\n{d.get('q_table', 'Unknown')}\n{d.get('training', 'Unknown')}"
            
            # Include player order in config
            d['config'] += f"\n{d.get('player_order', 'Unknown')}"
        
        # Sort data for better visualization
        c4_data.sort(key=lambda x: (x['agent_type'], x.get('config', '')))
        
        # Create plot
        plt.figure(figsize=(15, 10))
        
        # X positions
        configs = [d['config'] for d in c4_data]
        x = np.arange(len(configs))
        width = 0.25
        
        # Plot bars
        plt.bar(x - width, [d.get('avg_time', 0) for d in c4_data], width, label='Avg Time (s)')
        plt.bar(x, [d.get('min_time', 0) for d in c4_data], width, label='Min Time (s)')
        plt.bar(x + width, [d.get('max_time', 0) for d in c4_data], width, label='Max Time (s)')
        
        # Add details
        plt.xlabel('Agent Configuration')
        plt.ylabel('Time (seconds)')
        plt.title('Connect-4: Time Comparison')
        plt.xticks(x, configs, rotation=90)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(OUTPUT_DIR, 'c4_time_comparison.png'), dpi=300)
        plt.close()
    
    print(f"Time comparison plots saved to {OUTPUT_DIR}")

def plot_states_explored():
    """Create bar plots comparing states explored for minimax configurations."""
    # Find all summary files for minimax
    summary_files = glob.glob(os.path.join(RESULTS_DIR, '*minimax*_summary.txt'))
    
    if not summary_files:
        print("No minimax summary files found")
        return
    
    # Parse all summary files
    data = [parse_summary_file(file) for file in summary_files]
    # Remove None values that might be from failed parsing
    data = [d for d in data if d is not None]
    
    if not data:
        print("No valid data parsed from minimax summary files")
        return
    
    # Create dataframes for each game
    ttt_data = [d for d in data if d['game'] == 'Tic-Tac-Toe' and 'avg_states' in d]
    c4_data = [d for d in data if d['game'] == 'Connect-4' and 'avg_states' in d]
    
    # Plot for Tic-Tac-Toe
    if ttt_data:
        # Create a more informative label for each configuration
        for d in ttt_data:
            d['config'] = f"{d.get('pruning', 'Unknown')}\nDepth {d.get('depth', 'Unknown')}\n{d.get('player_order', 'Unknown')}"
        
        # Sort data for better visualization
        ttt_data.sort(key=lambda x: (x.get('pruning', ''), x.get('depth', ''), x.get('player_order', '')))
        
        # Create plot
        plt.figure(figsize=(15, 10))
        
        # X positions
        configs = [d['config'] for d in ttt_data]
        x = np.arange(len(configs))
        width = 0.25
        
        # Plot bars
        plt.bar(x - width, [d.get('avg_states', 0) for d in ttt_data], width, label='Avg States')
        plt.bar(x, [d.get('min_states', 0) for d in ttt_data], width, label='Min States')
        plt.bar(x + width, [d.get('max_states', 0) for d in ttt_data], width, label='Max States')
        
        # Add details
        plt.xlabel('Minimax Configuration')
        plt.ylabel('Number of States Explored')
        plt.title('Tic-Tac-Toe: States Explored by Minimax')
        plt.xticks(x, configs, rotation=90)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(OUTPUT_DIR, 'ttt_states_explored.png'), dpi=300)
        plt.close()
    else:
        print("No valid TTT minimax state exploration data found")
    
    # Plot for Connect-4
    if c4_data:
        # Create a more informative label for each configuration
        for d in c4_data:
            d['config'] = f"{d.get('pruning', 'Unknown')}\nDepth {d.get('depth', 'Unknown')}\n{d.get('player_order', 'Unknown')}"
        
        # Sort data for better visualization
        c4_data.sort(key=lambda x: (x.get('pruning', ''), x.get('depth', ''), x.get('player_order', '')))
        
        # Create plot
        plt.figure(figsize=(15, 10))
        
        # X positions
        configs = [d['config'] for d in c4_data]
        x = np.arange(len(configs))
        width = 0.25
        
        # Plot bars
        plt.bar(x - width, [d.get('avg_states', 0) for d in c4_data], width, label='Avg States')
        plt.bar(x, [d.get('min_states', 0) for d in c4_data], width, label='Min States')
        plt.bar(x + width, [d.get('max_states', 0) for d in c4_data], width, label='Max States')
        
        # Add details
        plt.xlabel('Minimax Configuration')
        plt.ylabel('Number of States Explored')
        plt.title('Connect-4: States Explored by Minimax')
        plt.xticks(x, configs, rotation=90)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(OUTPUT_DIR, 'c4_states_explored.png'), dpi=300)
        plt.close()
    else:
        print("No valid C4 minimax state exploration data found")
    
    print(f"States explored plots saved to {OUTPUT_DIR}")

def main():
    """Run all analyses."""
    print("Starting analysis...")
    
    # Create training plots
    plot_qlearning_training_rewards()
    plot_qlearning_win_rates()
    
    # Create comparison plots
    plot_moves_comparison()
    plot_time_comparison()
    plot_states_explored()
    
    print("Analysis complete. Results saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main() 