import os
import re
import pandas as pd

def parse_summary_file(filepath):
    with open(filepath, 'r') as f:
        text = f.read()

    data = {}

    # Extract file-based metadata
    data['filename'] = os.path.basename(filepath)
    if "ttt" in filepath:
        data['game'] = "Tic Tac Toe"
    elif "c4" in filepath:
        data['game'] = "Connect 4"
    else:
        data['game'] = "Unknown"

    # Determine if it's an agent vs agent or agent vs default opponent
    if "Agent 1" in text and "Agent 2" in text:
        # This is an agent vs agent file
        data['match_type'] = "agent_vs_agent"
        # Extract agents from file content
        agent1_match = re.search(r"Agent 1 \(Player 1\): (.+)", text)
        agent2_match = re.search(r"Agent 2 \(Player 2\): (.+)", text)
        
        if agent1_match and agent2_match:
            data['agent1_type'] = agent1_match.group(1).strip()
            data['agent2_type'] = agent2_match.group(1).strip()
            
            # Match outcomes
            data['agent1_wins'] = int(re.search(r"Agent 1 Wins: (\d+)", text).group(1))
            data['agent2_wins'] = int(re.search(r"Agent 2 Wins: (\d+)", text).group(1))
            data['draws'] = int(re.search(r"Draws: (\d+)", text).group(1))
            
            # Calculate total games played
            data['games_played'] = data['agent1_wins'] + data['agent2_wins'] + data['draws']
        
            # Moves
            agent1_moves = re.search(r"Agent 1 Metrics:\n.*?Average Moves Per Game: ([\d.]+)", text, re.DOTALL)
            agent2_moves = re.search(r"Agent 2 Metrics:\n.*?Average Moves Per Game: ([\d.]+)", text, re.DOTALL)
            data['agent1_avg_moves'] = float(agent1_moves.group(1)) if agent1_moves else None
            data['agent2_avg_moves'] = float(agent2_moves.group(1)) if agent2_moves else None
        
            # Time per move
            agent1_time = re.search(r"Agent 1 Metrics:\n.*?Average Time Per Move: ([\d.]+)", text, re.DOTALL)
            agent2_time = re.search(r"Agent 2 Metrics:\n.*?Average Time Per Move: ([\d.]+)", text, re.DOTALL)
            data['agent1_avg_time'] = float(agent1_time.group(1)) if agent1_time else None
            data['agent2_avg_time'] = float(agent2_time.group(1)) if agent2_time else None
    else:
        # This is an agent vs default opponent file
        data['match_type'] = "agent_vs_default"
        
        # Extract agent type and position
        agent_match = re.search(r"Agent: ([^,]+), (First|Second) Player", text)
        if agent_match:
            data['agent_type'] = agent_match.group(1).strip()
            is_first = agent_match.group(2) == "First"
            data['agent_player'] = "Player 1" if is_first else "Player 2"
            
            # Match outcomes
            wins_match = re.search(r"Agent Wins: (\d+) \(([\d.]+)%\)", text)
            losses_match = re.search(r"Agent Losses: (\d+) \(([\d.]+)%\)", text)
            draws_match = re.search(r"Draws: (\d+) \(([\d.]+)%\)", text)
            
            if wins_match and losses_match and draws_match:
                data['agent_wins'] = int(wins_match.group(1))
                data['agent_losses'] = int(losses_match.group(1))
                data['agent_draws'] = int(draws_match.group(1))
                data['agent_win_rate'] = float(wins_match.group(2)) / 100
                
                # Calculate or extract total games played
                games_played_match = re.search(r"Games Played: (\d+)", text)
                if games_played_match:
                    data['games_played'] = int(games_played_match.group(1))
                else:
                    # Calculate if not explicitly stated
                    data['games_played'] = data['agent_wins'] + data['agent_losses'] + data['agent_draws']
            
            # Move and time metrics
            moves_match = re.search(r"Average Moves Per Game: ([\d.]+)", text)
            time_match = re.search(r"Average Time Per Move: ([\d.]+)", text)
            
            if moves_match:
                data['avg_moves'] = float(moves_match.group(1))
            if time_match:
                data['avg_time_per_move'] = float(time_match.group(1))
                
            # Game duration
            duration_match = re.search(r"Average Game Duration: ([\d.]+)", text)
            if duration_match:
                data['avg_game_duration'] = float(duration_match.group(1))

    # States explored (if available)
    states_match = re.search(r"Total States Explored: (\d+)", text)
    data['states_explored'] = int(states_match.group(1)) if states_match else None

    # Game duration (if not already extracted)
    if 'avg_game_duration' not in data:
        duration_match = re.search(r"Average Game Duration: ([\d.]+)", text)
        data['avg_game_duration'] = float(duration_match.group(1)) if duration_match else None

    return data

def parse_all_summaries(directory):
    records = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and "summary" in filename:
            fullpath = os.path.join(directory, filename)
            try:
                records.append(parse_summary_file(fullpath))
            except Exception as e:
                print(f"Error parsing {filename}: {e}")
    return pd.DataFrame(records)

# Example usage
if __name__ == "__main__":
    # Get results from agent vs default opponents
    results_dir = "experiments/results"
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' does not exist.")
        exit(1)
    
    # List summary text files before parsing
    text_files = [f for f in os.listdir(results_dir) if f.endswith(".txt") and "summary" in f]
    if text_files:
        print(f"Found {len(text_files)} summary files in '{results_dir}':")
        for file in text_files:
            print(f"  - {file}")
    else:
        print(f"Warning: No summary files found in '{results_dir}'.")
    
    # Parse agent vs default opponent results
    vs_default_df = parse_all_summaries(results_dir)
    vs_default_df.to_csv("experiments/analysis/against-default-opponent/agent_summary_comparison.csv", index=False)
    print(f"Agent vs Default analysis complete. Found {len(vs_default_df)} records.")
    
    # Also parse agent vs agent results if the directory exists
    vs_agent_dir = "experiments/against-each-other-results"
    if os.path.exists(vs_agent_dir):
        # List agent vs agent summary files
        agent_text_files = [f for f in os.listdir(vs_agent_dir) if f.endswith(".txt") and "summary" in f]
        if agent_text_files:
            print(f"Found {len(agent_text_files)} summary files in '{vs_agent_dir}':")
            for file in agent_text_files:
                print(f"  - {file}")
                
            # Parse agent vs agent results
            vs_agent_df = parse_all_summaries(vs_agent_dir)
            
            # Create output directory if needed
            output_dir = "experiments/analysis/against-each-other"
            os.makedirs(output_dir, exist_ok=True)
            
            vs_agent_df.to_csv(f"{output_dir}/agent_summary_comparison.csv", index=False)
            print(f"Agent vs Agent analysis complete. Found {len(vs_agent_df)} records.")
    else:
        print(f"Directory '{vs_agent_dir}' does not exist. Skipping agent vs agent analysis.")
    
    print("Analysis complete!")
