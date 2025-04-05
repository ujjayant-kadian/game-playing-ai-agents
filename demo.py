#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import datetime
import time

def create_output_dir():
    """Create the output directory if it doesn't exist."""
    output_dir = "demo-outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def get_user_script_choice():
    """Ask the user which script they want to run."""
    print("\n=== Game AI Demo Script ===")
    print("1. Play vs Opponent (AI agent vs. default opponent)")
    print("2. Play vs Each Other (AI agent vs. AI agent)")
    print("3. Visualize Game (with UI)")
    
    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ")
        if choice in ["1", "2", "3"]:
            return int(choice)
        print("Invalid choice. Please enter 1, 2, or 3.")

def get_game_choice():
    """Ask the user which game they want to play."""
    print("\n=== Select Game ===")
    print("1. Tic-Tac-Toe (ttt)")
    print("2. Connect-4 (c4)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ")
        if choice == "1":
            return "ttt"
        elif choice == "2":
            return "c4"
        print("Invalid choice. Please enter 1 or 2.")

def get_agent_type(agent_num=None):
    """Ask the user which agent type they want to use."""
    if agent_num:
        print(f"\n=== Select Agent {agent_num} Type ===")
    else:
        print("\n=== Select Agent Type ===")
    print("1. Minimax")
    print("2. Q-Learning")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ")
        if choice == "1":
            return "minimax"
        elif choice == "2":
            return "qlearning"
        print("Invalid choice. Please enter 1 or 2.")

def get_agent_first():
    """Ask if the AI agent should go first."""
    print("\n=== Should the AI agent play first? ===")
    print("1. Yes (agent is player 1)")
    print("2. No (agent is player 2)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ")
        if choice == "1":
            return True
        elif choice == "2":
            return False
        print("Invalid choice. Please enter 1 or 2.")

def get_minimax_config(agent_num=None):
    """Get minimax agent configuration from user."""
    prefix = f"agent{agent_num}-" if agent_num else ""
    config = {}
    
    # Get pruning option
    print("\n=== Alpha-Beta Pruning ===")
    print("1. Enable pruning")
    print("2. Disable pruning")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ")
        if choice == "1":
            config[f"--{prefix}use-pruning"] = []
            break
        elif choice == "2":
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    # Get depth option
    print("\n=== Search Depth ===")
    print("1. Unlimited depth (may be very slow for Connect-4)")
    print("2. Specify a depth limit")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ")
        if choice == "1":
            # Use default (unlimited)
            break
        elif choice == "2":
            while True:
                depth = input("Enter depth limit (recommended: 3-6 for Connect-4, 5-9 for Tic-Tac-Toe): ")
                if depth.isdigit() and int(depth) > 0:
                    config[f"--{prefix}max-depth"] = [int(depth)]
                    break
                print("Invalid depth. Please enter a positive integer.")
            break
        print("Invalid choice. Please enter 1 or 2.")
        
    return config

def get_qlearning_config(agent_num=None):
    """Get Q-learning agent configuration from user."""
    prefix = f"agent{agent_num}-" if agent_num else ""
    config = {}
    
    # Ask about using a pre-trained Q-table
    print("\n=== Q-Learning Configuration ===")
    print("1. Use a pre-trained Q-table")
    print("2. Train a new Q-table")
    print("3. Use default (untrained)")
    
    choice = input("\nEnter your choice (1, 2, or 3): ")
    
    if choice == "1":
        while True:
            q_table_path = input("Enter path to Q-table file: ")
            if os.path.isfile(q_table_path):
                config[f"--{prefix}q-table"] = [q_table_path]
                break
            print(f"File not found: {q_table_path}")
    
    elif choice == "2":
        config[f"--{prefix}train"] = []
        
        # Get number of training episodes
        while True:
            episodes = input("Enter number of training episodes (default 10000): ")
            if not episodes:
                episodes = "10000"
            if episodes.isdigit() and int(episodes) > 0:
                config[f"--{prefix}train-episodes"] = [int(episodes)]
                break
            print("Invalid number. Please enter a positive integer.")
        
        # Ask if user wants to save the trained Q-table
        save_choice = input("Do you want to save the trained Q-table? (y/n): ").lower()
        if save_choice.startswith('y'):
            default_path = f"models/{prefix}qtable_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            path = input(f"Enter save path (default: {default_path}): ")
            if not path:
                path = default_path
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            config[f"--{prefix}save-q-table"] = [path]
    
    return config

def get_experiment_config():
    """Get general experiment configuration from user."""
    config = {}
    
    # Get number of games
    while True:
        num_games = input("\nEnter number of games to play (default 10): ")
        if not num_games:
            num_games = "10"
        if num_games.isdigit() and int(num_games) > 0:
            config["--num-games"] = [int(num_games)]
            break
        print("Invalid number. Please enter a positive integer.")
    
    # Get move timeout
    while True:
        timeout = input("\nEnter maximum seconds per move (default 300): ")
        if not timeout:
            timeout = "300"
        if timeout.isdigit() and int(timeout) > 0:
            config["--move-timeout"] = [int(timeout)]
            break
        print("Invalid number. Please enter a positive integer.")
    
    # Ask about verbose output
    verbose = input("\nShow verbose output? (y/n, default n): ").lower()
    if verbose.startswith('y'):
        config["--verbose"] = []
    
    # Ask about force flag for potentially slow configurations
    force = input("\nForce run even if configuration might be slow? (y/n, default n): ").lower()
    if force.startswith('y'):
        config["--force"] = []
    
    # Custom experiment name
    custom_name = input("\nEnter a custom experiment name (or leave blank for auto-generated): ")
    if custom_name:
        config["--experiment-name"] = [custom_name]
    
    return config

def build_command_play_vs_opponent():
    """Build command for play_vs_opponent.py based on user input."""
    cmd = ["python", "-m", "experiments.play_vs_opponent"]
    
    # Get game choice
    game = get_game_choice()
    cmd.extend(["--game", game])
    
    # Get agent type
    agent_type = get_agent_type()
    cmd.extend(["--agent-type", agent_type])
    
    # Get agent position
    if get_agent_first():
        cmd.append("--agent-first")
    
    # Get agent-specific configuration
    if agent_type == "minimax":
        minimax_config = get_minimax_config()
        for key, val in minimax_config.items():
            cmd.append(key)
            if val:  # For non-empty lists (flags don't have values)
                cmd.extend([str(v) for v in val])
    else:  # qlearning
        qlearning_config = get_qlearning_config()
        for key, val in qlearning_config.items():
            cmd.append(key)
            if val:  # For non-empty lists (flags don't have values)
                cmd.extend([str(v) for v in val])
    
    # Get experiment configuration
    exp_config = get_experiment_config()
    for key, val in exp_config.items():
        cmd.append(key)
        if val:  # For non-empty lists (flags don't have values)
            cmd.extend([str(v) for v in val])
    
    return cmd

def build_command_play_vs_each_other():
    """Build command for play_vs_each_other.py based on user input."""
    cmd = ["python", "-m", "experiments.play_vs_each_other"]
    
    # Get game choice
    game = get_game_choice()
    cmd.extend(["--game", game])
    
    # Get agent 1 configuration
    print("\n=== Configure Agent 1 (Player 1) ===")
    agent1_type = get_agent_type(1)
    cmd.extend(["--agent1-type", agent1_type])
    
    if agent1_type == "minimax":
        minimax_config = get_minimax_config(1)
        for key, val in minimax_config.items():
            cmd.append(key)
            if val:
                cmd.extend([str(v) for v in val])
    else:  # qlearning
        qlearning_config = get_qlearning_config(1)
        for key, val in qlearning_config.items():
            cmd.append(key)
            if val:
                cmd.extend([str(v) for v in val])
    
    # Get agent 2 configuration
    print("\n=== Configure Agent 2 (Player 2) ===")
    agent2_type = get_agent_type(2)
    cmd.extend(["--agent2-type", agent2_type])
    
    if agent2_type == "minimax":
        minimax_config = get_minimax_config(2)
        for key, val in minimax_config.items():
            cmd.append(key)
            if val:
                cmd.extend([str(v) for v in val])
    else:  # qlearning
        qlearning_config = get_qlearning_config(2)
        for key, val in qlearning_config.items():
            cmd.append(key)
            if val:
                cmd.extend([str(v) for v in val])
    
    # Get experiment configuration
    exp_config = get_experiment_config()
    for key, val in exp_config.items():
        cmd.append(key)
        if val:
            cmd.extend([str(v) for v in val])
    
    return cmd

def get_visualization_mode():
    """Ask the user which visualization mode they want to use."""
    print("\n=== Select Game Mode ===")
    print("1. Human vs Human")
    print("2. Human vs AI")
    print("3. Human vs Semi-Intelligent")
    print("4. AI vs Semi-Intelligent")
    print("5. AI vs AI")
    
    while True:
        choice = input("\nEnter your choice (1-5): ")
        if choice in ["1", "2", "3", "4", "5"]:
            return int(choice)
        print("Invalid choice. Please enter a number from 1 to 5.")

def get_ai_config_for_visualization(player_num, game_type):
    """Get AI agent configuration for visualization."""
    print(f"\n=== Configure AI for Player {player_num} ===")
    print("1. Minimax")
    print("2. Q-Learning")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ")
        if choice in ["1", "2"]:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    if choice == "1":
        # Configure Minimax
        print("\n=== Minimax Configuration ===")
        
        # Get pruning option
        print("1. Enable pruning")
        print("2. Disable pruning")
        
        pruning = False
        while True:
            pruning_choice = input("\nEnter your choice (1 or 2): ")
            if pruning_choice == "1":
                pruning = True
                break
            elif pruning_choice == "2":
                break
            print("Invalid choice. Please enter 1 or 2.")
        
        # Get depth option
        print("\n=== Search Depth ===")
        print("1. Unlimited depth (may be very slow for Connect-4)")
        print("2. Specify a depth limit")
        
        depth = None
        while True:
            depth_choice = input("\nEnter your choice (1 or 2): ")
            if depth_choice == "1":
                # Use unlimited depth
                break
            elif depth_choice == "2":
                while True:
                    depth_val = input("Enter depth limit (recommended: 3-6 for Connect-4, 5-9 for Tic-Tac-Toe): ")
                    if depth_val.isdigit() and int(depth_val) > 0:
                        depth = int(depth_val)
                        break
                    print("Invalid depth. Please enter a positive integer.")
                break
            print("Invalid choice. Please enter 1 or 2.")
        
        return {
            "type": "minimax",
            "pruning": pruning,
            "depth": depth
        }
        
    else:  # Q-Learning
        # For visualization, we'll use pre-trained Q-tables from the models directory
        if game_type == "ttt":
            q_table_path = "models/ttt_qtable.pkl"
        else:  # Connect-4
            q_table_path = "models/c4_qtable.pkl"
            
        if not os.path.isfile(q_table_path):
            print(f"\nWarning: Default Q-table not found at {q_table_path}")
            print("A new Q-learning agent will be initialized (may not play optimally).")
            q_table_path = None
        else:
            print(f"\nUsing pre-trained Q-table from: {q_table_path}")
            
        return {
            "type": "qlearning",
            "q_table_path": q_table_path
        }

def build_command_visualize_game():
    """Build command to run a game with visualization."""
    # Get game choice
    game_choice = get_game_choice()
    
    # Get visualization mode
    vis_mode = get_visualization_mode()
    
    # Create a Python script to run the visualization
    vis_script = f"visualize_{game_choice}_game.py"
    
    with open(vis_script, 'w') as f:
        f.write("#!/usr/bin/env python3\n")
        f.write("import sys\n")
        
        if game_choice == "ttt":
            f.write("from games import TicTacToe, TicTacToeUI, PlayerType, GameMode\n")
            ui_class = "TicTacToeUI"
            game_class = "TicTacToe"
        else:  # Connect-4
            f.write("from games import Connect4, Connect4UI, PlayerType, GameMode\n")
            ui_class = "Connect4UI"
            game_class = "Connect4"
        
        # Import appropriate AI agents based on mode
        if vis_mode in [2, 4, 5]:  # Modes with AI
            f.write("from agents import Minimax{}, QLearning{}\n".format(
                "TicTacToe" if game_choice == "ttt" else "Connect4",
                "TicTacToe" if game_choice == "ttt" else "Connect4"
            ))
        
        # Import default opponents for semi-intelligent agents
        if vis_mode in [3, 4]:  # Modes with semi-intelligent agents
            if game_choice == "ttt":
                f.write("from opponents.default_opponent_ttt import DefaultOpponentTTT\n")
            else:  # Connect-4
                f.write("from opponents.default_opponent_c4 import DefaultOpponentC4\n")
        
        # Start the script
        f.write("\n# Create UI instance\n")
        f.write(f"ui = {ui_class}()\n\n")
        
        # Configure the game mode
        f.write("# Set game mode\n")
        if vis_mode == 1:
            f.write("ui.set_game_mode(GameMode.HUMAN_VS_HUMAN)\n")
        elif vis_mode == 2:
            f.write("ui.set_game_mode(GameMode.HUMAN_VS_AI)\n")
            
            # Configure AI agent
            agent_config = get_ai_config_for_visualization(2, game_choice)
            
            if agent_config["type"] == "minimax":
                depth_param = f"max_depth={agent_config['depth']}" if agent_config["depth"] is not None else ""
                f.write(f"\n# Create AI agent\n")
                f.write(f"ai_agent = Minimax{game_class}({depth_param}, use_pruning={agent_config['pruning']})\n")
                f.write("ui.set_player2_agent(ai_agent)\n")
            else:  # qlearning
                f.write(f"\n# Create Q-learning agent\n")
                f.write(f"ai_agent = QLearning{game_class}()\n")
                if agent_config["q_table_path"]:
                    f.write(f"ai_agent.load('{agent_config['q_table_path']}')\n")
                f.write("ui.set_player2_agent(ai_agent)\n")
                
        elif vis_mode == 3:
            f.write("ui.set_game_mode(GameMode.HUMAN_VS_SEMI)\n")
            # Set up semi-intelligent agent
            if game_choice == "ttt":
                f.write("\n# Create semi-intelligent agent\n")
                f.write("semi_agent = DefaultOpponentTTT()\n")
                f.write("ui.set_player2_agent(semi_agent)\n")
            else:
                f.write("\n# Create semi-intelligent agent\n")
                f.write("semi_agent = DefaultOpponentC4()\n")
                f.write("ui.set_player2_agent(semi_agent)\n")
            
        elif vis_mode == 4:
            f.write("ui.set_game_mode(GameMode.AI_VS_SEMI)\n")
            
            # Configure AI agent
            agent_config = get_ai_config_for_visualization(1, game_choice)
            
            if agent_config["type"] == "minimax":
                depth_param = f"max_depth={agent_config['depth']}" if agent_config["depth"] is not None else ""
                f.write(f"\n# Create AI agent\n")
                f.write(f"ai_agent = Minimax{game_class}({depth_param}, use_pruning={agent_config['pruning']})\n")
                f.write("ui.set_player1_agent(ai_agent)\n")
            else:  # qlearning
                f.write(f"\n# Create Q-learning agent\n")
                f.write(f"ai_agent = QLearning{game_class}()\n")
                if agent_config["q_table_path"]:
                    f.write(f"ai_agent.load('{agent_config['q_table_path']}')\n")
                f.write("ui.set_player1_agent(ai_agent)\n")
            
            # Set up semi-intelligent agent
            if game_choice == "ttt":
                f.write("\n# Create semi-intelligent agent\n")
                f.write("semi_agent = DefaultOpponentTTT()\n")
                f.write("ui.set_player2_agent(semi_agent)\n")
            else:
                f.write("\n# Create semi-intelligent agent\n")
                f.write("semi_agent = DefaultOpponentC4()\n")
                f.write("ui.set_player2_agent(semi_agent)\n")
                
        elif vis_mode == 5:
            f.write("ui.set_game_mode(GameMode.AI_VS_AI)\n")
            
            # Configure first AI agent
            print("\n=== AI vs AI Configuration ===")
            agent1_config = get_ai_config_for_visualization(1, game_choice)
            
            if agent1_config["type"] == "minimax":
                depth_param = f"max_depth={agent1_config['depth']}" if agent1_config["depth"] is not None else ""
                f.write(f"\n# Create AI agent 1\n")
                f.write(f"ai_agent1 = Minimax{game_class}({depth_param}, use_pruning={agent1_config['pruning']})\n")
                f.write("ui.set_player1_agent(ai_agent1)\n")
            else:  # qlearning
                f.write(f"\n# Create Q-learning agent 1\n")
                f.write(f"ai_agent1 = QLearning{game_class}()\n")
                if agent1_config["q_table_path"]:
                    f.write(f"ai_agent1.load('{agent1_config['q_table_path']}')\n")
                f.write("ui.set_player1_agent(ai_agent1)\n")
            
            # Configure second AI agent
            agent2_config = get_ai_config_for_visualization(2, game_choice)
            
            if agent2_config["type"] == "minimax":
                depth_param = f"max_depth={agent2_config['depth']}" if agent2_config["depth"] is not None else ""
                f.write(f"\n# Create AI agent 2\n")
                f.write(f"ai_agent2 = Minimax{game_class}({depth_param}, use_pruning={agent2_config['pruning']})\n")
                f.write("ui.set_player2_agent(ai_agent2)\n")
            else:  # qlearning
                f.write(f"\n# Create Q-learning agent 2\n")
                f.write(f"ai_agent2 = QLearning{game_class}()\n")
                if agent2_config["q_table_path"]:
                    f.write(f"ai_agent2.load('{agent2_config['q_table_path']}')\n")
                f.write("ui.set_player2_agent(ai_agent2)\n")
        
        # Run the game
        f.write("\n# Run the game\n")
        f.write("ui.run()\n")
    
    # Make the script executable
    os.chmod(vis_script, 0o755)
    
    # Return the command to run the script
    return ["python", vis_script]

def run_command(cmd):
    """Run the command and capture output."""
    print("\n=== Running command ===")
    print(" ".join(cmd))
    print("\n=== Command output ===")
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    output_lines = []
    
    # Read and display output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end='')  # Display to console
        output_lines.append(line)  # Save for file
        sys.stdout.flush()
    
    # Wait for process to complete
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        print(f"\n=== Command failed with return code {return_code} ===")
    
    return "".join(output_lines)

def save_output(output, cmd, output_dir):
    """Save command output to a file."""
    # Create a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine script type for filename
    if "play_vs_opponent" in " ".join(cmd):
        script_type = "vs_opponent"
    elif "play_vs_each_other" in " ".join(cmd):
        script_type = "vs_each_other"
    else:
        script_type = "visualization"
    
    # Extract game type from command or visualization script
    game_type = "unknown"
    if script_type == "visualization":
        if "visualize_ttt_game.py" in " ".join(cmd):
            game_type = "ttt"
        elif "visualize_c4_game.py" in " ".join(cmd):
            game_type = "c4"
    else:
        for i, arg in enumerate(cmd):
            if arg == "--game" and i+1 < len(cmd):
                game_type = cmd[i+1]
    
    # Create a filename
    filename = f"{timestamp}_{script_type}_{game_type}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write("=== Command ===\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write("=== Output ===\n")
        f.write(output)
    
    return filepath

def main():
    """Main function to run the demo."""
    # Create output directory
    output_dir = create_output_dir()
    
    # Get user's script choice
    script_choice = get_user_script_choice()
    
    # Build command based on choice
    if script_choice == 1:
        cmd = build_command_play_vs_opponent()
    elif script_choice == 2:
        cmd = build_command_play_vs_each_other()
    else:  # script_choice == 3
        cmd = build_command_visualize_game()
    
    # Ask for confirmation
    print("\n=== Command to run ===")
    print(" ".join(cmd))
    confirmation = input("\nRun this command? (y/n): ").lower()
    
    if not confirmation.startswith('y'):
        print("Aborted.")
        return
    
    # Run the command
    output = run_command(cmd)
    
    # Save output to file
    output_file = save_output(output, cmd, output_dir)
    
    print(f"\n=== Output saved to {output_file} ===")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}") 