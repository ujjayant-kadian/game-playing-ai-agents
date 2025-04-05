# Game Playing Agents: Minimax and Q-Learning

This project implements AI agents for Tic-Tac-Toe and Connect-4 games using Minimax and Q-Learning algorithms.

## Project Structure

- **[agents/](./agents/)**: AI agent implementations
  - **[agents/minimax/](./agents/minimax/)**: Minimax algorithm with/without alpha-beta pruning
  - **[agents/qlearning/](./agents/qlearning/)**: Q-Learning with epsilon-greedy policy
- **[games/](./games/)**: Game environments
  - Tic-Tac-Toe and Connect-4 with game logic and UI components
- **[opponents/](./opponents/)**: Semi-intelligent opponents
  - Default opponents that can win, block, and make strategic moves
- **[experiments/](./experiments/)**: Scripts for running simulations and collecting results
  - Play vs opponent, Play vs each other, Analysis scripts
- **[utils/](./utils/)**: Utility functions for metrics calculation
- **[models/](./models/)**: Storage for trained Q-learning models
- **[demo-outputs/](./demo-outputs/)**: Output files from demo runs
- **[demo_video/](./demo_video/)**: Demo videos showing agent performance

## Running the Code

### Prerequisites

```bash
pip install -r requirements.txt
```

### Interactive Demo Script

The easiest way to run experiments is using the interactive demo script:

```bash
python demo.py
```

This script provides a menu-driven interface to:
1. Play an AI agent against the default opponent
2. Play two AI agents against each other
3. Visualize games with a graphical interface

Follow the on-screen prompts to configure:
- Game type (Tic-Tac-Toe or Connect-4)
- Agent types (Minimax or Q-Learning)
- Agent parameters (pruning, depth, training episodes)
- Experiment settings (number of games, move timeout)

### Running Specific Experiments

Alternatively, you can run the experiment scripts directly:

**AI vs Default Opponent:**
```bash
python -m experiments.play_vs_opponent --game ttt --agent minimax --use-pruning --max-depth 5 --num-games 100
```

**AI vs AI:**
```bash
python -m experiments.play_vs_each_other --game ttt --agent1 minimax --agent1-use-pruning --agent2 qlearning --agent2-train --agent2-train-episodes 10000 --num-games 100
```

## Output and Logs

### Results Storage

All experiment results are stored in structured directories:

- **`experiments/results/`**: Results of AI vs default opponent
  - Format: `{game}_{agent_type}_{config}_{ordering}_metrics.{ext}`
  - Example: `ttt_minimax_with_pruning_depth_5_agent_first_metrics.csv`

- **`experiments/against-each-other-results/`**: Results of AI vs AI experiments
  - Format: `{game}_{agent1_type}_vs_{agent2_type}_{config}_metrics.{ext}`

- **`demo-outputs/`**: Results from demo script runs
  - Demo runs create timestamped directories with all experiment outputs

### Log Files

Each experiment produces multiple log files:

1. **`metrics.csv`**: Game-by-game results with detailed metrics
   - Move times, states explored, game outcome

2. **`metrics.json`**: Aggregated metrics in JSON format
   - Win/loss/draw rates, time analysis, move statistics

3. **`summary.txt`**: Human-readable summary of experiment results
   - Performance metrics, configuration details

4. **Q-Learning Specific Logs**:
   - `qlearn_metrics_reward_log.csv`: Episode-by-episode rewards
   - `qlearn_metrics_eval_log.csv`: Periodic evaluation metrics

### Trained Models

Q-Learning models are saved as pickle files in the `models/` directory:
- Default format: `{game}_qtable.pkl`
- Custom names can be specified with the `--save-q-table` flag

## Visualization vs. Experiments

There are two primary ways to run the AI agents in this project:

### Experiments (Headless)

Experiments run games in headless mode (without UI) for rapid testing and data collection:

- **Play vs Opponent**: Tests an AI agent against a semi-intelligent default opponent
- **Play vs Each Other**: Tests two AI agents against each other

These are designed for:
- Running large numbers of games quickly
- Collecting performance metrics
- Training Q-learning agents
- Benchmarking different algorithms

### Visualization (with UI)

Visualization mode uses Pygame to provide a graphical interface showing:

- Interactive game boards
- Real-time gameplay
- Move animations
- Game statistics
- Current player indicators

Visualization is useful for:
- Observing agent behavior in real-time
- Educational demonstrations
- Debugging agent decision-making
- Manual play against AI

## Configuration Options

### Common Configuration Options

These options apply to both experiment scripts:

| Option | Description | Default |
|--------|-------------|---------|
| `--game` | Game type (`ttt` or `c4`) | Required |
| `--num-games` | Number of games to play | 10 |
| `--move-timeout` | Maximum seconds per move | 300 |
| `--verbose` | Show detailed output | False |
| `--experiment-name` | Custom name for experiment | Auto-generated |
| `--force` | Force run potentially slow configurations | False |

### Minimax Agent Options

| Option | Description | Default |
|--------|-------------|---------|
| `--use-pruning` | Enable alpha-beta pruning | False |
| `--max-depth` | Maximum search depth | Unlimited |

### Q-Learning Agent Options

| Option | Description | Default |
|--------|-------------|---------|
| `--q-table` | Path to pre-trained Q-table | None |
| `--train` | Train the agent before evaluation | False |
| `--train-episodes` | Number of training episodes | 10000 |
| `--eval-interval` | Evaluation Interval during training | 1000 |
| `--save-q-table` | Path to save trained Q-table | None |

### Play vs Each Other Additional Options

When using `experiments.play_vs_each_other`, agent options are prefixed:

| Prefix | Description |
|--------|-------------|
| `--agent1-` | Options for first agent (e.g., `--agent1-use-pruning`) |
| `--agent2-` | Options for second agent (e.g., `--agent2-train-episodes`) |


## Analysis Tools

The project includes tools to analyze experiment results:

```bash
python -m experiments.analyze_results
```

This script generates:
- Win rate comparisons between different agent configurations
- Performance metrics (states explored, move times)
- Learning curves for Q-learning agents
- Statistical analysis of game outcomes

The `experiments.parse_agent_summaries` script can be used to extract key metrics from all experiments for comparison:

```bash
python -m experiments.parse_agent_summaries
```

Visualization output is saved to the `experiments/analysis/` directory with separate folders for different experiment types.
