CS7IS2 - Assignment 3: Game Playing Agents using Minimax and Q-learning
Author: Ujjayant Kadian
Student ID: 22330954
Date: 03 April 2025

---

Contents
--------
1. Project Structure
2. How to Run the Code
3. Default Opponent Description
4. Output Format
5. Additional Notes

---

1. Project Structure
--------------------
games/               - Game environment logic for Tic Tac Toe and Connect 4
agents/minimax/      - Minimax algorithm implementations (generic + per-game)
agents/qlearning/    - Q-learning implementations (generic + per-game)
opponents/           - Semi-intelligent default opponents
experiments/         - Scripts to run simulations and collect results
report/              - PDF report and appendices with algorithm source code
demo_video/          - Demo video showing agent performance and functionality
utils/               - Helper scripts (e.g., display functions, metric calculations)

---

2. How to Run the Code
----------------------
Install dependencies:
$ pip install -r requirements.txt

To run Minimax agent vs default opponent in Tic Tac Toe:
$ python experiments/play_vs_opponent.py --game ttt --agent minimax

To run Q-learning agent vs default opponent in Connect 4:
$ python experiments/play_vs_opponent.py --game c4 --agent qlearning

To run Minimax vs Q-learning (Tic Tac Toe):
$ python experiments/play_vs_each_other.py --game ttt

Results will be stored in `experiments/data/` as CSVs.

Note: Add `--train` flag for Q-learning agents if training is needed before evaluation.

---

3. Default Opponent Description
-------------------------------
Default opponents for both games select:
- A winning move if available
- A blocking move to prevent a loss
- Otherwise, a legal random move

---

4. Output Format
----------------
The scripts output:
- Game-by-game results (win/loss/draw)
- Cumulative performance metrics
- Optional graphs saved to PDF/PNG
- Logs saved in CSVs under `experiments/data/`

---

5. Additional Notes
-------------------
- Minimax in Connect 4 is depth-limited with an evaluation function due to scalability
- Q-learning agents are trained using epsilon-greedy exploration
- See the report PDF in `report/` for detailed analysis, graphs, and results

To run everything at once (optional), use:
$ bash run_all.sh

