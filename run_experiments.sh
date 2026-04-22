#!/bin/bash

# 1. Navigate to the working directory
cd /jss_project/RL-Job-Shop-Scheduling/PPO || { echo "Failed to navigate to working directory"; exit 1; }

# 2. Activate the virtual environment
source /jss_project/venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

echo "Environment activated. Starting batch experiments..."

# 3. Execute each command sequentially
# For each command, we ensure the directory exists, then run the python script and route all logs into it.

# Run 1
mkdir -p checkpoint_results/PPO_ta42_warm_GA_22042026
python3 warm_start_GA.py --bks bks.json --instances instances/ta42 --evo-alg GA --iters 1000 --out checkpoint_results/PPO_ta42_warm_GA_22042026 --evo-gens 1000 --bc-epochs 50 --target-gap 15.0 --evo-early-stop > checkpoint_results/PPO_ta42_warm_GA_22042026/terminal_output.log 2>&1

# Run 2
mkdir -p checkpoint_results/PPO_ta42_22042026
python3 warm_start_GA.py --bks bks.json --instances instances/ta42 --evo-alg GA --iters 1000 --out checkpoint_results/PPO_ta42_22042026 --evo-gens 1000 --bc-epochs 50 --target-gap 15.0 --evo-early-stop --no-warmstart > checkpoint_results/PPO_ta42_22042026/terminal_output.log 2>&1

# Run 3
mkdir -p checkpoint_results/PPO_ta52_warm_GA_22042026
python3 warm_start_GA.py --bks bks.json --instances instances/ta52 --evo-alg GA --iters 1000 --out checkpoint_results/PPO_ta52_warm_GA_22042026 --evo-gens 1000 --bc-epochs 50 --target-gap 15.0 --evo-early-stop > checkpoint_results/PPO_ta52_warm_GA_22042026/terminal_output.log 2>&1

# Run 4
mkdir -p checkpoint_results/PPO_ta52_22042026
python3 warm_start_GA.py --bks bks.json --instances instances/ta52 --evo-alg GA --iters 1000 --out checkpoint_results/PPO_ta52_22042026 --evo-gens 1000 --bc-epochs 50 --target-gap 15.0 --evo-early-stop --no-warmstart > checkpoint_results/PPO_ta52_22042026/terminal_output.log 2>&1

# Run 5
mkdir -p checkpoint_results/PPO_ta62_warm_GA_22042026
python3 warm_start_GA.py --bks bks.json --instances instances/ta62 --evo-alg GA --iters 1000 --out checkpoint_results/PPO_ta62_warm_GA_22042026 --evo-gens 1000 --bc-epochs 50 --target-gap 15.0 --evo-early-stop > checkpoint_results/PPO_ta62_warm_GA_22042026/terminal_output.log 2>&1

# Run 6
mkdir -p checkpoint_results/PPO_ta62_22042026
python3 warm_start_GA.py --bks bks.json --instances instances/ta62 --evo-alg GA --iters 1000 --out checkpoint_results/PPO_ta62_22042026 --evo-gens 1000 --bc-epochs 50 --target-gap 15.0 --evo-early-stop --no-warmstart > checkpoint_results/PPO_ta62_22042026/terminal_output.log 2>&1

# Run 7
mkdir -p checkpoint_results/PPO_ta72_warm_GA_22042026
python3 warm_start_GA.py --bks bks.json --instances instances/ta72 --evo-alg GA --iters 1000 --out checkpoint_results/PPO_ta72_warm_GA_22042026 --evo-gens 1000 --bc-epochs 50 --target-gap 15.0 --evo-early-stop > checkpoint_results/PPO_ta72_warm_GA_22042026/terminal_output.log 2>&1

# Run 8
mkdir -p checkpoint_results/PPO_ta72_22042026
python3 warm_start_GA.py --bks bks.json --instances instances/ta72 --evo-alg GA --iters 1000 --out checkpoint_results/PPO_ta72_22042026 --evo-gens 1000 --bc-epochs 50 --target-gap 15.0 --evo-early-stop --no-warmstart > checkpoint_results/PPO_ta72_22042026/terminal_output.log 2>&1

echo "All batch experiments have completed."
