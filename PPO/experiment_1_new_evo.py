#!/usr/bin/env python3
"""
experiment_1_evolutionary_comparison.py

Compare GA, MA, and HGA against PPO baseline results on Taillard instances.
Each algorithm runs until it matches or beats the PPO target makespan,
or exhausts its generation budget. Records best makespan and wall-clock time.

Usage:
    python3 experiment_1_evolutionary_comparison.py

Inputs:
    - tabulated_ppo_results.csv   (columns: Instance, Achieved Result)
    - instances/<instance_id>     (Taillard format files)

Outputs:
    - evolutionary_comparison_results.csv
"""

import os
import time
import math
import random
from copy import deepcopy

import numpy as np
import pandas as pd

from run_ma_hga_taillard import (
    parse_taillard,
    compute_makespan,
    random_permutation,
    tournament_select,
    order_crossover,
    swap_mutation,
    simulated_annealing_improve,
    calibrate_sa_temperature,
    TOURNAMENT_K,
    SA_TEND,
)

# ===========================
# CONFIGURATION
# ===========================
POP_SIZE = 100
MAX_GENERATIONS = 500
CROSSOVER_P = 0.9
MUTATION_P = 0.1

# MA: SA applied to every offspring
MA_SA_ITERS = 50

# HGA: SA applied to top fraction of population
HGA_SA_FRACTION = 0.2
HGA_SA_ITERS = 25

PPO_RESULTS_FILE = "tabulated_ppo_results.csv"
INSTANCES_DIR = "instances"
OUTPUT_FILE = "evolutionary_comparison_results_new_evo.csv"


# ===========================
# ALGORITHM RUNNERS WITH EARLY STOPPING
# ===========================

def _init_population(jobs, pop_size, seed):
    """Create initial population and evaluate fitness."""
    random.seed(seed)
    np.random.seed(seed)
    pop = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = [compute_makespan(jobs, ind) for ind in pop]
    best_idx = int(np.argmin(fitness))
    return pop, fitness, deepcopy(pop[best_idx]), fitness[best_idx]


def _evolutionary_step(pop, fitness, best_ind, cross_p, mut_p, pop_size):
    """One generation of selection, crossover, mutation. Returns new population (unevaluated)."""
    new_pop = [deepcopy(best_ind)]  # elitism
    while len(new_pop) < pop_size:
        p1 = tournament_select(pop, fitness, TOURNAMENT_K)
        p2 = tournament_select(pop, fitness, TOURNAMENT_K)
        child = order_crossover(p1, p2) if random.random() < cross_p else deepcopy(p1)
        child = swap_mutation(child, mutation_rate=mut_p)
        new_pop.append(child)
    return new_pop


def run_ga_until_target(jobs, target, pop_size, generations, seed=42):
    """
    Pure GA with early stopping when target makespan is reached.
    """
    start = time.time()
    pop, fitness, best_ind, best_val = _init_population(jobs, pop_size, seed)
    best_gen = 0

    if best_val <= target:
        return {"best_makespan": best_val, "best_gen": 0,
                "time": time.time() - start, "status": "HIT"}

    for gen in range(1, generations + 1):
        pop = _evolutionary_step(pop, fitness, best_ind, CROSSOVER_P, MUTATION_P, pop_size)
        fitness = [compute_makespan(jobs, ind) for ind in pop]

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_val:
            best_val = fitness[gen_best_idx]
            best_ind = deepcopy(pop[gen_best_idx])
            best_gen = gen

        if best_val <= target:
            return {"best_makespan": best_val, "best_gen": gen,
                    "time": time.time() - start, "status": "HIT"}

    return {"best_makespan": best_val, "best_gen": best_gen,
            "time": time.time() - start, "status": "DNF"}


def run_ma_until_target(jobs, target, pop_size, generations, sa_iters, seed=42):
    """
    Memetic Algorithm (GA + SA on every offspring) with early stopping.
    """
    start = time.time()
    pop, fitness, best_ind, best_val = _init_population(jobs, pop_size, seed)
    best_gen = 0

    if best_val <= target:
        return {"best_makespan": best_val, "best_gen": 0,
                "time": time.time() - start, "status": "HIT"}

    # Calibrate SA temperature from a sample individual
    t0 = calibrate_sa_temperature(jobs, pop[0])

    for gen in range(1, generations + 1):
        new_pop = [deepcopy(best_ind)]  # elitism

        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = order_crossover(p1, p2) if random.random() < CROSSOVER_P else deepcopy(p1)
            child = swap_mutation(child, mutation_rate=MUTATION_P)

            # Lamarckian local search on every offspring
            child, _ = simulated_annealing_improve(
                jobs, child, iters=sa_iters, t0=t0, tend=SA_TEND
            )
            new_pop.append(child)

        pop = new_pop
        fitness = [compute_makespan(jobs, ind) for ind in pop]

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_val:
            best_val = fitness[gen_best_idx]
            best_ind = deepcopy(pop[gen_best_idx])
            best_gen = gen

        if best_val <= target:
            return {"best_makespan": best_val, "best_gen": gen,
                    "time": time.time() - start, "status": "HIT"}

    return {"best_makespan": best_val, "best_gen": best_gen,
            "time": time.time() - start, "status": "DNF"}


def run_hga_until_target(jobs, target, pop_size, generations, sa_fraction, sa_iters, seed=42):
    """
    Hybrid GA (GA + SA on top fraction of population) with early stopping.
    """
    start = time.time()
    pop, fitness, best_ind, best_val = _init_population(jobs, pop_size, seed)
    best_gen = 0

    if best_val <= target:
        return {"best_makespan": best_val, "best_gen": 0,
                "time": time.time() - start, "status": "HIT"}

    # Calibrate SA temperature from a sample individual
    t0 = calibrate_sa_temperature(jobs, pop[0])
    topk = max(1, int(sa_fraction * pop_size))

    for gen in range(1, generations + 1):
        pop = _evolutionary_step(pop, fitness, best_ind, CROSSOVER_P, MUTATION_P, pop_size)
        fitness = [compute_makespan(jobs, ind) for ind in pop]

        # Selective intensification: SA on the top-k individuals only
        idx_sorted = sorted(range(len(pop)), key=lambda i: fitness[i])
        for i in idx_sorted[:topk]:
            improved, val = simulated_annealing_improve(
                jobs, pop[i], iters=sa_iters, t0=t0, tend=SA_TEND
            )
            pop[i] = improved
            fitness[i] = val

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_val:
            best_val = fitness[gen_best_idx]
            best_ind = deepcopy(pop[gen_best_idx])
            best_gen = gen

        if best_val <= target:
            return {"best_makespan": best_val, "best_gen": gen,
                    "time": time.time() - start, "status": "HIT"}

    return {"best_makespan": best_val, "best_gen": best_gen,
            "time": time.time() - start, "status": "DNF"}


# ===========================
# EXPERIMENT DRIVER
# ===========================

def main():
    # Load PPO baseline results
    if not os.path.exists(PPO_RESULTS_FILE):
        raise FileNotFoundError(f"PPO results file not found: {PPO_RESULTS_FILE}")

    ppo_df = pd.read_csv(PPO_RESULTS_FILE)
    print(f"Loaded {len(ppo_df)} instances from {PPO_RESULTS_FILE}\n")

    results = []

    for _, row in ppo_df.iterrows():
        inst_id = row["Instance"]
        ppo_target = int(row["Achieved Result"])
        inst_path = os.path.join(INSTANCES_DIR, inst_id)

        if not os.path.exists(inst_path):
            print(f"WARNING: Instance file missing: {inst_path}, skipping.")
            continue

        jobs = parse_taillard(inst_path)
        print(f"{'='*60}")
        print(f"Instance: {inst_id}  |  PPO Target: {ppo_target}")
        print(f"{'='*60}")

        # --- GA ---
        ga = run_ga_until_target(
            jobs, ppo_target, pop_size=POP_SIZE, generations=MAX_GENERATIONS
        )
        ga_time_str = round(ga["time"], 4) if ga["status"] == "HIT" else "DNF"
        print(f"  GA  -> Makespan: {ga['best_makespan']:>6}  "
              f"Time: {ga_time_str!s:>10}  Gen: {ga['best_gen']:>4}  [{ga['status']}]")

        # --- MA ---
        ma = run_ma_until_target(
            jobs, ppo_target, pop_size=POP_SIZE, generations=MAX_GENERATIONS,
            sa_iters=MA_SA_ITERS
        )
        ma_time_str = round(ma["time"], 4) if ma["status"] == "HIT" else "DNF"
        print(f"  MA  -> Makespan: {ma['best_makespan']:>6}  "
              f"Time: {ma_time_str!s:>10}  Gen: {ma['best_gen']:>4}  [{ma['status']}]")

        # --- HGA ---
        hga = run_hga_until_target(
            jobs, ppo_target, pop_size=POP_SIZE, generations=MAX_GENERATIONS,
            sa_fraction=HGA_SA_FRACTION, sa_iters=HGA_SA_ITERS
        )
        hga_time_str = round(hga["time"], 4) if hga["status"] == "HIT" else "DNF"
        print(f"  HGA -> Makespan: {hga['best_makespan']:>6}  "
              f"Time: {hga_time_str!s:>10}  Gen: {hga['best_gen']:>4}  [{hga['status']}]")

        print()

        results.append({
            "Instance": inst_id,
            "PPO_Target": ppo_target,
            "GA_Makespan": ga["best_makespan"],
            "GA_Time_s": ga_time_str,
            "GA_Gen": ga["best_gen"],
            "GA_Status": ga["status"],
            "MA_Makespan": ma["best_makespan"],
            "MA_Time_s": ma_time_str,
            "MA_Gen": ma["best_gen"],
            "MA_Status": ma["status"],
            "HGA_Makespan": hga["best_makespan"],
            "HGA_Time_s": hga_time_str,
            "HGA_Gen": hga["best_gen"],
            "HGA_Status": hga["status"],
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
