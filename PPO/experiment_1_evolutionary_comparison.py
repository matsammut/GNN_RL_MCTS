#!/usr/bin/env python3
"""
Compare GA, MA, and HGA against PPO baseline results on Taillard instances.
Each algorithm runs until it matches or beats the PPO target makespan,
or exhausts its generation budget. Records best makespan and wall-clock time.

Usage:
    python3 experiment_1_evolutionary_comparison.py

Inputs:
    - tabulated_ppo_results.csv   (columns: Instance, Achieved Result, BKS, ...)
    - instances/<instance_id>     (Taillard format files)

Outputs:
    - ppo_vs_evolutionary_comparison.csv
"""

import os
import sys
import time
import logging
import multiprocessing as mp
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

from run_ma_hga_taillard import (
    parse_taillard,
    compute_makespan,
    random_permutation,
    tournament_select,
    order_crossover,
    mutate,
    simulated_annealing_improve,
    calibrate_sa_temperature,
    deduplicate_population,
    parallel_evaluate,
    parallel_sa_improve,
    TOURNAMENT_K,
    SA_TEND,
    N_WORKERS,
)


# ===========================
# CONFIG
# ===========================
POP_SIZE = 1250
MAX_GENERATIONS = 50
CROSSOVER_P = 0.9
MUTATION_P = 0.1

MA_SA_ITERS = 50
HGA_SA_FRACTION = 0.2
HGA_SA_ITERS = 25

PPO_RESULTS_FILE = "tabulated_results/tabulated_ppo_results.csv"
INSTANCES_DIR = "instances"
OUTPUT_FILE = "tabulated_results/ppo_vs_evolutionary_comparison_test.csv"

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


# ===========================
# ALGORITHM RUNNERS WITH EARLY STOPPING
# ===========================

def _init_population(jobs, pop_size, seed):
    """Create initial population and evaluate fitness in parallel."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    pop = [random_permutation(jobs) for _ in range(pop_size)]
    with mp.Pool(N_WORKERS) as pool:
        fitness = parallel_evaluate(pool, jobs, pop)
    best_idx = int(np.argmin(fitness))
    return pop, fitness, deepcopy(pop[best_idx]), fitness[best_idx]


def _evolutionary_step(pop, fitness, best_ind, cross_p, mut_p, pop_size):
    """One generation of selection, crossover, mutation. Returns unevaluated new population."""
    new_pop = [deepcopy(best_ind)]  # elitism
    while len(new_pop) < pop_size:
        p1 = tournament_select(pop, fitness, TOURNAMENT_K)
        p2 = tournament_select(pop, fitness, TOURNAMENT_K)
        import random
        child = order_crossover(p1, p2) if random.random() < cross_p else deepcopy(p1)
        child = mutate(child, mutation_rate=mut_p)
        new_pop.append(child)
    return new_pop


def run_ga_until_target(jobs, target, pop_size, generations, seed=42):
    """Pure GA with early stopping when target makespan is reached."""
    start = time.time()
    pop, fitness, best_ind, best_val = _init_population(jobs, pop_size, seed)
    best_gen = 0

    logger.info(f"    GA | Init | Best: {best_val} | Target: {target}")

    if best_val <= target:
        return {"best_makespan": best_val, "best_gen": 0,
                "time": time.time() - start, "status": "HIT"}

    for gen in range(1, generations + 1):
        pop = _evolutionary_step(pop, fitness, best_ind, CROSSOVER_P, MUTATION_P, pop_size)

        with mp.Pool(N_WORKERS) as pool:
            fitness = parallel_evaluate(pool, jobs, pop)

        pop, fitness = deduplicate_population(pop, fitness, jobs)

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_val:
            best_val = fitness[gen_best_idx]
            best_ind = deepcopy(pop[gen_best_idx])
            best_gen = gen
            logger.info(f"    GA | Gen {gen:>4}/{generations} | NEW BEST: {best_val}")

        if best_val <= target:
            logger.info(f"    GA | HIT target at gen {gen} | Best: {best_val}")
            return {"best_makespan": best_val, "best_gen": gen,
                    "time": time.time() - start, "status": "HIT"}

    logger.info(f"    GA | DNF after {generations} gens | Best: {best_val}")
    return {"best_makespan": best_val, "best_gen": best_gen,
            "time": time.time() - start, "status": "DNF"}


def run_ma_until_target(jobs, target, pop_size, generations, sa_iters, seed=42):
    """Memetic Algorithm (GA + SA on every offspring) with early stopping."""
    start = time.time()
    pop, fitness, best_ind, best_val = _init_population(jobs, pop_size, seed)
    best_gen = 0

    logger.info(f"    MA | Init | Best: {best_val} | Target: {target}")

    if best_val <= target:
        return {"best_makespan": best_val, "best_gen": 0,
                "time": time.time() - start, "status": "HIT"}

    for gen in range(1, generations + 1):
        import random
        new_pop = [deepcopy(best_ind)]
        offspring = []
        while len(offspring) < pop_size - 1:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = order_crossover(p1, p2) if random.random() < CROSSOVER_P else deepcopy(p1)
            child = mutate(child, mutation_rate=MUTATION_P)
            offspring.append(child)

        # Parallel SA on all offspring
        with mp.Pool(N_WORKERS) as pool:
            improved, vals = parallel_sa_improve(pool, jobs, offspring, sa_iters, SA_TEND)

        new_pop.extend(improved)
        pop = new_pop
        fitness = [compute_makespan(jobs, pop[0])] + vals

        pop, fitness = deduplicate_population(pop, fitness, jobs)

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_val:
            best_val = fitness[gen_best_idx]
            best_ind = deepcopy(pop[gen_best_idx])
            best_gen = gen
            logger.info(f"    MA | Gen {gen:>4}/{generations} | NEW BEST: {best_val}")

        if best_val <= target:
            logger.info(f"    MA | HIT target at gen {gen} | Best: {best_val}")
            return {"best_makespan": best_val, "best_gen": gen,
                    "time": time.time() - start, "status": "HIT"}

    logger.info(f"    MA | DNF after {generations} gens | Best: {best_val}")
    return {"best_makespan": best_val, "best_gen": best_gen,
            "time": time.time() - start, "status": "DNF"}


def run_hga_until_target(jobs, target, pop_size, generations, sa_fraction, sa_iters, seed=42):
    """Hybrid GA (GA + SA on top fraction) with early stopping."""
    start = time.time()
    pop, fitness, best_ind, best_val = _init_population(jobs, pop_size, seed)
    best_gen = 0
    topk = max(1, int(sa_fraction * pop_size))

    logger.info(f"    HGA | Init | Best: {best_val} | Target: {target}")

    if best_val <= target:
        return {"best_makespan": best_val, "best_gen": 0,
                "time": time.time() - start, "status": "HIT"}

    for gen in range(1, generations + 1):
        pop = _evolutionary_step(pop, fitness, best_ind, CROSSOVER_P, MUTATION_P, pop_size)

        with mp.Pool(N_WORKERS) as pool:
            fitness = parallel_evaluate(pool, jobs, pop)

        # Selective intensification on top-k
        idx_sorted = sorted(range(len(pop)), key=lambda i: fitness[i])
        elite_indices = idx_sorted[:topk]
        elite_inds = [pop[i] for i in elite_indices]

        with mp.Pool(N_WORKERS) as pool:
            improved, vals = parallel_sa_improve(pool, jobs, elite_inds, sa_iters, SA_TEND)

        for rank, idx in enumerate(elite_indices):
            pop[idx] = improved[rank]
            fitness[idx] = vals[rank]

        pop, fitness = deduplicate_population(pop, fitness, jobs)

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_val:
            best_val = fitness[gen_best_idx]
            best_ind = deepcopy(pop[gen_best_idx])
            best_gen = gen
            logger.info(f"    HGA | Gen {gen:>4}/{generations} | NEW BEST: {best_val}")

        if best_val <= target:
            logger.info(f"    HGA | HIT target at gen {gen} | Best: {best_val}")
            return {"best_makespan": best_val, "best_gen": gen,
                    "time": time.time() - start, "status": "HIT"}

    logger.info(f"    HGA | DNF after {generations} gens | Best: {best_val}")
    return {"best_makespan": best_val, "best_gen": best_gen,
            "time": time.time() - start, "status": "DNF"}


# ===========================
# MAIN
# ===========================

def main():
    # Load PPO data
    if not os.path.exists(PPO_RESULTS_FILE):
        logger.error(f"PPO results file not found: {PPO_RESULTS_FILE}")
        sys.exit(1)

    ppo_results = pd.read_csv(PPO_RESULTS_FILE)
    logger.info(f"Loaded {len(ppo_results)} rows from {PPO_RESULTS_FILE}")
    logger.info(f"Workers available: {N_WORKERS}")
    logger.info(f"Pop size: {POP_SIZE} | Max gens: {MAX_GENERATIONS}")
    logger.info(f"MA SA iters: {MA_SA_ITERS} | HGA SA iters: {HGA_SA_ITERS} | HGA fraction: {HGA_SA_FRACTION}")
    logger.info("=" * 70)

    comparison_data = []

    for row_idx, row in ppo_results.iterrows():
        inst_id = row["Instance"]
        ppo_target = int(row["Achieved Result"])
        bks = int(row["BKS"])
        inst_path = os.path.join(INSTANCES_DIR, inst_id)

        if not os.path.exists(inst_path):
            logger.warning(f"Instance file missing: {inst_path}, skipping.")
            continue

        logger.info("")
        logger.info(f"{'=' * 60}")
        logger.info(f"Instance: {inst_id}  |  BKS: {bks}  |  PPO Target: {ppo_target}")
        logger.info(f"{'=' * 60}")

        jobs_data = parse_taillard(inst_path)

        # --- GA ---
        logger.info(f"  Running GA...")
        ga = run_ga_until_target(
            jobs_data, ppo_target, pop_size=POP_SIZE, generations=MAX_GENERATIONS
        )
        ga_time_str = round(ga["time"], 4) if ga["status"] == "HIT" else "DNF"
        logger.info(f"  GA  -> Makespan: {ga['best_makespan']:>6}  "
                     f"Time: {ga_time_str!s:>10}  [{ga['status']}]")

        # --- MA ---
        logger.info(f"  Running MA...")
        ma = run_ma_until_target(
            jobs_data, ppo_target, pop_size=POP_SIZE, generations=MAX_GENERATIONS,
            sa_iters=MA_SA_ITERS
        )
        ma_time_str = round(ma["time"], 4) if ma["status"] == "HIT" else "DNF"
        logger.info(f"  MA  -> Makespan: {ma['best_makespan']:>6}  "
                     f"Time: {ma_time_str!s:>10}  [{ma['status']}]")

        # --- HGA ---
        logger.info(f"  Running HGA...")
        hga = run_hga_until_target(
            jobs_data, ppo_target, pop_size=POP_SIZE, generations=MAX_GENERATIONS,
            sa_fraction=HGA_SA_FRACTION, sa_iters=HGA_SA_ITERS
        )
        hga_time_str = round(hga["time"], 4) if hga["status"] == "HIT" else "DNF"
        logger.info(f"  HGA -> Makespan: {hga['best_makespan']:>6}  "
                     f"Time: {hga_time_str!s:>10}  [{hga['status']}]")

        comparison_data.append({
            "Instance": inst_id,
            "BKS": bks,
            "PPO Result": ppo_target,
            "GA Makespan": ga["best_makespan"],
            "GA Time (s)": ga_time_str,
            "GA Status": ga["status"],
            "MA Makespan": ma["best_makespan"],
            "MA Time (s)": ma_time_str,
            "MA Status": ma["status"],
            "HGA Makespan": hga["best_makespan"],
            "HGA Time (s)": hga_time_str,
            "HGA Status": hga["status"],
        })

    # Save and print
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(OUTPUT_FILE, index=False)

    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL COMPARISON TABLE")
    logger.info("=" * 70)
    logger.info("\n" + comparison_df.to_string(index=False))
    logger.info(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
