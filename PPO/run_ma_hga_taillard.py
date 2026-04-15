#!/usr/bin/env python3
"""
run_ma_hga_taillard.py

Run GA, MA (memetic = GA + local search), and HGA (GA + Simulated Annealing hybrid)
on Taillard JSSP instances with fixed-budget evaluation, calibrated SA,
critical-path neighbourhood, active-schedule decoding, and multiprocessing.

Usage:
    python3 run_ma_hga_taillard.py
    python3 run_ma_hga_taillard.py --instances ta41,ta42,ta43
    python3 run_ma_hga_taillard.py --instances ta50-ta60
    python3 run_ma_hga_taillard.py --instances ta42,ta52,ta62,ta72 --gap-target 15.0
    python3 run_ma_hga_taillard.py --bks bks.json --output-dir my_results

Requirements:
    python3.8+ (numpy, pandas)

Outputs:
    <output_dir>/ga_results_<date>/  containing per-algorithm CSVs, summary, and log.
"""

import os
import sys
import re
import random
import math
import json
import time
import logging
import argparse
import multiprocessing as mp
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd


# ===========================
# ARGUMENT PARSING
# ===========================

def parse_instance_spec(spec, available_keys):
    """
    Parse an instance specification string into a list of instance names.

    Supported formats:
        ta42,ta52,ta62,ta72   -> explicit comma-separated list
        ta50-ta60             -> inclusive range (numeric suffix)
        ta41,ta50-ta55,ta72   -> mixed
        all                   -> all keys from BKS
    """
    if spec.strip().lower() == "all":
        return sorted(available_keys)

    instances = []
    parts = [p.strip() for p in spec.split(",")]

    for part in parts:
        range_match = re.match(r'^(ta)(\d+)-(ta)?(\d+)$', part)
        if range_match:
            prefix = range_match.group(1)
            start = int(range_match.group(2))
            end = int(range_match.group(4))
            if start > end:
                start, end = end, start
            for i in range(start, end + 1):
                name = f"{prefix}{i}"
                if name in available_keys:
                    instances.append(name)
                else:
                    logging.warning(f"Instance '{name}' from range not found in BKS, skipping.")
        else:
            if part in available_keys:
                instances.append(part)
            else:
                logging.warning(f"Instance '{part}' not found in BKS, skipping.")

    # Preserve order, remove duplicates
    seen = set()
    unique = []
    for inst in instances:
        if inst not in seen:
            seen.add(inst)
            unique.append(inst)
    return unique


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run GA, MA, and HGA on Taillard JSSP instances.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Instance specification examples:
  ta42,ta52,ta62,ta72      Explicit list
  ta50-ta60                Inclusive range
  ta41,ta50-ta55,ta72      Mixed list and range
  all                      All instances in BKS file
        """,
    )
    parser.add_argument(
        "--bks", type=str, default="bks.json",
        help="Path to BKS JSON file (default: bks.json)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="ga_results",
        help="Base output directory (default: ga_results)"
    )
    parser.add_argument(
        "--instances", type=str, default=None,
        help="Instance specification: e.g. 'ta42,ta52', 'ta50-ta60', 'all' (default: all)"
    )
    parser.add_argument(
        "--gap-target", type=float, default=20.0,
        help="Early-stop when BKS gap (%%) falls below this threshold (default: 20.0)"
    )
    return parser


# ===========================
# CONFIG
# ===========================
INST_DIR = "instances"

# GA parameters — tuned for dual Xeon E5-2640 v4 (35 cores, 50 GB RAM)
TRIALS = 1
POP_SIZE = 1250
TOTAL_EVAL_BUDGET = 500_000
CROSSOVER_P = 0.9
MUTATION_P = 0.1
TOURNAMENT_K = 3

# SA parameters (used by MA and HGA)
SA_ITERS = 2000
SA_TEND = 1e-3
MA_SA_ITERS = 500
HGA_SA_ITERS = 250
HGA_SA_FRACTION = 0.2

# Multiprocessing: leave 2 cores free for OS and logging
N_WORKERS = max(1, mp.cpu_count() - 2)


# ===========================
# LOGGING SETUP
# ===========================

def setup_logging(output_dir):
    """Configure dual logging to both console and file."""
    log_path = os.path.join(output_dir, "run.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return log_path


# ===========================
# TAILLARD PARSER
# ===========================

def parse_taillard(path):
    with open(path, "r") as f:
        toks = f.read().strip().split()
    n_jobs = int(toks[0])
    n_machines = int(toks[1])
    nums = list(map(int, toks[2:]))
    jobs = []
    for j in range(n_jobs):
        ops = []
        for m in range(n_machines):
            machine = nums[(j * n_machines + m) * 2]
            dur = nums[(j * n_machines + m) * 2 + 1]
            ops.append((machine, dur))
        jobs.append(ops)
    return jobs


# ===========================
# ACTIVE-SCHEDULE DECODER (Giffler & Thompson)
# ===========================

def compute_makespan(jobs, perm):
    """
    Giffler & Thompson active-schedule decoder.
    Uses the permutation as a priority list: when multiple operations
    compete for a machine, the one appearing earliest in `perm` wins.
    """
    n_jobs = len(jobs)
    n_machines = len(jobs[0])
    job_end = [0] * n_jobs
    machine_end = [0] * n_machines
    scheduled = [0] * n_jobs
    total_ops = n_jobs * n_machines

    # Pre-compute priority map: (job, op_index) -> position in perm
    priority_map = {}
    job_count = [0] * n_jobs
    for pos, j in enumerate(perm):
        k = job_count[j]
        priority_map[(j, k)] = pos
        job_count[j] += 1

    ops_scheduled = 0
    while ops_scheduled < total_ops:
        eligible = []
        for j in range(n_jobs):
            if scheduled[j] < n_machines:
                op_idx = scheduled[j]
                machine, dur = jobs[j][op_idx]
                earliest_start = max(job_end[j], machine_end[machine])
                earliest_finish = earliest_start + dur
                eligible.append((j, op_idx, machine, dur, earliest_start, earliest_finish))

        if not eligible:
            break

        min_finish = min(e[5] for e in eligible)
        conflict_machine = None
        for e in eligible:
            if e[5] == min_finish:
                conflict_machine = e[2]
                break

        conflict_set = [
            e for e in eligible
            if e[2] == conflict_machine and e[4] < min_finish
        ]

        best_op = min(conflict_set, key=lambda e: priority_map[(e[0], e[1])])
        j, op_idx, machine, dur, _, _ = best_op
        start = max(job_end[j], machine_end[machine])
        finish = start + dur
        job_end[j] = finish
        machine_end[machine] = finish
        scheduled[j] += 1
        ops_scheduled += 1

    return max(job_end)


def random_permutation(jobs):
    n_jobs = len(jobs)
    n_machines = len(jobs[0])
    perm = []
    for j in range(n_jobs):
        perm += [j] * n_machines
    random.shuffle(perm)
    return perm


# ===========================
# SA TEMPERATURE CALIBRATION
# ===========================

def calibrate_sa_temperature(jobs, perm, n_samples=200):
    """
    Estimate starting temperature such that ~80% of uphill moves
    are accepted at T0.
    """
    deltas = []
    cur_val = compute_makespan(jobs, perm)
    cur = perm[:]
    for _ in range(n_samples):
        i, j = random.sample(range(len(cur)), 2)
        cur[i], cur[j] = cur[j], cur[i]
        val = compute_makespan(jobs, cur)
        d = val - cur_val
        if d > 0:
            deltas.append(d)
        cur[i], cur[j] = cur[j], cur[i]

    if not deltas:
        return 50.0
    median_delta = sorted(deltas)[len(deltas) // 2]
    t0 = -median_delta / math.log(0.8)
    return t0


# ===========================
# GA OPERATORS
# ===========================

def order_crossover(parent1, parent2):
    """Order Crossover (OX) for permutations with repeats."""
    L = len(parent1)
    a, b = sorted(random.sample(range(L), 2))
    child = [None] * L
    child[a:b + 1] = parent1[a:b + 1]
    fill_idx = (b + 1) % L
    p2_idx = (b + 1) % L
    while None in child:
        v = parent2[p2_idx]
        if child.count(v) < parent1.count(v):
            child[fill_idx] = v
            fill_idx = (fill_idx + 1) % L
        p2_idx = (p2_idx + 1) % L
    return child


def swap_mutation(chrom, mutation_rate=0.1):
    """Single swap mutation."""
    chrom = chrom[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


def insertion_mutation(chrom, mutation_rate=0.1):
    """Remove a gene from one position and insert at another."""
    chrom = chrom[:]
    if random.random() < mutation_rate:
        i = random.randrange(len(chrom))
        j = random.randrange(len(chrom))
        gene = chrom.pop(i)
        chrom.insert(j, gene)
    return chrom


def mutate(chrom, mutation_rate=0.1):
    """Apply swap or insertion mutation with equal probability."""
    if random.random() < 0.5:
        return swap_mutation(chrom, mutation_rate)
    else:
        return insertion_mutation(chrom, mutation_rate)


def tournament_select(pop, fitnesses, k=3):
    idxs = random.sample(range(len(pop)), k)
    best = min(idxs, key=lambda i: fitnesses[i])
    return deepcopy(pop[best])


# ===========================
# CRITICAL-PATH SA
# ===========================

def build_schedule_detail(jobs, perm):
    """Decode and return full schedule detail for critical path analysis."""
    n_jobs = len(jobs)
    n_machines = len(jobs[0])
    job_next_op = [0] * n_jobs
    job_end = [0] * n_jobs
    machine_end = [0] * n_machines
    machine_order = [[] for _ in range(n_machines)]
    op_start = {}
    op_finish = {}

    for job in perm:
        op_idx = job_next_op[job]
        machine, dur = jobs[job][op_idx]
        start = max(job_end[job], machine_end[machine])
        finish = start + dur
        job_end[job] = finish
        machine_end[machine] = finish
        op_start[(job, op_idx)] = start
        op_finish[(job, op_idx)] = finish
        machine_order[machine].append((job, op_idx))
        job_next_op[job] += 1
    makespan = max(job_end)
    return makespan, op_start, op_finish, machine_order


def find_critical_path_swaps(jobs, perm):
    """
    Identify candidate swap positions on the critical path.
    Returns (i, j) index pairs in perm corresponding to adjacent
    operations on the same machine along the critical path.
    """
    n_machines = len(jobs[0])
    makespan, op_start, op_finish, machine_order = build_schedule_detail(jobs, perm)

    critical_ops = set()
    for key, ft in op_finish.items():
        if ft == makespan:
            critical_ops.add(key)

    job_count = {}
    pos_map = {}
    for idx, job in enumerate(perm):
        cnt = job_count.get(job, 0)
        pos_map[(job, cnt)] = idx
        job_count[job] = cnt + 1

    swap_candidates = []
    for m in range(n_machines):
        order = machine_order[m]
        for k in range(len(order) - 1):
            op_a = order[k]
            op_b = order[k + 1]
            if op_a in critical_ops or op_b in critical_ops:
                pi = pos_map[op_a]
                pj = pos_map[op_b]
                swap_candidates.append((pi, pj))
    return swap_candidates


def simulated_annealing_improve(jobs, perm, iters=1000, t0=50.0, tend=1e-2):
    """SA with critical-path-biased neighbourhood and calibrated temperature."""
    best = perm[:]
    best_val = compute_makespan(jobs, best)
    cur = best[:]
    cur_val = best_val

    recompute_interval = max(1, iters // 10)
    cp_swaps = []

    for it in range(iters):
        T = t0 * ((tend / t0) ** (it / max(1, iters - 1)))

        if it % recompute_interval == 0:
            cp_swaps = find_critical_path_swaps(jobs, cur)

        if cp_swaps and random.random() < 0.7:
            i, j = random.choice(cp_swaps)
        else:
            i, j = random.sample(range(len(cur)), 2)

        cur[i], cur[j] = cur[j], cur[i]
        val = compute_makespan(jobs, cur)
        delta = val - cur_val

        if delta <= 0 or random.random() < math.exp(-delta / max(1e-12, T)):
            cur_val = val
            if val < best_val:
                best_val = val
                best = cur[:]
        else:
            cur[i], cur[j] = cur[j], cur[i]

    return best, best_val


# ===========================
# POPULATION DIVERSITY
# ===========================

def deduplicate_population(pop, fitness, jobs):
    """Replace duplicate individuals with random ones to maintain diversity."""
    seen = set()
    for i in range(len(pop)):
        fingerprint = (fitness[i], tuple(pop[i][:20]))
        if fingerprint in seen:
            pop[i] = random_permutation(jobs)
            fitness[i] = compute_makespan(jobs, pop[i])
        else:
            seen.add(fingerprint)
    return pop, fitness


# ===========================
# PARALLEL POPULATION EVALUATION
# ===========================

def _eval_individual(args):
    """Worker function for parallel makespan evaluation."""
    jobs, perm = args
    return compute_makespan(jobs, perm)


def _eval_individual_sa(args):
    """Worker function for parallel SA improvement."""
    jobs, perm, sa_iters, sa_tend = args
    t0 = calibrate_sa_temperature(jobs, perm, n_samples=200)
    improved, val = simulated_annealing_improve(jobs, perm, iters=sa_iters, t0=t0, tend=sa_tend)
    return improved, val


def parallel_evaluate(pool, jobs, pop):
    """Evaluate entire population in parallel."""
    tasks = [(jobs, ind) for ind in pop]
    return pool.map(_eval_individual, tasks)


def parallel_sa_improve(pool, jobs, individuals, sa_iters, sa_tend):
    """Apply SA to a list of individuals in parallel."""
    tasks = [(jobs, ind, sa_iters, sa_tend) for ind in individuals]
    results = pool.map(_eval_individual_sa, tasks)
    improved_inds = [r[0] for r in results]
    improved_vals = [r[1] for r in results]
    return improved_inds, improved_vals


# ===========================
# GA / MA / HGA MAIN LOOPS
# ===========================

def run_GA(jobs, bks_val, pop_size=100, generations=100, cross_p=0.9, mut_p=0.1,
           seed=None, gap_target=20.0, verbose=True):
    random.seed(seed)
    np.random.seed(seed)
    logger = logging.getLogger()

    logger.info(f"  GA | Initialising population (size={pop_size})")
    with mp.Pool(N_WORKERS) as pool:
        pop = [random_permutation(jobs) for _ in range(pop_size)]
        fitness = parallel_evaluate(pool, jobs, pop)

    best_val = min(fitness)
    best_ind = deepcopy(pop[fitness.index(best_val)])
    best_iter = 0
    gap = 100.0 * (best_val - bks_val) / bks_val
    logger.info(f"  GA | Init complete | Best: {best_val} | Gap: {gap:.2f}%")

    for gen in range(1, generations + 1):
        new_pop = [deepcopy(best_ind)]
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = order_crossover(p1, p2) if random.random() < cross_p else deepcopy(p1)
            child = mutate(child, mutation_rate=mut_p)
            new_pop.append(child)

        pop = new_pop
        with mp.Pool(N_WORKERS) as pool:
            fitness = parallel_evaluate(pool, jobs, pop)

        pop, fitness = deduplicate_population(pop, fitness, jobs)

        gen_best = min(fitness)
        if gen_best < best_val:
            best_val = gen_best
            best_ind = deepcopy(pop[fitness.index(gen_best)])
            best_iter = gen
            gap = 100.0 * (best_val - bks_val) / bks_val
            logger.info(f"  GA | Gen {gen:>5}/{generations} | NEW BEST: {best_val} | Gap: {gap:.2f}%")

        elif gen % max(1, generations // 20) == 0:
            gap = 100.0 * (best_val - bks_val) / bks_val
            logger.info(f"  GA | Gen {gen:>5}/{generations} | Best: {best_val} | Gap: {gap:.2f}%")

        # Early stopping
        gap = 100.0 * (best_val - bks_val) / bks_val
        if gap < gap_target:
            logger.info(f"  GA | EARLY STOP at gen {gen} | Gap {gap:.2f}% < target {gap_target}%")
            break

    return pop, fitness, best_ind, best_val, best_iter


def run_MA(jobs, bks_val, pop_size=100, generations=100, cross_p=0.9, mut_p=0.1,
           sa_iters=500, seed=None, gap_target=20.0, verbose=True):
    random.seed(seed)
    np.random.seed(seed)
    logger = logging.getLogger()

    logger.info(f"  MA | Initialising population (size={pop_size})")
    with mp.Pool(N_WORKERS) as pool:
        pop = [random_permutation(jobs) for _ in range(pop_size)]
        fitness = parallel_evaluate(pool, jobs, pop)

    best_val = min(fitness)
    best_ind = deepcopy(pop[fitness.index(best_val)])
    best_iter = 0
    gap = 100.0 * (best_val - bks_val) / bks_val
    logger.info(f"  MA | Init complete | Best: {best_val} | Gap: {gap:.2f}%")

    for gen in range(1, generations + 1):
        new_pop = [deepcopy(best_ind)]
        offspring = []
        while len(offspring) < pop_size - 1:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = order_crossover(p1, p2) if random.random() < cross_p else deepcopy(p1)
            child = mutate(child, mutation_rate=mut_p)
            offspring.append(child)

        # Parallel SA on all offspring
        with mp.Pool(N_WORKERS) as pool:
            improved, vals = parallel_sa_improve(pool, jobs, offspring, sa_iters, SA_TEND)

        new_pop.extend(improved)
        pop = new_pop
        fitness = [compute_makespan(jobs, pop[0])] + vals

        pop, fitness = deduplicate_population(pop, fitness, jobs)

        gen_best = min(fitness)
        if gen_best < best_val:
            best_val = gen_best
            best_ind = deepcopy(pop[fitness.index(gen_best)])
            best_iter = gen
            gap = 100.0 * (best_val - bks_val) / bks_val
            logger.info(f"  MA | Gen {gen:>5}/{generations} | NEW BEST: {best_val} | Gap: {gap:.2f}%")

        elif gen % max(1, generations // 20) == 0:
            gap = 100.0 * (best_val - bks_val) / bks_val
            logger.info(f"  MA | Gen {gen:>5}/{generations} | Best: {best_val} | Gap: {gap:.2f}%")

        gap = 100.0 * (best_val - bks_val) / bks_val
        if gap < gap_target:
            logger.info(f"  MA | EARLY STOP at gen {gen} | Gap {gap:.2f}% < target {gap_target}%")
            break

    return pop, fitness, best_ind, best_val, best_iter


def run_HGA(jobs, bks_val, pop_size=100, generations=100, cross_p=0.9, mut_p=0.1,
            sa_fraction=0.2, sa_iters=500, seed=None, gap_target=20.0, verbose=True):
    random.seed(seed)
    np.random.seed(seed)
    logger = logging.getLogger()

    logger.info(f"  HGA | Initialising population (size={pop_size})")
    with mp.Pool(N_WORKERS) as pool:
        pop = [random_permutation(jobs) for _ in range(pop_size)]
        fitness = parallel_evaluate(pool, jobs, pop)

    best_val = min(fitness)
    best_ind = deepcopy(pop[fitness.index(best_val)])
    best_iter = 0
    gap = 100.0 * (best_val - bks_val) / bks_val
    logger.info(f"  HGA | Init complete | Best: {best_val} | Gap: {gap:.2f}%")

    topk = max(1, int(sa_fraction * pop_size))

    for gen in range(1, generations + 1):
        new_pop = [deepcopy(best_ind)]
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = order_crossover(p1, p2) if random.random() < cross_p else deepcopy(p1)
            child = mutate(child, mutation_rate=mut_p)
            new_pop.append(child)

        pop = new_pop
        with mp.Pool(N_WORKERS) as pool:
            fitness = parallel_evaluate(pool, jobs, pop)

        # Selective intensification: SA on top-k individuals in parallel
        idx_sorted = sorted(range(len(pop)), key=lambda i: fitness[i])
        elite_indices = idx_sorted[:topk]
        elite_inds = [pop[i] for i in elite_indices]

        with mp.Pool(N_WORKERS) as pool:
            improved, vals = parallel_sa_improve(pool, jobs, elite_inds, sa_iters, SA_TEND)

        for rank, idx in enumerate(elite_indices):
            pop[idx] = improved[rank]
            fitness[idx] = vals[rank]

        pop, fitness = deduplicate_population(pop, fitness, jobs)

        gen_best = min(fitness)
        if gen_best < best_val:
            best_val = gen_best
            best_ind = deepcopy(pop[fitness.index(gen_best)])
            best_iter = gen
            gap = 100.0 * (best_val - bks_val) / bks_val
            logger.info(f"  HGA | Gen {gen:>5}/{generations} | NEW BEST: {best_val} | Gap: {gap:.2f}%")

        elif gen % max(1, generations // 20) == 0:
            gap = 100.0 * (best_val - bks_val) / bks_val
            logger.info(f"  HGA | Gen {gen:>5}/{generations} | Best: {best_val} | Gap: {gap:.2f}%")

        gap = 100.0 * (best_val - bks_val) / bks_val
        if gap < gap_target:
            logger.info(f"  HGA | EARLY STOP at gen {gen} | Gap {gap:.2f}% < target {gap_target}%")
            break

    return pop, fitness, best_ind, best_val, best_iter


# ===========================
# EXPERIMENT DRIVER
# ===========================

def compute_generation_budgets(pop_size, total_budget, ma_sa_iters, hga_sa_iters, hga_sa_fraction):
    """Compute per-algorithm generation counts under a fixed evaluation budget."""
    ga_gens = total_budget // pop_size
    ma_cost_per_gen = pop_size + (pop_size - 1) * ma_sa_iters
    ma_gens = max(1, total_budget // ma_cost_per_gen)
    hga_topk = max(1, int(hga_sa_fraction * pop_size))
    hga_cost_per_gen = pop_size + hga_topk * hga_sa_iters
    hga_gens = max(1, total_budget // hga_cost_per_gen)
    return ga_gens, ma_gens, hga_gens


def run_trials_for_instance(inst_name, inst_file, alg_name, run_func,
                            bks_val, trials=1, params=None):
    logger = logging.getLogger()
    jobs = parse_taillard(inst_file)
    out = []

    for t in range(trials):
        seed = 1000 + t
        logger.info(f"  {alg_name} | {inst_name} | Trial {t + 1}/{trials} | Seed: {seed}")
        start = time.time()

        _, _, ind, best_val, best_iter = run_func(jobs, bks_val, seed=seed, **(params or {}))

        elapsed = time.time() - start
        gap = 100.0 * (best_val - bks_val) / bks_val

        out.append({
            "instance": inst_name,
            "algorithm": alg_name,
            "trial": t + 1,
            "seed": seed,
            "best_makespan": best_val,
            "best_iter": best_iter,
            "bks": bks_val,
            "gap_%": round(gap, 4),
            "time_s": round(elapsed, 2),
        })

        logger.info(
            f"  {alg_name} | {inst_name} | Trial {t + 1}/{trials} DONE | "
            f"Best: {best_val} | Gap: {gap:.2f}% | "
            f"Best@Gen: {best_iter} | Time: {elapsed:.1f}s"
        )

    return out


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load BKS
    if not os.path.exists(args.bks):
        print(f"ERROR: BKS file not found: {args.bks}")
        sys.exit(1)

    with open(args.bks, "r") as f:
        bks_data = json.load(f)

    available_keys = set(bks_data.keys())

    # Parse instances
    if args.instances is None:
        instances = sorted(available_keys)
    else:
        instances = parse_instance_spec(args.instances, available_keys)

    if not instances:
        print("ERROR: No valid instances specified.")
        sys.exit(1)

    # Create output directory with date identifier
    date_id = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"ga_results_{date_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_path = setup_logging(output_dir)
    logger = logging.getLogger()

    # Log configuration
    logger.info("=" * 70)
    logger.info("JSSP Metaheuristic Experiment Runner")
    logger.info("=" * 70)
    logger.info(f"Date:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"BKS file:         {args.bks}")
    logger.info(f"Output directory:  {output_dir}")
    logger.info(f"Log file:          {log_path}")
    logger.info(f"Instances:         {', '.join(instances)}")
    logger.info(f"Gap target:        {args.gap_target}%")
    logger.info(f"Population size:   {POP_SIZE}")
    logger.info(f"Total eval budget: {TOTAL_EVAL_BUDGET:,}")
    logger.info(f"Trials per inst:   {TRIALS}")
    logger.info(f"Workers (cores):   {N_WORKERS}")
    logger.info(f"Python version:    {sys.version}")
    logger.info("-" * 70)

    # Compute generation budgets
    ga_gens, ma_gens, hga_gens = compute_generation_budgets(
        POP_SIZE, TOTAL_EVAL_BUDGET, MA_SA_ITERS, HGA_SA_ITERS, HGA_SA_FRACTION
    )

    logger.info(f"Budget allocation:")
    logger.info(f"  GA  generations:  {ga_gens}")
    logger.info(f"  MA  generations:  {ma_gens} (SA iters/offspring: {MA_SA_ITERS})")
    logger.info(f"  HGA generations:  {hga_gens} (SA iters/elite: {HGA_SA_ITERS}, "
                f"fraction: {HGA_SA_FRACTION})")
    logger.info("-" * 70)

    # Verify all instance files exist
    for inst in instances:
        f = os.path.join(INST_DIR, inst)
        if not os.path.exists(f):
            logger.error(f"Instance file missing: {f}")
            sys.exit(1)

    ga_params = {
        "pop_size": POP_SIZE, "generations": ga_gens,
        "cross_p": CROSSOVER_P, "mut_p": MUTATION_P,
        "gap_target": args.gap_target,
    }
    ma_params = {
        "pop_size": POP_SIZE, "generations": ma_gens,
        "cross_p": CROSSOVER_P, "mut_p": MUTATION_P,
        "sa_iters": MA_SA_ITERS, "gap_target": args.gap_target,
    }
    hga_params = {
        "pop_size": POP_SIZE, "generations": hga_gens,
        "cross_p": CROSSOVER_P, "mut_p": MUTATION_P,
        "sa_fraction": HGA_SA_FRACTION, "sa_iters": HGA_SA_ITERS,
        "gap_target": args.gap_target,
    }

    # Save run config for reproducibility
    config_dump = {
        "bks_file": args.bks,
        "instances": instances,
        "gap_target": args.gap_target,
        "pop_size": POP_SIZE,
        "total_eval_budget": TOTAL_EVAL_BUDGET,
        "trials": TRIALS,
        "crossover_p": CROSSOVER_P,
        "mutation_p": MUTATION_P,
        "tournament_k": TOURNAMENT_K,
        "ma_sa_iters": MA_SA_ITERS,
        "hga_sa_iters": HGA_SA_ITERS,
        "hga_sa_fraction": HGA_SA_FRACTION,
        "sa_tend": SA_TEND,
        "n_workers": N_WORKERS,
        "ga_generations": ga_gens,
        "ma_generations": ma_gens,
        "hga_generations": hga_gens,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dump, f, indent=2)
    logger.info(f"Configuration saved to {os.path.join(output_dir, 'config.json')}")
    logger.info("=" * 70)

    # Run experiments
    all_data = []

    for inst_idx, inst in enumerate(instances, 1):
        inst_file = os.path.join(INST_DIR, inst)
        bks_val = bks_data[inst]

        logger.info("")
        logger.info(f"{'#' * 70}")
        logger.info(f"# Instance {inst_idx}/{len(instances)}: {inst}  |  BKS: {bks_val}")
        logger.info(f"{'#' * 70}")

        # --- GA ---
        logger.info(f"\n--- Running GA on {inst} ---")
        out_ga = run_trials_for_instance(
            inst, inst_file, "GA", run_GA, bks_val,
            trials=TRIALS, params=ga_params
        )
        pd.DataFrame(out_ga).to_csv(
            os.path.join(output_dir, f"GA_{inst}.csv"), index=False
        )
        all_data.extend(out_ga)

        # --- MA ---
        logger.info(f"\n--- Running MA on {inst} ---")
        out_ma = run_trials_for_instance(
            inst, inst_file, "MA", run_MA, bks_val,
            trials=TRIALS, params=ma_params
        )
        pd.DataFrame(out_ma).to_csv(
            os.path.join(output_dir, f"MA_{inst}.csv"), index=False
        )
        all_data.extend(out_ma)

        # --- HGA ---
        logger.info(f"\n--- Running HGA on {inst} ---")
        out_hga = run_trials_for_instance(
            inst, inst_file, "HGA", run_HGA, bks_val,
            trials=TRIALS, params=hga_params
        )
        pd.DataFrame(out_hga).to_csv(
            os.path.join(output_dir, f"HGA_{inst}.csv"), index=False
        )
        all_data.extend(out_hga)

    # Build summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("GENERATING SUMMARY")
    logger.info("=" * 70)

    summary = []
    for alg in ["GA", "MA", "HGA"]:
        for inst in instances:
            csv_path = os.path.join(output_dir, f"{alg}_{inst}.csv")
            if not os.path.exists(csv_path):
                continue
            algdf = pd.read_csv(csv_path)
            bks_val = bks_data[inst]
            best_ms = algdf["best_makespan"].min()
            mean_ms = algdf["best_makespan"].mean()
            std_ms = algdf["best_makespan"].std() if len(algdf) > 1 else 0.0
            mean_time = algdf["time_s"].mean()
            gap_best = 100.0 * (best_ms - bks_val) / bks_val
            gap_mean = 100.0 * (mean_ms - bks_val) / bks_val

            summary.append({
                "algorithm": alg,
                "instance": inst,
                "bks": bks_val,
                "best": int(best_ms),
                "mean": round(mean_ms, 1),
                "std": round(std_ms, 1),
                "gap_best_%": round(gap_best, 2),
                "gap_mean_%": round(gap_mean, 2),
                "mean_time_s": round(mean_time, 1),
            })

            logger.info(
                f"  {alg:>3} | {inst} | Best: {int(best_ms):>5} | "
                f"Mean: {mean_ms:>7.1f} | Gap(best): {gap_best:>6.2f}% | "
                f"Gap(mean): {gap_mean:>6.2f}% | Time: {mean_time:.1f}s"
            )

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Also save all raw data
    all_data_path = os.path.join(output_dir, "all_results.csv")
    pd.DataFrame(all_data).to_csv(all_data_path, index=False)

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"COMPLETE. Results in: {output_dir}")
    logger.info(f"  Summary:     {summary_path}")
    logger.info(f"  All results: {all_data_path}")
    logger.info(f"  Log:         {log_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
