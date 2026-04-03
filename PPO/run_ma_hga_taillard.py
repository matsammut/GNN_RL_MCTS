#!/usr/bin/env python3
"""
run_ma_hga_taillard.py

Run GA, MA (memetic = GA + local search), and HGA (GA + Simulated Annealing hybrid)
on Taillard instances ta42, ta52, ta62, ta72. 10 trials per instance.

Usage:
    python3 run_ma_hga_taillard.py

Requirements:
    python3.8+ (numpy, pandas)

Outputs:
    results_{alg}.csv for each algorithm with per-trial best makespans and gaps.
"""
import os
import random
import math
import json
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd

# ===========================
# CONFIG
# ===========================
INST_DIR = "instances"
INSTANCES = ["ta42", "ta52", "ta62", "ta72"]
BKS = {"ta42": 1939, "ta52": 2756, "ta62": 2869, "ta72": 5181}

TRIALS = 10
POP_SIZE = 100
GENERATIONS = 500000
CROSSOVER_P = 0.9
MUTATION_P = 0.1
TOURNAMENT_K = 3

# Local search / SA params (used by MA and HGA)
SA_ITERS = 2000
SA_TEND = 1e-3
MA_INTENSIFICATION_ON_OFFSPRING = True  # run SA on each offspring (MA)
HGA_SA_FRACTION = 0.2   # fraction of population to apply SA to each generation (HGA)

OUTPUT_DIR = "ma_hga_results_active_31032026"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calibrate_sa_temperature(jobs, perm, n_samples=200):
    """
    Estimate a starting temperature such that ~80% of uphill moves
    are accepted at T0, following standard SA calibration practice.
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
        cur[i], cur[j] = cur[j], cur[i]  # revert

    if not deltas:
        return 50.0  # fallback

    # Set T0 such that acceptance probability of median uphill delta is ~0.8
    median_delta = sorted(deltas)[len(deltas) // 2]
    t0 = -median_delta / math.log(0.8)
    return t0

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
# SCHEDULE DECODER & MAKESPAN
# Permutation representation: list of job ids repeated m times.
# decode -> greedy forward scheduling
# ===========================
def compute_makespan(jobs, perm):
    """
    Giffler & Thompson active-schedule decoder.
    Uses the permutation as a priority list: when multiple operations
    compete for a machine, the one appearing earliest in `perm` wins.
    Guarantees the optimal schedule is within the search space.
    """
    n_jobs = len(jobs)
    n_machines = len(jobs[0])
    job_end = [0] * n_jobs
    machine_end = [0] * n_machines
    scheduled = [0] * n_jobs  # how many ops of each job have been scheduled
    total_ops = n_jobs * n_machines

    # Pre-compute priority: for each job j and its k-th occurrence in perm,
    # store the position index. Lower position = higher priority.
    # priority_map[(j, k)] = position in perm
    priority_map = {}  # type: Dict[Tuple[int, int], int]
    job_count = [0] * n_jobs
    for pos, j in enumerate(perm):
        k = job_count[j]
        priority_map[(j, k)] = pos
        job_count[j] += 1

    ops_scheduled = 0

    while ops_scheduled < total_ops:
        # Identify all eligible operations (next unscheduled op per job)
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

        # Find the minimum completion time among all eligible ops
        min_finish = min(e[5] for e in eligible)

        # Identify the machine that achieves this minimum
        conflict_machine = None
        for e in eligible:
            if e[5] == min_finish:
                conflict_machine = e[2]
                break

        # Conflict set: all eligible ops on that machine that could
        # start before min_finish (i.e. they overlap with the winner)
        conflict_set = [
            e for e in eligible
            if e[2] == conflict_machine and e[4] < min_finish
        ]

        # Resolve conflict using the pre-computed priority map:
        # pick the operation whose (job, op_idx) has the lowest
        # position in the original permutation
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
# GA operators: OX crossover, swap mutation, tournament selection
# ===========================
def order_crossover(parent1, parent2):
    """Order Crossover (OX) for permutations with repeats."""
    L = len(parent1)
    a, b = sorted(random.sample(range(L), 2))
    child = [None] * L
    # copy slice from parent1
    child[a:b+1] = parent1[a:b+1]
    # fill remaining with parent2 in order
    fill_idx = (b+1) % L
    p2_idx = (b+1) % L
    while None in child:
        v = parent2[p2_idx]
        # count occurrences in child and how many should appear (each job appears m times)
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
    """
    Remove a gene from one position and insert it at another.
    Produces a larger perturbation than swap while preserving feasibility.
    """
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
    best = min(idxs, key=lambda i: fitnesses[i])  # minimize makespan
    return deepcopy(pop[best])

# ===========================
# Simulated Annealing local search (operates on permutations)
# Small neighborhood: swap two positions, accept if better or by SA prob
# ===========================
def build_schedule_detail(jobs, perm):
    """Decode and return full schedule detail needed for critical path analysis."""
    n_jobs = len(jobs)
    n_machines = len(jobs[0])
    job_next_op = [0] * n_jobs
    job_end = [0] * n_jobs
    machine_end = [0] * n_machines
    machine_order = [[] for _ in range(n_machines)]  # ordered list of (job, op_idx) per machine
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
    Returns a list of (i, j) index pairs in the permutation that correspond
    to adjacent operations on the same machine along the critical path.
    """
    n_machines = len(jobs[0])
    makespan, op_start, op_finish, machine_order = build_schedule_detail(jobs, perm)

    # Find all operations on the critical path (finish == makespan, trace back)
    critical_ops = set()
    # Start from any op that finishes at makespan
    for key, ft in op_finish.items():
        if ft == makespan:
            critical_ops.add(key)

    # Build position index: map (job, op_idx) -> position in perm
    job_count = {}
    pos_map = {}
    for idx, job in enumerate(perm):
        cnt = job_count.get(job, 0)
        pos_map[(job, cnt)] = idx
        job_count[job] = cnt + 1

    # Find adjacent pairs on the same machine within the critical path
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

    # Recompute critical path candidates periodically
    recompute_interval = max(1, iters // 10)

    for it in range(iters):
        T = t0 * ((tend / t0) ** (it / max(1, iters - 1)))

        # Periodically recompute critical-path swaps
        if it % recompute_interval == 0:
            cp_swaps = find_critical_path_swaps(jobs, cur)

        # With 70% probability, use a critical-path swap; else random swap
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

def deduplicate_population(pop, fitness, jobs):
    """
    Remove duplicate individuals (by makespan + first 20 genes as fingerprint)
    and replace with random individuals to maintain diversity.
    """
    seen = set()
    for i in range(len(pop)):
        # Use makespan + partial chromosome as a lightweight fingerprint
        fingerprint = (fitness[i], tuple(pop[i][:20]))
        if fingerprint in seen:
            pop[i] = random_permutation(jobs)
            fitness[i] = compute_makespan(jobs, pop[i])
        else:
            seen.add(fingerprint)
    return pop, fitness
# ===========================
# GA / MA / HGA main loops
# ===========================
def run_GA(jobs, pop_size=100, generations=100, cross_p=0.9, mut_p=0.1, seed=None, verbose=False):
    random.seed(seed); np.random.seed(seed)
    # initial population
    pop = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = [compute_makespan(jobs, ind) for ind in pop]
    best_val = min(fitness); best_ind = deepcopy(pop[fitness.index(best_val)])
    best_iter = 0
    for gen in range(1, generations+1):
        new_pop = []
        # elitism: keep best
        new_pop.append(deepcopy(best_ind))
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            if random.random() < cross_p:
                child = order_crossover(p1, p2)
            else:
                child = deepcopy(p1)
            child = swap_mutation(child, mutation_rate=mut_p)
            new_pop.append(child)
        pop = new_pop
        fitness = [compute_makespan(jobs, ind) for ind in pop]
        gen_best = min(fitness)
        if gen_best < best_val:
            best_val = gen_best
            best_ind = deepcopy(pop[fitness.index(gen_best)])
            best_iter = gen
        if verbose and gen % 10 == 0:
            print(f"GA gen {gen} best {best_val}")
    pop, fitness = deduplicate_population(pop, fitness, jobs)
    return pop, fitness, best_ind, best_val, best_iter

def run_MA(jobs, pop_size=100, generations=100, cross_p=0.9, mut_p=0.1, sa_iters=500, seed=None, verbose=False):
    random.seed(seed); np.random.seed(seed)
    pop = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = [compute_makespan(jobs, ind) for ind in pop]
    best_val = min(fitness); best_ind = deepcopy(pop[fitness.index(best_val)])
    best_iter = 0
    for gen in range(1, generations+1):
        new_pop = []
        new_pop.append(deepcopy(best_ind))  # elitism
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = order_crossover(p1, p2) if random.random() < cross_p else deepcopy(p1)
            child = swap_mutation(child, mutation_rate=mut_p)
            # local improvement on child (Memetic)
            t0 = calibrate_sa_temperature(jobs, child, n_samples=200)
            child_improved, val =  simulated_annealing_improve(jobs, child, iters=sa_iters, t0=t0, tend=SA_TEND)
            new_pop.append(child_improved)
        pop = new_pop
        fitness = [compute_makespan(jobs, ind) for ind in pop]
        gen_best = min(fitness)
        if gen_best < best_val:
            best_val = gen_best
            best_ind = deepcopy(pop[fitness.index(gen_best)])
            best_iter = gen
        if verbose and gen % 10 == 0:
            print(f"MA gen {gen} best {best_val}")
    pop, fitness = deduplicate_population(pop, fitness, jobs)
    return pop, fitness, best_ind, best_val, best_iter

def run_HGA(jobs, pop_size=100, generations=100, cross_p=0.9, mut_p=0.1, sa_fraction=0.2, sa_iters=500, seed=None, verbose=False):
    random.seed(seed); np.random.seed(seed)
    pop = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = [compute_makespan(jobs, ind) for ind in pop]
    best_val = min(fitness); best_ind = deepcopy(pop[fitness.index(best_val)])
    best_iter = 0
    for gen in range(1, generations+1):
        new_pop = []
        new_pop.append(deepcopy(best_ind))
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = order_crossover(p1, p2) if random.random() < cross_p else deepcopy(p1)
            child = swap_mutation(child, mutation_rate=mut_p)
            new_pop.append(child)
        pop = new_pop
        fitness = [compute_makespan(jobs, ind) for ind in pop]
        idx_sorted = sorted(range(len(pop)), key=lambda i: fitness[i])
        topk = max(1, int(sa_fraction * pop_size))
        for i in idx_sorted[:topk]:
            t0 = calibrate_sa_temperature(jobs, child, n_samples=200)
            improved, val = simulated_annealing_improve(jobs, pop[i], iters=sa_iters, t0=t0, tend=SA_TEND)
            pop[i] = improved
            fitness[i] = val
        gen_best = min(fitness)
        if gen_best < best_val:
            best_val = gen_best
            best_ind = deepcopy(pop[fitness.index(gen_best)])
            best_iter = gen
        if verbose and gen % 10 == 0:
            print(f"HGA gen {gen} best {best_val}")
    pop, fitness = deduplicate_population(pop, fitness, jobs)
    return pop, fitness, best_ind, best_val, best_iter

# ===========================
# Experiment driver
# ===========================
def run_trials_for_instance(inst_name, inst_file, alg_name, run_func, trials=10, params=None):
    jobs = parse_taillard(inst_file)
    out = []
    for t in range(trials):
        seed = 1000 + t
        start = time.time()
        _ ,_ , ind, best_val, best_iter = run_func(jobs, seed=seed, **(params or {}))
        elapsed = time.time() - start
        gap = 100.0 * (best_val - BKS[inst_name]) / BKS[inst_name]
        out.append({
            "instance": inst_name,
            "trial": t+1,
            "seed": seed,
            "best_makespan": best_val,
            "best_iter": best_iter,
            "gap_%": gap,
            "time_s": round(elapsed, 2)
        })
        print(f"{alg_name} {inst_name} trial {t+1}/{trials} best {best_val} gap {gap:.2f}% elapsed {elapsed:.1f}s")
    return out

def main():
    data = []
    for inst in INSTANCES:
        f = os.path.join(INST_DIR, f"{inst}")
        if not os.path.exists(f):
            raise FileNotFoundError(f"Instance file missing: {f}")
    ga_gens = GENERATIONS // POP_SIZE 
    # MA: each generation costs pop_size + (pop_size - 1) * sa_iters_ma evaluations
    MA_SA_ITERS = 500
    ma_cost_per_gen = POP_SIZE + (POP_SIZE - 1) * MA_SA_ITERS
    ma_gens = GENERATIONS // ma_cost_per_gen

    # HGA: each generation costs pop_size + topk * sa_iters_hga
    HGA_SA_ITERS = 250
    hga_topk = int(HGA_SA_FRACTION * POP_SIZE)
    hga_cost_per_gen = POP_SIZE + hga_topk * HGA_SA_ITERS 
    hga_gens = GENERATIONS // hga_cost_per_gen 

    ga_params  = {"pop_size": POP_SIZE, "generations": ga_gens,  "cross_p": CROSSOVER_P, "mut_p": MUTATION_P}
    ma_params  = {"pop_size": POP_SIZE, "generations": ma_gens,  "cross_p": CROSSOVER_P, "mut_p": MUTATION_P, "sa_iters": MA_SA_ITERS}
    hga_params = {"pop_size": POP_SIZE, "generations": hga_gens, "cross_p": CROSSOVER_P, "mut_p": MUTATION_P, "sa_fraction": HGA_SA_FRACTION, "sa_iters": HGA_SA_ITERS}
    for inst in INSTANCES:
        f = os.path.join(INST_DIR, f"{inst}")

        out_ga = run_trials_for_instance(inst, f, "GA", run_GA, trials=TRIALS, params=ga_params)
        pd.DataFrame(out_ga).to_csv(os.path.join(OUTPUT_DIR, f"GA_{inst}.csv"), index=False)
        data += out_ga

        out_ma = run_trials_for_instance(inst, f, "MA", run_MA, trials=TRIALS, params=ma_params)
        pd.DataFrame(out_ma).to_csv(os.path.join(OUTPUT_DIR, f"MA_{inst}.csv"), index=False)
        data += out_ma

        out_hga = run_trials_for_instance(inst, f, "HGA", run_HGA, trials=TRIALS, params=hga_params)
        pd.DataFrame(out_hga).to_csv(os.path.join(OUTPUT_DIR, f"HGA_{inst}.csv"), index=False)
        data += out_hga

    df = pd.DataFrame(data)
    summary = []
    for alg in ["GA", "MA", "HGA"]:
        for inst in INSTANCES:
            d = df[(df["instance"] == inst) & (df["trial"].notnull()) & (df["algorithm"].isnull() if False else True)]
            algdf = pd.read_csv(os.path.join(OUTPUT_DIR, f"{alg}_{inst}.csv"))
            bests = algdf["best_makespan"].min()
            mean = algdf["best_makespan"].mean()
            std = algdf["best_makespan"].std()
            gap_best = 100.0 * (bests - BKS[inst]) / BKS[inst]
            gap_mean = 100.0 * (mean - BKS[inst]) / BKS[inst]
            summary.append({
                "algorithm": alg,
                "instance": inst,
                "best": bests,
                "mean": mean,
                "std": std,
                "gap_best_%": gap_best,
                "gap_mean_%": gap_mean
            })
    pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    print("Done. Results in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
