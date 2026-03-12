#!/usr/bin/env python3
"""
run_ma_hga_taillard.py

Run GA, MA (memetic = GA + local search), and HGA (GA + Simulated Annealing hybrid)
on Taillard instances ta42, ta52, ta62, ta72. 10 trials per instance.
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

TRIALS = 3
POP_SIZE = 100
GENERATIONS = 100
CROSSOVER_P = 0.9
MUTATION_P = 0.1
TOURNAMENT_K = 3

# Local search / SA params (used by MA and HGA)
SA_ITERS = 2000
SA_T0 = 1.5
SA_TEND = 1e-3
MA_INTENSIFICATION_ON_OFFSPRING = True  # run SA on each offspring (MA)
HGA_SA_FRACTION = 0.2   # fraction of population to apply SA to each generation (HGA)

OUTPUT_DIR = "ma_hga_results_07022026"
HILL_COEFFS = [1.0, 1.2, 1.5, 2.0, 2.5]   
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    n_jobs = len(jobs)
    n_machines = len(jobs[0])
    job_next_op = [0] * n_jobs
    job_end = [0] * n_jobs
    machine_end = [0] * n_machines
    for job in perm:
        op_idx = job_next_op[job]
        machine, dur = jobs[job][op_idx]
        start = max(job_end[job], machine_end[machine])
        finish = start + dur
        job_end[job] = finish
        machine_end[machine] = finish
        job_next_op[job] += 1
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

def swap_mutation(chrom, mutation_rate=0.02):
    chrom = chrom[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom

def tournament_select(pop, fitnesses, k=3):
    idxs = random.sample(range(len(pop)), k)
    best = min(idxs, key=lambda i: fitnesses[i])  # minimize makespan
    return deepcopy(pop[best])

# ===========================
# Simulated Annealing local search (operates on permutations)
# Small neighborhood: swap two positions, accept if better or by SA prob
# ===========================
def simulated_annealing_improve(jobs, perm, iters=1000, t0=1.0, tend=1e-3):
    best = perm[:]
    best_val = compute_makespan(jobs, best)
    cur = best[:]
    cur_val = best_val
    for it in range(iters):
        T = t0 * ((tend / t0) ** (it / max(1, iters-1)))
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
            # revert swap
            cur[i], cur[j] = cur[j], cur[i]
    return best, best_val

# GA / MA / HGA main loops
def run_GA(jobs, pop_size=100, generations=100, cross_p=0.9, mut_p=0.1, seed=None, verbose=False):
    random.seed(seed); np.random.seed(seed)
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
    return best_ind, best_val, best_iter

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
            child_improved, val = simulated_annealing_improve(jobs, child, iters=sa_iters, t0=SA_T0, tend=SA_TEND)
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
    return best_ind, best_val, best_iter

def run_HGA(jobs,target, pop_size=100, generations=100, cross_p=0.9, mut_p=0.1, sa_fraction=0.2, sa_iters=500, seed=None, verbose=False):
    # HGA: GA but run SA on the top fraction each generation (intensification)
    random.seed(seed); np.random.seed(seed)
    pop = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = [compute_makespan(jobs, ind) for ind in pop]
    best_val = min(fitness); best_ind = deepcopy(pop[fitness.index(best_val)])

    patience = 30
    no_imp_count = 0
    start_time = time.time()
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
        # apply SA on top fraction of population (sorted by fitness)
        pop = new_pop
        fitness = [compute_makespan(jobs, ind) for ind in pop]
        
        if no_imp_count >= 5: 
            idx_sorted = sorted(range(len(pop)), key=lambda i: fitness[i])
            topk = max(1, int(sa_fraction * pop_size))
            for i in idx_sorted[:topk]:
                # Using parameters from your original file
                improved, val = simulated_annealing_improve(jobs, pop[i], iters=sa_iters)
                pop[i] = improved
                fitness[i] = val
        
        gen_best = min(fitness)
        if gen_best < best_val:
            best_val = gen_best
            best_ind = deepcopy(pop[fitness.index(gen_best)])
            no_imp_count = 0 # Reset
        else:
            no_imp_count += 1
# 4. Target Check (PPO Exit)
        if best_val <= target:
            return {"best_makespan": best_val, "time": time.time() - start_time, "status": "Success"}

        # 5. EARLY STAGNATION EXIT
        if no_imp_count >= patience:
            return {"best_makespan": best_val, "time": time.time() - start_time, "status": "DNF"}

    return {"best_makespan": best_val, "time": time.time() - start_time, "status": "DNF"}
# Experiment driver
def run_trials_for_instance(inst_name, inst_file, alg_name, run_func, trials=10, params=None):
    jobs = parse_taillard(inst_file)
    out = []
    for t in range(trials):
        seed = 1000 + t
        start = time.time()
        ind, best_val, best_iter = run_func(jobs, seed=seed, **(params or {}))
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

    ga_params = {"pop_size": POP_SIZE, "generations": GENERATIONS, "cross_p": CROSSOVER_P, "mut_p": MUTATION_P}
    ma_params = {"pop_size": POP_SIZE, "generations": GENERATIONS, "cross_p": CROSSOVER_P, "mut_p": MUTATION_P, "sa_iters": int(SA_ITERS/4)}
    hga_params = {"pop_size": POP_SIZE, "generations": GENERATIONS, "cross_p": CROSSOVER_P, "mut_p": MUTATION_P, "sa_fraction": HGA_SA_FRACTION, "sa_iters": int(SA_ITERS/8)}

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
