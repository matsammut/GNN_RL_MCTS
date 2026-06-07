#!/usr/bin/env python3
"""
warm_start_MLP_CQL.py — v4.1 (parallel instances, gap-targeted, n-step, prioritised)

Fixes vs v4:
  1. Reward variance fix — quadratic terminal penalty replaces linear,
     raising σ from ~0.005 to ~0.05-0.15 so n-step returns can
     differentiate good from bad dispatching decisions.
  2. Default --evo-gens raised to 300 — previous default of 35 caused
     GA to hit the hard ceiling before reaching the target gap.
  3. Reward logging extended — now prints min/max alongside μ/σ, and
     warns if σ < 0.02 (signal too weak for n-step to be useful).

Parallelism (daemon-safe):
  - Outer pool: N instances collected simultaneously (instance-level parallelism).
  - Inner EA:   SERIAL per worker (daemon processes cannot spawn child pools).
  - Eval EA:    PARALLEL in main process (not a daemon, full pool available).
"""

import csv
import argparse
import os
import time
import json
import logging
import multiprocessing as mp
from copy import deepcopy
from collections import defaultdict

import numpy as np
import gym
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from dynamic_jss_wrapper import DynamicJSSWrapper
from run_ma_hga_taillard import (
    parse_taillard, compute_makespan, random_permutation,
    order_crossover, mutate, tournament_select,
    simulated_annealing_improve, calibrate_sa_temperature,
    deduplicate_population, TOURNAMENT_K, SA_TEND,
)

# ───────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="EA warm-start → MLP + CQL pipeline (v4.1).",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--eval-instances",       type=str,   nargs="+", required=True)
parser.add_argument("--num-train-instances",  type=int,   required=True)
parser.add_argument("--train-instance-seed",  type=int,   default=0)
parser.add_argument("--out",                  type=str,   default="checkpoint_mlp_cql_v4")
parser.add_argument("--bks",                  type=str,   required=True)

# Parallelism
parser.add_argument("--n-workers", type=int,
                    default=max(1, mp.cpu_count() - 2),
                    help=(
                        f"Parallel worker processes. Used at two levels: "
                        f"(1) N instances collected simultaneously, "
                        f"(2) parallel fitness eval in main-process EA. "
                        f"Default: cpu_count()-2 = {max(1, mp.cpu_count()-2)}"
                    ))

# EA
parser.add_argument("--evo-alg",         type=str,  default="HGA",
                    choices=["GA", "MA", "HGA"])
parser.add_argument("--evo-pop",         type=int,  default=50)
parser.add_argument("--evo-gens",        type=int,  default=300,
                    help="Hard ceiling on EA generations. Gap-based stopping "
                         "usually terminates before this. Default raised to 300 "
                         "to avoid hitting ceiling before target gap is reached.")
parser.add_argument("--evo-sa-iters",    type=int,  default=1000)
parser.add_argument("--evo-early-stop",  action="store_true", default=False)
parser.add_argument("--evo-target-gap",  type=float, default=15.0,
                    help="Stop EA when makespan is within this %% of LB (train) "
                         "or BKS (eval). (default: 15.0)")
parser.add_argument("--eval-target-gap", type=float, default=15.0,
                    help="Gap target for EA baseline on eval instances. (default: 15.0)")

# MLP
parser.add_argument("--mlp-hidden",   type=int, nargs="+", default=[1024, 1024, 512, 256])
parser.add_argument("--mlp-dropout",  type=float, default=0.1)
parser.add_argument("--mlp-residual", action="store_true", default=True)

# BC pre-training
parser.add_argument("--bc-pretrain-epochs", type=int,   default=20)
parser.add_argument("--bc-pretrain-lr",     type=float, default=1e-3)

# N-step returns
parser.add_argument("--nstep-returns", type=int, default=5)

# Prioritised replay
parser.add_argument("--priority-alpha", type=float, default=0.6)
parser.add_argument("--priority-beta",  type=float, default=0.4)
parser.add_argument("--priority-eps",   type=float, default=1e-6)

# CQL
parser.add_argument("--cql-epochs",               type=int,   default=400)
parser.add_argument("--cql-batch",                type=int,   default=1024)
parser.add_argument("--cql-lr",                   type=float, default=5e-5)
parser.add_argument("--cql-alpha",                type=float, default=0.2)
parser.add_argument("--cql-gamma",                type=float, default=0.99)
parser.add_argument("--cql-tau",                  type=float, default=0.001)
parser.add_argument("--cql-target-update-every",  type=int,   default=100)
parser.add_argument("--cql-demo-episodes",         type=int,   default=None)
parser.add_argument("--cql-epsilon",               type=float, default=0.05)
parser.add_argument("--buffer-cap",                type=int,   default=1_000_000)

# Eval
parser.add_argument("--eval-ea-baseline", action="store_true", default=True)
parser.add_argument("--eval-episodes",    type=int,   default=1)
parser.add_argument("--alpha",            type=float, default=0.05)

args = parser.parse_args()

# ───────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────
CHECKPOINT_DIR = os.path.abspath(args.out)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
RESULTS_TXT = os.path.join(CHECKPOINT_DIR, "results.txt")

_fmt     = "[%(asctime)s] %(levelname)-8s %(message)s"
_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=_fmt, datefmt=_datefmt)
logger = logging.getLogger()
_fh = logging.FileHandler(RESULTS_TXT, mode="w", encoding="utf-8")
_fh.setFormatter(logging.Formatter(_fmt, datefmt=_datefmt))
logger.addHandler(_fh)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ───────────────────────────────────────────────
# Dimensions
# ───────────────────────────────────────────────
EVAL_INSTANCE_PATHS = [os.path.abspath(p) for p in args.eval_instances]


def read_dimensions(path):
    with open(path) as f:
        t = f.read().split()
    return int(t[0]), int(t[1])


eval_dims = [read_dimensions(p) for p in EVAL_INSTANCE_PATHS]
if len(set(eval_dims)) != 1:
    raise ValueError("All eval instances must share the same (n_jobs, n_machines).")
NUM_JOBS, NUM_MACHINES = eval_dims[0]

if args.cql_demo_episodes is None:
    args.cql_demo_episodes = max(1, args.evo_pop // 2)

ENV_ID      = "JSSEnv:jss-v1"
CROSSOVER_P = 0.9
MUTATION_P  = 0.1
ALPHA       = args.alpha
MODEL_PATH  = os.path.join(CHECKPOINT_DIR, "mlp_cql_weights.pt")
BC_PATH     = os.path.join(CHECKPOINT_DIR, "mlp_bc_pretrain_weights.pt")

with open(args.bks) as f:
    bks_map = json.load(f)


def get_bks(instance_path):
    key = os.path.basename(instance_path).replace(".txt", "")
    return float(bks_map.get(key, 1.0))


def make_wrapped_env(instance_path):
    base = gym.make(ENV_ID, env_config={"instance_path": instance_path})
    return DynamicJSSWrapper(base)


# ───────────────────────────────────────────────
# JSSP lower bound
# ───────────────────────────────────────────────

def compute_jssp_lower_bound(jobs):
    """
    LB = max(max job total processing time, max machine total load).
    Tightest closed-form lower bound for JSSP makespan.
    Used as gap reference for synthetic training instances.
    """
    job_spans     = [sum(pt for _, pt in job) for job in jobs]
    machine_loads = defaultdict(float)
    for job in jobs:
        for m, pt in job:
            machine_loads[m] += pt
    return max(max(job_spans), max(machine_loads.values()))


# ───────────────────────────────────────────────
# Synthetic instance generation
# ───────────────────────────────────────────────

def generate_random_jssp_instance(n_jobs, n_machines, seed,
                                   proc_lo=1, proc_hi=99):
    rng   = np.random.default_rng(seed)
    lines = [f"{n_jobs} {n_machines}"]
    for _ in range(n_jobs):
        mo = rng.permutation(n_machines).tolist()
        pt = rng.integers(proc_lo, proc_hi + 1, size=n_machines).tolist()
        lines.append(" ".join(f"{m} {t}" for m, t in zip(mo, pt)))
    return "\n".join(lines) + "\n"


def create_synthetic_training_instances(n_instances, n_jobs, n_machines,
                                         master_seed, out_dir):
    synth_dir = os.path.join(out_dir, "synthetic_instances")
    os.makedirs(synth_dir, exist_ok=True)
    paths = []
    for i in range(n_instances):
        seed    = master_seed + i
        content = generate_random_jssp_instance(n_jobs, n_machines, seed=seed)
        fname   = f"synth_{n_jobs}x{n_machines}_seed{seed}.txt"
        fpath   = os.path.join(synth_dir, fname)
        with open(fpath, "w") as f:
            f.write(content)
        paths.append(fpath)
    logger.info(f"  Generated {n_instances} synthetic instance(s) "
                f"({n_jobs}×{n_machines}) in {synth_dir}")
    return paths


# ───────────────────────────────────────────────
# Flat observation helpers
# ───────────────────────────────────────────────

def extract_flat_obs(obs):
    if isinstance(obs, dict):
        return obs["real_obs"].flatten().astype(np.float32)
    return np.array(obs, dtype=np.float32).flatten()


def extract_action_mask(obs, n_jobs):
    if isinstance(obs, dict) and "action_mask" in obs:
        return obs["action_mask"].astype(np.float32)
    return np.ones(n_jobs + 1, dtype=np.float32)


# ───────────────────────────────────────────────
# Parallel EA primitives
# ───────────────────────────────────────────────

def _eval_one(args_tuple):
    """Top-level picklable wrapper for mp.Pool fitness evaluation."""
    jobs, ind = args_tuple
    return compute_makespan(jobs, ind)


def evaluate_population_parallel(jobs, population, n_workers):
    """
    Evaluate all individuals simultaneously.

    IMPORTANT: Only call from the MAIN process. Daemon worker processes
    cannot spawn child pools — doing so raises:
        AssertionError: daemonic processes are not allowed to have children
    Pass n_workers=1 when calling from inside a pool worker.
    """
    if n_workers <= 1 or len(population) <= 4:
        return [compute_makespan(jobs, ind) for ind in population]
    task_args = [(jobs, ind) for ind in population]
    with mp.Pool(processes=n_workers) as pool:
        return pool.map(_eval_one, task_args,
                        chunksize=max(1, len(population) // n_workers))


def _sa_improve_one(args_tuple):
    """Top-level picklable wrapper for SA improvement."""
    jobs, child, sa_iters = args_tuple
    t0       = calibrate_sa_temperature(jobs, child)
    improved, _ = simulated_annealing_improve(
        jobs, child, iters=sa_iters, t0=t0, tend=SA_TEND)
    return improved


def sa_improve_population_parallel(jobs, candidates, sa_iters, n_workers):
    """Run SA on all candidates. Serial when n_workers=1 (daemon-safe)."""
    if n_workers <= 1 or len(candidates) <= 2:
        result = []
        for c in candidates:
            t0 = calibrate_sa_temperature(jobs, c)
            improved, _ = simulated_annealing_improve(
                jobs, c, iters=sa_iters, t0=t0, tend=SA_TEND)
            result.append(improved)
        return result
    task_args = [(jobs, c, sa_iters) for c in candidates]
    with mp.Pool(processes=n_workers) as pool:
        return pool.map(_sa_improve_one, task_args,
                        chunksize=max(1, len(candidates) // n_workers))


# ───────────────────────────────────────────────
# EA variants with gap-based stopping
# ───────────────────────────────────────────────

def _run_ea_ga_gap(jobs, pop_size, max_gens, target_gap_pct,
                    lower_bound, n_workers=1, seed=42, early_stop=False):
    """
    GA with gap-based stopping.
    n_workers=1  → serial (safe inside daemon workers).
    n_workers>1  → parallel fitness eval (main process only).
    """
    import random
    random.seed(seed); np.random.seed(seed)

    pop      = [random_permutation(jobs) for _ in range(pop_size)]
    fitness  = evaluate_population_parallel(jobs, pop, n_workers)
    best_val = min(fitness)
    best_ind = deepcopy(pop[np.argmin(fitness)])
    patience = 0
    gen_used = 0

    for gen in range(1, max_gens + 1):
        gen_used = gen
        new_pop  = [deepcopy(best_ind)]

        while len(new_pop) < pop_size:
            p1    = tournament_select(pop, fitness, TOURNAMENT_K)
            p2    = tournament_select(pop, fitness, TOURNAMENT_K)
            child = (order_crossover(p1, p2)
                     if np.random.random() < CROSSOVER_P else deepcopy(p1))
            child = mutate(child, mutation_rate=MUTATION_P)
            new_pop.append(child)

        fitness      = evaluate_population_parallel(jobs, new_pop, n_workers)
        pop, fitness = deduplicate_population(new_pop, fitness, jobs)
        gen_best     = min(fitness)

        if gen_best < best_val:
            best_val, best_ind, patience = (
                gen_best, deepcopy(pop[fitness.index(gen_best)]), 0)
        else:
            patience += 1

        gap = (best_val - lower_bound) / lower_bound * 100.0
        print(f"    GA  | Gen {gen:>4}/{max_gens} | "
              f"Best: {best_val} | Gap: {gap:.2f}% "
              f"(target ≤{target_gap_pct}%) | workers={n_workers}", flush=True)

        if gap <= target_gap_pct:
            print(f"    GA  | Target gap reached at gen {gen}.", flush=True)
            break
        if early_stop and patience >= 3:
            print(f"    GA  | Stagnation stop at gen {gen}.", flush=True)
            break

    final_gap = (best_val - lower_bound) / lower_bound * 100.0
    return pop, fitness, best_ind, best_val, gen_used, final_gap


def _run_ea_ma_gap(jobs, pop_size, max_gens, sa_iters,
                    target_gap_pct, lower_bound, n_workers=1, seed=42):
    """MA with gap-based stopping. Serial when n_workers=1."""
    import random
    random.seed(seed); np.random.seed(seed)

    pop      = [random_permutation(jobs) for _ in range(pop_size)]
    fitness  = evaluate_population_parallel(jobs, pop, n_workers)
    best_val = min(fitness)
    best_ind = deepcopy(pop[np.argmin(fitness)])
    gen_used = 0

    for gen in range(1, max_gens + 1):
        gen_used  = gen
        offspring = []

        while len(offspring) < pop_size - 1:
            p1    = tournament_select(pop, fitness, TOURNAMENT_K)
            p2    = tournament_select(pop, fitness, TOURNAMENT_K)
            child = (order_crossover(p1, p2)
                     if np.random.random() < CROSSOVER_P else deepcopy(p1))
            child = mutate(child, mutation_rate=MUTATION_P)
            offspring.append(child)

        improved = sa_improve_population_parallel(
            jobs, offspring, sa_iters, n_workers)
        new_pop  = [deepcopy(best_ind)] + improved

        fitness      = evaluate_population_parallel(jobs, new_pop, n_workers)
        pop, fitness = deduplicate_population(new_pop, fitness, jobs)
        gen_best     = min(fitness)
        if gen_best < best_val:
            best_val, best_ind = gen_best, deepcopy(pop[fitness.index(gen_best)])

        gap = (best_val - lower_bound) / lower_bound * 100.0
        print(f"    MA  | Gen {gen:>4}/{max_gens} | "
              f"Best: {best_val} | Gap: {gap:.2f}% | workers={n_workers}",
              flush=True)
        if gap <= target_gap_pct:
            print(f"    MA  | Target gap reached at gen {gen}.", flush=True)
            break

    final_gap = (best_val - lower_bound) / lower_bound * 100.0
    return pop, fitness, best_ind, best_val, gen_used, final_gap


def _run_ea_hga_gap(jobs, pop_size, max_gens, sa_iters,
                     target_gap_pct, lower_bound, n_workers=1, seed=42):
    """
    HGA with gap-based stopping.
    n_workers=1  → fully serial (daemon-safe).
    n_workers>1  → parallel fitness + parallel SA (main process only).
    """
    import random
    random.seed(seed); np.random.seed(seed)

    pop      = [random_permutation(jobs) for _ in range(pop_size)]
    fitness  = evaluate_population_parallel(jobs, pop, n_workers)
    best_val = min(fitness)
    best_ind = deepcopy(pop[np.argmin(fitness)])
    gen_used = 0

    for gen in range(1, max_gens + 1):
        gen_used  = gen
        offspring = []

        while len(offspring) < pop_size - 1:
            p1    = tournament_select(pop, fitness, TOURNAMENT_K)
            p2    = tournament_select(pop, fitness, TOURNAMENT_K)
            child = (order_crossover(p1, p2)
                     if np.random.random() < CROSSOVER_P else deepcopy(p1))
            child = mutate(child, mutation_rate=MUTATION_P)
            offspring.append(child)

        off_fitness = evaluate_population_parallel(jobs, offspring, n_workers)
        ranked      = sorted(zip(off_fitness, offspring), key=lambda x: x[0])
        half        = len(ranked) // 2

        sa_candidates = [c for _, c in ranked[:half]]
        kept          = [c for _, c in ranked[half:]]
        improved      = sa_improve_population_parallel(
            jobs, sa_candidates, sa_iters, n_workers)

        new_pop      = [deepcopy(best_ind)] + improved + kept
        fitness      = evaluate_population_parallel(jobs, new_pop, n_workers)
        pop, fitness = deduplicate_population(new_pop, fitness, jobs)
        gen_best     = min(fitness)

        if gen_best < best_val:
            best_val, best_ind = gen_best, deepcopy(pop[fitness.index(gen_best)])

        gap = (best_val - lower_bound) / lower_bound * 100.0
        print(f"    HGA | Gen {gen:>4}/{max_gens} | "
              f"Best: {best_val} | Gap: {gap:.2f}% "
              f"(target ≤{target_gap_pct}%) | workers={n_workers}", flush=True)

        if gap <= target_gap_pct:
            print(f"    HGA | Target gap reached at gen {gen}.", flush=True)
            break

    final_gap = (best_val - lower_bound) / lower_bound * 100.0
    return pop, fitness, best_ind, best_val, gen_used, final_gap


def run_ea(jobs, args_ns, seed=42, target_gap_pct=None,
           lower_bound=None, n_workers_override=None):
    """
    Dispatch to chosen EA variant.

    n_workers_override=1  → force serial (use inside daemon workers).
    n_workers_override=None → use args_ns.n_workers (main process default).
    """
    tg = target_gap_pct   if target_gap_pct   is not None else args_ns.evo_target_gap
    lb = lower_bound      if lower_bound       is not None else compute_jssp_lower_bound(jobs)
    nw = (n_workers_override if n_workers_override is not None
          else getattr(args_ns, "n_workers", 1))

    if args_ns.evo_alg == "GA":
        return _run_ea_ga_gap(
            jobs, args_ns.evo_pop, args_ns.evo_gens, tg, lb, nw,
            seed=seed, early_stop=args_ns.evo_early_stop)
    elif args_ns.evo_alg == "MA":
        return _run_ea_ma_gap(
            jobs, args_ns.evo_pop, args_ns.evo_gens,
            args_ns.evo_sa_iters, tg, lb, nw, seed=seed)
    else:
        return _run_ea_hga_gap(
            jobs, args_ns.evo_pop, args_ns.evo_gens,
            args_ns.evo_sa_iters, tg, lb, nw, seed=seed)


# ───────────────────────────────────────────────
# N-step transition computation
# ───────────────────────────────────────────────

def compute_nstep_transitions(episode, n_steps, gamma):
    """
    Convert a flat episode list to n-step transitions.

    episode entries: (obs, action, reward, next_obs, done, mask, next_mask)

    Returns list of:
        (obs_t, action_t, G, next_obs_n, done_n, gamma_n, mask_t, next_mask_n)
    where G = sum_{k=0}^{n-1} gamma^k * r_{t+k}
    """
    T           = len(episode)
    transitions = []

    for t in range(T):
        G = 0.0; actual_n = 0; terminated = False

        for k in range(min(n_steps, T - t)):
            _, _, r, _, done_k, _, _ = episode[t + k]
            G       += (gamma ** k) * r
            actual_n = k + 1
            if done_k:
                terminated = True
                break

        n_idx = t + actual_n - 1
        _, _, _, next_obs_n, _, _, next_mask_n = episode[n_idx]
        obs_t, action_t, _, _, _, mask_t, _    = episode[t]

        transitions.append((
            obs_t, action_t, G, next_obs_n,
            float(terminated), gamma ** actual_n,
            mask_t, next_mask_n,
        ))

    return transitions


# ───────────────────────────────────────────────
# Per-step reward — FIX: quadratic terminal penalty
# ───────────────────────────────────────────────

def compute_step_reward(action, done, makespan, lower_bound, n_jobs):
    """
    LB-normalised per-step reward with quadratic terminal penalty.

    FIX vs v4: The previous linear terminal penalty produced σ≈0.005
    across all transitions — too uniform for n-step returns to distinguish
    good from bad decisions. The quadratic penalty creates variance:

        gap=0.10 → terminal = -0.05   (good schedule, near target)
        gap=0.20 → terminal = -0.20   (moderate)
        gap=0.30 → terminal = -0.45   (poor)
        gap=0.50 → terminal = -1.25   (very poor)

    This raises σ from ~0.005 to ~0.05-0.15 so n-step returns carry
    meaningful credit assignment signal back to early dispatching decisions.

    Per-step components (unchanged):
        +0.02  dispatched a real job (machine utilisation)
        -0.05  no-op / idle step
    """
    step_r = 0.02 if action < n_jobs else -0.05

    if done:
        gap     = (makespan - lower_bound) / max(lower_bound, 1.0)
        # Quadratic: penalises large gaps disproportionately more than small ones
        step_r += max(-3.0, -(gap ** 2) * 5.0)

    return step_r


# ───────────────────────────────────────────────
# N-step prioritised replay buffer
# ───────────────────────────────────────────────

class NStepPrioritizedBuffer(Dataset):
    """
    Replay buffer with per-transition priorities and n-step returns.
    float16 storage halves RAM vs float32.
    """

    def __init__(self, cap=None, priority_eps=1e-6):
        self._cap         = cap
        self.priority_eps = priority_eps

        self.states            = []
        self.next_states       = []
        self.actions           = []
        self.nstep_rewards     = []
        self.nstep_gammas      = []
        self.dones             = []
        self.action_masks      = []
        self.next_action_masks = []
        self.priorities        = []

    def _drop_oldest(self):
        self.states.pop(0);            self.next_states.pop(0)
        self.actions.pop(0);           self.nstep_rewards.pop(0)
        self.nstep_gammas.pop(0);      self.dones.pop(0)
        self.action_masks.pop(0);      self.next_action_masks.pop(0)
        self.priorities.pop(0)

    def add(self, state, action, nstep_reward, next_state, done,
            nstep_gamma, action_mask, next_action_mask, priority=1.0):
        if self._cap and len(self.actions) >= self._cap:
            self._drop_oldest()
        self.states.append(state.astype(np.float16))
        self.next_states.append(next_state.astype(np.float16))
        self.actions.append(int(action))
        self.nstep_rewards.append(float(nstep_reward))
        self.nstep_gammas.append(float(nstep_gamma))
        self.dones.append(float(done))
        self.action_masks.append(action_mask.astype(np.float32))
        self.next_action_masks.append(next_action_mask.astype(np.float32))
        self.priorities.append(float(priority))

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = float(abs(err)) + self.priority_eps

    def get_sampler(self, priority_alpha):
        prios   = np.array(self.priorities, dtype=np.float32) + self.priority_eps
        weights = torch.tensor(prios ** priority_alpha, dtype=torch.double)
        return WeightedRandomSampler(weights, num_samples=len(self),
                                     replacement=True)

    def estimate_ram_mb(self):
        if not self.states:
            return 0
        return (len(self) * 2 * self.states[0].nbytes) / (1024 ** 2)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return dict(
            buffer_idx       = idx,
            state            = torch.tensor(self.states[idx],      dtype=torch.float32),
            next_state       = torch.tensor(self.next_states[idx], dtype=torch.float32),
            action           = self.actions[idx],
            nstep_reward     = self.nstep_rewards[idx],
            nstep_gamma      = self.nstep_gammas[idx],
            done             = self.dones[idx],
            action_mask      = torch.tensor(self.action_masks[idx],      dtype=torch.float32),
            next_action_mask = torch.tensor(self.next_action_masks[idx], dtype=torch.float32),
        )


def nstep_collate(batch):
    return dict(
        buffer_indices    = torch.tensor([b["buffer_idx"]       for b in batch], dtype=torch.long),
        states            = torch.stack( [b["state"]            for b in batch]),
        next_states       = torch.stack( [b["next_state"]       for b in batch]),
        actions           = torch.tensor([b["action"]           for b in batch], dtype=torch.long),
        nstep_rewards     = torch.tensor([b["nstep_reward"]     for b in batch], dtype=torch.float32),
        nstep_gammas      = torch.tensor([b["nstep_gamma"]      for b in batch], dtype=torch.float32),
        dones             = torch.tensor([b["done"]             for b in batch], dtype=torch.float32),
        action_masks      = torch.stack( [b["action_mask"]      for b in batch]),
        next_action_masks = torch.stack( [b["next_action_mask"] for b in batch]),
    )


# ───────────────────────────────────────────────
# Single-instance collection (daemon-safe, serial EA)
# ───────────────────────────────────────────────

def collect_offline_data_for_instance(instance_path, n_episodes,
                                       epsilon, all_buffer,
                                       n_steps, gamma, args_ns,
                                       ea_n_workers=1, label=None):
    """
    Collect n-step demonstration data for one instance.

    ea_n_workers=1  (default): EA runs serially — safe inside daemon pool workers.
    ea_n_workers>1:            EA runs in parallel — main process only.

    FIX: reward logging now includes min/max and warns if σ < 0.02.
    """
    tag  = label or os.path.basename(instance_path)
    jobs = parse_taillard(instance_path)
    lb   = compute_jssp_lower_bound(jobs)

    t0 = time.time()
    pop, fitness, best_ind, best_val, gens_used, final_gap = run_ea(
        jobs, args_ns, seed=42,
        lower_bound=lb,
        n_workers_override=ea_n_workers)
    elapsed = time.time() - t0
    print(f"  [{tag}] EA done in {elapsed:.1f}s | "
          f"makespan={best_val} LB={lb:.0f} "
          f"gap={final_gap:.1f}% gens={gens_used}", flush=True)

    ranked    = sorted(zip(fitness, pop), key=lambda x: x[0])
    n_experts = max(1, args_ns.evo_pop // 2)
    experts   = [(f, p) for f, p in ranked[:n_experts]]

    env         = make_wrapped_env(instance_path)
    total_trans = 0
    reward_vals = []

    for ep in range(n_episodes):
        _, perm   = experts[ep % n_experts]
        remaining = list(perm)
        obs       = env.reset()
        done      = False
        episode   = []

        while not done:
            flat_obs    = extract_flat_obs(obs)
            action_mask = extract_action_mask(obs, NUM_JOBS)
            legal       = [a for a in range(NUM_JOBS) if action_mask[a] == 1]

            if legal:
                if epsilon > 0.0 and np.random.random() < epsilon:
                    action = int(np.random.choice(legal))
                else:
                    def first_pos(job):
                        for i, j in enumerate(remaining):
                            if j == job: return i
                        return float("inf")
                    action = min(legal, key=first_pos)
                for i, j in enumerate(remaining):
                    if j == action:
                        remaining.pop(i); break
            else:
                action = NUM_JOBS

            next_obs, _, done, _ = env.step(action)
            makespan  = (getattr(env.unwrapped, "last_time_step", float("inf"))
                         if done else 0.0)

            # FIX: uses updated compute_step_reward with quadratic terminal penalty
            reward = compute_step_reward(action, done, makespan, lb, NUM_JOBS)
            reward_vals.append(reward)

            next_flat  = extract_flat_obs(next_obs)
            next_amask = extract_action_mask(next_obs, NUM_JOBS)
            episode.append((flat_obs, action, reward, next_flat, done,
                            action_mask, next_amask))
            obs = next_obs

        for (s, a, G, s_n, done_n, gn, mask, next_mask_n) in \
                compute_nstep_transitions(episode, n_steps, gamma):
            all_buffer.add(s, a, G, s_n, done_n, gn, mask, next_mask_n,
                           priority=1.0)
        total_trans += len(episode)

    env.close()

    # FIX: extended reward stats with min/max and low-variance warning
    r_arr = np.array(reward_vals)
    print(f"  [{tag}] {total_trans:,} transitions | "
          f"reward μ={r_arr.mean():.4f} σ={r_arr.std():.4f} "
          f"min={r_arr.min():.4f} max={r_arr.max():.4f}", flush=True)

    if r_arr.std() < 0.02:
        print(f"  [{tag}] ⚠  Low reward σ={r_arr.std():.4f} — "
              f"n-step returns may not differentiate decisions effectively.",
              flush=True)


# ───────────────────────────────────────────────
# Parallel instance collection
# ───────────────────────────────────────────────

def _collect_one_instance_worker(packed):
    """
    Top-level picklable function — runs in a daemon pool worker.

    Passes ea_n_workers=1 so the EA inside runs SERIALLY.
    Daemon processes cannot spawn child pools — doing so raises:
        AssertionError: daemonic processes are not allowed to have children

    The instance-level parallelism (N instances simultaneously) provides
    the throughput improvement without requiring nested pools.
    """
    (inst_path, n_episodes, epsilon,
     n_steps, gamma, idx, total, args_dict) = packed

    import types
    args_ns = types.SimpleNamespace(**args_dict)

    label     = f"synth_{idx+1}/{total}  ({os.path.basename(inst_path)})"
    local_buf = NStepPrioritizedBuffer(cap=None,
                                        priority_eps=args_ns.priority_eps)

    collect_offline_data_for_instance(
        inst_path, n_episodes, epsilon,
        local_buf, n_steps, gamma, args_ns,
        ea_n_workers=1,    # ← SERIAL: daemon workers cannot spawn child pools
        label=label,
    )

    return list(zip(
        local_buf.states,
        local_buf.actions,
        local_buf.nstep_rewards,
        local_buf.next_states,
        local_buf.dones,
        local_buf.nstep_gammas,
        local_buf.action_masks,
        local_buf.next_action_masks,
        local_buf.priorities,
    ))


def collect_all_instances_parallel(train_paths, n_episodes, epsilon,
                                    n_steps, gamma, all_buffer,
                                    n_workers, args_ns):
    """
    Collect offline data from all synthetic instances in parallel.

    Outer pool: n_workers daemon processes, one full instance per worker.
    Inner EA:   serial per worker (daemon-safe, no nested pools).

    CPU utilisation: n_workers cores active simultaneously, each running
    a complete serial EA + rollout sequence for one instance.
    """
    logger.info(f"  Parallel collection: {n_workers} workers  "
                f"({len(train_paths)} instances)")
    logger.info(f"  EA per worker: SERIAL (daemon-safe — no nested pools).")
    logger.info(f"  Throughput: ~{n_workers} instances processed simultaneously.")

    args_dict = {k: v for k, v in vars(args_ns).items()
                 if not k.startswith("_")}

    task_args = [
        (path, n_episodes, epsilon, n_steps, gamma, i, len(train_paths), args_dict)
        for i, path in enumerate(train_paths)
    ]

    with mp.Pool(processes=n_workers) as pool:
        all_results = pool.map(_collect_one_instance_worker, task_args,
                               chunksize=1)

    total = 0
    for transitions in all_results:
        for (s, a, G, s_n, done, gn, mask, next_mask, prio) in transitions:
            all_buffer.add(s, a, G, s_n, done, gn, mask, next_mask,
                           priority=float(prio))
            total += 1

    logger.info(f"  Merged {total:,} transitions from {len(train_paths)} instances  "
                f"(buffer RAM ≈ {all_buffer.estimate_ram_mb():.0f} MB)")


# ───────────────────────────────────────────────
# MLP Q-Network with residual connections
# ───────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Residual block: x → LayerNorm → Linear → ReLU → Dropout → Linear → +x → ReLU
    Skip connections prevent vanishing gradients in deeper networks.
    """
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net  = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(F.relu(x + self.net(self.norm(x))))


class MLPQNetwork(nn.Module):
    """
    Dueling MLP Q-network with optional residual connections.

    Architecture:
        BatchNorm1d → Projection(obs→hidden[0]) →
        [ResidualBlock | Linear+ReLU+Drop] × n →
        Value head(hidden→1) + Advantage head(hidden→n_actions)
        Q = V + A − mean_legal(A)
    """

    def __init__(self, obs_dim, n_actions,
                 hidden_layers=(1024, 1024, 512, 256),
                 dropout=0.1, use_residual=True):
        super().__init__()
        self.n_actions    = n_actions
        self.input_norm   = nn.BatchNorm1d(obs_dim)
        self.use_residual = use_residual

        self.proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_layers[0]),
            nn.ReLU(), nn.Dropout(dropout),
        )

        blocks = []
        dims   = hidden_layers
        for i in range(len(dims)):
            in_d  = dims[i]
            out_d = dims[i + 1] if i + 1 < len(dims) else dims[i]
            if use_residual and in_d == out_d:
                blocks.append(ResidualBlock(in_d, dropout))
            else:
                blocks.append(nn.Sequential(
                    nn.Linear(in_d, out_d), nn.ReLU(), nn.Dropout(dropout)))
        self.trunk = nn.ModuleList(blocks)

        final_dim           = dims[-1]
        self.value_head     = nn.Linear(final_dim, 1)
        self.advantage_head = nn.Linear(final_dim, n_actions)

    def forward(self, obs, action_mask=None):
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)

        x = self.input_norm(obs)
        x = self.proj(x)
        for block in self.trunk:
            x = block(x)

        V = self.value_head(x)
        A = self.advantage_head(x)

        if action_mask is not None:
            A_masked    = A.clone()
            A_masked[action_mask == 0] = -1e9
            legal_count = action_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            A_mean      = (A_masked * action_mask).sum(dim=-1, keepdim=True) / legal_count
        else:
            A_mean = A.mean(dim=-1, keepdim=True)

        Q = V + A - A_mean
        if action_mask is not None:
            Q = Q.masked_fill(action_mask == 0, -1e9)
        if squeeze:
            Q = Q.squeeze(0)
        return Q


# ───────────────────────────────────────────────
# MLP-CQL Trainer
# ───────────────────────────────────────────────

class MLPCQLTrainer:
    """
    CQL with BC pre-training, n-step Double-DQN, prioritised replay,
    periodic hard target copy, fixed alpha, cosine LR decay.
    """

    def __init__(self, obs_dim, n_actions, hidden_layers, dropout, use_residual,
                 lr, gamma, tau, cql_alpha, n_epochs, target_update_every,
                 n_steps, priority_alpha, priority_beta, priority_eps):
        self.gamma               = gamma
        self.tau                 = tau
        self.cql_alpha           = cql_alpha
        self.target_update_every = target_update_every
        self.n_steps             = n_steps
        self.priority_alpha      = priority_alpha
        self.priority_beta       = priority_beta
        self.priority_eps        = priority_eps
        self._batch_count        = 0

        self.q_net = MLPQNetwork(
            obs_dim, n_actions, hidden_layers, dropout, use_residual).to(DEVICE)
        self.target_q_net = MLPQNetwork(
            obs_dim, n_actions, hidden_layers, dropout, use_residual).to(DEVICE)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(
            self.q_net.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(n_epochs, 1), eta_min=lr * 0.1)

    def _update_target(self):
        self._batch_count += 1
        if self._batch_count % self.target_update_every == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        else:
            for p, tp in zip(self.q_net.parameters(),
                             self.target_q_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def _td_loss_nstep(self, batch):
        """
        N-step Double-DQN Bellman loss.
        Target = G_t + γ^n * (1 − done_n) * Q_target(s_{t+n}, argmax Q_online)
        """
        states        = batch["states"].to(DEVICE)
        next_states   = batch["next_states"].to(DEVICE)
        actions       = batch["actions"].to(DEVICE)
        nstep_rewards = batch["nstep_rewards"].to(DEVICE)
        nstep_gammas  = batch["nstep_gammas"].to(DEVICE)
        dones         = batch["dones"].to(DEVICE)
        next_masks    = batch["next_action_masks"].to(DEVICE)
        curr_masks    = batch["action_masks"].to(DEVICE)

        q_values = self.q_net(states, curr_masks)
        q_taken  = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            nq_online = self.q_net(next_states, next_masks)
            nq_online = nq_online.masked_fill(next_masks == 0, -1e9)
            best_a    = nq_online.argmax(dim=1)
            nq_target = self.target_q_net(next_states, next_masks)
            nq        = nq_target.gather(1, best_a.unsqueeze(1)).squeeze(1)
            target    = nstep_rewards + nstep_gammas * (1.0 - dones) * nq

        td_errors = (q_taken - target).detach().abs().cpu().numpy()
        return F.mse_loss(q_taken, target), q_values, td_errors

    def _cql_loss(self, q_values, batch):
        """
        CQL penalty: α * E_s[ logsumexp_{a∈legal} Q(s,a) − Q(s, a_data) ]
        Restricted to legal actions only.
        """
        actions = batch["actions"].to(DEVICE)
        masks   = batch["action_masks"].to(DEVICE)

        masked_q             = q_values.clone()
        masked_q[masks == 0] = -1e9
        lse    = torch.logsumexp(masked_q, dim=1)
        q_data = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        return self.cql_alpha * (lse - q_data).mean()

    def pretrain_bc(self, buffer, n_epochs, lr, log_interval=5):
        """
        Supervised BC pre-training: CrossEntropy(Q(s,·), a_expert).
        Prevents Q-divergence in early CQL epochs by warming up the network
        to predict expert actions before Bellman bootstrapping begins.
        """
        if n_epochs <= 0:
            logger.info("  BC pre-training disabled.")
            return

        logger.info(f"\n  ── BC Pre-Training ({n_epochs} epochs, lr={lr}) ──")
        dl     = DataLoader(buffer, batch_size=512, shuffle=True,
                            collate_fn=nstep_collate, num_workers=0)
        bc_opt = optim.Adam(self.q_net.parameters(), lr=lr)
        best_acc = 0.0
        t_start  = time.perf_counter()

        for epoch in range(1, n_epochs + 1):
            self.q_net.train()
            total_loss = correct = total = 0

            for batch in dl:
                states  = batch["states"].to(DEVICE)
                actions = batch["actions"].to(DEVICE)
                masks   = batch["action_masks"].to(DEVICE)

                bc_opt.zero_grad()
                logits = self.q_net(states, masks)
                loss   = F.cross_entropy(logits, actions)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                bc_opt.step()

                total_loss += loss.item()
                correct    += (logits.argmax(1) == actions).sum().item()
                total      += len(actions)

            acc = 100.0 * correct / max(total, 1)
            if acc > best_acc:
                best_acc = acc
                torch.save(self.q_net.state_dict(), BC_PATH)

            if epoch % log_interval == 0 or epoch == n_epochs or epoch == 1:
                logger.info(
                    f"    BC Epoch {epoch:>3}/{n_epochs} | "
                    f"Loss={total_loss/len(dl):.4f} | "
                    f"Acc={acc:.1f}% (best={best_acc:.1f}%) | "
                    f"{time.perf_counter()-t_start:.0f}s")

        if os.path.exists(BC_PATH):
            self.q_net.load_state_dict(torch.load(BC_PATH, map_location=DEVICE))
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self._batch_count = 0
        logger.info(f"  BC pre-training complete | best acc: {best_acc:.1f}%")
        if best_acc < 20.0:
            logger.warning("  ⚠  BC acc < 20% — observation may lack discriminating signal.")

    def train_epoch(self, buffer, epoch, cql_batch, log_interval=20):
        """
        One CQL epoch with prioritised sampling.
        Sampler rebuilt each epoch from updated priorities.
        Priorities updated after each batch with per-sample |TD error|.
        """
        self.q_net.train()
        sampler = buffer.get_sampler(self.priority_alpha)
        dl = DataLoader(buffer, batch_size=cql_batch,
                        sampler=sampler, collate_fn=nstep_collate,
                        num_workers=0, pin_memory=False)

        etd = ecql = etot = 0.0
        running_td = running_cql = running_tot = 0.0
        n = 0
        t_data = t_fwd = t_bwd = 0.0
        t_start       = time.perf_counter()
        t_batch_start = time.perf_counter()
        total_batches = len(dl)

        for batch_idx, batch in enumerate(dl, start=1):
            t_data += time.perf_counter() - t_batch_start

            t0 = time.perf_counter()
            self.optimizer.zero_grad()
            td_loss, qv, td_errors = self._td_loss_nstep(batch)
            cql_loss               = self._cql_loss(qv, batch)
            loss                   = td_loss + cql_loss
            t_fwd                 += time.perf_counter() - t0

            if not torch.isfinite(loss):
                logger.error(f"[Ep {epoch}|B {batch_idx}] Non-finite loss "
                             f"TD={td_loss.item():.4f} CQL={cql_loss.item():.4f}")
                t_batch_start = time.perf_counter()
                continue

            t0 = time.perf_counter()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            self._update_target()
            t_bwd += time.perf_counter() - t0

            indices = batch["buffer_indices"].numpy()
            buffer.update_priorities(indices, td_errors)

            td_v, cql_v, tot_v = td_loss.item(), cql_loss.item(), loss.item()
            etd += td_v; ecql += cql_v; etot += tot_v
            running_td += td_v; running_cql += cql_v; running_tot += tot_v
            n += 1

            if batch_idx % log_interval == 0 or batch_idx == total_batches:
                w       = min(log_interval, batch_idx)
                elapsed = time.perf_counter() - t_start
                bps     = batch_idx / elapsed if elapsed > 0 else 0
                eta     = (total_batches - batch_idx) / bps if bps > 0 else 0
                hard_in = (self.target_update_every
                           - self._batch_count % self.target_update_every)
                logger.info(
                    f"[Ep {epoch:>3}|B {batch_idx:>5}/{total_batches}] "
                    f"α={self.cql_alpha:.2f} | "
                    f"Loss={running_tot/w:.4f} "
                    f"(TD={running_td/w:.4f} CQL={running_cql/w:.4f}) | "
                    f"∇={grad_norm:.3f} | "
                    f"Q(μ={qv.mean().item():.3f} "
                    f"σ={qv.std().item():.3f} "
                    f"min={qv.min().item():.3f} "
                    f"max={qv.max().item():.3f}) | "
                    f"LR={self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"tgt_in={hard_in}b | "
                    f"{bps:.1f} b/s ETA {eta:.0f}s | "
                    f"[data={t_data:.1f}s fwd={t_fwd:.1f}s bwd={t_bwd:.1f}s]"
                )
                running_td = running_cql = running_tot = 0.0

            t_batch_start = time.perf_counter()

        n = max(n, 1)
        epoch_t = time.perf_counter() - t_start
        logger.info(
            f"[Ep {epoch:>3} DONE] "
            f"AvgLoss={etot/n:.4f} (TD={etd/n:.4f} CQL={ecql/n:.4f}) | "
            f"{epoch_t:.1f}s ({n} batches) | "
            f"[data={t_data:.1f}s fwd={t_fwd:.1f}s bwd={t_bwd:.1f}s]"
        )
        return etot/n, etd/n, ecql/n

    def select_action(self, flat_obs, action_mask, greedy=True):
        self.q_net.eval()
        with torch.no_grad():
            obs_t  = torch.tensor(flat_obs,    dtype=torch.float32,
                                   device=DEVICE).unsqueeze(0)
            mask_t = torch.tensor(action_mask, dtype=torch.float32,
                                   device=DEVICE).unsqueeze(0)
            qv     = self.q_net(obs_t, mask_t).squeeze(0)
            if greedy:
                return qv.argmax().item()
            legal = torch.where(mask_t.squeeze(0) == 1)[0]
            probs = F.softmax(qv[legal], dim=0)
            return legal[torch.multinomial(probs, 1).item()].item()

    def save(self, path):
        torch.save(dict(
            q_net        = self.q_net.state_dict(),
            target_q_net = self.target_q_net.state_dict(),
            optimizer    = self.optimizer.state_dict(),
            batch_count  = self._batch_count,
        ), path)
        logger.info(f"  Model saved → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_q_net.load_state_dict(ckpt["target_q_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._batch_count = ckpt.get("batch_count", 0)


# ───────────────────────────────────────────────
# Training orchestration
# ───────────────────────────────────────────────

def train_cql(buffer, obs_dim, n_actions, args_ns):
    logger.info(f"  Dataset:           {len(buffer):,} n-step transitions  "
                f"(RAM ≈ {buffer.estimate_ram_mb():.0f} MB)")
    logger.info(f"  MLP arch:          {args_ns.mlp_hidden}  "
                f"dropout={args_ns.mlp_dropout}  residual={args_ns.mlp_residual}")
    logger.info(f"  N-step returns:    n={args_ns.nstep_returns}")
    logger.info(f"  Priority α:        {args_ns.priority_alpha}")
    logger.info(f"  CQL α (fixed):     {args_ns.cql_alpha}")
    logger.info(f"  Target update:     hard every {args_ns.cql_target_update_every} batches")
    logger.info(f"  Training:          {args_ns.cql_epochs} epochs  "
                f"batch={args_ns.cql_batch}  lr={args_ns.cql_lr}")

    trainer = MLPCQLTrainer(
        obs_dim              = obs_dim,
        n_actions            = n_actions,
        hidden_layers        = tuple(args_ns.mlp_hidden),
        dropout              = args_ns.mlp_dropout,
        use_residual         = args_ns.mlp_residual,
        lr                   = args_ns.cql_lr,
        gamma                = args_ns.cql_gamma,
        tau                  = args_ns.cql_tau,
        cql_alpha            = args_ns.cql_alpha,
        n_epochs             = args_ns.cql_epochs,
        target_update_every  = args_ns.cql_target_update_every,
        n_steps              = args_ns.nstep_returns,
        priority_alpha       = args_ns.priority_alpha,
        priority_beta        = args_ns.priority_beta,
        priority_eps         = args_ns.priority_eps,
    )

    n_params = sum(p.numel() for p in trainer.q_net.parameters() if p.requires_grad)
    logger.info(f"  Parameters:        {n_params:,}")

    trainer.pretrain_bc(buffer, args_ns.bc_pretrain_epochs, args_ns.bc_pretrain_lr)

    logger.info(f"\n  ── CQL Fine-Tuning ({args_ns.cql_epochs} epochs) ──")
    best_loss, patience, patience_limit = float("inf"), 0, 20

    for epoch in range(1, args_ns.cql_epochs + 1):
        tot, td, cql = trainer.train_epoch(
            buffer, epoch=epoch,
            cql_batch=args_ns.cql_batch, log_interval=20)
        trainer.scheduler.step()

        if tot < best_loss:
            best_loss, patience = tot, 0
            trainer.save(MODEL_PATH)
        else:
            patience += 1

        if patience >= patience_limit:
            logger.info(f"  Early stopping at epoch {epoch}.")
            break

    if os.path.exists(MODEL_PATH):
        trainer.load(MODEL_PATH)
    logger.info(f"  CQL training complete | Best loss: {best_loss:.4f}")
    return trainer


# ───────────────────────────────────────────────
# Evaluation with timing
# ───────────────────────────────────────────────

def greedy_rollout_timed(trainer, instance_path):
    """
    One greedy CQL rollout recording wall-clock time and step count.
    Returns: (makespan, steps, elapsed_seconds)
    """
    env   = make_wrapped_env(instance_path)
    obs   = env.reset()
    done  = False
    steps = 0
    t0    = time.perf_counter()

    while not done:
        flat   = extract_flat_obs(obs)
        amask  = extract_action_mask(obs, NUM_JOBS)
        action = trainer.select_action(flat, amask, greedy=True)
        obs, _, done, _ = env.step(action)
        steps += 1

    elapsed  = time.perf_counter() - t0
    makespan = getattr(env.unwrapped, "last_time_step", float("inf"))
    env.close()
    return float(makespan), steps, elapsed


def evaluate_instance(instance_path, trainer, ea_baseline,
                       n_episodes, eval_target_gap, args_ns):
    inst_key = os.path.basename(instance_path).replace(".txt", "")
    inst_bks = get_bks(instance_path)
    jobs     = parse_taillard(instance_path)
    lb       = compute_jssp_lower_bound(jobs)

    logger.info(f"  ── {inst_key}  (BKS={inst_bks:.0f}  LB={lb:.0f}) ──")

    makespans     = []
    step_counts   = []
    rollout_times = []

    for r in range(n_episodes):
        ms, steps, elapsed = greedy_rollout_timed(trainer, instance_path)
        makespans.append(ms)
        step_counts.append(steps)
        rollout_times.append(elapsed)
        logger.info(
            f"    Rollout {r+1}/{n_episodes}: makespan={ms:.0f}  "
            f"gap_bks={100*(ms-inst_bks)/inst_bks:+.2f}%  "
            f"gap_lb={(ms-lb)/lb*100:+.2f}%  "
            f"steps={steps}  time={elapsed:.3f}s"
        )

    avg       = float(np.mean(makespans))
    best      = float(np.min(makespans))
    avg_time  = float(np.mean(rollout_times))
    avg_steps = float(np.mean(step_counts))

    logger.info(f"    CQL | avg={avg:.0f} "
                f"gap_bks={100*(avg-inst_bks)/inst_bks:+.2f}% "
                f"gap_lb={(avg-lb)/lb*100:+.2f}% | "
                f"time={avg_time:.3f}s steps={avg_steps:.0f}")

    ea_best = ea_gens = ea_time_s = ea_gap_lb = None
    if ea_baseline:
        t0 = time.perf_counter()
        # Main process → uses full parallel pool (not a daemon process)
        _, _, _, ea_best_ms, ea_gens, ea_gap_bks = run_ea(
            jobs, args_ns, seed=0,
            target_gap_pct=eval_target_gap,
            lower_bound=inst_bks,
            n_workers_override=args_ns.n_workers)
        ea_time_s = time.perf_counter() - t0
        ea_best   = float(ea_best_ms)
        ea_gap_lb = (ea_best - lb) / lb * 100.0
        logger.info(
            f"    EA  | best={ea_best:.0f} "
            f"gap_bks={100*(ea_best-inst_bks)/inst_bks:+.2f}% "
            f"gap_lb={ea_gap_lb:+.2f}% | "
            f"gens={ea_gens}  time={ea_time_s:.1f}s"
        )

    return dict(
        instance       = inst_key,
        bks            = inst_bks,
        lb             = lb,
        makespans      = makespans,
        avg            = avg,
        best           = best,
        gap_bks_avg    = 100*(avg  - inst_bks)/inst_bks,
        gap_bks_best   = 100*(best - inst_bks)/inst_bks,
        gap_lb_avg     = (avg  - lb)/lb*100,
        gap_lb_best    = (best - lb)/lb*100,
        rollout_time_s = avg_time,
        rollout_steps  = int(avg_steps),
        ea_best        = ea_best,
        ea_gap_bks     = (100*(ea_best-inst_bks)/inst_bks if ea_best else None),
        ea_gap_lb      = ea_gap_lb,
        ea_gens        = ea_gens,
        ea_time_s      = ea_time_s,
    )


def print_and_save_summary(results, label):
    logger.info("")
    logger.info("─" * 95)
    logger.info(f"  {label} Summary")
    logger.info("─" * 95)

    has_ea = any(r["ea_best"] for r in results)
    hdr    = (f"  {'Instance':<12} {'BKS':>7} {'LB':>7}  "
              f"{'CQL avg':>8} {'vsBKS':>7} {'vsLB':>7}  "
              f"{'t(s)':>7} {'steps':>6}"
              + (f"  {'EA best':>8} {'vsBKS':>7} {'gens':>5} {'t(s)':>7}"
                 if has_ea else ""))
    logger.info(hdr)
    logger.info("  " + "─" * (len(hdr) - 2))

    for r in results:
        row = (f"  {r['instance']:<12} {r['bks']:>7.0f} {r['lb']:>7.0f}  "
               f"{r['avg']:>8.0f} {r['gap_bks_avg']:>+7.2f}% "
               f"{r['gap_lb_avg']:>+7.2f}%  "
               f"{r['rollout_time_s']:>7.3f} {r['rollout_steps']:>6}")
        if has_ea and r["ea_best"]:
            row += (f"  {r['ea_best']:>8.0f} {r['ea_gap_bks']:>+7.2f}% "
                    f"{r['ea_gens']:>5} {r['ea_time_s']:>7.1f}s")
        logger.info(row)

    bks_gaps = [r["gap_bks_avg"] for r in results]
    lb_gaps  = [r["gap_lb_avg"]  for r in results]
    logger.info("  " + "─" * (len(hdr) - 2))
    logger.info(f"  {'MEAN':<12} {'':>7} {'':>7}  "
                f"{'':>8} {np.mean(bks_gaps):>+7.2f}% {np.mean(lb_gaps):>+7.2f}%")

    if has_ea:
        ea_bks = [r["ea_gap_bks"] for r in results if r["ea_best"]]
        ea_t   = [r["ea_time_s"]  for r in results if r["ea_best"]]
        logger.info(f"  {'MEAN (EA)':<12} {'':>7} {'':>7}  "
                    f"{'':>8} {'':>7} {'':>7}  {'':>7} {'':>6}  "
                    f"{'':>8} {np.mean(ea_bks):>+7.2f}% "
                    f"{'':>5} {np.mean(ea_t):>7.1f}s")
    logger.info("─" * 95)

    slug     = label.lower().replace(" ", "_")
    csv_path = os.path.join(CHECKPOINT_DIR, f"mlp_cql_eval_{slug}.csv")
    fn = ["instance", "bks", "lb",
          "avg", "best", "gap_bks_avg", "gap_bks_best",
          "gap_lb_avg", "gap_lb_best",
          "rollout_time_s", "rollout_steps",
          "ea_best", "ea_gap_bks", "ea_gap_lb", "ea_gens", "ea_time_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else
                            (v if v is not None else ""))
                        for k, v in r.items() if k in fn})
    logger.info(f"  CSV → {csv_path}")


# ───────────────────────────────────────────────
# Statistical tests
# ───────────────────────────────────────────────

def _choose_test(x, y=None):
    n = len(x)
    if n < 5:
        diffs  = np.array(x) if y is None else np.array(x) - np.array(y)
        n_pos  = int((diffs > 0).sum())
        n_ties = int((diffs == 0).sum())
        n_eff  = n - n_ties
        p      = (float(2 * stats.binom.cdf(
                      min(n_pos-n_ties, n_eff-(n_pos-n_ties)), n_eff, 0.5))
                  if n_eff > 0 else 1.0)
        return float(n_pos), p, "sign test"
    elif n < 20:
        fn   = stats.wilcoxon
        s, p = (fn(x, alternative="two-sided") if y is None
                else fn(x, y, alternative="two-sided"))
        return float(s), float(p), "Wilcoxon signed-rank"
    else:
        fn   = stats.ttest_1samp if y is None else stats.ttest_rel
        s, p = fn(x, popmean=0) if y is None else fn(x, y)
        return float(s), float(p), "paired t-test"


def run_statistical_tests(results, alpha):
    logger.info("")
    logger.info("═" * 80)
    logger.info("  Statistical Tests  (out-of-sample)")
    logger.info(f"  α={alpha}  two-sided")
    logger.info("═" * 80)

    bks_gaps = [r["gap_bks_avg"] for r in results]
    lb_gaps  = [r["gap_lb_avg"]  for r in results]
    ea_bks   = [r["ea_gap_bks"]  for r in results if r["ea_best"]]
    n        = len(bks_gaps)

    stat, p, tn = _choose_test(bks_gaps)
    logger.info(f"\n  T1 CQL vs BKS  (n={n}, {tn})")
    logger.info(f"     mean={np.mean(bks_gaps):+.2f}%  p={p:.4f}  "
                + ("SIGNIFICANT" if p < alpha else "not significant"))

    stat, p, tn = _choose_test(lb_gaps)
    logger.info(f"\n  T2 CQL vs LB  (n={n}, {tn})")
    logger.info(f"     mean={np.mean(lb_gaps):+.2f}%  p={p:.4f}  "
                + ("SIGNIFICANT" if p < alpha else "not significant"))

    if ea_bks:
        n_ea    = len(ea_bks)
        stat, p, tn = _choose_test(bks_gaps[:n_ea], y=ea_bks)
        diff    = np.mean(bks_gaps[:n_ea]) - np.mean(ea_bks)
        better  = diff < 0
        logger.info(f"\n  T3 CQL vs EA  (n={n_ea}, {tn})")
        logger.info(f"     CQL={np.mean(bks_gaps[:n_ea]):+.2f}%  "
                    f"EA={np.mean(ea_bks):+.2f}%  "
                    f"diff={diff:+.2f}%  p={p:.4f}  "
                    + (f"SIGNIFICANT — CQL {'BETTER' if better else 'WORSE'} than EA"
                       if p < alpha else "not significant"))

    logger.info("")
    logger.info("  ── Descriptive Statistics ──")
    logger.info(f"  {'Metric':<45} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    logger.info("  " + "─" * 81)

    def _desc(lbl, vals):
        a = np.array(vals, dtype=float)
        logger.info(f"  {lbl:<45} {a.mean():>+8.2f} {a.std():>8.2f} "
                    f"{a.min():>+8.2f} {a.max():>+8.2f}")

    _desc("CQL gap% vs BKS",           bks_gaps)
    _desc("CQL gap% vs LB",            lb_gaps)
    if ea_bks:
        _desc("EA  gap% vs BKS",       ea_bks)
        _desc("CQL − EA (neg = CQL better)",
              (np.array(bks_gaps[:len(ea_bks)]) - np.array(ea_bks)).tolist())
    _desc("CQL rollout time (s)", [r["rollout_time_s"] for r in results])
    _desc("CQL rollout steps",    [r["rollout_steps"]  for r in results])
    if ea_bks:
        _desc("EA  run time (s)", [r["ea_time_s"] for r in results if r["ea_best"]])
    logger.info("═" * 80)


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    logger.info("=" * 80)
    logger.info("EA Warm-Start → MLP + CQL  (v4.1: parallel, gap-targeted, "
                "n-step, prioritised, quadratic reward)")
    logger.info("=" * 80)
    logger.info(f"Problem:             {NUM_JOBS} jobs × {NUM_MACHINES} machines")
    logger.info(f"Train instances:     {args.num_train_instances} synthetic")
    logger.info(f"Eval instances:      "
                f"{[os.path.basename(p) for p in EVAL_INSTANCE_PATHS]}")
    logger.info(f"Parallel workers:    {args.n_workers} of {mp.cpu_count()} cores")
    logger.info(f"  ↳ Collection:      {args.n_workers} instances simultaneously")
    logger.info(f"  ↳ EA per worker:   SERIAL (daemon-safe, no nested pools)")
    logger.info(f"  ↳ Eval EA:         PARALLEL ({args.n_workers} workers, main process)")
    logger.info(f"EA:                  {args.evo_alg}  pop={args.evo_pop}  "
                f"max_gens={args.evo_gens}  sa_iters={args.evo_sa_iters}")
    logger.info(f"EA target gap:       ≤{args.evo_target_gap}% of LB (train) | "
                f"≤{args.eval_target_gap}% of BKS (eval)")
    logger.info(f"Reward:              quadratic terminal penalty (σ target ~0.05-0.15)")
    logger.info(f"MLP:                 {args.mlp_hidden}  "
                f"dropout={args.mlp_dropout}  residual={args.mlp_residual}")
    logger.info(f"N-step returns:      n={args.nstep_returns}")
    logger.info(f"Priority α:          {args.priority_alpha}")
    logger.info(f"BC pre-train:        {args.bc_pretrain_epochs} epochs  "
                f"lr={args.bc_pretrain_lr}")
    logger.info(f"CQL α (fixed):       {args.cql_alpha}")
    logger.info(f"CQL target update:   hard every {args.cql_target_update_every} b  "
                f"τ={args.cql_tau}")
    logger.info(f"Training:            {args.cql_epochs} epochs  "
                f"batch={args.cql_batch}  lr={args.cql_lr}")
    logger.info(f"Buffer cap:          {args.buffer_cap:,}")
    logger.info(f"Device:              {DEVICE}")
    logger.info(f"Output:              {CHECKPOINT_DIR}")
    logger.info("=" * 80)

    # Stage 1: Synthetic instances
    logger.info("\n═══ Stage 1: Generating Synthetic Training Instances ═══")
    train_paths = create_synthetic_training_instances(
        args.num_train_instances, NUM_JOBS, NUM_MACHINES,
        args.train_instance_seed, CHECKPOINT_DIR)

    # Stage 2: Parallel offline data collection
    logger.info("\n═══ Stage 2: Collecting Offline EA Demonstrations (parallel) ═══")
    logger.info(f"  {args.n_workers} instances collected simultaneously")
    logger.info(f"  EA stops at ≤{args.evo_target_gap}% of LB "
                f"(max {args.evo_gens} gens — FIX: raised from 35 to 300)")

    all_buffer = NStepPrioritizedBuffer(
        cap=args.buffer_cap, priority_eps=args.priority_eps)

    env_tmp   = make_wrapped_env(train_paths[0])
    obs_tmp   = env_tmp.reset()
    obs_dim   = len(extract_flat_obs(obs_tmp))
    n_actions = env_tmp.action_space.n
    env_tmp.close()
    logger.info(f"  obs_dim={obs_dim}  n_actions={n_actions}")

    collect_all_instances_parallel(
        train_paths = train_paths,
        n_episodes  = args.cql_demo_episodes,
        epsilon     = args.cql_epsilon,
        n_steps     = args.nstep_returns,
        gamma       = args.cql_gamma,
        all_buffer  = all_buffer,
        n_workers   = args.n_workers,
        args_ns     = args,
    )

    # Stage 3: BC pre-training + CQL
    logger.info("\n═══ Stage 3: BC Pre-Training + MLP Conservative Q-Learning ═══")
    trainer = train_cql(all_buffer, obs_dim, n_actions, args)

    # Stage 4: Evaluation
    logger.info("\n═══ Stage 4: Out-of-Sample Evaluation ═══")
    logger.info(f"  EA baseline: parallel ({args.n_workers} workers), "
                f"targets ≤{args.eval_target_gap}% of BKS")
    logger.info("  NOTE: eval instances not seen during training.")
    results = [
        evaluate_instance(p, trainer,
                          ea_baseline     = args.eval_ea_baseline,
                          n_episodes      = args.eval_episodes,
                          eval_target_gap = args.eval_target_gap,
                          args_ns         = args)
        for p in EVAL_INSTANCE_PATHS
    ]
    print_and_save_summary(results, "out_of_sample")

    # Stage 5: Statistical tests
    run_statistical_tests(results, alpha=ALPHA)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Complete.")
    logger.info(f"  CQL weights: {MODEL_PATH}")
    logger.info(f"  BC weights:  {BC_PATH}")
    logger.info(f"  Log:         {RESULTS_TXT}")
    logger.info("=" * 80)
