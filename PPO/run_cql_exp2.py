#!/usr/bin/env python3
"""
warm_start_MLP_CQL.py — v6.2

Changes over v6.1:
  BUG FIX: last-batch loss window divided by log_interval instead of actual batch count
           → added window_n counter, reset after each log interval

  BUG FIX: adaptive alpha was gated behind `epoch > warmup_epochs`, meaning α=1.0
           (full CQL pressure) during warmup epochs regardless of gap.
           Now alpha adapts from epoch 1; LR warmup is separate.

  NEW:     Stagnation detection + weight perturbation to escape flat Q-value basins.
           Triggered when gradient norm stays below --stag-grad-thresh for
           --stag-window consecutive batches.

  NEW:     CosineAnnealingWarmRestarts replaces CosineAnnealingLR so LR is
           periodically reset, allowing escape from flat regions.

  NEW:     --lr-scheduler {cosine_restart|cosine|constant|linear_decay}
           controls how the LR evolves after warmup.

  NEW:     --lr-restart-period  (T_0 for CosineAnnealingWarmRestarts, in epochs)
  NEW:     --lr-restart-mult    (T_mult — period doubling factor)
  NEW:     --lr-min-ratio       (eta_min = lr * this value, default 0.02)
  NEW:     --lr-decay-epochs    (for linear_decay: full decay over this many epochs)

  NEW:     --stag-grad-thresh   gradient norm below which a batch is "stagnant"
  NEW:     --stag-window        consecutive stagnant batches before perturbation
  NEW:     --stag-noise         base weight noise scale on perturbation

  IMPROVED: logging — window_n tracks actual batch count per log window
  IMPROVED: evaluate_instance indentation + best_ms bug (from v6.1) retained
  IMPROVED: _cql_loss uses Q(s, a_data) — standard CQL (from v6.1) retained
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
    description="EA warm-start → MLP + CQL (v6.2).",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--eval-instances",       type=str, nargs="+", required=True)
parser.add_argument("--num-train-instances",  type=int, required=True)
parser.add_argument("--train-instance-seed",  type=int, default=0)
parser.add_argument("--out",                  type=str, default="checkpoint_mlp_cql_v62")
parser.add_argument("--bks",                  type=str, required=True)

parser.add_argument("--n-workers", type=int, default=max(1, mp.cpu_count() - 2))

# EA
parser.add_argument("--evo-alg",         type=str,  default="GA", choices=["GA","MA","HGA"])
parser.add_argument("--evo-pop",         type=int,  default=50)
parser.add_argument("--evo-gens",        type=int,  default=300)
parser.add_argument("--evo-sa-iters",    type=int,  default=1000)
parser.add_argument("--evo-early-stop",  action="store_true", default=False)
parser.add_argument("--evo-target-gap",  type=float, default=15.0)
parser.add_argument("--eval-target-gap", type=float, default=15.0)

# MLP
parser.add_argument("--mlp-hidden",   type=int, nargs="+", default=[512, 512, 256, 128])
parser.add_argument("--mlp-dropout",  type=float, default=0.05)
parser.add_argument("--mlp-residual", action="store_true", default=True)

# Mixed demonstrations
parser.add_argument("--expert-episodes", type=int,   default=25)
parser.add_argument("--random-episodes", type=int,   default=25)
parser.add_argument("--cql-epsilon",     type=float, default=0.05)

# N-step + priority
parser.add_argument("--nstep-returns",  type=int,   default=15,
                    help="N-step return horizon. For ~300-step episodes use 20-30. "
                         "For ~750-step episodes use 15-20. Never use 3.")
parser.add_argument("--priority-alpha", type=float, default=0.5)
parser.add_argument("--priority-beta",  type=float, default=0.4)
parser.add_argument("--priority-eps",   type=float, default=1e-6)

# CQL
parser.add_argument("--cql-epochs",              type=int,   default=1000)
parser.add_argument("--cql-target-gap",          type=float, default=15.0)
parser.add_argument("--cql-eval-every",          type=int,   default=5)
parser.add_argument("--cql-patience",            type=int,   default=80,
                    help="Loss plateau patience (secondary stop).")
parser.add_argument("--cql-gap-patience",        type=int,   default=30,
                    help="Primary stop: eval checks without gap improvement.")
parser.add_argument("--cql-batch",               type=int,   default=32768)
parser.add_argument("--cql-lr",                  type=float, default=5e-4)
parser.add_argument("--cql-alpha",               type=float, default=0.25)
parser.add_argument("--cql-gamma",               type=float, default=0.99)
parser.add_argument("--cql-tau",                 type=float, default=0.01)
parser.add_argument("--cql-target-update-every", type=int,   default=200)
parser.add_argument("--buffer-cap",              type=int,   default=3_000_000)
parser.add_argument("--online-refresh-every",    type=int,   default=20)
parser.add_argument("--online-refresh-episodes", type=int,   default=5)
parser.add_argument("--dl-workers",              type=int,   default=4)

# LR scheduling — NEW in v6.2
parser.add_argument("--warmup-epochs",   type=int,   default=0,
                    help="Linear LR warmup epochs. Default 0 (no warmup) so alpha "
                         "adapts from epoch 1. Set >0 only if you need it.")
parser.add_argument("--lr-scheduler",   type=str,   default="cosine_restart",
                    choices=["cosine_restart", "cosine", "constant", "linear_decay"],
                    help="LR schedule after warmup.\n"
                         "  cosine_restart : CosineAnnealingWarmRestarts — periodic LR "
                         "bumps to escape flat regions (recommended).\n"
                         "  cosine         : CosineAnnealingLR — monotone decay "
                         "(original v6.0 behaviour, decays too fast on small instances).\n"
                         "  constant       : No decay at all after warmup.\n"
                         "  linear_decay   : Linear decay over --lr-decay-epochs.")
parser.add_argument("--lr-restart-period", type=int,   default=25,
                    help="[cosine_restart] T_0: epochs per LR cycle. "
                         "LR returns to cql-lr every this many epochs. "
                         "Smaller = more frequent restarts = more escape attempts. "
                         "Larger = more time to converge per cycle. "
                         "Rule of thumb: 10-20%% of total epochs.")
parser.add_argument("--lr-restart-mult",   type=float, default=2.0,
                    help="[cosine_restart] T_mult: cycle length multiplier after each "
                         "restart. 1.0=fixed period, 2.0=period doubles each restart. "
                         "Use 1.0 for persistent oscillation on hard instances.")
parser.add_argument("--lr-min-ratio",      type=float, default=0.02,
                    help="eta_min = cql-lr * lr-min-ratio. Controls how low the LR "
                         "drops at the bottom of each cosine cycle. "
                         "0.02 = drops to 2%% of base LR (default). "
                         "0.1  = drops to 10%% (shallower decay, stays more active). "
                         "1.0  = constant LR (same as --lr-scheduler constant). "
                         "Increase this to slow LR decay on small instances.")
parser.add_argument("--lr-decay-epochs",   type=int,   default=None,
                    help="[linear_decay] Epochs over which LR decays from cql-lr to "
                         "cql-lr * lr-min-ratio. Defaults to cql-epochs if not set.")

# Stagnation detection + perturbation — NEW in v6.2
parser.add_argument("--stag-grad-thresh", type=float, default=0.015,
                    help="Gradient L2-norm below this for --stag-window consecutive "
                         "batches triggers weight perturbation. "
                         "Lower = more sensitive. Typical healthy ∇ > 0.02.")
parser.add_argument("--stag-window",      type=int,   default=50,
                    help="Number of consecutive stagnant batches before perturbation. "
                         "50 ≈ 1 epoch for small instances (63 batches). "
                         "Increase if perturbations are too frequent.")
parser.add_argument("--stag-noise",       type=float, default=0.005,
                    help="Base weight-noise scale for perturbation. "
                         "Actual noise = stag-noise * (1 + epoch/50). "
                         "Too large → destabilises good Q-values. "
                         "Too small → cannot escape flat basin.")
parser.add_argument("--stag-disable",     action="store_true", default=False,
                    help="Disable stagnation detection entirely.")

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
RESULTS_TXT    = os.path.join(CHECKPOINT_DIR, "results.txt")
WORKER_TMP_DIR = os.path.join(CHECKPOINT_DIR, "_worker_tmp")
os.makedirs(WORKER_TMP_DIR, exist_ok=True)

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
    raise ValueError("All eval instances must share (n_jobs, n_machines).")
NUM_JOBS, NUM_MACHINES = eval_dims[0]
N_ACTIONS = NUM_JOBS + 1

ENV_ID      = "JSSEnv:jss-v1"
CROSSOVER_P = 0.9
MUTATION_P  = 0.1
ALPHA       = args.alpha
MODEL_PATH  = os.path.join(CHECKPOINT_DIR, "mlp_cql_weights.pt")

with open(args.bks) as f:
    bks_map = json.load(f)

def get_bks(path):
    key = os.path.basename(path).replace(".txt", "")
    return float(bks_map.get(key, 1.0))

def make_wrapped_env(path):
    base = gym.make(ENV_ID, env_config={"instance_path": path})
    return DynamicJSSWrapper(base)

# ───────────────────────────────────────────────
# JSSP lower bound
# ───────────────────────────────────────────────

def compute_jssp_lower_bound(jobs):
    job_spans     = [sum(pt for _, pt in job) for job in jobs]
    machine_loads = defaultdict(float)
    for job in jobs:
        for m, pt in job:
            machine_loads[m] += pt
    return max(max(job_spans), max(machine_loads.values()))

# ───────────────────────────────────────────────
# Synthetic instance generation
# ───────────────────────────────────────────────

def generate_random_jssp_instance(n_jobs, n_machines, seed, proc_lo=1, proc_hi=99):
    rng   = np.random.default_rng(seed)
    lines = [f"{n_jobs} {n_machines}"]
    for _ in range(n_jobs):
        mo = rng.permutation(n_machines).tolist()
        pt = rng.integers(proc_lo, proc_hi + 1, size=n_machines).tolist()
        lines.append(" ".join(f"{m} {t}" for m, t in zip(mo, pt)))
    return "\n".join(lines) + "\n"

def create_synthetic_training_instances(n_instances, n_jobs, n_machines, master_seed, out_dir):
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
    logger.info(f"  Generated {n_instances} synthetic instance(s) ({n_jobs}×{n_machines})")
    return paths

# ───────────────────────────────────────────────
# Observation helpers
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
    jobs, ind = args_tuple
    return compute_makespan(jobs, ind)

def evaluate_population_parallel(jobs, population, n_workers):
    if n_workers <= 1 or len(population) <= 4:
        return [compute_makespan(jobs, ind) for ind in population]
    with mp.Pool(processes=n_workers) as pool:
        return pool.map(_eval_one, [(jobs, ind) for ind in population],
                        chunksize=max(1, len(population)//n_workers))

def _sa_one(args_tuple):
    jobs, child, sa_iters = args_tuple
    t0 = calibrate_sa_temperature(jobs, child)
    improved, _ = simulated_annealing_improve(jobs, child, iters=sa_iters, t0=t0, tend=SA_TEND)
    return improved

def sa_improve_population_parallel(jobs, candidates, sa_iters, n_workers):
    if n_workers <= 1 or len(candidates) <= 2:
        result = []
        for c in candidates:
            t0 = calibrate_sa_temperature(jobs, c)
            improved, _ = simulated_annealing_improve(jobs, c, iters=sa_iters,
                                                      t0=t0, tend=SA_TEND)
            result.append(improved)
        return result
    with mp.Pool(processes=n_workers) as pool:
        return pool.map(_sa_one, [(jobs, c, sa_iters) for c in candidates],
                        chunksize=max(1, len(candidates)//n_workers))

# ───────────────────────────────────────────────
# EA variants
# ───────────────────────────────────────────────

def _run_ea_ga_gap(jobs, pop_size, max_gens, target_gap_pct,
                   lower_bound, n_workers=1, seed=42, early_stop=False):
    import random; random.seed(seed); np.random.seed(seed)
    pop     = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = evaluate_population_parallel(jobs, pop, n_workers)
    best    = min(fitness); best_ind = deepcopy(pop[np.argmin(fitness)])
    patience = 0; gen_used = 0
    for gen in range(1, max_gens + 1):
        gen_used = gen; new_pop = [deepcopy(best_ind)]
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = order_crossover(p1, p2) if np.random.random() < CROSSOVER_P else deepcopy(p1)
            new_pop.append(mutate(child, mutation_rate=MUTATION_P))
        fitness = evaluate_population_parallel(jobs, new_pop, n_workers)
        pop, fitness = deduplicate_population(new_pop, fitness, jobs)
        gb = min(fitness)
        if gb < best: best, best_ind, patience = gb, deepcopy(pop[fitness.index(gb)]), 0
        else:         patience += 1
        gap = (best - lower_bound) / lower_bound * 100
        print(f"    GA  | Gen {gen:>4}/{max_gens} | Best:{best} | "
              f"Gap:{gap:.2f}% | workers={n_workers}", flush=True)
        if gap <= target_gap_pct:
            print(f"    GA  | Target reached gen {gen}.", flush=True); break
        if early_stop and patience >= 3:
            print(f"    GA  | Stagnation gen {gen}.", flush=True); break
    return pop, fitness, best_ind, best, gen_used, (best - lower_bound) / lower_bound * 100

def _run_ea_hga_gap(jobs, pop_size, max_gens, sa_iters,
                    target_gap_pct, lower_bound, n_workers=1, seed=42):
    import random; random.seed(seed); np.random.seed(seed)
    pop     = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = evaluate_population_parallel(jobs, pop, n_workers)
    best    = min(fitness); best_ind = deepcopy(pop[np.argmin(fitness)]); gen_used = 0
    for gen in range(1, max_gens + 1):
        gen_used = gen; offspring = []
        while len(offspring) < pop_size - 1:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = order_crossover(p1, p2) if np.random.random() < CROSSOVER_P else deepcopy(p1)
            offspring.append(mutate(child, mutation_rate=MUTATION_P))
        off_fit  = evaluate_population_parallel(jobs, offspring, n_workers)
        ranked   = sorted(zip(off_fit, offspring), key=lambda x: x[0])
        half     = len(ranked) // 2
        improved = sa_improve_population_parallel(
            jobs, [c for _, c in ranked[:half]], sa_iters, n_workers)
        new_pop  = [deepcopy(best_ind)] + improved + [c for _, c in ranked[half:]]
        fitness  = evaluate_population_parallel(jobs, new_pop, n_workers)
        pop, fitness = deduplicate_population(new_pop, fitness, jobs)
        gb = min(fitness)
        if gb < best: best, best_ind = gb, deepcopy(pop[fitness.index(gb)])
        gap = (best - lower_bound) / lower_bound * 100
        print(f"    HGA | Gen {gen:>4}/{max_gens} | Best:{best} | "
              f"Gap:{gap:.2f}% | workers={n_workers}", flush=True)
        if gap <= target_gap_pct:
            print(f"    HGA | Target reached gen {gen}.", flush=True); break
    return pop, fitness, best_ind, best, gen_used, (best - lower_bound) / lower_bound * 100

def _run_ea_ma_gap(jobs, pop_size, max_gens, sa_iters,
                   target_gap_pct, lower_bound, n_workers=1, seed=42):
    import random; random.seed(seed); np.random.seed(seed)
    pop     = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = evaluate_population_parallel(jobs, pop, n_workers)
    best    = min(fitness); best_ind = deepcopy(pop[np.argmin(fitness)]); gen_used = 0
    for gen in range(1, max_gens + 1):
        gen_used = gen; offspring = []
        while len(offspring) < pop_size - 1:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = order_crossover(p1, p2) if np.random.random() < CROSSOVER_P else deepcopy(p1)
            offspring.append(mutate(child, mutation_rate=MUTATION_P))
        improved = sa_improve_population_parallel(jobs, offspring, sa_iters, n_workers)
        new_pop  = [deepcopy(best_ind)] + improved
        fitness  = evaluate_population_parallel(jobs, new_pop, n_workers)
        pop, fitness = deduplicate_population(new_pop, fitness, jobs)
        gb = min(fitness)
        if gb < best: best, best_ind = gb, deepcopy(pop[fitness.index(gb)])
        gap = (best - lower_bound) / lower_bound * 100
        print(f"    MA  | Gen {gen:>4}/{max_gens} | Best:{best} | "
              f"Gap:{gap:.2f}% | workers={n_workers}", flush=True)
        if gap <= target_gap_pct:
            print(f"    MA  | Target reached gen {gen}.", flush=True); break
    return pop, fitness, best_ind, best, gen_used, (best - lower_bound) / lower_bound * 100

def run_ea(jobs, args_ns, seed=42, target_gap_pct=None,
           lower_bound=None, n_workers_override=None):
    tg = target_gap_pct if target_gap_pct is not None else args_ns.evo_target_gap
    lb = lower_bound    if lower_bound    is not None else compute_jssp_lower_bound(jobs)
    nw = n_workers_override if n_workers_override is not None else getattr(args_ns, "n_workers", 1)
    if args_ns.evo_alg == "GA":
        return _run_ea_ga_gap(jobs, args_ns.evo_pop, args_ns.evo_gens, tg, lb, nw,
                              seed=seed, early_stop=args_ns.evo_early_stop)
    elif args_ns.evo_alg == "MA":
        return _run_ea_ma_gap(jobs, args_ns.evo_pop, args_ns.evo_gens,
                              args_ns.evo_sa_iters, tg, lb, nw, seed=seed)
    else:
        return _run_ea_hga_gap(jobs, args_ns.evo_pop, args_ns.evo_gens,
                               args_ns.evo_sa_iters, tg, lb, nw, seed=seed)

# ───────────────────────────────────────────────
# N-step transitions
# ───────────────────────────────────────────────

def compute_nstep_transitions(episode, n_steps, gamma):
    """
    Compute N-step returns.

    For JSSP episodes of length T:
      n_steps=3  → only last 3 transitions carry terminal reward (bad for long eps)
      n_steps=15 → reasonable for T≈750 episodes
      n_steps=30 → recommended for T≈300 episodes (20×15 instances)
      n_steps=T  → full Monte-Carlo returns

    episode: list of (obs, action, reward, next_obs, done, mask, next_mask)
    """
    T = len(episode)
    transitions = []
    for t in range(T):
        G          = 0.0
        actual_n   = 0
        terminated = False
        for k in range(min(n_steps, T - t)):
            _, _, r, _, done_k, _, _ = episode[t + k]
            G       += (gamma ** k) * r
            actual_n = k + 1
            if done_k:
                terminated = True
                break
        n_idx = t + actual_n - 1
        _, _, _, next_obs_n, _, _, next_mask_n = episode[n_idx]
        obs_t, action_t, _, _, _, mask_t, _   = episode[t]
        transitions.append((obs_t, action_t, G, next_obs_n,
                             float(terminated), gamma ** actual_n, mask_t, next_mask_n))
    return transitions

# ───────────────────────────────────────────────
# Reward shaping
# ───────────────────────────────────────────────

def shape_episode_rewards(episode_raw, makespan, lb):
    """
    Reward design for expert / random contrast.

      Terminal reward : -gap   (expert≈-0.15, random≈-0.60)
      Per-step reward : quality / T  (quality-scaled, always positive for job dispatch)
      No-op penalty   : -0.02 / T

    Expected episode totals:
      Expert (15% gap)  ≈ +0.72
      Random (60% gap)  ≈ +0.03
    """
    T       = len(episode_raw)
    gap     = max(0.0, (makespan - lb) / max(lb, 1.0))
    quality = 1.0 / (1.0 + gap)
    per_step_base = quality / max(T, 1)

    shaped = []
    for i, (s, a, s_n, done, mask, next_mask) in enumerate(episode_raw):
        if a < NUM_JOBS:
            r = per_step_base
        else:
            r = -0.02 / max(T, 1)
        if done:
            r += -gap
        shaped.append((s, a, r, s_n, done, mask, next_mask))
    return shaped

# ───────────────────────────────────────────────
# NumPy ring-buffer replay buffer
# ───────────────────────────────────────────────

class NStepPrioritizedBuffer(Dataset):
    """
    Contiguous NumPy arrays with ring-buffer semantics.
    O(1) random access, ~5× faster DataLoader throughput vs Python lists.
    """

    def __init__(self, cap, priority_eps=1e-6):
        self._cap          = cap
        self._size         = 0
        self._ptr          = 0
        self.priority_eps  = priority_eps
        self._initialized  = False
        self._is_expert    = None

    def _init_arrays(self, obs_dim, n_actions):
        if self._initialized:
            return
        c = self._cap
        self._states            = np.zeros((c, obs_dim),   dtype=np.float16)
        self._next_states       = np.zeros((c, obs_dim),   dtype=np.float16)
        self._actions           = np.zeros(c,              dtype=np.int32)
        self._nstep_rewards     = np.zeros(c,              dtype=np.float32)
        self._nstep_gammas      = np.zeros(c,              dtype=np.float32)
        self._dones             = np.zeros(c,              dtype=np.float32)
        self._action_masks      = np.zeros((c, n_actions), dtype=np.float32)
        self._next_action_masks = np.zeros((c, n_actions), dtype=np.float32)
        self._priorities        = np.ones(c,               dtype=np.float32)
        self._is_expert         = np.zeros(c,              dtype=np.bool_)
        self._obs_dim           = obs_dim
        self._n_actions         = n_actions
        self._initialized       = True

    def _write_slice(self, start, end,
                     states, next_states, actions, rewards, gammas,
                     dones, masks, next_masks, is_expert=None):
        self._states[start:end]             = states.astype(np.float16)
        self._next_states[start:end]        = next_states.astype(np.float16)
        self._actions[start:end]            = actions
        self._nstep_rewards[start:end]      = rewards
        self._nstep_gammas[start:end]       = gammas
        self._dones[start:end]              = dones
        self._action_masks[start:end]       = masks
        self._next_action_masks[start:end]  = next_masks
        self._priorities[start:end]         = 1.0
        self._is_expert[start:end]          = is_expert if is_expert is not None else False

    def bulk_extend(self, states, actions, nstep_rewards, next_states,
                    dones, nstep_gammas, action_masks, next_action_masks,
                    is_expert=None):
        n = len(actions)
        if n == 0:
            return
        obs_dim   = states.shape[1] if states.ndim > 1 else states[0].size
        n_actions = action_masks.shape[1]
        self._init_arrays(obs_dim, n_actions)

        if is_expert is None:
            is_expert = np.zeros(n, dtype=np.bool_)

        end = self._ptr + n
        if end <= self._cap:
            self._write_slice(self._ptr, end,
                              states, next_states, actions, nstep_rewards,
                              nstep_gammas, dones, action_masks, next_action_masks,
                              is_expert)
        else:
            first = self._cap - self._ptr
            self._write_slice(self._ptr, self._cap,
                              states[:first], next_states[:first], actions[:first],
                              nstep_rewards[:first], nstep_gammas[:first],
                              dones[:first], action_masks[:first],
                              next_action_masks[:first], is_expert[:first])
            rest = n - first
            self._write_slice(0, rest,
                              states[first:], next_states[first:], actions[first:],
                              nstep_rewards[first:], nstep_gammas[first:],
                              dones[first:], action_masks[first:],
                              next_action_masks[first:], is_expert[first:])

        self._ptr  = end % self._cap
        self._size = min(self._size + n, self._cap)

    def update_priorities(self, indices, td_errors):
        idx   = np.asarray(indices)
        valid = (idx >= 0) & (idx < self._size)
        self._priorities[idx[valid]] = (
            np.abs(td_errors[valid]) + self.priority_eps
        ).astype(np.float32)

    def get_sampler(self, priority_alpha):
        prios   = self._priorities[:self._size] ** priority_alpha
        weights = torch.tensor(prios, dtype=torch.double)
        return WeightedRandomSampler(weights, num_samples=self._size, replacement=True)

    def estimate_ram_mb(self):
        if not self._initialized:
            return 0
        return (self._size * (2 * self._obs_dim * 2 +
                              self._n_actions * 2 * 4 +
                              5 * 4)) / (1024 ** 2)

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return dict(
            buffer_idx        = idx,
            state             = torch.from_numpy(self._states[idx].astype(np.float32)),
            next_state        = torch.from_numpy(self._next_states[idx].astype(np.float32)),
            action            = int(self._actions[idx]),
            nstep_reward      = float(self._nstep_rewards[idx]),
            nstep_gamma       = float(self._nstep_gammas[idx]),
            done              = float(self._dones[idx]),
            action_mask       = torch.from_numpy(self._action_masks[idx].copy()),
            next_action_mask  = torch.from_numpy(self._next_action_masks[idx].copy()),
            is_expert         = bool(self._is_expert[idx]),
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
        is_expert         = torch.tensor([b["is_expert"]        for b in batch], dtype=torch.bool),
    )

# ───────────────────────────────────────────────
# Offline collection — mixed demonstrations
# ───────────────────────────────────────────────

def _run_episode(env, policy_fn):
    obs  = env.reset(); done = False; episode = []
    while not done:
        flat   = extract_flat_obs(obs)
        mask   = extract_action_mask(obs, NUM_JOBS)
        action = policy_fn(obs, flat, mask)
        next_obs, _, done, _ = env.step(action)
        episode.append((flat, action, extract_flat_obs(next_obs), done,
                        mask, extract_action_mask(next_obs, NUM_JOBS)))
        obs = next_obs
    makespan = getattr(env.unwrapped, "last_time_step", float("inf"))
    return episode, float(makespan)

def collect_offline_data_for_instance(instance_path, expert_episodes, random_episodes,
                                      epsilon, n_steps, gamma, args_ns,
                                      ea_n_workers=1, label=None):
    tag  = label or os.path.basename(instance_path)
    jobs = parse_taillard(instance_path)
    lb   = compute_jssp_lower_bound(jobs)

    t0 = time.time()
    pop, fitness, best_ind, best_val, gens_used, final_gap = run_ea(
        jobs, args_ns, seed=42, lower_bound=lb, n_workers_override=ea_n_workers)
    print(f"  [{tag}] EA done in {time.time()-t0:.1f}s | "
          f"makespan={best_val} LB={lb:.0f} gap={final_gap:.1f}% gens={gens_used}",
          flush=True)

    ranked    = sorted(zip(fitness, pop), key=lambda x: x[0])
    n_experts = max(1, args_ns.evo_pop // 2)
    experts   = ranked[:n_experts]

    env = make_wrapped_env(instance_path)
    all_s=[]; all_a=[]; all_G=[]; all_sn=[]
    all_dn=[]; all_gn=[]; all_m=[]; all_mn=[]
    all_is_expert = []
    expert_rewards=[]; random_rewards=[]

    # Expert rollouts
    for ep in range(expert_episodes):
        _, perm    = experts[ep % n_experts]
        remaining  = list(perm)
        def expert_policy(obs, flat, mask, _rem=remaining):
            legal = [a for a in range(NUM_JOBS) if mask[a] == 1]
            if not legal: return NUM_JOBS
            if epsilon > 0.0 and np.random.random() < epsilon:
                return int(np.random.choice(legal))
            def fp(job):
                for i, j in enumerate(_rem):
                    if j == job: return i
                return float("inf")
            action = min(legal, key=fp)
            for i, j in enumerate(_rem):
                if j == action: _rem.pop(i); break
            return action
        episode_raw, makespan = _run_episode(env, expert_policy)
        shaped = shape_episode_rewards(episode_raw, makespan, lb)
        expert_rewards.extend([r for _, _, r, _, _, _, _ in shaped])
        for (s, a, G, s_n, d_n, gn, mask, nmask) in compute_nstep_transitions(
                shaped, n_steps, gamma):
            all_s.append(s); all_a.append(a); all_G.append(G)
            all_sn.append(s_n); all_dn.append(d_n); all_gn.append(gn)
            all_m.append(mask); all_mn.append(nmask)
            all_is_expert.append(True)

    # Random rollouts
    for ep in range(random_episodes):
        def random_policy(obs, flat, mask):
            legal = [a for a in range(NUM_JOBS) if mask[a] == 1]
            return int(np.random.choice(legal)) if legal else NUM_JOBS
        episode_raw, makespan = _run_episode(env, random_policy)
        shaped = shape_episode_rewards(episode_raw, makespan, lb)
        random_rewards.extend([r for _, _, r, _, _, _, _ in shaped])
        for (s, a, G, s_n, d_n, gn, mask, nmask) in compute_nstep_transitions(
                shaped, n_steps, gamma):
            all_s.append(s); all_a.append(a); all_G.append(G)
            all_sn.append(s_n); all_dn.append(d_n); all_gn.append(gn)
            all_m.append(mask); all_mn.append(nmask)
            all_is_expert.append(False)

    env.close()
    er    = np.array(expert_rewards, dtype=np.float32)
    rr    = np.array(random_rewards,  dtype=np.float32)
    all_r = np.concatenate([er, rr])
    contrast = er.mean() - rr.mean()
    status   = ("✓ healthy" if all_r.std() >= 0.05
                else ("⚠  low" if all_r.std() >= 0.01 else "✗ BAD"))
    print(f"  [{tag}] {len(all_a):,} transitions | "
          f"expert μ={er.mean():.4f} | random μ={rr.mean():.4f} | "
          f"contrast={contrast:.4f} | σ={all_r.std():.4f} {status}", flush=True)

    return (np.array(all_s,         dtype=np.float16),
            np.array(all_a,         dtype=np.int32),
            np.array(all_G,         dtype=np.float32),
            np.array(all_sn,        dtype=np.float16),
            np.array(all_dn,        dtype=np.float32),
            np.array(all_gn,        dtype=np.float32),
            np.array(all_m,         dtype=np.float32),
            np.array(all_mn,        dtype=np.float32),
            np.array(all_is_expert, dtype=np.bool_))

# ───────────────────────────────────────────────
# Worker + parallel collection
# ───────────────────────────────────────────────

def _collect_worker(packed):
    (inst_path, expert_eps, random_eps, epsilon,
     n_steps, gamma, idx, total, tmp_dir, args_dict) = packed
    import types; args_ns = types.SimpleNamespace(**args_dict)
    label  = f"synth_{idx+1}/{total}  ({os.path.basename(inst_path)})"
    arrays = collect_offline_data_for_instance(
        inst_path, expert_eps, random_eps, epsilon, n_steps, gamma, args_ns,
        ea_n_workers=1, label=label)
    npz_path = os.path.join(tmp_dir, f"worker_{idx:04d}.npz")
    np.savez_compressed(
        npz_path,
        states=arrays[0], actions=arrays[1], nstep_rewards=arrays[2],
        next_states=arrays[3], dones=arrays[4], nstep_gammas=arrays[5],
        action_masks=arrays[6], next_action_masks=arrays[7],
        is_expert=arrays[8])
    print(f"  [{label}] saved {npz_path}", flush=True)
    return npz_path

def collect_all_instances_parallel(train_paths, expert_episodes, random_episodes,
                                   epsilon, n_steps, gamma, all_buffer,
                                   n_workers, args_ns, tmp_dir):
    logger.info(f"  Parallel collection: {n_workers} workers  ({len(train_paths)} instances)")
    logger.info(f"  Expert: {expert_episodes}  Random: {random_episodes}  "
                f"Total per instance: {expert_episodes+random_episodes}")

    args_dict = {k: v for k, v in vars(args_ns).items() if not k.startswith("_")}
    task_args = [(path, expert_episodes, random_episodes, epsilon, n_steps, gamma,
                  i, len(train_paths), tmp_dir, args_dict)
                 for i, path in enumerate(train_paths)]

    with mp.Pool(processes=n_workers) as pool:
        npz_paths = pool.map(_collect_worker, task_args, chunksize=1)

    logger.info(f"  Workers done. Merging {len(npz_paths)} files (bulk_extend)...")
    total = 0
    for fi, npz_path in enumerate(npz_paths, 1):
        if not npz_path or not os.path.exists(npz_path):
            logger.warning(f"  Missing: {npz_path}"); continue
        t0      = time.perf_counter()
        data    = np.load(npz_path)
        n       = len(data["actions"])
        ie      = data["is_expert"] if "is_expert" in data else None
        all_buffer.bulk_extend(
            states=data["states"], actions=data["actions"],
            nstep_rewards=data["nstep_rewards"], next_states=data["next_states"],
            dones=data["dones"], nstep_gammas=data["nstep_gammas"],
            action_masks=data["action_masks"], next_action_masks=data["next_action_masks"],
            is_expert=ie)
        total  += n
        elapsed = time.perf_counter() - t0
        data.close()
        try:    os.remove(npz_path)
        except OSError: pass
        logger.info(f"  [{fi:>3}/{len(npz_paths)}] {os.path.basename(npz_path)} "
                    f"({n:,} trans, {elapsed:.1f}s) | total:{total:,}  "
                    f"RAM≈{all_buffer.estimate_ram_mb():.0f}MB")

    logger.info(f"  Merge complete: {total:,} transitions  "
                f"RAM≈{all_buffer.estimate_ram_mb():.0f}MB")

# ───────────────────────────────────────────────
# MLP Q-Network
# ───────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net  = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(F.relu(x + self.net(self.norm(x))))


class MLPQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions,
                 hidden_layers=(512, 512, 256, 128), dropout=0.05, use_residual=True):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(obs_dim)
        self.proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_layers[0]), nn.ReLU(), nn.Dropout(dropout))
        blocks = []
        for i in range(len(hidden_layers)):
            in_d  = hidden_layers[i]
            out_d = hidden_layers[i+1] if i+1 < len(hidden_layers) else hidden_layers[i]
            if use_residual and in_d == out_d:
                blocks.append(ResidualBlock(in_d, dropout))
            else:
                blocks.append(nn.Sequential(
                    nn.Linear(in_d, out_d), nn.ReLU(), nn.Dropout(dropout)))
        self.trunk          = nn.ModuleList(blocks)
        self.value_head     = nn.Linear(hidden_layers[-1], 1)
        self.advantage_head = nn.Linear(hidden_layers[-1], n_actions)

    def forward(self, obs, action_mask=None):
        squeeze = obs.dim() == 1
        if squeeze: obs = obs.unsqueeze(0)
        x = self.proj(self.input_norm(obs))
        for block in self.trunk:
            x = block(x)
        V = self.value_head(x)
        A = self.advantage_head(x)

        if action_mask is not None:
            lc     = action_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            A_mean = (A * action_mask).sum(dim=-1, keepdim=True) / lc
        else:
            A_mean = A.mean(dim=-1, keepdim=True)

        Q = V + A - A_mean
        if action_mask is not None:
            Q = Q.masked_fill(action_mask == 0, -1e9)
        if squeeze: Q = Q.squeeze(0)
        return Q

# ───────────────────────────────────────────────
# v6.2 CQL Trainer
# ───────────────────────────────────────────────

def _build_lr_scheduler(optimizer, args_ns, base_lr):
    """
    Build the LR scheduler based on --lr-scheduler.

    cosine_restart  (recommended for small instances):
        CosineAnnealingWarmRestarts — LR periodically returns to base_lr.
        Controlled by --lr-restart-period, --lr-restart-mult, --lr-min-ratio.
        Prevents LR from ever permanently reaching near-zero.

    cosine (original):
        CosineAnnealingLR — monotone decay over all epochs.
        For small instances with few batches/epoch this reaches eta_min
        within the first 10-20 epochs and stays there. Not recommended.

    constant:
        No decay. LR stays at base_lr throughout. Useful for debugging
        or when you want to rely only on gradient clipping for stability.

    linear_decay:
        Linear decay from base_lr → base_lr*lr-min-ratio over lr-decay-epochs.
        More predictable than cosine; use when cosine restarts cause instability.
    """
    eta_min      = base_lr * args_ns.lr_min_ratio
    sched_type   = args_ns.lr_scheduler
    n_epochs     = args_ns.cql_epochs
    warmup       = args_ns.warmup_epochs

    effective_epochs = max(n_epochs - warmup, 1)

    if sched_type == "cosine_restart":
        T_0  = args_ns.lr_restart_period
        Tmul = args_ns.lr_restart_mult
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=int(Tmul) if Tmul == int(Tmul) else 1,
            eta_min=eta_min)
        logger.info(f"  LR scheduler:  cosine_restart  T_0={T_0}  T_mult={Tmul}  "
                    f"eta_min={eta_min:.2e}")

    elif sched_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=effective_epochs, eta_min=eta_min)
        logger.info(f"  LR scheduler:  cosine  T_max={effective_epochs}  eta_min={eta_min:.2e}")

    elif sched_type == "constant":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
        logger.info(f"  LR scheduler:  constant  LR={base_lr:.2e}")

    elif sched_type == "linear_decay":
        decay_epochs = args_ns.lr_decay_epochs or effective_epochs
        min_ratio    = args_ns.lr_min_ratio
        def linear_lambda(epoch):
            if epoch >= decay_epochs:
                return min_ratio
            return 1.0 - (1.0 - min_ratio) * (epoch / decay_epochs)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lambda)
        logger.info(f"  LR scheduler:  linear_decay  over {decay_epochs} epochs  "
                    f"min_ratio={min_ratio}")

    else:
        raise ValueError(f"Unknown --lr-scheduler: {sched_type}")

    return scheduler


class MLPCQLTrainer:
    def __init__(self, obs_dim, n_actions, hidden_layers, dropout, use_residual,
                 lr, gamma, tau, cql_alpha, n_epochs, target_update_every,
                 n_steps, priority_alpha, priority_beta, priority_eps,
                 warmup_epochs, args_ns):
        self.gamma               = gamma
        self.tau                 = tau
        self.cql_alpha           = cql_alpha
        self.target_update_every = target_update_every
        self.n_steps             = n_steps
        self.priority_alpha      = priority_alpha
        self.priority_beta       = priority_beta
        self.priority_eps        = priority_eps
        self._batch_count        = 0
        self.warmup_epochs       = warmup_epochs
        self._base_lr            = lr
        self._args               = args_ns

        # Stagnation detection state
        self._stag_count         = 0
        self._stag_threshold     = args_ns.stag_grad_thresh
        self._stag_window        = args_ns.stag_window
        self._stag_noise         = args_ns.stag_noise
        self._stag_disable       = args_ns.stag_disable
        self._current_epoch      = 0

        self.q_net        = MLPQNetwork(obs_dim, n_actions, hidden_layers,
                                        dropout, use_residual).to(DEVICE)
        self.target_q_net = MLPQNetwork(obs_dim, n_actions, hidden_layers,
                                        dropout, use_residual).to(DEVICE)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer    = optim.AdamW(self.q_net.parameters(), lr=lr, weight_decay=1e-5)
        self._scheduler   = _build_lr_scheduler(self.optimizer, args_ns, lr)

    def step_scheduler(self):
        """Called once per epoch. Handles warmup manually then delegates."""
        self._current_epoch += 1
        if self._current_epoch <= self.warmup_epochs:
            # Linear LR warmup — set directly so it overrides scheduler
            factor = self._current_epoch / max(self.warmup_epochs, 1)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self._base_lr * factor
        else:
            self._scheduler.step()

    def _update_target(self):
        self._batch_count += 1
        if self._batch_count % self.target_update_every == 0:
            # Hard copy every target_update_every batches
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        else:
            # Soft update every batch
            for p, tp in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def _maybe_perturb(self, grad_norm: float) -> bool:
        """
        Stagnation detection + weight perturbation.

        Counts consecutive batches where gradient norm < stag_grad_thresh.
        After stag_window such batches, applies Gaussian noise to all weights.
        Noise scale grows with epoch to overcome increasingly entrenched fixed points.

        Returns True if perturbation was applied.
        """
        if self._stag_disable:
            return False

        if grad_norm < self._stag_threshold:
            self._stag_count += 1
        else:
            self._stag_count = 0

        if self._stag_count >= self._stag_window:
            # Noise grows gently with epoch so early perturbations are smaller
            noise_scale = self._stag_noise * (1.0 + self._current_epoch / 50.0)
            with torch.no_grad():
                for p in self.q_net.parameters():
                    p.add_(torch.randn_like(p) * noise_scale)
            self._stag_count = 0
            logger.info(
                f"  [Ep{self._current_epoch}] ⚡ Stagnation detected "
                f"(∇<{self._stag_threshold} for {self._stag_window} batches) — "
                f"weight perturbation applied (noise={noise_scale:.5f})")
            return True
        return False

    def _td_loss(self, batch):
        s  = batch["states"].to(DEVICE)
        ns = batch["next_states"].to(DEVICE)
        a  = batch["actions"].to(DEVICE)
        G  = batch["nstep_rewards"].to(DEVICE)
        gn = batch["nstep_gammas"].to(DEVICE)
        d  = batch["dones"].to(DEVICE)
        nm = batch["next_action_masks"].to(DEVICE)
        cm = batch["action_masks"].to(DEVICE)

        qv = self.q_net(s, cm)
        qt = qv.gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double-DQN target selection
            nqo        = self.q_net(ns, nm)
            nqo_masked = nqo.masked_fill(nm == 0, -1e9)
            best_a     = nqo_masked.argmax(1, keepdim=True)
            nqt        = self.target_q_net(ns, nm).gather(1, best_a).squeeze(1)
            # Wider clip: ±50 prevents blocking TD learning on hard instances
            tgt = G + gn * (1.0 - d) * nqt.clamp(-50.0, 50.0)

        td_errors = (qt - tgt).detach().abs().cpu().numpy()
        return F.smooth_l1_loss(qt, tgt), qv, cm, a, td_errors

    def _cql_loss(self, qv, action_masks, actions):
        """
        Standard CQL: E[logsumexp Q(s,·)] − E[Q(s, a_data)]

        Uses Q at the *data action* (not average Q over valid actions).
        Average-Q dilutes the penalty and conflicts with the BC loss.
        """
        mq  = qv.clone()
        mq[action_masks == 0] = -1e9
        lse    = torch.logsumexp(mq, dim=1)
        q_data = qv.gather(1, actions.unsqueeze(1)).squeeze(1)
        return self.cql_alpha * (lse - q_data).mean()

    def _bc_loss(self, qv, batch):
        """
        Auxiliary BC loss on expert transitions only.
        Maximises Q(s, a_expert) relative to other valid actions via cross-entropy.
        """
        is_exp = batch["is_expert"].to(DEVICE)
        if not is_exp.any():
            return torch.tensor(0.0, device=DEVICE)
        a      = batch["actions"].to(DEVICE)
        cm     = batch["action_masks"].to(DEVICE)
        qv_exp = qv[is_exp]
        a_exp  = a[is_exp]
        cm_exp = cm[is_exp]
        qv_masked           = qv_exp.clone()
        qv_masked[cm_exp == 0] = -1e9
        return F.cross_entropy(qv_masked, a_exp) * 0.1

    def train_epoch(self, buffer, epoch, cql_batch, log_interval=20, dl_workers=4):
        self.q_net.train()
        sampler     = buffer.get_sampler(self.priority_alpha)
        use_workers = dl_workers if dl_workers > 0 else 0
        persistent  = (use_workers > 0)
        dl = DataLoader(
            buffer,
            batch_size=cql_batch,
            sampler=sampler,
            collate_fn=nstep_collate,
            num_workers=use_workers,
            pin_memory=(DEVICE.type == "cuda"),
            persistent_workers=persistent,
            prefetch_factor=2 if use_workers > 0 else None,
        )

        etd = ecql = ebc = etot = 0.0   # epoch totals
        rtd = rcql = rbc = rtot = 0.0   # rolling window totals
        window_n = 0                     # FIX: actual batch count in current window
        n        = 0                     # total batches processed
        t0       = time.perf_counter()
        nb       = len(dl)

        for bi, batch in enumerate(dl, 1):
            self.optimizer.zero_grad(set_to_none=True)

            tdl, qv, cm, a, tde = self._td_loss(batch)
            cql_l = self._cql_loss(qv, cm, a)
            bc_l  = self._bc_loss(qv, batch)
            loss  = tdl + cql_l + bc_l

            if not torch.isfinite(loss):
                logger.error(f"[Ep{epoch}|B{bi}] non-finite loss, skip")
                continue

            loss.backward()
            gn_val = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()
            self._update_target()
            # Stagnation check after each batch
            self._maybe_perturb(float(gn_val))
            buffer.update_priorities(batch["buffer_indices"].numpy(), tde)

            tv, cv, bv, lv = tdl.item(), cql_l.item(), bc_l.item(), loss.item()
            etd  += tv; ecql += cv; ebc  += bv; etot += lv
            rtd  += tv; rcql += cv; rbc  += bv; rtot += lv
            n    += 1
            window_n += 1   # FIX: track actual window size

            if bi % log_interval == 0 or bi == nb:
                w   = window_n   # FIX: use actual count, not min(log_interval, bi)
                el  = time.perf_counter() - t0
                bps = bi / el if el > 0 else 0
                eta = (nb - bi) / bps if bps > 0 else 0
                with torch.no_grad():
                    cm_b     = batch["action_masks"].to(DEVICE)
                    qv_b     = qv.detach()
                    valid    = (cm_b == 1)
                    qv_valid = qv_b[valid]
                    qmu  = qv_valid.mean().item()  if qv_valid.numel() > 0 else float("nan")
                    qsig = qv_valid.std().item()   if qv_valid.numel() > 0 else float("nan")
                    qmin = qv_valid.min().item()   if qv_valid.numel() > 0 else float("nan")
                    qmax = qv_valid.max().item()   if qv_valid.numel() > 0 else float("nan")
                logger.info(
                    f"[Ep{epoch:>4}|B{bi:>5}/{nb}] α={self.cql_alpha:.3f} | "
                    f"Loss={rtot/w:.4f}(TD={rtd/w:.4f} CQL={rcql/w:.4f} BC={rbc/w:.4f}) | "
                    f"∇={gn_val:.3f} | Q(μ={qmu:.3f} σ={qsig:.3f} "
                    f"min={qmin:.3f} max={qmax:.3f}) | "
                    f"LR={self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"{bps:.1f}b/s ETA{eta:.0f}s")
                rtd = rcql = rbc = rtot = 0.0
                window_n = 0   # FIX: reset window counter after logging

        n   = max(n, 1)
        et  = time.perf_counter() - t0
        logger.info(f"[Ep{epoch:>4} DONE] AvgLoss={etot/n:.4f} "
                    f"(TD={etd/n:.4f} CQL={ecql/n:.4f} BC={ebc/n:.4f}) | "
                    f"{et:.1f}s ({n} batches)")
        return etot / n, etd / n, ecql / n

    def greedy_gap_vs_bks(self, instance_paths):
        """Greedy rollout on all eval instances, returns mean gap vs BKS."""
        self.q_net.eval()
        gaps = []
        for path in instance_paths:
            bks  = get_bks(path)
            env  = make_wrapped_env(path)
            obs  = env.reset()
            done = False
            with torch.no_grad():
                while not done:
                    flat   = extract_flat_obs(obs)
                    mask   = extract_action_mask(obs, NUM_JOBS)
                    ot     = torch.tensor(flat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    mt     = torch.tensor(mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    action = self.q_net(ot, mt).squeeze(0).argmax().item()
                    obs, _, done, _ = env.step(action)
            makespan = getattr(env.unwrapped, "last_time_step", float("inf"))
            env.close()
            gaps.append(100.0 * (makespan - bks) / bks)
        return float(np.mean(gaps)), gaps

    def collect_online_rollouts(self, instance_paths, n_episodes,
                                n_steps, gamma, epsilon=0.1):
        """Collect fresh ε-greedy rollouts from the current policy."""
        self.q_net.eval()
        all_s=[]; all_a=[]; all_G=[]; all_sn=[]
        all_dn=[]; all_gn=[]; all_m=[]; all_mn=[]
        all_ie=[]

        for path in instance_paths:
            jobs = parse_taillard(path)
            lb   = compute_jssp_lower_bound(jobs)
            env  = make_wrapped_env(path)
            for _ in range(n_episodes):
                obs  = env.reset(); done = False; episode = []
                while not done:
                    flat  = extract_flat_obs(obs)
                    mask  = extract_action_mask(obs, NUM_JOBS)
                    legal = [a for a in range(NUM_JOBS) if mask[a] == 1]
                    if not legal:
                        action = NUM_JOBS
                    elif np.random.random() < epsilon:
                        action = int(np.random.choice(legal))
                    else:
                        with torch.no_grad():
                            ot     = torch.tensor(flat, dtype=torch.float32,
                                                  device=DEVICE).unsqueeze(0)
                            mt     = torch.tensor(mask, dtype=torch.float32,
                                                  device=DEVICE).unsqueeze(0)
                            action = self.q_net(ot, mt).squeeze(0).argmax().item()
                    next_obs, _, done, _ = env.step(action)
                    episode.append((flat, action, extract_flat_obs(next_obs), done,
                                    mask, extract_action_mask(next_obs, NUM_JOBS)))
                    obs = next_obs
                makespan = getattr(env.unwrapped, "last_time_step", float("inf"))
                shaped   = shape_episode_rewards(episode, makespan, lb)
                gap      = (makespan - lb) / max(lb, 1.0)
                is_good  = gap < 0.20
                for (s, a, G, s_n, d_n, gn, mask, nmask) in compute_nstep_transitions(
                        shaped, n_steps, gamma):
                    all_s.append(s); all_a.append(a); all_G.append(G)
                    all_sn.append(s_n); all_dn.append(d_n); all_gn.append(gn)
                    all_m.append(mask); all_mn.append(nmask)
                    all_ie.append(is_good)
            env.close()

        if not all_a:
            return None
        return (np.array(all_s,  dtype=np.float16),
                np.array(all_a,  dtype=np.int32),
                np.array(all_G,  dtype=np.float32),
                np.array(all_sn, dtype=np.float16),
                np.array(all_dn, dtype=np.float32),
                np.array(all_gn, dtype=np.float32),
                np.array(all_m,  dtype=np.float32),
                np.array(all_mn, dtype=np.float32),
                np.array(all_ie, dtype=np.bool_))

    def select_action(self, flat_obs, action_mask, greedy=True):
        self.q_net.eval()
        with torch.no_grad():
            ot = torch.tensor(flat_obs,    dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mt = torch.tensor(action_mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            qv = self.q_net(ot, mt).squeeze(0)
            if greedy:
                return qv.argmax().item()
            legal = torch.where(mt.squeeze(0) == 1)[0]
            return legal[torch.multinomial(F.softmax(qv[legal], dim=0), 1).item()].item()

    def save(self, path):
        torch.save(dict(
            q_net=self.q_net.state_dict(),
            target_q_net=self.target_q_net.state_dict(),
            optimizer=self.optimizer.state_dict(),
            batch_count=self._batch_count,
            epoch=self._current_epoch,
        ), path)
        logger.info(f"  Model saved → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_q_net.load_state_dict(ckpt["target_q_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._batch_count   = ckpt.get("batch_count", 0)
        self._current_epoch = ckpt.get("epoch", 0)

# ───────────────────────────────────────────────
# Gap-targeted CQL training loop (v6.2)
# ───────────────────────────────────────────────

def train_cql(buffer, obs_dim, n_actions, args_ns, eval_paths):
    logger.info(f"  Dataset:        {len(buffer):,} transitions  "
                f"RAM≈{buffer.estimate_ram_mb():.0f}MB")
    logger.info(f"  MLP:            {args_ns.mlp_hidden}  dropout={args_ns.mlp_dropout}")
    logger.info(f"  CQL α:          {args_ns.cql_alpha}  (adaptive from epoch 1)")
    logger.info(f"  N-step:         {args_ns.nstep_returns}  "
                f"priority_α={args_ns.priority_alpha}")
    logger.info(f"  Max epochs:     {args_ns.cql_epochs}")
    logger.info(f"  Target gap:     ≤{args_ns.cql_target_gap:.1f}% vs BKS "
                f"(checked every {args_ns.cql_eval_every} epochs)")
    logger.info(f"  Loss patience:  {args_ns.cql_patience} epochs (secondary stop)")
    logger.info(f"  Gap patience:   {args_ns.cql_gap_patience} eval checks (primary stop)")
    logger.info(f"  LR:             {args_ns.cql_lr}  batch={args_ns.cql_batch}")
    logger.info(f"  LR min ratio:   {args_ns.lr_min_ratio}  "
                f"(eta_min={args_ns.cql_lr * args_ns.lr_min_ratio:.2e})")
    logger.info(f"  Warmup:         {args_ns.warmup_epochs} epochs  "
                f"(alpha adapts from epoch 1 regardless)")
    logger.info(f"  DL workers:     {args_ns.dl_workers}")
    logger.info(f"  Online refresh: every {args_ns.online_refresh_every} epochs, "
                f"{args_ns.online_refresh_episodes} episodes")
    logger.info(f"  Stagnation:     {'DISABLED' if args_ns.stag_disable else 'ENABLED'}  "
                f"thresh=∇<{args_ns.stag_grad_thresh}  "
                f"window={args_ns.stag_window}  noise={args_ns.stag_noise}")

    trainer = MLPCQLTrainer(
        obs_dim=obs_dim, n_actions=n_actions,
        hidden_layers=tuple(args_ns.mlp_hidden),
        dropout=args_ns.mlp_dropout, use_residual=args_ns.mlp_residual,
        lr=args_ns.cql_lr, gamma=args_ns.cql_gamma, tau=args_ns.cql_tau,
        cql_alpha=args_ns.cql_alpha, n_epochs=args_ns.cql_epochs,
        target_update_every=args_ns.cql_target_update_every,
        n_steps=args_ns.nstep_returns, priority_alpha=args_ns.priority_alpha,
        priority_beta=args_ns.priority_beta, priority_eps=args_ns.priority_eps,
        warmup_epochs=args_ns.warmup_epochs,
        args_ns=args_ns)

    n_params = sum(p.numel() for p in trainer.q_net.parameters() if p.requires_grad)
    logger.info(f"  Parameters:     {n_params:,}")
    logger.info(f"\n  ── CQL Training (max {args_ns.cql_epochs} epochs) ──")

    best_loss      = float("inf"); loss_patience  = 0
    best_gap       = float("inf"); best_epoch     = 0
    gap_no_improve = 0
    gap_history    = []
    stop_reason    = "max_epochs"
    t_train_start  = time.perf_counter()

    base_alpha    = args_ns.cql_alpha
    current_alpha = base_alpha

    gap_csv_path = os.path.join(CHECKPOINT_DIR, "cql_gap_trajectory.csv")
    with open(gap_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "mean_gap_bks", "cql_alpha"] +
                   [os.path.basename(p).replace(".txt", "") for p in eval_paths] +
                   ["elapsed_s"])

    # Initial gap before any training
    logger.info("  Initial gap check (before training)...")
    t_eval = time.perf_counter()
    mean_gap0, gaps0 = trainer.greedy_gap_vs_bks(eval_paths)
    logger.info(f"  [Ep  0 INIT] mean={mean_gap0:+.2f}%  "
                f"eval={time.perf_counter()-t_eval:.1f}s")
    best_gap   = mean_gap0
    best_epoch = 0
    trainer.save(MODEL_PATH)

    for epoch in range(1, args_ns.cql_epochs + 1):

        # ── Adaptive CQL alpha — active from epoch 1 (FIX: removed warmup gate) ──
        # Low alpha when gap is large so TD can build useful Q-structure first.
        # High alpha when near target to prevent Q-overestimation.
        if best_gap > 30.0:
            current_alpha = base_alpha * 0.25
        elif best_gap > 25.0:
            current_alpha = base_alpha * 0.5
        elif best_gap > 20.0:
            current_alpha = base_alpha * 0.75
        elif best_gap > 17.0:
            current_alpha = base_alpha * 1.0
        elif best_gap > 15.0:
            current_alpha = base_alpha * 1.5
        else:
            current_alpha = base_alpha * 2.0
        trainer.cql_alpha = current_alpha

        # ── Online buffer refresh ──
        if (epoch > 1 and
                args_ns.online_refresh_every > 0 and
                epoch % args_ns.online_refresh_every == 0 and
                best_gap < 30.0):
            logger.info(f"  [Ep{epoch}] Online refresh: collecting "
                        f"{args_ns.online_refresh_episodes} rollout(s) per eval instance...")
            t_ref = time.perf_counter()
            online_arrays = trainer.collect_online_rollouts(
                eval_paths,
                n_episodes=args_ns.online_refresh_episodes,
                n_steps=args_ns.nstep_returns,
                gamma=args_ns.cql_gamma,
                epsilon=max(0.05, 0.3 * (1.0 - epoch / args_ns.cql_epochs)))
            if online_arrays is not None:
                n_new = len(online_arrays[1])
                buffer.bulk_extend(*online_arrays)
                logger.info(f"  [Ep{epoch}] Added {n_new:,} online transitions | "
                             f"Buffer: {len(buffer):,} | "
                             f"{time.perf_counter()-t_ref:.1f}s")

        # ── Train one epoch ──
        tot, td, cql_v = trainer.train_epoch(
            buffer, epoch=epoch, cql_batch=args_ns.cql_batch,
            log_interval=20, dl_workers=args_ns.dl_workers)
        trainer.step_scheduler()

        # Loss-plateau tracking (secondary)
        if tot < best_loss: best_loss = tot; loss_patience = 0
        else:               loss_patience += 1

        # ── Gap evaluation ──
        if epoch % args_ns.cql_eval_every == 0 or epoch == 1:
            t_eval = time.perf_counter()
            mean_gap, per_inst_gaps = trainer.greedy_gap_vs_bks(eval_paths)
            elapsed_eval  = time.perf_counter() - t_eval
            elapsed_total = time.perf_counter() - t_train_start
            gap_history.append((epoch, mean_gap))

            if mean_gap < best_gap:
                best_gap       = mean_gap
                best_epoch     = epoch
                gap_no_improve = 0
                trainer.save(MODEL_PATH)
            else:
                gap_no_improve += 1

            inst_str = "  ".join(
                f"{os.path.basename(p).replace('.txt','')}:{g:+.1f}%"
                for p, g in zip(eval_paths, per_inst_gaps))
            status = ("✓ TARGET REACHED" if mean_gap <= args_ns.cql_target_gap
                      else f"target≤{args_ns.cql_target_gap:.1f}%")
            logger.info(
                f"  [Gap check Ep{epoch:>4}] mean={mean_gap:+.2f}% {status} | "
                f"best={best_gap:+.2f}%@ep{best_epoch} | α={current_alpha:.3f} | "
                f"no-improve={gap_no_improve}/{args_ns.cql_gap_patience} | "
                f"LR={trainer.optimizer.param_groups[0]['lr']:.2e} | "
                f"eval={elapsed_eval:.1f}s total={elapsed_total:.0f}s")
            logger.info(f"    {inst_str}")

            with open(gap_csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch, f"{mean_gap:.4f}", f"{current_alpha:.4f}"] +
                           [f"{g:.4f}" for g in per_inst_gaps] +
                           [f"{elapsed_total:.1f}"])

            if mean_gap <= args_ns.cql_target_gap:
                stop_reason = (f"gap_target={args_ns.cql_target_gap:.1f}% "
                               f"reached at epoch {epoch}")
                logger.info("  ✓ Gap target reached — stopping CQL training.")
                break

            # Primary stop: gap plateau
            if gap_no_improve >= args_ns.cql_gap_patience:
                stop_reason = (f"gap_plateau at epoch {epoch}: no improvement for "
                               f"{gap_no_improve} eval checks "
                               f"({gap_no_improve * args_ns.cql_eval_every} epochs)")
                logger.info("  Gap plateau — stopping CQL training.")
                break

        # Secondary stop: loss plateau
        if loss_patience >= args_ns.cql_patience:
            stop_reason = (f"loss_plateau at epoch {epoch} "
                           f"(patience={args_ns.cql_patience})")
            logger.info("  Loss plateau — stopping CQL training.")
            break

    total_elapsed = time.perf_counter() - t_train_start
    logger.info(f"\n  CQL complete | stop: {stop_reason}")
    logger.info(f"  Best gap:   {best_gap:+.2f}% at epoch {best_epoch}")
    logger.info(f"  Best loss:  {best_loss:.4f}")
    logger.info(f"  Total time: {total_elapsed:.0f}s")
    logger.info(f"  Gap CSV:    {gap_csv_path}")

    if os.path.exists(MODEL_PATH):
        trainer.load(MODEL_PATH)
        logger.info(f"  Loaded best model from epoch {best_epoch}")

    return trainer, stop_reason, best_gap, best_epoch, gap_history

# ───────────────────────────────────────────────
# Evaluation
# ───────────────────────────────────────────────

def greedy_rollout_timed(trainer, instance_path):
    env  = make_wrapped_env(instance_path)
    obs  = env.reset(); done = False; steps = 0
    t0   = time.perf_counter()
    while not done:
        action = trainer.select_action(
            extract_flat_obs(obs), extract_action_mask(obs, NUM_JOBS), greedy=True)
        obs, _, done, _ = env.step(action)
        steps += 1
    elapsed  = time.perf_counter() - t0
    makespan = getattr(env.unwrapped, "last_time_step", float("inf"))
    env.close()
    return float(makespan), steps, elapsed

def evaluate_instance(instance_path, trainer, ea_baseline, n_episodes,
                      eval_target_gap, args_ns):
    inst_key = os.path.basename(instance_path).replace(".txt", "")
    inst_bks = get_bks(instance_path)
    jobs     = parse_taillard(instance_path)
    lb       = compute_jssp_lower_bound(jobs)
    logger.info(f"  ── {inst_key}  (BKS={inst_bks:.0f}  LB={lb:.0f}) ──")

    makespans=[]; step_counts=[]; rollout_times=[]
    for r in range(n_episodes):
        ms, steps, elapsed = greedy_rollout_timed(trainer, instance_path)
        makespans.append(ms); step_counts.append(steps); rollout_times.append(elapsed)
        logger.info(f"    Rollout {r+1}/{n_episodes}: makespan={ms:.0f}  "
                    f"gap_bks={100*(ms-inst_bks)/inst_bks:+.2f}%  "
                    f"gap_lb={(ms-lb)/lb*100:+.2f}%  steps={steps}  time={elapsed:.3f}s")

    avg       = float(np.mean(makespans))
    best_ms   = float(np.min(makespans))
    avg_time  = float(np.mean(rollout_times))
    avg_steps = float(np.mean(step_counts))
    logger.info(f"    CQL | avg={avg:.0f} gap_bks={100*(avg-inst_bks)/inst_bks:+.2f}% "
                f"gap_lb={(avg-lb)/lb*100:+.2f}% | time={avg_time:.3f}s")

    ea_best=ea_gens=ea_time_s=ea_gap_lb=None
    if ea_baseline:
        t0 = time.perf_counter()
        _, _, _, ea_ms, ea_gens, _ = run_ea(
            jobs, args_ns, seed=0,
            target_gap_pct=eval_target_gap, lower_bound=inst_bks,
            n_workers_override=args_ns.n_workers)
        # FIX (v6.1): was indented 10 spaces causing SyntaxError
        ea_time_s = time.perf_counter() - t0
        ea_best   = float(ea_ms)
        ea_gap_lb = (ea_best - lb) / lb * 100.0
        logger.info(f"    EA  | best={ea_best:.0f} "
                    f"gap_bks={100*(ea_best-inst_bks)/inst_bks:+.2f}% "
                    f"gap_lb={ea_gap_lb:+.2f}% | gens={ea_gens}  time={ea_time_s:.1f}s")

    return dict(
        instance=inst_key, bks=inst_bks, lb=lb, makespans=makespans,
        avg=avg, best=best_ms,           # FIX (v6.1): was 'best' (undefined)
        gap_bks_avg=100*(avg-inst_bks)/inst_bks,
        gap_bks_best=100*(best_ms-inst_bks)/inst_bks,
        gap_lb_avg=(avg-lb)/lb*100, gap_lb_best=(best_ms-lb)/lb*100,
        rollout_time_s=avg_time, rollout_steps=int(avg_steps),
        ea_best=ea_best,
        ea_gap_bks=(100*(ea_best-inst_bks)/inst_bks if ea_best else None),
        ea_gap_lb=ea_gap_lb, ea_gens=ea_gens, ea_time_s=ea_time_s)

def print_and_save_summary(results, label):
    logger.info(""); logger.info("─"*95)
    logger.info(f"  {label} Summary"); logger.info("─"*95)
    has_ea = any(r["ea_best"] for r in results)
    hdr = (f"  {'Instance':<12} {'BKS':>7} {'LB':>7}  "
           f"{'CQL avg':>8} {'vsBKS':>7} {'vsLB':>7}  {'t(s)':>7} {'steps':>6}"
           + (f"  {'EA best':>8} {'vsBKS':>7} {'gens':>5} {'t(s)':>7}" if has_ea else ""))
    logger.info(hdr); logger.info("  "+"─"*(len(hdr)-2))
    for r in results:
        row = (f"  {r['instance']:<12} {r['bks']:>7.0f} {r['lb']:>7.0f}  "
               f"{r['avg']:>8.0f} {r['gap_bks_avg']:>+7.2f}% {r['gap_lb_avg']:>+7.2f}%  "
               f"{r['rollout_time_s']:>7.3f} {r['rollout_steps']:>6}")
        if has_ea and r["ea_best"]:
            row += (f"  {r['ea_best']:>8.0f} {r['ea_gap_bks']:>+7.2f}% "
                    f"{r['ea_gens']:>5} {r['ea_time_s']:>7.1f}s")
        logger.info(row)
    bks_gaps = [r["gap_bks_avg"] for r in results]
    lb_gaps  = [r["gap_lb_avg"]  for r in results]
    logger.info("  "+"─"*(len(hdr)-2))
    logger.info(f"  {'MEAN':<12} {'':>7} {'':>7}  "
                f"{'':>8} {np.mean(bks_gaps):>+7.2f}% {np.mean(lb_gaps):>+7.2f}%")
    if has_ea:
        ea_bks = [r["ea_gap_bks"] for r in results if r["ea_best"]]
        ea_t   = [r["ea_time_s"]  for r in results if r["ea_best"]]
        logger.info(f"  {'MEAN (EA)':<75} {np.mean(ea_bks):>+7.2f}% "
                    f"{'':>5} {np.mean(ea_t):>7.1f}s")
    logger.info("─"*95)
    csv_path = os.path.join(CHECKPOINT_DIR,
                            f"mlp_cql_eval_{label.lower().replace(' ','_')}.csv")
    fn = ["instance","bks","lb","avg","best","gap_bks_avg","gap_bks_best",
          "gap_lb_avg","gap_lb_best","rollout_time_s","rollout_steps",
          "ea_best","ea_gap_bks","ea_gap_lb","ea_gens","ea_time_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn); w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float)
                            else (v if v is not None else ""))
                        for k, v in r.items() if k in fn})
    logger.info(f"  CSV → {csv_path}")

def run_statistical_tests(results, alpha):
    logger.info(""); logger.info("═"*80)
    logger.info("  Statistical Tests"); logger.info(f"  α={alpha}"); logger.info("═"*80)
    bks_gaps = [r["gap_bks_avg"] for r in results]
    ea_bks   = [r["ea_gap_bks"]  for r in results if r["ea_best"]]
    n = len(bks_gaps)
    def _test(x, y=None):
        if len(x) < 5:
            d   = np.array(x) if y is None else np.array(x) - np.array(y)
            np_ = int((d > 0).sum()); nt = int((d == 0).sum()); ne = len(d) - nt
            p   = float(2*stats.binom.cdf(min(np_-nt, ne-(np_-nt)), ne, 0.5)) if ne > 0 else 1.0
            return float(np_), p, "sign test"
        elif len(x) < 20:
            try:
                s, p = (stats.wilcoxon(x, alternative="two-sided") if y is None
                        else stats.wilcoxon(x, y, alternative="two-sided"))
            except:
                s, p = 0.0, 1.0
            return float(s), float(p), "Wilcoxon"
        else:
            s, p = (stats.ttest_1samp(x, popmean=0) if y is None else stats.ttest_rel(x, y))
            return float(s), float(p), "paired t-test"
    _, p, tn = _test(bks_gaps)
    logger.info(f"\n  T1 CQL vs BKS  (n={n}, {tn})  mean={np.mean(bks_gaps):+.2f}%  "
                f"p={p:.4f}  " + ("SIGNIFICANT" if p < alpha else "not significant"))
    if ea_bks:
        _, p, tn = _test(bks_gaps[:len(ea_bks)], y=ea_bks)
        diff = np.mean(bks_gaps[:len(ea_bks)]) - np.mean(ea_bks)
        logger.info(f"\n  T2 CQL vs EA  (n={len(ea_bks)}, {tn})  "
                    f"CQL={np.mean(bks_gaps[:len(ea_bks)]):+.2f}%  "
                    f"EA={np.mean(ea_bks):+.2f}%  diff={diff:+.2f}%  p={p:.4f}  "
                    + (f"SIGNIFICANT — CQL {'BETTER' if diff < 0 else 'WORSE'} than EA"
                       if p < alpha else "not significant"))
    logger.info("═"*80)

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    logger.info("="*80)
    logger.info("EA Warm-Start → MLP + CQL  (v6.2)")
    logger.info("="*80)
    logger.info(f"Problem:          {NUM_JOBS} jobs × {NUM_MACHINES} machines")
    logger.info(f"Train instances:  {args.num_train_instances} synthetic")
    logger.info(f"Eval instances:   {[os.path.basename(p) for p in EVAL_INSTANCE_PATHS]}")
    logger.info(f"Workers:          {args.n_workers} of {mp.cpu_count()} cores")
    logger.info(f"EA:               {args.evo_alg}  pop={args.evo_pop}  "
                f"gens={args.evo_gens}")
    logger.info(f"Demonstrations:   {args.expert_episodes} expert + "
                f"{args.random_episodes} random")
    nstep_warn = (" ⚠ VERY SHORT for long episodes!" if args.nstep_returns < 10 else "")
    logger.info(f"N-step:           {args.nstep_returns}{nstep_warn}")
    logger.info(f"CQL target gap:   ≤{args.cql_target_gap:.1f}% vs BKS "
                f"(evaluated every {args.cql_eval_every} epochs)")
    logger.info(f"CQL max epochs:   {args.cql_epochs}")
    logger.info(f"CQL patience:     {args.cql_patience} epochs (loss plateau, secondary)")
    logger.info(f"Gap patience:     {args.cql_gap_patience} eval checks (primary stop)")
    logger.info(f"CQL α:            {args.cql_alpha}  lr={args.cql_lr}  "
                f"batch={args.cql_batch}")
    logger.info(f"LR scheduler:     {args.lr_scheduler}  "
                f"min_ratio={args.lr_min_ratio}  "
                f"restart_period={args.lr_restart_period}  "
                f"restart_mult={args.lr_restart_mult}")
    logger.info(f"Warmup epochs:    {args.warmup_epochs}  "
                f"(alpha adapts from epoch 1 regardless)")
    logger.info(f"Stagnation:       {'off' if args.stag_disable else 'on'}  "
                f"thresh={args.stag_grad_thresh}  "
                f"window={args.stag_window}  noise={args.stag_noise}")
    logger.info(f"Device:           {DEVICE}")
    logger.info(f"Output:           {CHECKPOINT_DIR}")
    logger.info("="*80)

    # Stage 1
    logger.info("\n═══ Stage 1: Generating Synthetic Training Instances ═══")
    train_paths = create_synthetic_training_instances(
        args.num_train_instances, NUM_JOBS, NUM_MACHINES,
        args.train_instance_seed, CHECKPOINT_DIR)

    # Stage 2
    logger.info("\n═══ Stage 2: Collecting Mixed Demonstrations (parallel) ═══")
    all_buffer = NStepPrioritizedBuffer(cap=args.buffer_cap, priority_eps=args.priority_eps)
    env_tmp    = make_wrapped_env(train_paths[0]); obs_tmp = env_tmp.reset()
    obs_dim    = len(extract_flat_obs(obs_tmp))
    n_actions  = env_tmp.action_space.n
    env_tmp.close()
    logger.info(f"  obs_dim={obs_dim}  n_actions={n_actions}")

    collect_all_instances_parallel(
        train_paths=train_paths,
        expert_episodes=args.expert_episodes,
        random_episodes=args.random_episodes,
        epsilon=args.cql_epsilon,
        n_steps=args.nstep_returns,
        gamma=args.cql_gamma,
        all_buffer=all_buffer,
        n_workers=args.n_workers,
        args_ns=args,
        tmp_dir=WORKER_TMP_DIR)

    # Stage 3
    logger.info("\n═══ Stage 3: MLP Conservative Q-Learning ═══")
    trainer, stop_reason, best_gap, best_epoch, gap_history = train_cql(
        all_buffer, obs_dim, n_actions, args, EVAL_INSTANCE_PATHS)

    logger.info(f"\n  Training stopped: {stop_reason}")
    logger.info(f"  Best eval gap:    {best_gap:+.2f}% vs BKS at epoch {best_epoch}")

    # Stage 4
    logger.info("\n═══ Stage 4: Out-of-Sample Evaluation ═══")
    results = [evaluate_instance(p, trainer,
                                 ea_baseline=args.eval_ea_baseline,
                                 n_episodes=args.eval_episodes,
                                 eval_target_gap=args.eval_target_gap,
                                 args_ns=args)
               for p in EVAL_INSTANCE_PATHS]
    print_and_save_summary(results, "out_of_sample")
    run_statistical_tests(results, alpha=ALPHA)

    logger.info("")
    logger.info("="*80)
    logger.info("Complete.")
    logger.info(f"  CQL weights:    {MODEL_PATH}")
    logger.info(f"  Gap trajectory: "
                f"{os.path.join(CHECKPOINT_DIR,'cql_gap_trajectory.csv')}")
    logger.info(f"  Log:            {RESULTS_TXT}")
    logger.info("="*80)
