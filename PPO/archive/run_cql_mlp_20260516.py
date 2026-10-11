#!/usr/bin/env python3
"""
warm_start_MLP_CQL.py  —  v2 (stabilised)

Changes vs previous version:
  1. BC pre-training warms up the Q-network before CQL starts, preventing
     the Q-divergence seen from epoch 2 onwards in v1.
  2. Periodic hard target update (every N batches) replaces per-batch soft
     Polyak update — more stable for offline RL.
  3. Utilisation-aware per-step reward replaces the near-constant -0.01 signal.
     Every step now carries a meaningful gradient.
  4. Fixed CQL alpha (no linear schedule) — conservatism applied from the start
     at a moderate level rather than ramping into instability.
  5. Lower LR (5e-5), slower target (τ=0.001), smaller batch (1024).
  6. HGA recommended for demonstrations — produces solutions ~10% closer to BKS
     than GA alone.

Usage:
    python3 warm_obj2_mlp_cql.py \\
        --bks bks.json \\
        --eval-instances instances/ta41 instances/ta42 instances/ta43 \\
                         instances/ta44 instances/ta45 instances/ta46 \\
                         instances/ta47 instances/ta48 instances/ta49 \\
        --num-train-instances 20 \\
        --evo-alg HGA --evo-gens 10 --evo-sa-iters 300 --evo-early-stop \\
        --cql-epochs 50 --cql-alpha 0.5 \\
        --cql-lr 5e-5 --cql-batch 1024 --cql-tau 0.001 \\
        --bc-pretrain-epochs 10 \\
        --mlp-hidden 512 512 256 \\
        --out checkpoint_results/mlp_cql_run2
"""

import csv
import argparse
import os
import time
import json
import logging
from copy import deepcopy

import numpy as np
import gym
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
    description="EA warm-start → MLP + CQL pipeline (stabilised v2).",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--eval-instances",       type=str,   nargs="+", required=True)
parser.add_argument("--num-train-instances",  type=int,   required=True)
parser.add_argument("--train-instance-seed",  type=int,   default=0)
parser.add_argument("--out",                  type=str,   default="checkpoint_mlp_cql")
parser.add_argument("--bks",                  type=str,   required=True)

# EA
parser.add_argument("--evo-alg",        type=str,  default="HGA",
                    choices=["GA", "MA", "HGA"])
parser.add_argument("--evo-pop",        type=int,  default=50)
parser.add_argument("--evo-gens",       type=int,  default=10)
parser.add_argument("--evo-sa-iters",   type=int,  default=300)
parser.add_argument("--evo-early-stop", action="store_true", default=False)

# MLP architecture
parser.add_argument("--mlp-hidden",  type=int, nargs="+", default=[512, 512, 256])
parser.add_argument("--mlp-dropout", type=float, default=0.1)

# BC pre-training
parser.add_argument("--bc-pretrain-epochs", type=int, default=10,
                    help="Supervised BC epochs before CQL (0 = disabled). "
                         "Prevents Q-divergence by giving CQL a warm start. "
                         "(default: 10)")
parser.add_argument("--bc-pretrain-lr", type=float, default=1e-3,
                    help="Learning rate for BC pre-training (default: 1e-3)")

# CQL
parser.add_argument("--cql-epochs",          type=int,   default=50)
parser.add_argument("--cql-batch",           type=int,   default=1024)
parser.add_argument("--cql-lr",              type=float, default=5e-5)
parser.add_argument("--cql-alpha",           type=float, default=0.5,
                    help="Fixed CQL conservatism coefficient (default: 0.5). "
                         "Linear schedule removed — a fixed moderate value is "
                         "more stable when the Q-function is still settling.")
parser.add_argument("--cql-gamma",           type=float, default=0.99)
parser.add_argument("--cql-tau",             type=float, default=0.001,
                    help="Polyak coefficient for soft target update. "
                         "Only used between hard updates. (default: 0.001)")
parser.add_argument("--cql-target-update-every", type=int, default=100,
                    help="Hard-copy online → target network every N batches. "
                         "More stable than per-batch soft update. (default: 100)")
parser.add_argument("--cql-demo-episodes",   type=int,   default=None)
parser.add_argument("--cql-epsilon",         type=float, default=0.05)
parser.add_argument("--buffer-cap",          type=int,   default=500_000)

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
TOTAL_OPS = NUM_JOBS * NUM_MACHINES

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
# Offline replay buffer
# ───────────────────────────────────────────────

class FlatReplayBuffer(Dataset):
    """
    Memory-efficient buffer storing float16 flat observations.

    Memory for 300k transitions (obs_dim=300):
        300k × 2 × 300 × 2 bytes  ≈  360 MB
    """

    def __init__(self, cap=None):
        self._cap              = cap
        self.states            = []
        self.next_states       = []
        self.actions           = []
        self.rewards           = []
        self.dones             = []
        self.action_masks      = []
        self.next_action_masks = []

    def add(self, state, action, reward, next_state, done,
            action_mask, next_action_mask):
        if self._cap and len(self.actions) >= self._cap:
            self.states.pop(0);            self.next_states.pop(0)
            self.actions.pop(0);           self.rewards.pop(0)
            self.dones.pop(0)
            self.action_masks.pop(0);      self.next_action_masks.pop(0)

        self.states.append(state.astype(np.float16))
        self.next_states.append(next_state.astype(np.float16))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.action_masks.append(action_mask.astype(np.float32))
        self.next_action_masks.append(next_action_mask.astype(np.float32))

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return dict(
            state            = torch.tensor(self.states[idx],
                                            dtype=torch.float32),
            next_state       = torch.tensor(self.next_states[idx],
                                            dtype=torch.float32),
            action           = self.actions[idx],
            reward           = self.rewards[idx],
            done             = self.dones[idx],
            action_mask      = torch.tensor(self.action_masks[idx],
                                            dtype=torch.float32),
            next_action_mask = torch.tensor(self.next_action_masks[idx],
                                            dtype=torch.float32),
        )

    def estimate_ram_mb(self):
        if not self.states:
            return 0
        return (len(self) * 2 * self.states[0].nbytes) / (1024 ** 2)


def flat_collate(batch):
    return dict(
        states            = torch.stack([b["state"]            for b in batch]),
        next_states       = torch.stack([b["next_state"]       for b in batch]),
        actions           = torch.tensor([b["action"]          for b in batch],
                                          dtype=torch.long),
        rewards           = torch.tensor([b["reward"]          for b in batch],
                                          dtype=torch.float32),
        dones             = torch.tensor([b["done"]            for b in batch],
                                          dtype=torch.float32),
        action_masks      = torch.stack([b["action_mask"]      for b in batch]),
        next_action_masks = torch.stack([b["next_action_mask"] for b in batch]),
    )


# ───────────────────────────────────────────────
# EA variants
# ───────────────────────────────────────────────

def _run_ea_ga(jobs, pop_size, generations, seed=42, early_stop=False):
    import random
    random.seed(seed); np.random.seed(seed)
    pop     = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = [compute_makespan(jobs, ind) for ind in pop]
    best_val, best_ind = min(fitness), deepcopy(pop[np.argmin(fitness)])
    patience = 0
    for gen in range(1, generations + 1):
        new_pop = [deepcopy(best_ind)]
        while len(new_pop) < pop_size:
            p1    = tournament_select(pop, fitness, TOURNAMENT_K)
            p2    = tournament_select(pop, fitness, TOURNAMENT_K)
            child = (order_crossover(p1, p2)
                     if np.random.random() < CROSSOVER_P else deepcopy(p1))
            child = mutate(child, mutation_rate=MUTATION_P)
            new_pop.append(child)
        pop, fitness = new_pop, [compute_makespan(jobs, ind) for ind in new_pop]
        pop, fitness = deduplicate_population(pop, fitness, jobs)
        gen_best = min(fitness)
        if gen_best < best_val:
            best_val, best_ind, patience = (
                gen_best, deepcopy(pop[fitness.index(gen_best)]), 0)
        else:
            patience += 1
        logger.info(f"    GA | Gen {gen}/{generations} | Best: {best_val}")
        if early_stop and patience >= 3:
            logger.info(f"    GA | Early stopping at gen {gen}.")
            break
    return pop, fitness, best_ind, best_val


def _run_ea_ma(jobs, pop_size, generations, sa_iters, seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    pop     = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = [compute_makespan(jobs, ind) for ind in pop]
    best_val, best_ind = min(fitness), deepcopy(pop[np.argmin(fitness)])
    for gen in range(1, generations + 1):
        new_pop = [deepcopy(best_ind)]
        while len(new_pop) < pop_size:
            p1    = tournament_select(pop, fitness, TOURNAMENT_K)
            p2    = tournament_select(pop, fitness, TOURNAMENT_K)
            child = (order_crossover(p1, p2)
                     if np.random.random() < CROSSOVER_P else deepcopy(p1))
            child = mutate(child, mutation_rate=MUTATION_P)
            t0    = calibrate_sa_temperature(jobs, child)
            child, _ = simulated_annealing_improve(jobs, child, iters=sa_iters,
                                                    t0=t0, tend=SA_TEND)
            new_pop.append(child)
        pop, fitness = new_pop, [compute_makespan(jobs, ind) for ind in new_pop]
        pop, fitness = deduplicate_population(pop, fitness, jobs)
        gen_best = min(fitness)
        if gen_best < best_val:
            best_val, best_ind = gen_best, deepcopy(pop[fitness.index(gen_best)])
        logger.info(f"    MA | Gen {gen}/{generations} | Best: {best_val}")
    return pop, fitness, best_ind, best_val


def _run_ea_hga(jobs, pop_size, generations, sa_iters, seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    pop     = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = [compute_makespan(jobs, ind) for ind in pop]
    best_val, best_ind = min(fitness), deepcopy(pop[np.argmin(fitness)])
    for gen in range(1, generations + 1):
        new_pop   = [deepcopy(best_ind)]
        offspring = []
        while len(offspring) < pop_size - 1:
            p1    = tournament_select(pop, fitness, TOURNAMENT_K)
            p2    = tournament_select(pop, fitness, TOURNAMENT_K)
            child = (order_crossover(p1, p2)
                     if np.random.random() < CROSSOVER_P else deepcopy(p1))
            child = mutate(child, mutation_rate=MUTATION_P)
            offspring.append(child)
        ranked = sorted(
            zip([compute_makespan(jobs, c) for c in offspring], offspring),
            key=lambda x: x[0])
        half = len(ranked) // 2
        for i, (_, c) in enumerate(ranked):
            if i < half:
                t0 = calibrate_sa_temperature(jobs, c)
                c, _ = simulated_annealing_improve(jobs, c, iters=sa_iters,
                                                    t0=t0, tend=SA_TEND)
            new_pop.append(c)
        pop, fitness = new_pop, [compute_makespan(jobs, ind) for ind in new_pop]
        pop, fitness = deduplicate_population(pop, fitness, jobs)
        gen_best = min(fitness)
        if gen_best < best_val:
            best_val, best_ind = gen_best, deepcopy(pop[fitness.index(gen_best)])
        logger.info(f"    HGA | Gen {gen}/{generations} | Best: {best_val}")
    return pop, fitness, best_ind, best_val


def run_ea(jobs, args, seed=42):
    if args.evo_alg == "GA":
        return _run_ea_ga(jobs, args.evo_pop, args.evo_gens,
                          seed=seed, early_stop=args.evo_early_stop)
    elif args.evo_alg == "MA":
        return _run_ea_ma(jobs, args.evo_pop, args.evo_gens,
                          args.evo_sa_iters, seed=seed)
    else:
        return _run_ea_hga(jobs, args.evo_pop, args.evo_gens,
                           args.evo_sa_iters, seed=seed)


# ───────────────────────────────────────────────
# Offline data collection — utilisation-aware reward
# ───────────────────────────────────────────────

def compute_step_reward(action, done, makespan, best_val, n_jobs):
    """
    Utilisation-aware per-step reward.

    Every step now carries a distinct signal:
      +0.02  dispatching a real job (machine utilisation)
      -0.05  no-op / idle step (machine left idle when work available)
      terminal: normalised gap to EA best, bounded to [-2, 0]
                0.0  if the episode matches the EA best makespan
                -2.0 if makespan is 200%+ worse than EA best

    This replaces the near-constant (reward - 0.01) that gave
    CQL almost no per-step differentiation signal.
    """
    if action < n_jobs:
        step_r = 0.02
    else:
        step_r = -0.05

    if done:
        gap    = (makespan - best_val) / max(best_val, 1.0)
        step_r += max(-2.0, -gap * 5.0)

    return step_r


def collect_offline_data_for_instance(instance_path, n_episodes,
                                       epsilon, all_buffer, label=None):
    tag = label or os.path.basename(instance_path)
    logger.info(f"  ── {tag} ──")
    jobs = parse_taillard(instance_path)

    t0 = time.time()
    pop, fitness, best_ind, best_val = run_ea(jobs, args, seed=42)
    logger.info(f"    EA finished in {time.time()-t0:.1f}s | "
                f"best makespan={best_val}")

    ranked    = sorted(zip(fitness, pop), key=lambda x: x[0])
    n_experts = max(1, args.evo_pop // 2)
    experts   = [(f, p) for f, p in ranked[:n_experts]]
    logger.info(f"    Experts: top {n_experts} | "
                f"Collecting {n_episodes} episodes (ε={epsilon})")

    env         = make_wrapped_env(instance_path)
    total_trans = 0
    reward_stats = []

    for ep in range(n_episodes):
        _, perm   = experts[ep % n_experts]
        remaining = list(perm)
        obs       = env.reset()
        done      = False
        ep_trans  = []

        while not done:
            flat_obs    = extract_flat_obs(obs)
            action_mask = extract_action_mask(obs, NUM_JOBS)
            legal       = [a for a in range(NUM_JOBS)
                           if action_mask[a] == 1]

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
            makespan = (getattr(env.unwrapped, 'last_time_step', float('inf'))
                        if done else 0.0)

            shaped_reward = compute_step_reward(
                action, done, makespan, best_val, NUM_JOBS)
            reward_stats.append(shaped_reward)

            next_flat       = extract_flat_obs(next_obs)
            next_amask      = extract_action_mask(next_obs, NUM_JOBS)

            ep_trans.append((flat_obs, action, shaped_reward,
                             next_flat, float(done),
                             action_mask, next_amask))
            obs = next_obs

        for t in ep_trans:
            all_buffer.add(*t)
        total_trans += len(ep_trans)

    env.close()

    r_arr = np.array(reward_stats)
    logger.info(f"    {tag}: {total_trans:,} transitions  "
                f"(buffer RAM ≈ {all_buffer.estimate_ram_mb():.0f} MB)")
    logger.info(f"    Reward stats — mean={r_arr.mean():.4f}  "
                f"std={r_arr.std():.4f}  "
                f"min={r_arr.min():.4f}  "
                f"max={r_arr.max():.4f}")


# ───────────────────────────────────────────────
# MLP Q-Network (dueling)
# ───────────────────────────────────────────────

class MLPQNetwork(nn.Module):
    """
    Dueling MLP Q-network.

    Input:  flat obs [B, obs_dim]
    Output: Q-values [B, n_actions]

    Architecture:
        BatchNorm1d → [Linear → ReLU → Dropout] × n → Value + Advantage heads

    Dueling: Q(s,a) = V(s) + A(s,a) − mean(A(s,·) over legal actions)
    This stabilises CQL because V(s) provides a shared baseline so the
    advantage head only needs to learn relative preferences between actions.
    """

    def __init__(self, obs_dim, n_actions,
                 hidden_layers=(512, 512, 256), dropout=0.1):
        super().__init__()
        self.n_actions  = n_actions
        self.input_norm = nn.BatchNorm1d(obs_dim)

        trunk  = []
        in_dim = obs_dim
        for h in hidden_layers:
            trunk += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.trunk          = nn.Sequential(*trunk)
        self.value_head     = nn.Linear(in_dim, 1)
        self.advantage_head = nn.Linear(in_dim, n_actions)

    def forward(self, obs, action_mask=None):
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)

        x = self.input_norm(obs)
        x = self.trunk(x)

        V = self.value_head(x)       # [B, 1]
        A = self.advantage_head(x)   # [B, n_actions]

        # Mean over legal actions only to avoid contamination from illegal ones
        if action_mask is not None:
            A_for_mean        = A.clone()
            A_for_mean[action_mask == 0] = -1e9
            # Per-sample mean over legal actions
            legal_counts      = action_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            A_sum             = (A_for_mean * action_mask).sum(dim=-1, keepdim=True)
            A_mean            = A_sum / legal_counts
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
    CQL with:
      - BC pre-training phase (warms up Q before offline RL)
      - Fixed alpha (no schedule — avoids ramp-into-instability)
      - Periodic hard target update every N batches
      - Double-DQN Bellman target
      - Cosine LR annealing
    """

    def __init__(self, obs_dim, n_actions, hidden_layers, dropout,
                 lr, gamma, tau, cql_alpha, n_epochs, target_update_every):
        self.gamma               = gamma
        self.tau                 = tau
        self.cql_alpha           = cql_alpha
        self.target_update_every = target_update_every
        self._batch_count        = 0

        self.q_net = MLPQNetwork(
            obs_dim, n_actions, hidden_layers, dropout).to(DEVICE)
        self.target_q_net = MLPQNetwork(
            obs_dim, n_actions, hidden_layers, dropout).to(DEVICE)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(
            self.q_net.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(n_epochs, 1), eta_min=lr * 0.1)

        self.train_losses = []
        self.td_losses    = []
        self.cql_losses   = []

    def _update_target(self):
        """
        Hard copy online → target every target_update_every batches.
        Between hard updates a single soft Polyak step is also applied
        so the target does not stagnate completely.
        """
        self._batch_count += 1
        if self._batch_count % self.target_update_every == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        else:
            for p, tp in zip(self.q_net.parameters(),
                             self.target_q_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def _td_loss(self, batch):
        states       = batch["states"].to(DEVICE)
        next_states  = batch["next_states"].to(DEVICE)
        actions      = batch["actions"].to(DEVICE)
        rewards      = batch["rewards"].to(DEVICE)
        dones        = batch["dones"].to(DEVICE)
        next_masks   = batch["next_action_masks"].to(DEVICE)
        curr_masks   = batch["action_masks"].to(DEVICE)

        q_values = self.q_net(states, curr_masks)
        q_taken  = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: action selection with online, evaluation with target
            nq_online              = self.q_net(next_states, next_masks)
            nq_online              = nq_online.masked_fill(next_masks == 0, -1e9)
            best_a                 = nq_online.argmax(dim=1)
            nq_target              = self.target_q_net(next_states, next_masks)
            nq                     = nq_target.gather(
                                         1, best_a.unsqueeze(1)).squeeze(1)
            target = rewards + (1.0 - dones) * self.gamma * nq

        return F.mse_loss(q_taken, target), q_values

    def _cql_loss(self, q_values, batch):
        """
        CQL penalty over legal actions only:
            E_s[ logsumexp_a∈legal Q(s,a) − Q(s, a_data) ]

        Masking illegal actions before logsumexp prevents the penalty
        from trivially rewarding suppression of illegal action Q-values.
        """
        actions = batch["actions"].to(DEVICE)
        masks   = batch["action_masks"].to(DEVICE)

        masked_q            = q_values.clone()
        masked_q[masks == 0] = -1e9
        lse    = torch.logsumexp(masked_q, dim=1)
        q_data = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        return self.cql_alpha * (lse - q_data).mean()

    def pretrain_bc(self, buffer, n_epochs, lr, log_interval=5):
        """
        Supervised behaviour cloning pre-training.

        Treats the offline dataset as a classification problem:
            minimise  CrossEntropy(Q(s, ·), a_expert)

        This gives the Q-network a meaningful initialisation before
        CQL starts. Without it the Q-function diverges within 2 epochs
        because TD bootstrapping pushes values up faster than the CQL
        penalty can pull them down.

        After pre-training, target network weights are hard-copied from
        the online network so both start from the same good initialisation.
        """
        if n_epochs <= 0:
            logger.info("  BC pre-training disabled (--bc-pretrain-epochs 0).")
            return

        logger.info(f"\n  ── BC Pre-Training ({n_epochs} epochs, lr={lr}) ──")
        logger.info("  Purpose: warm-up Q to expert actions before CQL starts.")
        logger.info("  Without this, Q diverges from epoch 2 due to random init.")

        dl     = DataLoader(buffer, batch_size=512, shuffle=True,
                            collate_fn=flat_collate, num_workers=0)
        bc_opt = optim.Adam(self.q_net.parameters(), lr=lr)

        best_acc    = 0.0
        t_start     = time.perf_counter()

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
                torch.nn.utils.clip_grad_norm_(
                    self.q_net.parameters(), max_norm=1.0)
                bc_opt.step()

                total_loss += loss.item()
                preds       = logits.argmax(dim=1)
                correct    += (preds == actions).sum().item()
                total      += len(actions)

            acc = 100.0 * correct / max(total, 1)
            if acc > best_acc:
                best_acc = acc
                torch.save(self.q_net.state_dict(), BC_PATH)

            if epoch % log_interval == 0 or epoch == n_epochs or epoch == 1:
                elapsed = time.perf_counter() - t_start
                logger.info(
                    f"    BC Epoch {epoch:>3}/{n_epochs} | "
                    f"Loss={total_loss/len(dl):.4f} | "
                    f"Acc={acc:.1f}% (best={best_acc:.1f}%) | "
                    f"{elapsed:.0f}s elapsed"
                )

        # Load best BC weights into both networks
        if os.path.exists(BC_PATH):
            self.q_net.load_state_dict(
                torch.load(BC_PATH, map_location=DEVICE))
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self._batch_count = 0   # reset so first CQL epoch starts fresh

        logger.info(f"  BC pre-training complete | best accuracy: {best_acc:.1f}%")
        if best_acc < 20.0:
            logger.warning(
                "  ⚠  BC accuracy < 20% — the flat observation may not "
                "contain enough per-step signal. Consider richer features.")

    def train_epoch(self, dataloader, epoch, log_interval=20):
        self.q_net.train()

        etd = ecql = etot = 0.0
        running_td = running_cql = running_tot = 0.0
        n = 0

        t_data = t_fwd = t_bwd = 0.0
        t_start       = time.perf_counter()
        t_batch_start = time.perf_counter()
        total_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader, start=1):
            t_data += time.perf_counter() - t_batch_start

            t0 = time.perf_counter()
            self.optimizer.zero_grad()
            td_loss, qv = self._td_loss(batch)
            cql_loss    = self._cql_loss(qv, batch)
            loss        = td_loss + cql_loss      # alpha already in cql_loss
            t_fwd      += time.perf_counter() - t0

            if not torch.isfinite(loss):
                logger.error(f"[Ep {epoch}|B {batch_idx}] "
                             f"Non-finite loss — "
                             f"TD={td_loss.item():.4f} "
                             f"CQL={cql_loss.item():.4f}. Skipping batch.")
                t_batch_start = time.perf_counter()
                continue

            t0 = time.perf_counter()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            self._update_target()
            t_bwd += time.perf_counter() - t0

            td_v, cql_v, tot_v = (td_loss.item(),
                                   cql_loss.item(), loss.item())
            etd  += td_v;  ecql += cql_v;  etot += tot_v
            running_td  += td_v
            running_cql += cql_v
            running_tot += tot_v
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
                    f"α={self.cql_alpha:.3f} | "
                    f"Loss={running_tot/w:.4f} "
                    f"(TD={running_td/w:.4f} "
                    f"CQL={running_cql/w:.4f}) | "
                    f"∇={grad_norm:.3f} | "
                    f"Q(μ={qv.mean().item():.3f} "
                    f"σ={qv.std().item():.3f} "
                    f"min={qv.min().item():.3f} "
                    f"max={qv.max().item():.3f}) | "
                    f"LR={self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"target_in={hard_in}b | "
                    f"{bps:.1f} b/s ETA {eta:.0f}s | "
                    f"[data={t_data:.1f}s "
                    f"fwd={t_fwd:.1f}s "
                    f"bwd={t_bwd:.1f}s]"
                )
                running_td = running_cql = running_tot = 0.0

            t_batch_start = time.perf_counter()

        n = max(n, 1)
        epoch_t = time.perf_counter() - t_start
        avg_td, avg_cql, avg_tot = etd/n, ecql/n, etot/n

        logger.info(
            f"[Ep {epoch:>3} DONE] "
            f"AvgLoss={avg_tot:.4f} "
            f"(TD={avg_td:.4f} CQL={avg_cql:.4f}) | "
            f"{epoch_t:.1f}s ({n} batches) | "
            f"[data={t_data:.1f}s "
            f"fwd={t_fwd:.1f}s "
            f"bwd={t_bwd:.1f}s]"
        )
        return avg_tot, avg_td, avg_cql

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

def train_cql(buffer, obs_dim, n_actions, args):
    logger.info(f"  Dataset:        {len(buffer):,} transitions  "
                f"(RAM ≈ {buffer.estimate_ram_mb():.0f} MB)")
    logger.info(f"  MLP arch:       {args.mlp_hidden}  "
                f"dropout={args.mlp_dropout}")
    logger.info(f"  CQL α (fixed):  {args.cql_alpha}  "
                f"(no schedule — avoids ramp-into-instability)")
    logger.info(f"  Target update:  hard copy every "
                f"{args.cql_target_update_every} batches")
    logger.info(f"  Training:       {args.cql_epochs} epochs  "
                f"batch={args.cql_batch}  lr={args.cql_lr}")
    logger.info(f"  BC pre-train:   {args.bc_pretrain_epochs} epochs  "
                f"lr={args.bc_pretrain_lr}")

    dl = DataLoader(
        buffer,
        batch_size  = args.cql_batch,
        shuffle     = True,
        collate_fn  = flat_collate,
        num_workers = 0,
        pin_memory  = False,
    )

    trainer = MLPCQLTrainer(
        obs_dim              = obs_dim,
        n_actions            = n_actions,
        hidden_layers        = tuple(args.mlp_hidden),
        dropout              = args.mlp_dropout,
        lr                   = args.cql_lr,
        gamma                = args.cql_gamma,
        tau                  = args.cql_tau,
        cql_alpha            = args.cql_alpha,
        n_epochs             = args.cql_epochs,
        target_update_every  = args.cql_target_update_every,
    )

    n_params = sum(p.numel() for p in trainer.q_net.parameters()
                   if p.requires_grad)
    logger.info(f"  Parameters:     {n_params:,}")

    # ── Phase 1: BC pre-training ──────────────────────────────────────────────
    trainer.pretrain_bc(
        buffer   = buffer,
        n_epochs = args.bc_pretrain_epochs,
        lr       = args.bc_pretrain_lr,
    )

    # ── Phase 2: CQL fine-tuning ──────────────────────────────────────────────
    logger.info(f"\n  ── CQL Fine-Tuning ({args.cql_epochs} epochs) ──")

    best_loss, patience, patience_limit = float("inf"), 0, 20

    for epoch in range(1, args.cql_epochs + 1):
        tot, td, cql = trainer.train_epoch(dl, epoch=epoch, log_interval=20)
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
# Evaluation
# ───────────────────────────────────────────────

def greedy_rollout(trainer, instance_path):
    env  = make_wrapped_env(instance_path)
    obs  = env.reset()
    done = False
    while not done:
        flat   = extract_flat_obs(obs)
        amask  = extract_action_mask(obs, NUM_JOBS)
        action = trainer.select_action(flat, amask, greedy=True)
        obs, _, done, _ = env.step(action)
    ms = getattr(env.unwrapped, "last_time_step", float("inf"))
    env.close()
    return float(ms)


def evaluate_instance(instance_path, trainer, ea_baseline, n_episodes):
    inst_key = os.path.basename(instance_path).replace(".txt", "")
    inst_bks = get_bks(instance_path)
    logger.info(f"  ── {inst_key}  (BKS={inst_bks:.0f}) ──")

    makespans = [greedy_rollout(trainer, instance_path)
                 for _ in range(n_episodes)]
    avg  = float(np.mean(makespans))
    best = float(np.min(makespans))
    logger.info(f"    MLP-CQL | avg={avg:.0f} "
                f"(gap {100*(avg-inst_bks)/inst_bks:+.2f}%) | "
                f"best={best:.0f} "
                f"(gap {100*(best-inst_bks)/inst_bks:+.2f}%)")

    ea_best = None
    if ea_baseline:
        jobs = parse_taillard(instance_path)
        t0   = time.time()
        _, _, _, ea_best = run_ea(jobs, args, seed=0)
        ea_best = float(ea_best)
        logger.info(f"    EA      | best={ea_best:.0f} "
                    f"(gap {100*(ea_best-inst_bks)/inst_bks:+.2f}%) "
                    f"in {time.time()-t0:.1f}s")

    return dict(
        instance    = inst_key,
        bks         = inst_bks,
        makespans   = makespans,
        avg         = avg,
        best        = best,
        gap_avg     = 100*(avg  - inst_bks)/inst_bks,
        gap_best    = 100*(best - inst_bks)/inst_bks,
        ea_best     = ea_best,
        ea_gap_best = (100*(ea_best-inst_bks)/inst_bks
                       if ea_best else None),
    )


def print_and_save_summary(results, label):
    logger.info("")
    logger.info("─" * 70)
    logger.info(f"  {label} Summary")
    logger.info("─" * 70)

    has_ea = any(r["ea_best"] for r in results)
    hdr    = (f"  {'Instance':<12} {'BKS':>7}  "
              f"{'MLP-CQL':>8} {'gap%':>7}  {'Best':>8} {'gap%':>7}"
              + (f"  {'EA best':>8} {'gap%':>7}" if has_ea else ""))
    logger.info(hdr)
    logger.info("  " + "─" * (len(hdr) - 2))

    for r in results:
        row = (f"  {r['instance']:<12} {r['bks']:>7.0f}  "
               f"{r['avg']:>8.0f} {r['gap_avg']:>+7.2f}%  "
               f"{r['best']:>8.0f} {r['gap_best']:>+7.2f}%")
        if has_ea and r["ea_best"]:
            row += f"  {r['ea_best']:>8.0f} {r['ea_gap_best']:>+7.2f}%"
        logger.info(row)

    gaps = [r["gap_avg"] for r in results]
    logger.info("  " + "─" * (len(hdr) - 2))
    logger.info(f"  {'MEAN':<12} {'':>7}  {'':>8} {np.mean(gaps):>+7.2f}%")

    if has_ea:
        ea_gaps = [r["ea_gap_best"] for r in results if r["ea_best"]]
        logger.info(f"  {'MEAN (EA)':<12} {'':>7}  {'':>8} {'':>7}  "
                    f"{'':>8} {'':>7}  {'':>8} {np.mean(ea_gaps):>+7.2f}%")
    logger.info("─" * 70)

    slug     = label.lower().replace(" ", "_")
    csv_path = os.path.join(CHECKPOINT_DIR,
                             f"mlp_cql_eval_{slug}.csv")
    fn       = ["instance", "bks", "avg", "gap_avg", "best",
                "gap_best", "ea_best", "ea_gap_best"]
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
                      min(n_pos-n_ties, n_eff-(n_pos-n_ties)),
                      n_eff, 0.5))
                  if n_eff > 0 else 1.0)
        return float(n_pos), p, "sign test"
    elif n < 20:
        fn     = stats.wilcoxon
        s, p   = (fn(x, alternative="two-sided") if y is None
                  else fn(x, y, alternative="two-sided"))
        return float(s), float(p), "Wilcoxon signed-rank"
    else:
        fn     = stats.ttest_1samp if y is None else stats.ttest_rel
        s, p   = fn(x, popmean=0) if y is None else fn(x, y)
        return float(s), float(p), "paired t-test"


def run_statistical_tests(results, alpha):
    logger.info("")
    logger.info("═" * 70)
    logger.info("  Statistical Tests")
    logger.info(f"  α={alpha}  two-sided")
    logger.info("═" * 70)

    gaps    = [r["gap_avg"]     for r in results]
    ea_gaps = [r["ea_gap_best"] for r in results if r["ea_best"]]
    n       = len(gaps)

    stat, p, tn = _choose_test(gaps)
    mg = np.mean(gaps)
    logger.info(f"\n  T1 MLP-CQL vs BKS  (n={n}, test={tn})")
    logger.info(f"     mean gap={mg:+.2f}%  p={p:.4f}  "
                + ("SIGNIFICANT" if p < alpha else "not significant"))

    if ea_gaps:
        n_ea    = len(ea_gaps)
        stat, p, tn = _choose_test(gaps[:n_ea], y=ea_gaps)
        diff    = np.mean(gaps[:n_ea]) - np.mean(ea_gaps)
        better  = diff < 0
        logger.info(f"\n  T2 MLP-CQL vs EA  (n={n_ea}, test={tn})")
        logger.info(f"     MLP-CQL mean={np.mean(gaps[:n_ea]):+.2f}%  "
                    f"EA mean={np.mean(ea_gaps):+.2f}%  "
                    f"diff={diff:+.2f}%  p={p:.4f}  "
                    + (("SIGNIFICANT — CQL " + ("BETTER" if better else "WORSE")
                        + " than EA")
                       if p < alpha else "not significant"))

    # ── Descriptive ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("  ── Descriptive Statistics ──")
    logger.info(f"  {'Metric':<42} {'Mean':>8} {'Std':>8} "
                f"{'Min':>8} {'Max':>8}")
    logger.info("  " + "─" * 78)

    def _desc(lbl, vals):
        a = np.array(vals, dtype=float)
        logger.info(f"  {lbl:<42} {a.mean():>+8.2f} {a.std():>8.2f} "
                    f"{a.min():>+8.2f} {a.max():>+8.2f}")

    _desc("MLP-CQL gap% (avg per instance)", gaps)
    if ea_gaps:
        _desc("EA gap% (best per instance)", ea_gaps)
        _desc("CQL − EA gap% (neg = CQL better)",
              (np.array(gaps[:len(ea_gaps)]) - np.array(ea_gaps)).tolist())

    logger.info("═" * 70)


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info("EA Warm-Start → MLP + CQL  (stabilised v2)")
    logger.info("=" * 70)
    logger.info(f"Problem:           {NUM_JOBS} jobs × {NUM_MACHINES} machines")
    logger.info(f"Train instances:   {args.num_train_instances} synthetic")
    logger.info(f"Eval instances:    "
                f"{[os.path.basename(p) for p in EVAL_INSTANCE_PATHS]}")
    logger.info(f"EA:                {args.evo_alg}  "
                f"pop={args.evo_pop}  gens={args.evo_gens}  "
                f"sa_iters={args.evo_sa_iters}")
    logger.info(f"MLP:               {args.mlp_hidden}  "
                f"dropout={args.mlp_dropout}")
    logger.info(f"BC pre-training:   {args.bc_pretrain_epochs} epochs  "
                f"lr={args.bc_pretrain_lr}")
    logger.info(f"CQL α (fixed):     {args.cql_alpha}")
    logger.info(f"CQL target update: hard every "
                f"{args.cql_target_update_every} batches  "
                f"(τ={args.cql_tau} between hard updates)")
    logger.info(f"Training:          {args.cql_epochs} epochs  "
                f"batch={args.cql_batch}  lr={args.cql_lr}")
    logger.info(f"Demo eps/inst:     {args.cql_demo_episodes}  "
                f"ε={args.cql_epsilon}")
    logger.info(f"Buffer cap:        {args.buffer_cap:,}")
    logger.info(f"Device:            {DEVICE}")
    logger.info(f"Output:            {CHECKPOINT_DIR}")
    logger.info("=" * 70)

    # ── Stage 1: Synthetic instances ─────────────────────────────────────────
    logger.info("\n═══ Stage 1: Generating Synthetic Training Instances ═══")
    train_paths = create_synthetic_training_instances(
        args.num_train_instances, NUM_JOBS, NUM_MACHINES,
        args.train_instance_seed, CHECKPOINT_DIR)

    # ── Stage 2: Collect offline data ────────────────────────────────────────
    logger.info("\n═══ Stage 2: Collecting Offline EA Demonstrations ═══")
    all_buffer = FlatReplayBuffer(cap=args.buffer_cap)

    env_tmp   = make_wrapped_env(train_paths[0])
    obs_tmp   = env_tmp.reset()
    obs_dim   = len(extract_flat_obs(obs_tmp))
    n_actions = env_tmp.action_space.n
    env_tmp.close()
    logger.info(f"  obs_dim={obs_dim}  n_actions={n_actions}")

    for i, inst_path in enumerate(train_paths):
        collect_offline_data_for_instance(
            inst_path,
            n_episodes = args.cql_demo_episodes,
            epsilon    = args.cql_epsilon,
            all_buffer = all_buffer,
            label      = (f"synth_{i+1}/{args.num_train_instances}  "
                          f"({os.path.basename(inst_path)})"),
        )

    logger.info(f"  Pooled {len(all_buffer):,} transitions  "
                f"(RAM ≈ {all_buffer.estimate_ram_mb():.0f} MB)")

    # ── Stage 3: BC pre-training + CQL ───────────────────────────────────────
    logger.info("\n═══ Stage 3: BC Pre-Training + MLP Conservative Q-Learning ═══")
    trainer = train_cql(all_buffer, obs_dim, n_actions, args)

    # ── Stage 4: Evaluation ───────────────────────────────────────────────────
    logger.info("\n═══ Stage 4: Out-of-Sample Evaluation ═══")
    logger.info("  NOTE: eval instances not seen during training.")
    results = [
        evaluate_instance(p, trainer,
                          ea_baseline = args.eval_ea_baseline,
                          n_episodes  = args.eval_episodes)
        for p in EVAL_INSTANCE_PATHS
    ]
    print_and_save_summary(results, "out_of_sample")

    # ── Stage 5: Statistical tests ────────────────────────────────────────────
    run_statistical_tests(results, alpha=ALPHA)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Complete.")
    logger.info(f"  CQL weights:    {MODEL_PATH}")
    logger.info(f"  BC weights:     {BC_PATH}")
    logger.info(f"  Log:            {RESULTS_TXT}")
    logger.info("=" * 70)
