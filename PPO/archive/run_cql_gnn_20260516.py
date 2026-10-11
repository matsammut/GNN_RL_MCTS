#!/usr/bin/env python3
"""
warm_start_GNN_CQL.py  —  SAGEConv architecture (replaces GATv2)

Key change vs previous version:
    GATv2Conv replaced with SAGEConv throughout GNNQNetwork.
    SAGEConv: h_i = W1·h_i + W2·MEAN(h_j for j in N(i))
    No attention coefficients → no O(|E|×d) materialisation
    → ~20-40x faster on CPU for the same edge count

Expected epoch time: 2-5 minutes instead of 100 minutes.
"""

import csv
import argparse
import os
import time
import json
import logging
from copy import deepcopy
from collections import defaultdict

import numpy as np
import gym
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv

from dynamic_jss_wrapper import DynamicJSSWrapper

from run_ma_hga_taillard import (
    parse_taillard,
    compute_makespan,
    random_permutation,
    order_crossover,
    mutate,
    tournament_select,
    simulated_annealing_improve,
    calibrate_sa_temperature,
    deduplicate_population,
    TOURNAMENT_K,
    SA_TEND,
)

# ───────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="EA warm-start → GNN(SAGE) + CQL pipeline for Taillard JSSP.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument("--eval-instances",       type=str,   nargs="+", required=True)
parser.add_argument("--num-train-instances",  type=int,   required=True)
parser.add_argument("--train-instance-seed",  type=int,   default=0)
parser.add_argument("--out",                  type=str,   default="checkpoint_results_gnn_cql")
parser.add_argument("--bks",                  type=str,   required=True)

parser.add_argument("--evo-alg",       type=str, default="HGA", choices=["GA","MA","HGA"])
parser.add_argument("--evo-pop",       type=int, default=50)
parser.add_argument("--evo-gens",      type=int, default=5)
parser.add_argument("--evo-sa-iters",  type=int, default=500)
parser.add_argument("--evo-early-stop",action="store_true", default=False)

# GNN — note: no --gnn-heads (SAGEConv has no attention heads)
parser.add_argument("--gnn-layers",  type=int, default=3)
parser.add_argument("--gnn-hidden",  type=int, default=128)

parser.add_argument("--cql-epochs",        type=int,   default=100)
parser.add_argument("--cql-batch",         type=int,   default=256)
parser.add_argument("--cql-lr",            type=float, default=3e-4)
parser.add_argument("--cql-alpha",         type=float, default=1.0)
parser.add_argument("--cql-gamma",         type=float, default=0.99)
parser.add_argument("--cql-tau",           type=float, default=0.005)
parser.add_argument("--cql-demo-episodes", type=int,   default=None)
parser.add_argument("--cql-epsilon",       type=float, default=0.05)
parser.add_argument("--buffer-cap",        type=int,   default=200_000)

parser.add_argument("--eval-ea-baseline", action="store_true", default=True)
parser.add_argument("--eval-episodes",    type=int,   default=1)
parser.add_argument("--alpha",            type=float, default=0.05)

args = parser.parse_args()

# ───────────────────────────────────────────────
# Output + logging
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
MODEL_PATH  = os.path.join(CHECKPOINT_DIR, "cql_gnn_weights.pt")

with open(args.bks) as f:
    bks_map = json.load(f)

def get_bks(instance_path):
    key = os.path.basename(instance_path).replace(".txt","")
    return float(bks_map.get(key, 1.0))

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


def make_wrapped_env(instance_path):
    base = gym.make(ENV_ID, env_config={"instance_path": instance_path})
    return DynamicJSSWrapper(base)


# ───────────────────────────────────────────────
# Graph builder — conjunctive + assignment only
# (disjunctive edges removed: O(n²m) → 0 edges)
# Machine conflict is captured via 2-hop path:
#   op_i → machine_k ← op_j  (requires ≥ 2 GNN layers)
# ───────────────────────────────────────────────

class JSSPGraphBuilder:
    """
    Unified node matrix:
        rows 0 … n_ops-1                → operation nodes
        rows n_ops … n_ops+n_machines-1 → machine nodes

    Edge types retained:
        conj_edge_index   – op→op   (job precedence)         O(n×m)
        assign_edge_index – op→machine + machine→op (both)   O(n×m)

    Disjunctive edges deliberately excluded:
        Without them ops competing for the same machine
        communicate via: op_i → machine → op_j  (2 hops).
        This requires ≥ 2 GNN layers but saves O(n²×m) edges.
        For 30×20: 17,400 edges → 0.  For 50×15: 15,960 → 0.

    data.num_nodes set explicitly → no PyG batching inference errors.
    """

    def __init__(self, n_jobs, n_machines, instance_path):
        self.n_jobs     = n_jobs
        self.n_machines = n_machines
        self.n_ops      = n_jobs * n_machines
        self.n_nodes    = self.n_ops + n_machines
        self.n_nodes_per_graph = self.n_nodes
        self.instance_path = instance_path

        self.jobs = parse_taillard(instance_path)

        # Feature layout (operation rows):
        # proc_time_norm(1) + remaining_norm(1) + machine_onehot(M)
        # + is_scheduled(1) + is_eligible(1) + time_ph(1)
        self.unified_feat_dim = 6 + n_machines

        self.proc_times  = np.zeros(self.n_ops, dtype=np.float32)
        self.op_machines = np.zeros(self.n_ops, dtype=np.int64)
        for j in range(n_jobs):
            for pos in range(n_machines):
                op_idx = j * n_machines + pos
                self.op_machines[op_idx] = self.jobs[j][pos][0]
                self.proc_times[op_idx]  = self.jobs[j][pos][1]
        self.max_proc_time = max(float(self.proc_times.max()), 1.0)

        total_load = np.zeros(n_machines, dtype=np.float32)
        for op_idx in range(self.n_ops):
            total_load[self.op_machines[op_idx]] += self.proc_times[op_idx]
        self._max_load   = max(float(total_load.max()), 1.0)
        self._sum_load   = max(float(total_load.sum()), 1.0)
        self._total_load = total_load

        self._build_static_edges()

    def _build_static_edges(self):
        conj_src, conj_dst     = [], []
        assign_src, assign_dst = [], []

        for j in range(self.n_jobs):
            for pos in range(self.n_machines):
                op_idx = j * self.n_machines + pos
                mid    = int(self.op_machines[op_idx])

                # op → machine (assignment)
                assign_src.append(op_idx)
                assign_dst.append(self.n_ops + mid)
                # machine → op (reverse assignment — gives machine context to op)
                assign_src.append(self.n_ops + mid)
                assign_dst.append(op_idx)

                # precedence within job
                if pos > 0:
                    conj_src.append(j * self.n_machines + pos - 1)
                    conj_dst.append(op_idx)

        def _ei(s, d):
            return (torch.tensor([s, d], dtype=torch.long)
                    if s else torch.zeros((2, 0), dtype=torch.long))

        self.conj_edge_index   = _ei(conj_src,   conj_dst)
        self.assign_edge_index = _ei(assign_src, assign_dst)

        # Disjunctive edges: disabled (zero edges)
        self.disj_edge_index   = torch.zeros((2, 0), dtype=torch.long)

        n_conj   = self.conj_edge_index.size(1)
        n_assign = self.assign_edge_index.size(1)
        logger.debug(f"  Edges: conj={n_conj}, assign={n_assign}, disj=0 (disabled)")

    def build_feature_array(self, action_mask):
        """float16 numpy — converted to float32 in collate."""
        feat = np.zeros((self.n_nodes, self.unified_feat_dim), dtype=np.float16)

        for j in range(self.n_jobs):
            for pos in range(self.n_machines):
                op_idx = j * self.n_machines + pos
                mid    = int(self.op_machines[op_idx])
                f, col = feat[op_idx], 0
                f[col] = self.proc_times[op_idx] / self.max_proc_time; col += 1
                f[col] = (self.n_machines - pos) / self.n_machines;    col += 1
                f[col + mid] = 1.0;                                      col += self.n_machines
                f[col] = 0.0; col += 1
                if (action_mask is not None
                        and j < len(action_mask)
                        and action_mask[j] == 1
                        and pos == 0):
                    f[col] = 1.0
                col += 1
                f[col] = 0.0

        for m in range(self.n_machines):
            row    = feat[self.n_ops + m]
            row[0] = self._total_load[m] / self._max_load
            row[1] = 0.0
            row[2] = self._total_load[m] / self._sum_load
            row[3] = 0.0

        return feat

    def obs_to_graph(self, obs, env=None):
        """Used during eval rollouts only — not stored in buffer."""
        action_mask = obs.get("action_mask") if isinstance(obs, dict) else None
        x = self.build_feature_array(action_mask)

        data = Data()
        data.x                 = torch.tensor(x, dtype=torch.float32)
        data.num_nodes         = self.n_nodes
        data.conj_edge_index   = self.conj_edge_index.clone()
        data.disj_edge_index   = self.disj_edge_index.clone()
        data.assign_edge_index = self.assign_edge_index.clone()

        amask = (action_mask if action_mask is not None
                 else np.ones(self.n_jobs + 1))
        data.action_mask = torch.tensor(amask, dtype=torch.float32)
        return data


# ───────────────────────────────────────────────
# GNN Q-Network — SAGEConv (replaces GATv2Conv)
#
# SAGEConv: h_i = W1·h_i + W2·MEAN{h_j : j∈N(i)}
# No attention coefficients → no O(|E|×d) materialisation
# Backward pass: simple linear grads only
# Expected speedup vs GATv2: 20–40× on CPU
# ───────────────────────────────────────────────

class GNNQNetwork(nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers, n_jobs, n_machines):
        super().__init__()
        self.n_jobs            = n_jobs
        self.n_machines        = n_machines
        self.n_ops             = n_jobs * n_machines
        self.n_nodes_per_graph = n_jobs * n_machines + n_machines

        self.node_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # SAGEConv layers — much faster than GATv2 on CPU
        self.sage_layers = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.sage_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        # Per-job readout: mean+max of job's op embeddings → Q
        self.job_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # No-op readout: global mean+max → Q
        self.noop_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x = self.node_encoder(data.x)

        # Merge conjunctive + assignment edges (disjunctive is empty)
        parts = []
        for attr in ("conj_edge_index", "assign_edge_index", "disj_edge_index"):
            ei = getattr(data, attr, None)
            if ei is not None and ei.numel() > 0:
                parts.append(ei)
        combined = (torch.cat(parts, dim=1) if parts
                    else torch.zeros((2, 0), dtype=torch.long, device=x.device))

        # SAGEConv message passing with residual
        for sage, norm in zip(self.sage_layers, self.sage_norms):
            x = norm(x + sage(x, combined))

        # ── Vectorised readout (no Python loops) ──────────────────────────
        if hasattr(data, 'batch') and data.batch is not None:
            B = int(data.batch.max().item()) + 1
        else:
            B = 1

        H    = x.size(-1)
        # [total_nodes, H] → [B, n_nodes_per_graph, H]
        # Valid because PyG Batch lays graphs out contiguously
        x_b  = x.view(B, self.n_nodes_per_graph, H)

        # Operation embeddings: [B, n_ops, H]
        ops  = x_b[:, :self.n_ops, :]

        # Per-job pooling: [B, n_jobs, n_machines, H] → mean+max → [B, n_jobs]
        ops_j    = ops.view(B, self.n_jobs, self.n_machines, H)
        job_mean = ops_j.mean(dim=2)                        # [B, n_jobs, H]
        job_max  = ops_j.max(dim=2).values                  # [B, n_jobs, H]
        job_feat = torch.cat([job_mean, job_max], dim=-1)   # [B, n_jobs, 2H]
        job_q    = self.job_head(job_feat).squeeze(-1)      # [B, n_jobs]

        # Global pool for no-op: [B, H]
        g_mean  = x_b.mean(dim=1)
        g_max   = x_b.max(dim=1).values
        g_feat  = torch.cat([g_mean, g_max], dim=-1)
        noop_q  = self.noop_head(g_feat)                    # [B, 1]

        return torch.cat([job_q, noop_q], dim=-1)           # [B, n_jobs+1]


# ───────────────────────────────────────────────
# Offline Replay Buffer (memory-efficient)
# ───────────────────────────────────────────────

class OfflineReplayBuffer(Dataset):
    """
    Stores float16 numpy feature arrays + edge indices once per instance.
    Full PyG Data objects are reconstructed only at collate time.
    """

    def __init__(self, cap=None):
        self._cap              = cap
        self.state_x           = []
        self.next_state_x      = []
        self.actions           = []
        self.rewards           = []
        self.dones             = []
        self.action_masks      = []
        self.next_action_masks = []
        self.edge_idx          = []
        self._edge_store       = []
        self._n_nodes_store    = []

    def register_edges(self, graph_builder):
        idx = len(self._edge_store)
        self._edge_store.append((
            graph_builder.conj_edge_index.cpu(),
            graph_builder.disj_edge_index.cpu(),
            graph_builder.assign_edge_index.cpu(),
        ))
        self._n_nodes_store.append(graph_builder.n_nodes)
        return idx

    def add(self, state_x, action, reward, next_state_x, done,
            action_mask, next_action_mask, edge_store_idx):
        if self._cap and len(self.actions) >= self._cap:
            self.state_x.pop(0);       self.next_state_x.pop(0)
            self.actions.pop(0);       self.rewards.pop(0)
            self.dones.pop(0);         self.action_masks.pop(0)
            self.next_action_masks.pop(0); self.edge_idx.pop(0)

        self.state_x.append(state_x)
        self.next_state_x.append(next_state_x)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_masks.append(action_mask)
        self.next_action_masks.append(next_action_mask)
        self.edge_idx.append(edge_store_idx)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        ei = self.edge_idx[idx]
        conj, disj, assign = self._edge_store[ei]
        n_nodes = self._n_nodes_store[ei]
        return dict(
            state_x           = torch.tensor(self.state_x[idx],      dtype=torch.float32),
            next_state_x      = torch.tensor(self.next_state_x[idx], dtype=torch.float32),
            conj_edge_index   = conj,
            disj_edge_index   = disj,
            assign_edge_index = assign,
            n_nodes           = n_nodes,
            action            = self.actions[idx],
            reward            = self.rewards[idx],
            done              = self.dones[idx],
            action_mask       = self.action_masks[idx],
            next_action_mask  = self.next_action_masks[idx],
        )

    def estimate_ram_mb(self):
        if not self.state_x:
            return 0
        return (len(self) * 2 * self.state_x[0].nbytes) / (1024 ** 2)


def collate_transitions(batch):
    def _make_data_list(x_key):
        out = []
        for b in batch:
            d = Data()
            d.x                 = b[x_key]
            d.num_nodes         = b["n_nodes"]
            d.conj_edge_index   = b["conj_edge_index"]
            d.disj_edge_index   = b["disj_edge_index"]
            d.assign_edge_index = b["assign_edge_index"]
            out.append(d)
        return out

    return dict(
        states            = Batch.from_data_list(_make_data_list("state_x")),
        next_states       = Batch.from_data_list(_make_data_list("next_state_x")),
        actions           = torch.tensor([b["action"]  for b in batch], dtype=torch.long),
        rewards           = torch.tensor([b["reward"]  for b in batch], dtype=torch.float32),
        dones             = torch.tensor([b["done"]    for b in batch], dtype=torch.float32),
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
            child = (order_crossover(p1, p2) if np.random.random() < CROSSOVER_P
                     else deepcopy(p1))
            child = mutate(child, mutation_rate=MUTATION_P)
            new_pop.append(child)
        pop, fitness = new_pop, [compute_makespan(jobs, ind) for ind in new_pop]
        pop, fitness = deduplicate_population(pop, fitness, jobs)
        gen_best = min(fitness)
        if gen_best < best_val:
            best_val, best_ind, patience = (gen_best,
                                             deepcopy(pop[fitness.index(gen_best)]), 0)
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
            child = (order_crossover(p1, p2) if np.random.random() < CROSSOVER_P
                     else deepcopy(p1))
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
            child = (order_crossover(p1, p2) if np.random.random() < CROSSOVER_P
                     else deepcopy(p1))
            child = mutate(child, mutation_rate=MUTATION_P)
            offspring.append(child)
        ranked = sorted(zip([compute_makespan(jobs, c) for c in offspring], offspring),
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
# Offline data collection
# ───────────────────────────────────────────────

def collect_offline_data_for_instance(instance_path, n_episodes, epsilon,
                                       all_buffer, label=None):
    tag = label or os.path.basename(instance_path)
    logger.info(f"  ── {tag} ──")
    jobs = parse_taillard(instance_path)

    t0 = time.time()
    pop, fitness, best_ind, best_val = run_ea(jobs, args, seed=42)
    logger.info(f"    EA finished in {time.time()-t0:.1f}s | best makespan={best_val}")

    ranked    = sorted(zip(fitness, pop), key=lambda x: x[0])
    n_experts = max(1, args.evo_pop // 2)
    experts   = [(f, p) for f, p in ranked[:n_experts]]
    logger.info(f"    Experts: top {n_experts} | "
                f"Collecting {n_episodes} episodes (ε={epsilon})")

    graph_builder  = JSSPGraphBuilder(NUM_JOBS, NUM_MACHINES, instance_path)
    edge_store_idx = all_buffer.register_edges(graph_builder)

    env         = make_wrapped_env(instance_path)
    total_trans = 0

    for ep in range(n_episodes):
        _, perm   = experts[ep % n_experts]
        remaining = list(perm)
        obs       = env.reset()
        done      = False

        while not done:
            action_mask = obs["action_mask"] if isinstance(obs, dict) else None
            state_x     = graph_builder.build_feature_array(action_mask)
            amask_t     = torch.tensor(
                action_mask if action_mask is not None else np.ones(NUM_JOBS + 1),
                dtype=torch.float32)

            legal      = (set(np.where(action_mask == 1)[0])
                          if action_mask is not None else set(range(NUM_JOBS)))
            real_legal = [a for a in legal if a < NUM_JOBS]

            if real_legal:
                if epsilon > 0.0 and np.random.random() < epsilon:
                    action = int(np.random.choice(real_legal))
                else:
                    def first_pos(job):
                        for i, j in enumerate(remaining):
                            if j == job: return i
                        return float("inf")
                    action = min(real_legal, key=first_pos)
                for i, j in enumerate(remaining):
                    if j == action:
                        remaining.pop(i); break
            else:
                action = NUM_JOBS

            next_obs, reward, done, _ = env.step(action)

            shaped_reward = reward - 0.01
            if done:
                makespan       = getattr(env.unwrapped, 'last_time_step', float('inf'))
                shaped_reward += -makespan / (best_val * 10)

            next_action_mask = next_obs["action_mask"] if isinstance(next_obs, dict) else None
            next_state_x     = graph_builder.build_feature_array(next_action_mask)
            next_amask_t     = torch.tensor(
                next_action_mask if next_action_mask is not None
                else np.ones(NUM_JOBS + 1),
                dtype=torch.float32)

            all_buffer.add(state_x, action, shaped_reward, next_state_x,
                           float(done), amask_t, next_amask_t, edge_store_idx)
            obs = next_obs
            total_trans += 1

    env.close()
    logger.info(f"    {tag}: {total_trans:,} transitions  "
                f"(buffer RAM ≈ {all_buffer.estimate_ram_mb():.0f} MB)")


# ───────────────────────────────────────────────
# CQL Trainer
# ───────────────────────────────────────────────

class CQLTrainer:
    def __init__(self, feat_dim, hidden_dim, n_layers,
                 n_jobs, n_machines, lr, gamma, tau, cql_alpha):
        self.gamma     = gamma
        self.tau       = tau
        self.cql_alpha = cql_alpha

        self.q_net = GNNQNetwork(
            feat_dim, hidden_dim, n_layers, n_jobs, n_machines
        ).to(DEVICE)
        self.target_q_net = GNNQNetwork(
            feat_dim, hidden_dim, n_layers, n_jobs, n_machines
        ).to(DEVICE)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer    = optim.Adam(self.q_net.parameters(), lr=lr)
        self.train_losses = []
        self.td_losses    = []
        self.cql_losses   = []

    def _soft_update(self):
        for p, tp in zip(self.q_net.parameters(),
                         self.target_q_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def _td_loss(self, batch):
        states            = batch["states"].to(DEVICE)
        next_states       = batch["next_states"].to(DEVICE)
        actions           = batch["actions"].to(DEVICE)
        rewards           = batch["rewards"].to(DEVICE)
        dones             = batch["dones"].to(DEVICE)
        next_action_masks = batch["next_action_masks"].to(DEVICE)

        q_values = self.q_net(states)
        q_taken  = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            nq_online = self.q_net(next_states)
            nq_online[next_action_masks == 0] = -float('inf')
            best_a    = nq_online.argmax(dim=1)
            nq_target = self.target_q_net(next_states)
            nq        = nq_target.gather(1, best_a.unsqueeze(1)).squeeze(1)
            target    = rewards + (1.0 - dones) * self.gamma * nq

        return F.mse_loss(q_taken, target), q_values

    def _cql_loss(self, q_values, batch):
        actions      = batch["actions"].to(DEVICE)
        action_masks = batch["action_masks"].to(DEVICE)
        masked_q     = q_values.clone()
        masked_q[action_masks == 0] = -float('inf')
        lse    = torch.logsumexp(masked_q, dim=1)
        q_data = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        return (lse - q_data).mean()

    def train_epoch(self, dataloader, epoch=None, log_interval=10):
        self.q_net.train()
        etd = ecql = etot = 0.0
        running_td = running_cql = running_total = 0.0
        n = 0
        t_data = t_fwd = t_bwd = 0.0
        t_epoch_start = time.perf_counter()
        t_batch_start = time.perf_counter()
        total_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader, start=1):
            t_data += time.perf_counter() - t_batch_start

            t0 = time.perf_counter()
            self.optimizer.zero_grad()
            td_loss, qv = self._td_loss(batch)
            cql_loss    = self._cql_loss(qv, batch)
            loss        = td_loss + self.cql_alpha * cql_loss
            t_fwd      += time.perf_counter() - t0

            if not torch.isfinite(loss):
                logger.error(f"[Epoch {epoch} | Batch {batch_idx}] "
                             f"Non-finite loss TD={td_loss.item():.4f} "
                             f"CQL={cql_loss.item():.4f}")
                t_batch_start = time.perf_counter()
                continue

            t0 = time.perf_counter()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), 1.0)
            self.optimizer.step()
            self._soft_update()
            t_bwd += time.perf_counter() - t0

            td_val, cql_val, total_val = (td_loss.item(),
                                           cql_loss.item(), loss.item())
            etd  += td_val;  ecql += cql_val;  etot += total_val
            running_td += td_val; running_cql += cql_val
            running_total += total_val
            n += 1

            if batch_idx % log_interval == 0 or batch_idx == total_batches:
                window        = min(log_interval, batch_idx)
                elapsed       = time.perf_counter() - t_epoch_start
                secs_per_bat  = elapsed / batch_idx
                eta           = (total_batches - batch_idx) * secs_per_bat
                bps           = 1.0 / secs_per_bat if secs_per_bat > 0 else 0

                logger.info(
                    f"[Epoch {epoch:>3} | Batch {batch_idx:>4}/{total_batches}] "
                    f"Loss={running_total/window:.4f} "
                    f"(TD={running_td/window:.4f}, "
                    f"CQL={running_cql/window:.4f}) | "
                    f"GradNorm={grad_norm:.4f} | "
                    f"Q(μ={qv.mean().item():.3f}, "
                    f"σ={qv.std().item():.3f}, "
                    f"min={qv.min().item():.3f}, "
                    f"max={qv.max().item():.3f}) | "
                    f"LR={self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"{bps:.2f} batch/s | ETA {eta:.0f}s | "
                    f"[data={t_data:.1f}s "
                    f"fwd={t_fwd:.1f}s "
                    f"bwd={t_bwd:.1f}s]"
                )
                running_td = running_cql = running_total = 0.0

            t_batch_start = time.perf_counter()

        n = max(n, 1)
        epoch_time = time.perf_counter() - t_epoch_start
        avg_td, avg_cql, avg_tot = etd/n, ecql/n, etot/n

        logger.info(
            f"[Epoch {epoch:>3} DONE] "
            f"AvgLoss={avg_tot:.4f} (TD={avg_td:.4f}, CQL={avg_cql:.4f}) | "
            f"Epoch time={epoch_time:.1f}s "
            f"(data={t_data:.1f}s, fwd={t_fwd:.1f}s, bwd={t_bwd:.1f}s) | "
            f"{n} batches"
        )
        return avg_tot, avg_td, avg_cql

    def select_action(self, state_graph, action_mask, greedy=True):
        self.q_net.eval()
        with torch.no_grad():
            qv   = self.q_net(state_graph.to(DEVICE)).squeeze(0)
            mask = (action_mask if isinstance(action_mask, torch.Tensor)
                    else torch.tensor(action_mask, dtype=torch.float32)).to(DEVICE)
            qv[mask == 0] = -float('inf')
            if greedy:
                return qv.argmax().item()
            legal = torch.where(mask == 1)[0]
            probs = F.softmax(qv[legal], dim=0)
            return legal[torch.multinomial(probs, 1).item()].item()

    def save(self, path):
        torch.save(dict(
            q_net        = self.q_net.state_dict(),
            target_q_net = self.target_q_net.state_dict(),
            optimizer    = self.optimizer.state_dict(),
        ), path)
        logger.info(f"  CQL model saved → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_q_net.load_state_dict(ckpt['target_q_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])


# ───────────────────────────────────────────────
# CQL training loop
# ───────────────────────────────────────────────

def train_cql(buffer, graph_builder, args):
    logger.info(f"  Dataset size: {len(buffer):,} transitions  "
                f"(RAM ≈ {buffer.estimate_ram_mb():.0f} MB)")
    logger.info(f"  GNN (SAGEConv): {args.gnn_layers} layers, "
                f"{args.gnn_hidden} hidden  [no attention heads]")
    logger.info(f"  CQL: α={args.cql_alpha}, γ={args.cql_gamma}, τ={args.cql_tau}")
    logger.info(f"  Training: {args.cql_epochs} epochs, "
                f"batch={args.cql_batch}, lr={args.cql_lr}")

    dl = DataLoader(
        buffer,
        batch_size  = args.cql_batch,
        shuffle     = True,
        collate_fn  = collate_transitions,
        num_workers = 0,
        pin_memory  = False,
    )

    trainer = CQLTrainer(
        feat_dim   = graph_builder.unified_feat_dim,
        hidden_dim = args.gnn_hidden,
        n_layers   = args.gnn_layers,
        n_jobs     = NUM_JOBS,
        n_machines = NUM_MACHINES,
        lr         = args.cql_lr,
        gamma      = args.cql_gamma,
        tau        = args.cql_tau,
        cql_alpha  = args.cql_alpha,
    )

    n_params = sum(p.numel() for p in trainer.q_net.parameters()
                   if p.requires_grad)
    logger.info(f"  Q-network parameters: {n_params:,}")

    best_loss, patience, patience_limit = float('inf'), 0, 15

    for epoch in range(1, args.cql_epochs + 1):
        t0 = time.time()
        tot, td, cql = trainer.train_epoch(dl, epoch=epoch, log_interval=10)

        logger.info(f"    Epoch {epoch:>3}/{args.cql_epochs} | "
                    f"Loss: {tot:.4f} (TD: {td:.4f}, CQL: {cql:.4f}) | "
                    f"{time.time()-t0:.1f}s")

        if tot < best_loss:
            best_loss, patience = tot, 0
            trainer.save(MODEL_PATH)
        else:
            patience += 1

        if patience >= patience_limit:
            logger.info(f"    Early stopping at epoch {epoch}.")
            break

    if os.path.exists(MODEL_PATH):
        trainer.load(MODEL_PATH)
    logger.info(f"  CQL training complete | Best loss: {best_loss:.4f}")
    return trainer


# ───────────────────────────────────────────────
# Evaluation
# ───────────────────────────────────────────────

def cql_greedy_rollout(trainer, instance_path):
    gb   = JSSPGraphBuilder(NUM_JOBS, NUM_MACHINES, instance_path)
    env  = make_wrapped_env(instance_path)
    obs  = env.reset()
    done = False
    while not done:
        sg      = gb.obs_to_graph(obs, env).to(DEVICE)
        amask   = obs["action_mask"] if isinstance(obs, dict) else None
        amask_t = torch.tensor(
            amask if amask is not None else np.ones(NUM_JOBS + 1),
            dtype=torch.float32)
        action  = trainer.select_action(sg, amask_t, greedy=True)
        obs, _, done, _ = env.step(action)
    ms = getattr(env.unwrapped, 'last_time_step', float('inf'))
    env.close()
    return ms


def evaluate_instance(instance_path, trainer, ea_baseline, n_episodes):
    inst_key = os.path.basename(instance_path).replace(".txt","")
    inst_bks = get_bks(instance_path)
    logger.info(f"  ── Evaluating: {inst_key}  (BKS={inst_bks}) ──")

    cql_makespans = [cql_greedy_rollout(trainer, instance_path)
                     for _ in range(n_episodes)]
    cql_avg  = float(np.mean(cql_makespans))
    cql_best = float(np.min(cql_makespans))
    logger.info(f"    CQL-GNN | avg={cql_avg:.0f} "
                f"(gap {100*(cql_avg-inst_bks)/inst_bks:+.2f}%) | "
                f"best={cql_best:.0f} "
                f"(gap {100*(cql_best-inst_bks)/inst_bks:+.2f}%)")

    ea_best = None
    if ea_baseline:
        jobs = parse_taillard(instance_path)
        t0   = time.time()
        _, _, _, ea_best = run_ea(jobs, args, seed=0)
        ea_best = float(ea_best)
        logger.info(f"    EA  | best={ea_best:.0f} "
                    f"(gap {100*(ea_best-inst_bks)/inst_bks:+.2f}%) "
                    f"in {time.time()-t0:.1f}s")

    return dict(
        instance      = inst_key,
        bks           = inst_bks,
        cql_makespans = cql_makespans,
        cql_avg       = cql_avg,
        cql_best      = cql_best,
        cql_gap_avg   = 100*(cql_avg  - inst_bks)/inst_bks,
        cql_gap_best  = 100*(cql_best - inst_bks)/inst_bks,
        ea_best       = ea_best,
        ea_gap_best   = (100*(ea_best-inst_bks)/inst_bks
                         if ea_best else None),
    )


def print_and_save_summary(results, label):
    logger.info("")
    logger.info("─" * 70)
    logger.info(f"  {label} Summary")
    logger.info("─" * 70)

    has_ea = any(r["ea_best"] is not None for r in results)
    hdr    = (f"  {'Instance':<12} {'BKS':>7}  "
              f"{'CQL avg':>8} {'gap%':>7}  {'CQL best':>8} {'gap%':>7}"
              + (f"  {'EA best':>8} {'gap%':>7}" if has_ea else ""))
    logger.info(hdr)
    logger.info("  " + "─" * (len(hdr) - 2))

    for r in results:
        row = (f"  {r['instance']:<12} {r['bks']:>7.0f}  "
               f"{r['cql_avg']:>8.0f} {r['cql_gap_avg']:>+7.2f}%  "
               f"{r['cql_best']:>8.0f} {r['cql_gap_best']:>+7.2f}%")
        if has_ea and r["ea_best"]:
            row += f"  {r['ea_best']:>8.0f} {r['ea_gap_best']:>+7.2f}%"
        logger.info(row)

    cql_gaps = [r["cql_gap_avg"] for r in results]
    logger.info("  " + "─" * (len(hdr) - 2))
    logger.info(f"  {'MEAN':<12} {'':>7}  {'':>8} {np.mean(cql_gaps):>+7.2f}%")
    if has_ea:
        ea_gaps = [r["ea_gap_best"] for r in results if r["ea_best"]]
        logger.info(f"  {'MEAN (EA)':<12} {'':>7}  {'':>8} {'':>7}  "
                    f"{'':>8} {'':>7}  {'':>8} {np.mean(ea_gaps):>+7.2f}%")
    logger.info("─" * 70)

    slug     = label.lower().replace(" ", "_")
    csv_path = os.path.join(CHECKPOINT_DIR, f"cql_eval_{slug}.csv")
    fn       = ["instance","bks","cql_avg","cql_gap_avg",
                "cql_best","cql_gap_best","ea_best","ea_gap_best"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else
                            (v if v is not None else ""))
                        for k, v in r.items() if k in fn})
    logger.info(f"  CSV saved → {csv_path}")


# ───────────────────────────────────────────────
# Statistical tests
# ───────────────────────────────────────────────

def _interpret(p, alpha, a_mean, b_mean, a_label, b_label):
    if p < alpha:
        d = "BETTER" if a_mean < b_mean else "WORSE"
        return (f"SIGNIFICANT (p={p:.4f} < α={alpha}) — "
                f"{a_label} is {d} than {b_label} "
                f"(mean gap: {a_mean:+.2f}% vs {b_mean:+.2f}%)")
    return (f"NOT significant (p={p:.4f} ≥ α={alpha}) — "
            f"no reliable difference ({a_mean:+.2f}% vs {b_mean:+.2f}%)")


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
        fn      = stats.wilcoxon
        stat, p = (fn(x, alternative="two-sided") if y is None
                   else fn(x, y, alternative="two-sided"))
        return float(stat), float(p), "Wilcoxon signed-rank"
    else:
        fn      = stats.ttest_1samp if y is None else stats.ttest_rel
        stat, p = fn(x, popmean=0) if y is None else fn(x, y)
        return float(stat), float(p), "paired t-test"


def _log_test(title, stat, p, test_name, interpretation, n):
    logger.info(f"\n  ── {title} ──")
    logger.info(f"     Test:           {test_name}  (n={n})")
    logger.info(f"     Statistic:      {stat:.4f}")
    logger.info(f"     p-value:        {p:.4f}")
    logger.info(f"     Interpretation: {interpretation}")
    if n < 5:
        logger.warning(f"     ⚠  Only {n} instances — low statistical power.")


def run_statistical_tests(eval_results, alpha):
    logger.info("")
    logger.info("═" * 70)
    logger.info("  Statistical Tests  (out-of-sample eval instances)")
    logger.info(f"  α = {alpha}  |  two-sided tests")
    logger.info("═" * 70)

    cql_gaps = [r["cql_gap_avg"]  for r in eval_results]
    ea_gaps  = [r["ea_gap_best"]  for r in eval_results
                if r["ea_best"] is not None]
    n_oos, n_ea = len(cql_gaps), len(ea_gaps)

    logger.info(f"\n  T1  CQL-GNN gap% vs BKS  (H₀: mean gap == 0)")
    logger.info(f"      Values: {[f'{g:.2f}%' for g in cql_gaps]}")
    stat, p, tn = _choose_test(cql_gaps)
    mg = float(np.mean(cql_gaps))
    interp = (f"SIGNIFICANT (p={p:.4f}) — "
              f"{'above BKS' if mg>0 else 'below BKS'}, mean={mg:+.2f}%"
              if p < alpha else
              f"NOT significant (p={p:.4f}) — mean={mg:+.2f}%")
    _log_test("T1: CQL vs BKS", stat, p, tn, interp, n_oos)

    if n_ea > 0:
        cp = cql_gaps[:n_ea]
        logger.info(f"\n  T2  CQL vs EA (paired)")
        stat, p, tn = _choose_test(cp, y=ea_gaps)
        interp = _interpret(p, alpha, float(np.mean(cp)),
                            float(np.mean(ea_gaps)), "CQL-GNN", "EA")
        _log_test("T2: CQL vs EA", stat, p, tn, interp, n_ea)

    if args.eval_episodes > 1 and n_ea > 0:
        ep_cql, ep_ea = [], []
        for r in eval_results:
            if r["ea_best"] is not None:
                for ms in r["cql_makespans"]:
                    ep_cql.append(100*(ms-r["bks"])/r["bks"])
                    ep_ea.append(r["ea_gap_best"])
        stat, p, tn = _choose_test(ep_cql, y=ep_ea)
        interp = _interpret(p, alpha, float(np.mean(ep_cql)),
                            float(np.mean(ep_ea)), "CQL episodes", "EA best")
        _log_test("T3: CQL episodes vs EA (pooled)",
                  stat, p, tn, interp, len(ep_cql))

    logger.info("")
    logger.info("  ── Descriptive Statistics ──")
    logger.info(f"  {'Metric':<42} {'Mean':>8} {'Std':>8} "
                f"{'Min':>8} {'Max':>8}")
    logger.info("  " + "─" * 78)

    def _desc(lbl, vals):
        a = np.array(vals, dtype=float)
        logger.info(f"  {lbl:<42} {a.mean():>+8.2f} {a.std():>8.2f} "
                    f"{a.min():>+8.2f} {a.max():>+8.2f}")

    _desc("CQL-GNN gap% (avg per instance)", cql_gaps)
    if ea_gaps:
        _desc("EA gap% (best per instance)", ea_gaps)
        _desc("CQL − EA gap% (positive = CQL worse)",
              (np.array(cql_gaps[:n_ea]) - np.array(ea_gaps)).tolist())
    logger.info("═" * 70)


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info("EA Warm-Start → GNN(SAGEConv) + CQL Pipeline")
    logger.info("=" * 70)
    logger.info(f"Problem:          {NUM_JOBS} jobs × {NUM_MACHINES} machines")
    logger.info(f"Train instances:  {args.num_train_instances} synthetic")
    logger.info(f"Eval instances:   {[os.path.basename(p) for p in EVAL_INSTANCE_PATHS]}")
    logger.info(f"EA:               {args.evo_alg}  pop={args.evo_pop}  "
                f"gens={args.evo_gens}")
    logger.info(f"GNN (SAGEConv):   {args.gnn_layers} layers  "
                f"hidden={args.gnn_hidden}  [no attention — fast on CPU]")
    logger.info(f"CQL:              α={args.cql_alpha}  γ={args.cql_gamma}  "
                f"τ={args.cql_tau}")
    logger.info(f"Training:         {args.cql_epochs} epochs  "
                f"batch={args.cql_batch}  lr={args.cql_lr}")
    logger.info(f"Demo eps/inst:    {args.cql_demo_episodes}  "
                f"ε={args.cql_epsilon}")
    logger.info(f"Buffer cap:       {args.buffer_cap:,}")
    logger.info(f"Device:           {DEVICE}")
    logger.info(f"Output:           {CHECKPOINT_DIR}")
    logger.info("=" * 70)

    # Stage 1
    logger.info("\n═══ Stage 1: Generating Synthetic Training Instances ═══")
    train_instance_paths = create_synthetic_training_instances(
        args.num_train_instances, NUM_JOBS, NUM_MACHINES,
        args.train_instance_seed, CHECKPOINT_DIR)

    # Stage 2
    logger.info("\n═══ Stage 2: Collecting Offline EA Data ═══")
    all_buffer        = OfflineReplayBuffer(cap=args.buffer_cap)
    ref_graph_builder = JSSPGraphBuilder(
        NUM_JOBS, NUM_MACHINES, train_instance_paths[0])

    for i, inst_path in enumerate(train_instance_paths):
        collect_offline_data_for_instance(
            inst_path,
            n_episodes = args.cql_demo_episodes,
            epsilon    = args.cql_epsilon,
            all_buffer = all_buffer,
            label      = (f"synth_{i+1}/{args.num_train_instances}  "
                          f"({os.path.basename(inst_path)})"),
        )

    logger.info(f"  Pooled {len(all_buffer):,} transitions  "
                f"(buffer RAM ≈ {all_buffer.estimate_ram_mb():.0f} MB)")

    # Stage 3
    logger.info("\n═══ Stage 3: Conservative Q-Learning (CQL) + SAGEConv GNN ═══")
    trainer = train_cql(all_buffer, ref_graph_builder, args)

    # Stage 4
    logger.info("\n═══ Stage 4: Out-of-Sample Evaluation ═══")
    logger.info("  NOTE: eval instances not seen during training.")
    eval_results = [
        evaluate_instance(p, trainer,
                          ea_baseline = args.eval_ea_baseline,
                          n_episodes  = args.eval_episodes)
        for p in EVAL_INSTANCE_PATHS
    ]
    print_and_save_summary(eval_results, label="out_of_sample")

    # Stage 5
    run_statistical_tests(eval_results, alpha=ALPHA)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Pipeline complete.")
    logger.info(f"  Weights: {MODEL_PATH}")
    logger.info(f"  CSV:     "
                f"{os.path.join(CHECKPOINT_DIR,'cql_eval_out_of_sample.csv')}")
    logger.info(f"  Log:     {RESULTS_TXT}")
    logger.info("=" * 70)
