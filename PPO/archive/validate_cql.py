#!/usr/bin/env python3
"""
validate_cql.py

Standalone validation script for CQL-GNN checkpoints.
Works with any checkpoint saved during training — even if training was
interrupted at epoch 1. The checkpoint only needs cql_gnn_weights.pt
to exist.

Usage:
    # Basic validation on eval instances
    python3 validate_cql.py \\
        --checkpoint checkpoint_results/20260506_obj2/cql_gnn_weights.pt \\
        --bks bks.json \\
        --eval-instances instances/ta56 instances/ta57 instances/ta58 \\
        --n-jobs 30 --n-machines 20

    # Compare against EA baseline too
    python3 validate_cql.py \\
        --checkpoint checkpoint_results/20260506_obj2/cql_gnn_weights.pt \\
        --bks bks.json \\
        --eval-instances instances/ta56 instances/ta57 instances/ta58 \\
        --n-jobs 30 --n-machines 20 \\
        --ea-baseline --evo-alg GA --evo-pop 50 --evo-gens 5

    # Multiple rollouts per instance for variance estimate
    python3 validate_cql.py \\
        --checkpoint checkpoint_results/20260506_obj2/cql_gnn_weights.pt \\
        --bks bks.json \\
        --eval-instances instances/ta56 instances/ta57 instances/ta58 \\
        --n-jobs 30 --n-machines 20 \\
        --n-rollouts 5

    # Inspect checkpoint metadata without running rollouts
    python3 validate_cql.py \\
        --checkpoint checkpoint_results/20260506_obj2/cql_gnn_weights.pt \\
        --inspect-only
"""

import argparse
import os
import sys
import json
import time
import logging
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv

# ── optional env imports (only needed for rollouts) ───────────────────────────
try:
    import gym
    from dynamic_jss_wrapper import DynamicJSSWrapper
    from run_ma_hga_taillard import (
        parse_taillard, compute_makespan, random_permutation,
        order_crossover, mutate, tournament_select,
        simulated_annealing_improve, calibrate_sa_temperature,
        deduplicate_population, TOURNAMENT_K, SA_TEND,
    )
    ENV_AVAILABLE = True
except ImportError as e:
    ENV_AVAILABLE = False
    _ENV_ERR = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Validate a saved CQL-GNN checkpoint.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to cql_gnn_weights.pt")
parser.add_argument("--bks", type=str, default=None,
                    help="Path to bks.json (required for gap% calculation)")
parser.add_argument("--eval-instances", type=str, nargs="+", default=[],
                    help="Taillard instance files to evaluate on")
parser.add_argument("--n-jobs", type=int, default=None,
                    help="Number of jobs (inferred from instances if omitted)")
parser.add_argument("--n-machines", type=int, default=None,
                    help="Number of machines (inferred from instances if omitted)")

# ── model arch (must match training) ──────────────────────────────────────────
parser.add_argument("--gnn-layers", type=int, default=3)
parser.add_argument("--gnn-hidden", type=int, default=128)
parser.add_argument("--gnn-heads",  type=int, default=4)

# ── rollout options ────────────────────────────────────────────────────────────
parser.add_argument("--n-rollouts", type=int, default=1,
                    help="Greedy rollouts per eval instance (default: 1)")
parser.add_argument("--ea-baseline", action="store_true", default=False,
                    help="Also run EA on each eval instance for comparison")
parser.add_argument("--evo-alg",  type=str, default="GA",
                    choices=["GA", "MA", "HGA"])
parser.add_argument("--evo-pop",  type=int, default=50)
parser.add_argument("--evo-gens", type=int, default=5)
parser.add_argument("--evo-sa-iters", type=int, default=500)

# ── misc ──────────────────────────────────────────────────────────────────────
parser.add_argument("--inspect-only", action="store_true", default=False,
                    help="Print checkpoint metadata and exit without rollouts")
parser.add_argument("--env-id", type=str, default="JSSEnv:jss-v1")
parser.add_argument("--out-csv", type=str, default=None,
                    help="Optional path to save results CSV")

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_checkpoint(path):
    """Print everything stored in the checkpoint without loading the model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  Checkpoint: {path}")
    logger.info(f"{'='*60}")

    if not os.path.exists(path):
        logger.error(f"  File not found: {path}")
        return None

    size_mb = os.path.getsize(path) / (1024 ** 2)
    logger.info(f"  File size:  {size_mb:.2f} MB")

    ckpt = torch.load(path, map_location="cpu")
    logger.info(f"  Keys:       {list(ckpt.keys())}")

    if "q_net" in ckpt:
        sd = ckpt["q_net"]
        n_params = sum(v.numel() for v in sd.values())
        logger.info(f"  Parameters: {n_params:,}")
        logger.info(f"  Layers:")
        for k, v in sd.items():
            logger.info(f"    {k:<55} {str(tuple(v.shape))}")

    if "meta" in ckpt:
        logger.info(f"  Metadata:")
        for k, v in ckpt["meta"].items():
            logger.info(f"    {k}: {v}")

    if "optimizer" in ckpt:
        opt = ckpt["optimizer"]
        if "param_groups" in opt:
            for i, g in enumerate(opt["param_groups"]):
                logger.info(f"  Optimizer group {i}: lr={g.get('lr')}")

    logger.info(f"{'='*60}\n")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Model definition  (must match warm_obj2_20260506.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

class GNNQNetwork(nn.Module):
    """
    Vectorised GNNQNetwork — must match the architecture used during training.
    If you changed gnn-hidden/layers/heads during training, pass the same
    values via CLI flags.
    """
    def __init__(self, feat_dim, hidden_dim, n_heads, n_layers, n_jobs, n_machines):
        super().__init__()
        self.n_jobs              = n_jobs
        self.n_machines          = n_machines
        self.n_ops               = n_jobs * n_machines
        self.n_nodes_per_graph   = n_jobs * n_machines + n_machines

        assert hidden_dim % n_heads == 0
        head_dim = hidden_dim // n_heads

        self.node_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gat_layers = nn.ModuleList([
            GATv2Conv(hidden_dim, head_dim, heads=n_heads,
                      concat=True, add_self_loops=True)
            for _ in range(n_layers)
        ])
        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.job_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.noop_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x = self.node_encoder(data.x)

        parts = [getattr(data, a) for a in
                 ("conj_edge_index", "disj_edge_index", "assign_edge_index")
                 if getattr(data, a, None) is not None
                 and getattr(data, a).numel() > 0]
        combined = (torch.cat(parts, dim=1) if parts
                    else torch.zeros((2, 0), dtype=torch.long, device=x.device))

        for gat, norm in zip(self.gat_layers, self.gat_norms):
            x = norm(x + gat(x, combined))

        if hasattr(data, 'batch') and data.batch is not None:
            B = int(data.batch.max().item()) + 1
        else:
            B = 1

        H    = x.size(-1)
        x_b  = x.view(B, self.n_nodes_per_graph, H)
        ops  = x_b[:, :self.n_ops, :]

        ops_j    = ops.view(B, self.n_jobs, self.n_machines, H)
        job_mean = ops_j.mean(dim=2)
        job_max  = ops_j.max(dim=2).values
        job_feat = torch.cat([job_mean, job_max], dim=-1)
        job_q    = self.job_head(job_feat).squeeze(-1)

        g_mean  = x_b.mean(dim=1)
        g_max   = x_b.max(dim=1).values
        g_feat  = torch.cat([g_mean, g_max], dim=-1)
        noop_q  = self.noop_head(g_feat)

        return torch.cat([job_q, noop_q], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder  (identical to training script)
# ─────────────────────────────────────────────────────────────────────────────

class JSSPGraphBuilder:
    def __init__(self, n_jobs, n_machines, instance_path):
        self.n_jobs     = n_jobs
        self.n_machines = n_machines
        self.n_ops      = n_jobs * n_machines
        self.n_nodes    = self.n_ops + n_machines

        self.jobs = parse_taillard(instance_path)
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
        disj_src, disj_dst     = [], []
        assign_src, assign_dst = [], []

        machine_ops = defaultdict(list)
        for j in range(self.n_jobs):
            for pos in range(self.n_machines):
                op_idx = j * self.n_machines + pos
                mid    = int(self.op_machines[op_idx])
                machine_ops[mid].append(op_idx)
                assign_src.append(op_idx)
                assign_dst.append(self.n_ops + mid)
                if pos > 0:
                    conj_src.append(j * self.n_machines + pos - 1)
                    conj_dst.append(op_idx)

        for ops in machine_ops.values():
            for i in range(len(ops)):
                for k in range(i + 1, len(ops)):
                    disj_src += [ops[i], ops[k]]
                    disj_dst += [ops[k], ops[i]]

        def _ei(s, d):
            return (torch.tensor([s, d], dtype=torch.long)
                    if s else torch.zeros((2, 0), dtype=torch.long))

        self.conj_edge_index   = _ei(conj_src,   conj_dst)
        self.disj_edge_index   = _ei(disj_src,   disj_dst)
        self.assign_edge_index = _ei(assign_src, assign_dst)

    def obs_to_graph(self, obs, env=None):
        action_mask = obs.get("action_mask") if isinstance(obs, dict) else None
        x = self._build_node_features(action_mask)
        data = Data()
        data.x                 = torch.tensor(x, dtype=torch.float32)
        data.num_nodes         = self.n_nodes
        data.conj_edge_index   = self.conj_edge_index.clone()
        data.disj_edge_index   = self.disj_edge_index.clone()
        data.assign_edge_index = self.assign_edge_index.clone()
        amask = action_mask if action_mask is not None else np.ones(self.n_jobs + 1)
        data.action_mask = torch.tensor(amask, dtype=torch.float32)
        return data

    def _build_node_features(self, action_mask):
        feat = np.zeros((self.n_nodes, self.unified_feat_dim), dtype=np.float32)
        for j in range(self.n_jobs):
            for pos in range(self.n_machines):
                op_idx = j * self.n_machines + pos
                mid    = int(self.op_machines[op_idx])
                f, col = feat[op_idx], 0
                f[col] = self.proc_times[op_idx] / self.max_proc_time;  col += 1
                f[col] = (self.n_machines - pos)  / self.n_machines;     col += 1
                f[col + mid] = 1.0;                                       col += self.n_machines
                f[col] = 0.0;  col += 1
                if (action_mask is not None and j < len(action_mask)
                        and action_mask[j] == 1 and pos == 0):
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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load checkpoint → GNNQNetwork
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, n_jobs, n_machines,
               hidden_dim=128, n_heads=4, n_layers=3):
    feat_dim = 6 + n_machines

    model = GNNQNetwork(
        feat_dim   = feat_dim,
        hidden_dim = hidden_dim,
        n_heads    = n_heads,
        n_layers   = n_layers,
        n_jobs     = n_jobs,
        n_machines = n_machines,
    ).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    if "q_net" in ckpt:
        model.load_state_dict(ckpt["q_net"])
        logger.info(f"  Loaded q_net weights from {checkpoint_path}")
    else:
        # Bare state dict
        model.load_state_dict(ckpt)
        logger.info(f"  Loaded bare state dict from {checkpoint_path}")

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Single greedy rollout
# ─────────────────────────────────────────────────────────────────────────────

def greedy_rollout(model, instance_path, n_jobs, n_machines, env_id):
    gb  = JSSPGraphBuilder(n_jobs, n_machines, instance_path)
    env = gym.make(env_id, env_config={"instance_path": instance_path})
    env = DynamicJSSWrapper(env)

    obs  = env.reset()
    done = False
    steps = 0

    while not done:
        sg      = gb.obs_to_graph(obs, env).to(DEVICE)
        amask   = obs["action_mask"] if isinstance(obs, dict) else None
        amask_t = torch.tensor(
            amask if amask is not None else np.ones(n_jobs + 1),
            dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            qv = model(sg).squeeze(0)
        qv[amask_t == 0] = -float('inf')
        action = qv.argmax().item()

        obs, _, done, _ = env.step(action)
        steps += 1

    makespan = getattr(env.unwrapped, 'last_time_step', float('inf'))
    env.close()
    return float(makespan), steps


# ─────────────────────────────────────────────────────────────────────────────
# EA baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_ea_on_instance(instance_path, alg, pop, gens, sa_iters):
    jobs    = parse_taillard(instance_path)
    np.random.seed(0)

    population = [random_permutation(jobs) for _ in range(pop)]
    fitness    = [compute_makespan(jobs, ind) for ind in population]
    best_val   = min(fitness)
    best_ind   = deepcopy(population[np.argmin(fitness)])

    CROSSOVER_P, MUTATION_P = 0.9, 0.1

    for gen in range(1, gens + 1):
        new_pop = [deepcopy(best_ind)]
        while len(new_pop) < pop:
            p1    = tournament_select(population, fitness, TOURNAMENT_K)
            p2    = tournament_select(population, fitness, TOURNAMENT_K)
            child = (order_crossover(p1, p2) if np.random.random() < CROSSOVER_P
                     else deepcopy(p1))
            child = mutate(child, mutation_rate=MUTATION_P)
            if alg in ("MA", "HGA"):
                t0    = calibrate_sa_temperature(jobs, child)
                child, _ = simulated_annealing_improve(
                    jobs, child, iters=sa_iters, t0=t0, tend=SA_TEND)
            new_pop.append(child)

        population = new_pop
        fitness    = [compute_makespan(jobs, ind) for ind in population]
        population, fitness = deduplicate_population(population, fitness, jobs)
        gen_best   = min(fitness)
        if gen_best < best_val:
            best_val = gen_best
            best_ind = deepcopy(population[fitness.index(gen_best)])

    return float(best_val)


# ─────────────────────────────────────────────────────────────────────────────
# Validation runner
# ─────────────────────────────────────────────────────────────────────────────

def validate(model, eval_paths, bks_map, n_jobs, n_machines, n_rollouts,
             ea_baseline, env_id):
    results = []

    for inst_path in eval_paths:
        inst_key = os.path.basename(inst_path).replace(".txt", "")
        inst_bks = float(bks_map.get(inst_key, 1.0)) if bks_map else None

        logger.info(f"\n  ── {inst_key}"
                    + (f"  (BKS={inst_bks:.0f})" if inst_bks else "") + " ──")

        # ── CQL rollouts ───────────────────────────────────────────────────
        makespans = []
        for r in range(n_rollouts):
            t0 = time.perf_counter()
            ms, steps = greedy_rollout(model, inst_path, n_jobs, n_machines, env_id)
            elapsed   = time.perf_counter() - t0
            makespans.append(ms)
            gap_str = (f"  gap={100*(ms-inst_bks)/inst_bks:+.2f}%"
                       if inst_bks else "")
            logger.info(f"    Rollout {r+1}/{n_rollouts}: "
                        f"makespan={ms:.0f}{gap_str}  "
                        f"steps={steps}  ({elapsed:.1f}s)")

        cql_avg  = float(np.mean(makespans))
        cql_best = float(np.min(makespans))
        cql_std  = float(np.std(makespans))

        if inst_bks:
            logger.info(f"    CQL avg  : {cql_avg:.0f}  "
                        f"gap={100*(cql_avg-inst_bks)/inst_bks:+.2f}%")
            logger.info(f"    CQL best : {cql_best:.0f}  "
                        f"gap={100*(cql_best-inst_bks)/inst_bks:+.2f}%")
            if n_rollouts > 1:
                logger.info(f"    CQL std  : {cql_std:.1f}")
        else:
            logger.info(f"    CQL avg  : {cql_avg:.0f}  best={cql_best:.0f}")

        # ── EA baseline ────────────────────────────────────────────────────
        ea_best = None
        if ea_baseline:
            t0 = time.perf_counter()
            ea_best = run_ea_on_instance(
                inst_path, args.evo_alg, args.evo_pop,
                args.evo_gens, args.evo_sa_iters)
            elapsed = time.perf_counter() - t0
            if inst_bks:
                logger.info(f"    EA  best : {ea_best:.0f}  "
                            f"gap={100*(ea_best-inst_bks)/inst_bks:+.2f}%  "
                            f"({elapsed:.1f}s)")
            else:
                logger.info(f"    EA  best : {ea_best:.0f}  ({elapsed:.1f}s)")

        results.append(dict(
            instance     = inst_key,
            bks          = inst_bks,
            cql_avg      = cql_avg,
            cql_best     = cql_best,
            cql_std      = cql_std,
            cql_gap_avg  = 100*(cql_avg-inst_bks)/inst_bks  if inst_bks else None,
            cql_gap_best = 100*(cql_best-inst_bks)/inst_bks if inst_bks else None,
            ea_best      = ea_best,
            ea_gap_best  = 100*(ea_best-inst_bks)/inst_bks
                           if (ea_best and inst_bks) else None,
        ))

    return results


def print_summary(results):
    logger.info(f"\n{'='*70}")
    logger.info("  Validation Summary")
    logger.info(f"{'='*70}")

    has_bks = any(r["bks"] for r in results)
    has_ea  = any(r["ea_best"] for r in results)

    hdr = (f"  {'Instance':<14} {'Makespan':>10}"
           + (f"  {'gap%':>7}" if has_bks else "")
           + (f"  {'EA best':>10} {'EA gap%':>7}" if has_ea else ""))
    logger.info(hdr)
    logger.info("  " + "─" * (len(hdr) - 2))

    for r in results:
        row = f"  {r['instance']:<14} {r['cql_avg']:>10.0f}"
        if has_bks and r["cql_gap_avg"] is not None:
            row += f"  {r['cql_gap_avg']:>+7.2f}%"
        if has_ea and r["ea_best"]:
            row += f"  {r['ea_best']:>10.0f}"
            if r["ea_gap_best"] is not None:
                row += f" {r['ea_gap_best']:>+7.2f}%"
        logger.info(row)

    if has_bks:
        valid_gaps = [r["cql_gap_avg"] for r in results if r["cql_gap_avg"] is not None]
        if valid_gaps:
            logger.info("  " + "─" * (len(hdr) - 2))
            logger.info(f"  {'MEAN':<14} {'':>10}  {np.mean(valid_gaps):>+7.2f}%")

    logger.info(f"{'='*70}")


def save_csv(results, path):
    import csv
    fields = ["instance", "bks", "cql_avg", "cql_best", "cql_std",
              "cql_gap_avg", "cql_gap_best", "ea_best", "ea_gap_best"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else
                            (v if v is not None else ""))
                        for k, v in r.items() if k in fields})
    logger.info(f"  CSV saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── 1. Inspect checkpoint ─────────────────────────────────────────────────
    ckpt_info = inspect_checkpoint(args.checkpoint)
    if args.inspect_only:
        sys.exit(0)

    # ── 2. Check env imports ──────────────────────────────────────────────────
    if not ENV_AVAILABLE:
        logger.error(f"Cannot run rollouts — missing imports: {_ENV_ERR}")
        sys.exit(1)

    if not args.eval_instances:
        logger.error("No --eval-instances provided. Use --inspect-only to just "
                     "examine the checkpoint.")
        sys.exit(1)

    # ── 3. Infer or validate dimensions ───────────────────────────────────────
    def _dims(path):
        with open(path) as f:
            t = f.read().split()
        return int(t[0]), int(t[1])

    if args.n_jobs is None or args.n_machines is None:
        dims = [_dims(p) for p in args.eval_instances]
        if len(set(dims)) != 1:
            logger.error("Eval instances have different dimensions — "
                         "pass --n-jobs and --n-machines explicitly.")
            sys.exit(1)
        n_jobs, n_machines = dims[0]
        logger.info(f"Inferred dimensions: {n_jobs} jobs × {n_machines} machines")
    else:
        n_jobs, n_machines = args.n_jobs, args.n_machines

    # ── 4. Load BKS ───────────────────────────────────────────────────────────
    bks_map = {}
    if args.bks:
        with open(args.bks) as f:
            bks_map = json.load(f)
        logger.info(f"BKS loaded from {args.bks}  ({len(bks_map)} entries)")
    else:
        logger.warning("No --bks provided; gap% will not be reported.")

    # ── 5. Load model ─────────────────────────────────────────────────────────
    logger.info(f"\nLoading model: {args.checkpoint}")
    model = load_model(
        args.checkpoint,
        n_jobs     = n_jobs,
        n_machines = n_machines,
        hidden_dim = args.gnn_hidden,
        n_heads    = args.gnn_heads,
        n_layers   = args.gnn_layers,
    )

    # ── 6. Run validation ─────────────────────────────────────────────────────
    eval_paths = [os.path.abspath(p) for p in args.eval_instances]

    logger.info(f"\nValidating on {len(eval_paths)} instance(s), "
                f"{args.n_rollouts} rollout(s) each …")

    results = validate(
        model        = model,
        eval_paths   = eval_paths,
        bks_map      = bks_map,
        n_jobs       = n_jobs,
        n_machines   = n_machines,
        n_rollouts   = args.n_rollouts,
        ea_baseline  = args.ea_baseline,
        env_id       = args.env_id,
    )

    # ── 7. Summary + CSV ──────────────────────────────────────────────────────
    print_summary(results)

    if args.out_csv:
        save_csv(results, args.out_csv)
