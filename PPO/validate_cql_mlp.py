#!/usr/bin/env python3
"""
evaluate_cql_vs_ea.py — Standalone CQL vs EA comparison script.

Loads a trained MLP-CQL model and compares it against an EA baseline
on a set of evaluation instances. EA runs until it matches or beats
the CQL makespan (or hits a time/gen budget), reporting exactly how
long and how many generations it needed.

Usage:
    python3 evaluate_cql_vs_ea.py \
        --weights  checkpoint_results/.../mlp_cql_weights.pt \
        --instances instances/ta61 instances/ta62 ... \
        --bks       bks.json \
        --evo-alg   HGA \
        --evo-pop   50 \
        --evo-gens  500 \
        --evo-sa-iters 1000 \
        --time-limit 300
"""

import argparse
import os
import time
import json
import csv
import multiprocessing as mp
from copy import deepcopy
from collections import defaultdict

import numpy as np
import gym
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    description="Evaluate CQL weights vs EA on JSSP instances.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--weights",     type=str, required=True,
                    help="Path to mlp_cql_weights.pt")
parser.add_argument("--instances",   type=str, nargs="+", required=True,
                    help="Instance file paths to evaluate")
parser.add_argument("--bks",         type=str, required=True,
                    help="Path to bks.json")
parser.add_argument("--mlp-hidden",  type=int, nargs="+", default=[1024, 1024, 512, 256])
parser.add_argument("--mlp-dropout", type=float, default=0.1)
parser.add_argument("--mlp-residual",action="store_true", default=True)

# EA settings
parser.add_argument("--evo-alg",     type=str, default="HGA", choices=["GA","MA","HGA"])
parser.add_argument("--evo-pop",     type=int, default=50)
parser.add_argument("--evo-gens",    type=int, default=500,
                    help="Max generations (hard ceiling)")
parser.add_argument("--evo-sa-iters",type=int, default=1000)
parser.add_argument("--n-workers",   type=int, default=max(1, mp.cpu_count()-2),
                    help="Parallel workers for EA fitness eval")
parser.add_argument("--time-limit",  type=float, default=300.0,
                    help="Max wall-clock seconds for EA per instance (default: 300)")

# Targets
parser.add_argument("--ea-target-gap", type=float, default=None,
                    help="EA stops when gap%% ≤ this vs BKS. "
                         "If not set, EA runs until it matches/beats CQL makespan.")
parser.add_argument("--cql-rollouts", type=int, default=1,
                    help="Number of CQL greedy rollouts per instance (default: 1)")

# Output
parser.add_argument("--out-csv",     type=str, default="cql_vs_ea_results.csv")
parser.add_argument("--alpha",       type=float, default=0.05)
parser.add_argument("--seed",        type=int, default=42)

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_ID      = "JSSEnv:jss-v1"
CROSSOVER_P = 0.9
MUTATION_P  = 0.1

with open(args.bks) as f:
    bks_map = json.load(f)

INSTANCE_PATHS = [os.path.abspath(p) for p in args.instances]


def read_dimensions(path):
    with open(path) as f:
        t = f.read().split()
    return int(t[0]), int(t[1])


dims = [read_dimensions(p) for p in INSTANCE_PATHS]
if len(set(dims)) != 1:
    raise ValueError("All instances must share (n_jobs, n_machines).")
NUM_JOBS, NUM_MACHINES = dims[0]

print(f"Problem: {NUM_JOBS} jobs × {NUM_MACHINES} machines")
print(f"Device:  {DEVICE}")
print(f"Workers: {args.n_workers}")
print()


def get_bks(path):
    key = os.path.basename(path).replace(".txt", "")
    return float(bks_map.get(key, float("inf")))


def make_env(path):
    base = gym.make(ENV_ID, env_config={"instance_path": path})
    return DynamicJSSWrapper(base)


def extract_flat_obs(obs):
    if isinstance(obs, dict):
        return obs["real_obs"].flatten().astype(np.float32)
    return np.array(obs, dtype=np.float32).flatten()


def extract_action_mask(obs):
    if isinstance(obs, dict) and "action_mask" in obs:
        return obs["action_mask"].astype(np.float32)
    return np.ones(NUM_JOBS + 1, dtype=np.float32)


def compute_jssp_lower_bound(jobs):
    job_spans     = [sum(pt for _, pt in job) for job in jobs]
    machine_loads = defaultdict(float)
    for job in jobs:
        for m, pt in job:
            machine_loads[m] += pt
    return max(max(job_spans), max(machine_loads.values()))


# ───────────────────────────────────────────────
# MLP Q-Network (must match training architecture)
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
                 hidden_layers=(1024,1024,512,256),
                 dropout=0.1, use_residual=True):
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
        for block in self.trunk: x = block(x)
        V = self.value_head(x); A = self.advantage_head(x)
        if action_mask is not None:
            Am = A.clone(); Am[action_mask == 0] = -1e9
            lc = action_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            A_mean = (Am * action_mask).sum(dim=-1, keepdim=True) / lc
        else:
            A_mean = A.mean(dim=-1, keepdim=True)
        Q = V + A - A_mean
        if action_mask is not None: Q = Q.masked_fill(action_mask == 0, -1e9)
        if squeeze: Q = Q.squeeze(0)
        return Q


def load_model(weights_path, obs_dim, n_actions):
    """Load CQL weights into MLPQNetwork."""
    net = MLPQNetwork(
        obs_dim, n_actions,
        hidden_layers=tuple(args.mlp_hidden),
        dropout=args.mlp_dropout,
        use_residual=args.mlp_residual,
    ).to(DEVICE)
    net.eval()

    ckpt = torch.load(weights_path, map_location=DEVICE)
    # Supports both raw state_dict and trainer checkpoint format
    state = ckpt.get("q_net", ckpt)
    net.load_state_dict(state)
    print(f"Loaded weights from: {weights_path}")
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    return net


# ───────────────────────────────────────────────
# CQL greedy rollout
# ───────────────────────────────────────────────

def cql_rollout(net, instance_path):
    """Single greedy CQL rollout. Returns (makespan, steps, elapsed_s)."""
    env  = make_env(instance_path)
    obs  = env.reset(); done = False; steps = 0
    t0   = time.perf_counter()

    with torch.no_grad():
        while not done:
            flat  = extract_flat_obs(obs)
            mask  = extract_action_mask(obs)
            obs_t = torch.tensor(flat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            msk_t = torch.tensor(mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action = net(obs_t, msk_t).squeeze(0).argmax().item()
            obs, _, done, _ = env.step(action)
            steps += 1

    elapsed  = time.perf_counter() - t0
    makespan = getattr(env.unwrapped, "last_time_step", float("inf"))
    env.close()
    return float(makespan), steps, elapsed


# ───────────────────────────────────────────────
# EA with gap% + CQL-match + time-limit stopping
# ───────────────────────────────────────────────

def _eval_one(args_tuple):
    jobs, ind = args_tuple
    return compute_makespan(jobs, ind)


def _eval_pop(jobs, pop, n_workers):
    if n_workers <= 1 or len(pop) <= 4:
        return [compute_makespan(jobs, ind) for ind in pop]
    with mp.Pool(processes=n_workers) as pool:
        return pool.map(_eval_one, [(jobs, ind) for ind in pop],
                        chunksize=max(1, len(pop)//n_workers))


def _sa_one(args_tuple):
    jobs, child, sa_iters = args_tuple
    t0 = calibrate_sa_temperature(jobs, child)
    improved, _ = simulated_annealing_improve(jobs, child, iters=sa_iters, t0=t0, tend=SA_TEND)
    return improved


def _sa_pop(jobs, candidates, sa_iters, n_workers):
    if n_workers <= 1 or len(candidates) <= 2:
        result = []
        for c in candidates:
            t0 = calibrate_sa_temperature(jobs, c)
            improved, _ = simulated_annealing_improve(jobs, c, iters=sa_iters, t0=t0, tend=SA_TEND)
            result.append(improved)
        return result
    with mp.Pool(processes=n_workers) as pool:
        return pool.map(_sa_one, [(jobs, c, sa_iters) for c in candidates],
                        chunksize=max(1, len(candidates)//n_workers))


def run_ea_vs_cql(jobs, bks, cql_makespan, lb,
                  alg, pop_size, max_gens, sa_iters,
                  n_workers, time_limit, ea_target_gap, seed):
    """
    Run EA with three stopping conditions (whichever triggers first):
      1. gap% vs BKS ≤ ea_target_gap  (if --ea-target-gap is set)
      2. makespan ≤ cql_makespan       (EA matched or beat CQL)
      3. wall-clock time ≥ time_limit
      4. max_gens reached

    Returns dict with full trajectory of gap% per generation,
    plus which condition stopped it and when.
    """
    import random
    random.seed(seed); np.random.seed(seed)

    pop     = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = _eval_pop(jobs, pop, n_workers)
    best    = min(fitness)
    best_ind= deepcopy(pop[np.argmin(fitness)])

    trajectory = []   # (gen, best_makespan, gap_vs_bks, gap_vs_lb, elapsed_s)
    t_start    = time.perf_counter()
    stop_reason= f"max_gens={max_gens}"
    gen_used   = 0

    def _record(gen, best_val):
        elapsed      = time.perf_counter() - t_start
        gap_vs_bks   = (best_val - bks)  / bks  * 100.0
        gap_vs_lb    = (best_val - lb)   / lb   * 100.0
        trajectory.append(dict(gen=gen, makespan=best_val,
                               gap_bks=gap_vs_bks, gap_lb=gap_vs_lb,
                               elapsed=elapsed))

    _record(0, best)

    for gen in range(1, max_gens + 1):
        gen_used  = gen
        elapsed   = time.perf_counter() - t_start

        # Time limit check
        if elapsed >= time_limit:
            stop_reason = f"time_limit={time_limit:.0f}s at gen {gen}"
            break

        # Generate offspring
        if alg == "GA":
            new_pop = [deepcopy(best_ind)]
            while len(new_pop) < pop_size:
                p1    = tournament_select(pop, fitness, TOURNAMENT_K)
                p2    = tournament_select(pop, fitness, TOURNAMENT_K)
                child = order_crossover(p1, p2) if np.random.random() < CROSSOVER_P else deepcopy(p1)
                new_pop.append(mutate(child, mutation_rate=MUTATION_P))
            fitness      = _eval_pop(jobs, new_pop, n_workers)
            pop, fitness = deduplicate_population(new_pop, fitness, jobs)

        elif alg == "MA":
            offspring = []
            while len(offspring) < pop_size - 1:
                p1    = tournament_select(pop, fitness, TOURNAMENT_K)
                p2    = tournament_select(pop, fitness, TOURNAMENT_K)
                child = order_crossover(p1, p2) if np.random.random() < CROSSOVER_P else deepcopy(p1)
                offspring.append(mutate(child, mutation_rate=MUTATION_P))
            improved     = _sa_pop(jobs, offspring, sa_iters, n_workers)
            new_pop      = [deepcopy(best_ind)] + improved
            fitness      = _eval_pop(jobs, new_pop, n_workers)
            pop, fitness = deduplicate_population(new_pop, fitness, jobs)

        else:  # HGA
            offspring = []
            while len(offspring) < pop_size - 1:
                p1    = tournament_select(pop, fitness, TOURNAMENT_K)
                p2    = tournament_select(pop, fitness, TOURNAMENT_K)
                child = order_crossover(p1, p2) if np.random.random() < CROSSOVER_P else deepcopy(p1)
                offspring.append(mutate(child, mutation_rate=MUTATION_P))
            off_fit  = _eval_pop(jobs, offspring, n_workers)
            ranked   = sorted(zip(off_fit, offspring), key=lambda x: x[0])
            half     = len(ranked) // 2
            improved = _sa_pop(jobs, [c for _, c in ranked[:half]], sa_iters, n_workers)
            new_pop  = [deepcopy(best_ind)] + improved + [c for _, c in ranked[half:]]
            fitness      = _eval_pop(jobs, new_pop, n_workers)
            pop, fitness = deduplicate_population(new_pop, fitness, jobs)

        gen_best = min(fitness)
        if gen_best < best:
            best     = gen_best
            best_ind = deepcopy(pop[fitness.index(gen_best)])

        _record(gen, best)

        gap_bks = (best - bks) / bks * 100.0
        gap_lb  = (best - lb)  / lb  * 100.0

        print(f"    {alg} | Gen {gen:>4}/{max_gens} | "
              f"makespan={best} | gap_bks={gap_bks:+.2f}% | "
              f"gap_lb={gap_lb:+.2f}% | "
              f"t={time.perf_counter()-t_start:.1f}s", flush=True)

        # Stopping conditions
        if ea_target_gap is not None and gap_bks <= ea_target_gap:
            stop_reason = f"gap_target={ea_target_gap:.1f}% reached at gen {gen}"
            break

        if best <= cql_makespan:
            stop_reason = f"matched/beat CQL ({cql_makespan:.0f}) at gen {gen}"
            break

    final_elapsed = time.perf_counter() - t_start
    final_gap_bks = (best - bks) / bks * 100.0
    final_gap_lb  = (best - lb)  / lb  * 100.0

    # Find the generation where EA first matched CQL
    matched_gen = matched_time = None
    for entry in trajectory:
        if entry["makespan"] <= cql_makespan:
            matched_gen  = entry["gen"]
            matched_time = entry["elapsed"]
            break

    return dict(
        best_makespan  = best,
        final_gap_bks  = final_gap_bks,
        final_gap_lb   = final_gap_lb,
        gens_used      = gen_used,
        elapsed_s      = final_elapsed,
        stop_reason    = stop_reason,
        matched_cql_gen  = matched_gen,
        matched_cql_time = matched_time,
        trajectory     = trajectory,
    )


# ───────────────────────────────────────────────
# Per-instance evaluation
# ───────────────────────────────────────────────

def evaluate_instance(instance_path, net):
    key = os.path.basename(instance_path).replace(".txt", "")
    bks = get_bks(instance_path)
    jobs = parse_taillard(instance_path)
    lb   = compute_jssp_lower_bound(jobs)

    print(f"\n{'='*70}")
    print(f"  Instance: {key}  BKS={bks:.0f}  LB={lb:.0f}")
    print(f"{'='*70}")

    # ── CQL rollout(s) ──────────────────────────────────────────────────
    print(f"  Running {args.cql_rollouts} CQL rollout(s)...")
    cql_makespans = []; cql_times = []; cql_steps_list = []
    for r in range(args.cql_rollouts):
        ms, steps, elapsed = cql_rollout(net, instance_path)
        cql_makespans.append(ms)
        cql_times.append(elapsed)
        cql_steps_list.append(steps)
        print(f"    Rollout {r+1}: makespan={ms:.0f}  "
              f"gap_bks={100*(ms-bks)/bks:+.2f}%  "
              f"gap_lb={(ms-lb)/lb*100:+.2f}%  "
              f"steps={steps}  time={elapsed:.3f}s")

    cql_avg      = float(np.mean(cql_makespans))
    cql_best     = float(np.min(cql_makespans))
    cql_avg_time = float(np.mean(cql_times))

    print(f"\n  CQL summary:")
    print(f"    avg makespan : {cql_avg:.0f}  gap_bks={100*(cql_avg-bks)/bks:+.2f}%  "
          f"gap_lb={(cql_avg-lb)/lb*100:+.2f}%")
    print(f"    best makespan: {cql_best:.0f}  gap_bks={100*(cql_best-bks)/bks:+.2f}%  "
          f"gap_lb={(cql_best-lb)/lb*100:+.2f}%")
    print(f"    avg time     : {cql_avg_time:.3f}s")

    # ── EA run ──────────────────────────────────────────────────────────
    target_str = (f"gap≤{args.ea_target_gap:.1f}% vs BKS"
                  if args.ea_target_gap is not None
                  else f"match/beat CQL ({cql_best:.0f})")
    print(f"\n  Running {args.evo_alg} EA (target: {target_str}, "
          f"time_limit={args.time_limit:.0f}s, max_gens={args.evo_gens})...")

    ea = run_ea_vs_cql(
        jobs         = jobs,
        bks          = bks,
        cql_makespan = cql_best,
        lb           = lb,
        alg          = args.evo_alg,
        pop_size     = args.evo_pop,
        max_gens     = args.evo_gens,
        sa_iters     = args.evo_sa_iters,
        n_workers    = args.n_workers,
        time_limit   = args.time_limit,
        ea_target_gap= args.ea_target_gap,
        seed         = args.seed,
    )

    print(f"\n  EA summary:")
    print(f"    best makespan: {ea['best_makespan']:.0f}  "
          f"gap_bks={ea['final_gap_bks']:+.2f}%  "
          f"gap_lb={ea['final_gap_lb']:+.2f}%")
    print(f"    generations  : {ea['gens_used']}")
    print(f"    wall time    : {ea['elapsed_s']:.1f}s")
    print(f"    stop reason  : {ea['stop_reason']}")

    if ea["matched_cql_gen"] is not None:
        print(f"    ✓ EA matched/beat CQL at gen {ea['matched_cql_gen']} "
              f"({ea['matched_cql_time']:.1f}s)")
    else:
        print(f"    ✗ EA did NOT match CQL within budget  "
              f"(CQL={cql_best:.0f}, EA best={ea['best_makespan']:.0f}, "
              f"gap={ea['best_makespan']-cql_best:.0f})")

    # ── Head-to-head ────────────────────────────────────────────────────
    cql_wins = cql_best < ea["best_makespan"]
    ea_wins  = ea["best_makespan"] < cql_best
    tie      = cql_best == ea["best_makespan"]
    winner   = "CQL" if cql_wins else ("EA" if ea_wins else "TIE")

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  Head-to-head:  CQL={cql_best:.0f}  EA={ea['best_makespan']:.0f}  "
          f"→  {winner} wins  │")
    print(f"  └─────────────────────────────────────────────┘")

    return dict(
        instance        = key,
        bks             = bks,
        lb              = lb,

        cql_avg         = cql_avg,
        cql_best        = cql_best,
        cql_gap_bks_avg = 100*(cql_avg  - bks)/bks,
        cql_gap_bks_best= 100*(cql_best - bks)/bks,
        cql_gap_lb_avg  = (cql_avg  - lb)/lb*100,
        cql_gap_lb_best = (cql_best - lb)/lb*100,
        cql_time_s      = cql_avg_time,

        ea_best         = ea["best_makespan"],
        ea_gap_bks      = ea["final_gap_bks"],
        ea_gap_lb       = ea["final_gap_lb"],
        ea_gens         = ea["gens_used"],
        ea_time_s       = ea["elapsed_s"],
        ea_stop_reason  = ea["stop_reason"],

        ea_matched_cql_gen  = ea["matched_cql_gen"],
        ea_matched_cql_time = ea["matched_cql_time"],

        winner          = winner,
        trajectory      = ea["trajectory"],
    )


# ───────────────────────────────────────────────
# Summary table
# ───────────────────────────────────────────────

def print_summary(results):
    print(f"\n\n{'='*100}")
    print("  FINAL SUMMARY")
    print(f"{'='*100}")

    hdr = (f"  {'Instance':<10} {'BKS':>6} {'LB':>6}  "
           f"{'CQL best':>9} {'vsBKS':>7} {'vsLB':>7} {'t(s)':>6}  "
           f"{'EA best':>9} {'vsBKS':>7} {'vsLB':>7} {'gens':>5} {'t(s)':>7}  "
           f"{'Matched@':>8} {'MatchT':>7}  {'Winner':>6}")
    print(hdr)
    print("  " + "─"*(len(hdr)-2))

    for r in results:
        mg = f"gen{r['ea_matched_cql_gen']}"  if r["ea_matched_cql_gen"] else "never"
        mt = f"{r['ea_matched_cql_time']:.1f}s" if r["ea_matched_cql_time"] else "—"
        print(
            f"  {r['instance']:<10} {r['bks']:>6.0f} {r['lb']:>6.0f}  "
            f"{r['cql_best']:>9.0f} {r['cql_gap_bks_best']:>+7.2f}% "
            f"{r['cql_gap_lb_best']:>+7.2f}% {r['cql_time_s']:>6.3f}  "
            f"{r['ea_best']:>9.0f} {r['ea_gap_bks']:>+7.2f}% "
            f"{r['ea_gap_lb']:>+7.2f}% {r['ea_gens']:>5} {r['ea_time_s']:>7.1f}  "
            f"{mg:>8} {mt:>7}  {r['winner']:>6}"
        )

    print("  " + "─"*(len(hdr)-2))

    cql_gaps  = [r["cql_gap_bks_best"] for r in results]
    ea_gaps   = [r["ea_gap_bks"]       for r in results]
    ea_times  = [r["ea_time_s"]        for r in results]
    ea_gens   = [r["ea_gens"]          for r in results]
    matched   = [r for r in results if r["ea_matched_cql_gen"] is not None]

    print(f"\n  MEAN  CQL gap vs BKS : {np.mean(cql_gaps):+.2f}%")
    print(f"  MEAN  EA  gap vs BKS : {np.mean(ea_gaps):+.2f}%")
    print(f"  MEAN  EA  wall time  : {np.mean(ea_times):.1f}s")
    print(f"  MEAN  EA  gens used  : {np.mean(ea_gens):.1f}")
    print(f"\n  EA matched/beat CQL  : {len(matched)}/{len(results)} instances")

    if matched:
        match_gens  = [r["ea_matched_cql_gen"]  for r in matched]
        match_times = [r["ea_matched_cql_time"] for r in matched]
        print(f"    avg gen to match : {np.mean(match_gens):.1f}")
        print(f"    avg time to match: {np.mean(match_times):.1f}s")

    cql_wins = sum(1 for r in results if r["winner"] == "CQL")
    ea_wins  = sum(1 for r in results if r["winner"] == "EA")
    ties     = sum(1 for r in results if r["winner"] == "TIE")
    print(f"\n  Head-to-head: CQL wins={cql_wins}  EA wins={ea_wins}  Ties={ties}")

    # Statistical test
    if len(results) >= 3:
        diffs = np.array(cql_gaps) - np.array(ea_gaps)
        if len(results) < 20:
            try:
                stat, p = stats.wilcoxon(cql_gaps, ea_gaps, alternative="two-sided")
                test_name = "Wilcoxon signed-rank"
            except Exception:
                stat, p = float("nan"), 1.0; test_name = "n/a"
        else:
            stat, p = stats.ttest_rel(cql_gaps, ea_gaps)
            test_name = "paired t-test"
        sig = "SIGNIFICANT" if p < args.alpha else "not significant"
        print(f"\n  Statistical test ({test_name}, α={args.alpha}):")
        print(f"    CQL−EA mean diff = {np.mean(diffs):+.2f}%  p={p:.4f}  {sig}")
        if p < args.alpha:
            better = "CQL" if np.mean(diffs) < 0 else "EA"
            print(f"    → {better} is significantly better")

    print(f"{'='*100}")


def save_csv(results, path):
    """Save flat results (excluding trajectory) to CSV."""
    fields = [
        "instance","bks","lb",
        "cql_avg","cql_best","cql_gap_bks_avg","cql_gap_bks_best",
        "cql_gap_lb_avg","cql_gap_lb_best","cql_time_s",
        "ea_best","ea_gap_bks","ea_gap_lb",
        "ea_gens","ea_time_s","ea_stop_reason",
        "ea_matched_cql_gen","ea_matched_cql_time","winner",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: (f"{v:.4f}" if isinstance(v, float) else
                       (v if v is not None else ""))
                   for k, v in r.items() if k in fields}
            w.writerow(row)
    print(f"\n  CSV saved → {path}")


def save_trajectory_csv(results, base_path):
    """Save per-generation EA trajectory for each instance."""
    traj_path = base_path.replace(".csv", "_trajectory.csv")
    with open(traj_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instance","gen","makespan",
                                           "gap_bks","gap_lb","elapsed"])
        w.writeheader()
        for r in results:
            for entry in r["trajectory"]:
                w.writerow({"instance": r["instance"], **entry})
    print(f"  Trajectory CSV → {traj_path}")


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    # Probe obs/action dims from first instance
    env_tmp   = make_env(INSTANCE_PATHS[0])
    obs_tmp   = env_tmp.reset()
    obs_dim   = len(extract_flat_obs(obs_tmp))
    n_actions = env_tmp.action_space.n
    env_tmp.close()
    print(f"obs_dim={obs_dim}  n_actions={n_actions}\n")

    # Load model
    net = load_model(args.weights, obs_dim, n_actions)

    print(f"\nEvaluation config:")
    print(f"  EA:          {args.evo_alg}  pop={args.evo_pop}  "
          f"max_gens={args.evo_gens}  sa_iters={args.evo_sa_iters}")
    print(f"  Time limit:  {args.time_limit:.0f}s per instance")
    print(f"  EA target:   "
          + (f"gap ≤ {args.ea_target_gap:.1f}% vs BKS"
             if args.ea_target_gap else "match/beat CQL makespan"))
    print(f"  CQL rollouts:{args.cql_rollouts}")
    print(f"  Instances:   {len(INSTANCE_PATHS)}")

    # Evaluate each instance
    results = []
    for path in INSTANCE_PATHS:
        r = evaluate_instance(path, net)
        results.append(r)

    # Summary
    print_summary(results)

    # Save outputs
    save_csv(results, args.out_csv)
    save_trajectory_csv(results, args.out_csv)

    print("\nDone.")
