#!/usr/bin/env python3
"""
warm_start_GA.py

Two-stage pipeline for Taillard JSSP instances:
  1. Evolutionary Algorithm (GA/MA/HGA) runs across synthetically generated
     training instances (same dimensionality as eval instances, random machine
     orders and processing times) to produce a pooled set of demonstration
     schedules.
  2. Behaviour Cloning (BC) trains a neural network to imitate the EA's
     dispatching decisions via supervised learning.

After training, the BC model is evaluated on real held-out Taillard instances
whose BKS is known, enabling a meaningful optimality gap comparison.
Statistical tests validate whether BC performance differs from the EA baseline.
All output is mirrored to results.txt.

Usage:
    # Train on 5 synthetic instances, evaluate on ta56-ta60 (all 100x10)
    python3 warm_start_GA.py \\
        --bks bks.json \\
        --eval-instances instances/ta56 instances/ta57 instances/ta58 instances/ta59 instances/ta60 \\
        --num-train-instances 5

    # Larger synthetic training set, MA algorithm, more generations
    python3 warm_start_GA.py \\
        --bks bks.json \\
        --eval-instances instances/ta56 instances/ta57 instances/ta58 \\
        --num-train-instances 10 \\
        --evo-alg MA --evo-gens 10 --evo-sa-iters 1000

    # Reproducible run with explicit seed, HGA, 50-epoch BC
    python3 warm_start_GA.py \\
        --bks bks.json \\
        --eval-instances instances/ta56 instances/ta57 instances/ta58 instances/ta59 instances/ta60 \\
        --num-train-instances 8 \\
        --train-instance-seed 42 \\
        --evo-alg HGA --evo-gens 5 \\
        --bc-epochs 50 --bc-hidden 512 512 256 \\
        --bc-eval-episodes 3 \\
        --alpha 0.05

    # Quick smoke-test (tiny settings)
    python3 warm_start_GA.py \\
        --bks bks.json \\
        --eval-instances instances/ta56 \\
        --num-train-instances 2 \\
        --evo-pop 10 --evo-gens 2 --bc-epochs 5 --bc-demo-episodes 20
"""

import csv
import argparse
import os
import time
import json
import logging
import tempfile
from copy import deepcopy

import numpy as np
import gym
from scipy import stats

import tensorflow as tf
from tensorflow import keras

from dynamic_jss_wrapper import DynamicJSSWrapper
from models import FCMaskedActionsModelTF

# ── Import EA primitives ──────────────────────────────────────────────────────
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
# CLI arguments (parsed early so CHECKPOINT_DIR is known before logging setup)
# ───────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description=(
        "EA warm-start → Behaviour Cloning pipeline for Taillard JSSP.\n"
        "Training instances are synthetically generated with the same\n"
        "dimensionality as the (real, BKS-known) evaluation instances."
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
──────────────────────────────────────────────────────────────────────
CLI EXAMPLES
──────────────────────────────────────────────────────────────────────

1. Basic run — 5 synthetic training instances, eval on ta56-ta60 (100×10):

   python3 warm_start_GA.py \\
       --bks bks.json \\
       --eval-instances instances/ta56 instances/ta57 instances/ta58 \\
                        instances/ta59 instances/ta60 \\
       --num-train-instances 5

2. More synthetic data, Memetic Algorithm, longer EA:

   python3 warm_start_GA.py \\
       --bks bks.json \\
       --eval-instances instances/ta56 instances/ta57 instances/ta58 \\
       --num-train-instances 10 \\
       --evo-alg MA --evo-gens 10 --evo-sa-iters 1000

3. Reproducible run, explicit seed, wider BC network, 3 eval episodes:

   python3 warm_start_GA.py \\
       --bks bks.json \\
       --eval-instances instances/ta56 instances/ta57 instances/ta58 \\
                        instances/ta59 instances/ta60 \\
       --num-train-instances 8 \\
       --train-instance-seed 42 \\
       --evo-alg HGA --evo-gens 5 \\
       --bc-epochs 50 --bc-hidden 512 512 256 \\
       --bc-eval-episodes 3 \\
       --alpha 0.05

4. Pure-expert demos (no epsilon noise), small hidden network:

   python3 warm_start_GA.py \\
       --bks bks.json \\
       --eval-instances instances/ta56 instances/ta57 \\
       --num-train-instances 5 \\
       --bc-epsilon 0.0 --bc-hidden 256 256

5. Quick smoke-test (tiny settings for debugging):

   python3 warm_start_GA.py \\
       --bks bks.json \\
       --eval-instances instances/ta56 \\
       --num-train-instances 2 \\
       --evo-pop 10 --evo-gens 2 \\
       --bc-epochs 5 --bc-demo-episodes 20
──────────────────────────────────────────────────────────────────────
    """,
)

# ── Instance arguments ───────────────────────────────────────────────────────
parser.add_argument(
    "--eval-instances", type=str, nargs="+", required=True,
    help=(
        "Paths to real Taillard instance files used for out-of-sample evaluation. "
        "BKS must be present in --bks for each instance. All must share the same "
        "(n_jobs, n_machines); this dimensionality is used to generate training instances."
    ),
)
parser.add_argument(
    "--num-train-instances", type=int, required=True,
    help=(
        "Number of synthetic training instances to generate. Each is a random JSSP "
        "with the same (n_jobs, n_machines) as the eval instances but different "
        "machine orderings and processing times drawn uniformly from [1, 99]."
    ),
)
parser.add_argument(
    "--train-instance-seed", type=int, default=0,
    help=(
        "Master seed for synthetic instance generation. "
        "Instance i uses seed (train_instance_seed + i) for reproducibility. "
        "(default: 0)"
    ),
)

# ── Output / BKS ─────────────────────────────────────────────────────────────
parser.add_argument("--out", type=str, default="checkpoint_results",
                    help="Output directory (default: checkpoint_results)")
parser.add_argument("--bks", type=str, required=True,
                    help="Path to bks.json")

# ── EA arguments ─────────────────────────────────────────────────────────────
parser.add_argument("--evo-alg", type=str, default="HGA", choices=["GA", "MA", "HGA"],
                    help="EA variant for warm-start demonstrations (default: HGA)")
parser.add_argument("--evo-pop", type=int, default=50,
                    help="EA population size (default: 50)")
parser.add_argument("--evo-gens", type=int, default=5,
                    help="EA generations (default: 5)")
parser.add_argument("--evo-sa-iters", type=int, default=500,
                    help="SA iterations per offspring/elite in MA/HGA (default: 500)")
parser.add_argument("--evo-early-stop", action="store_true", default=False,
                    help="Stop EA early if best makespan doesn't improve for 3 generations")

# ── BC arguments ─────────────────────────────────────────────────────────────
parser.add_argument("--bc-epochs", type=int, default=50,
                    help="Behaviour cloning training epochs (default: 50)")
parser.add_argument("--bc-hidden", type=int, nargs="+", default=[512, 512, 256],
                    help="Hidden layer sizes for BC network (default: 512 512 256)")
parser.add_argument("--bc-demo-episodes", type=int, default=None,
                    help=(
                        "Demo episodes per training instance. "
                        "Defaults to 500 if epsilon > 0, else evo_pop // 2."
                    ))
parser.add_argument("--bc-epsilon", type=float, default=0.05,
                    help=(
                        "Epsilon for epsilon-greedy noise during demo collection. "
                        "0.0 = pure expert, 0.05 = 5%% random actions (default: 0.05)"
                    ))
parser.add_argument("--bc-eval-episodes", type=int, default=1,
                    help="Greedy rollout episodes per eval instance (default: 1)")

# ── Evaluation / stats arguments ─────────────────────────────────────────────
parser.add_argument("--eval-ea-baseline", action="store_true", default=True,
                    help="Run EA fresh on each eval instance as a comparison baseline (default: True)")
parser.add_argument("--alpha", type=float, default=0.05,
                    help="Significance level for statistical tests (default: 0.05)")

args = parser.parse_args()

# ───────────────────────────────────────────────
# Output directory + logging (file + console)
# ───────────────────────────────────────────────
CHECKPOINT_DIR  = os.path.abspath(args.out)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
RESULTS_TXT     = os.path.join(CHECKPOINT_DIR, "results.txt")

_fmt     = "[%(asctime)s] %(levelname)-8s %(message)s"
_datefmt = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=_fmt, datefmt=_datefmt)
logger = logging.getLogger()

_file_handler = logging.FileHandler(RESULTS_TXT, mode="w", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(_fmt, datefmt=_datefmt))
logger.addHandler(_file_handler)

# ───────────────────────────────────────────────
# Resolve eval instance paths
# ───────────────────────────────────────────────
EVAL_INSTANCE_PATHS = [os.path.abspath(p) for p in args.eval_instances]

# ───────────────────────────────────────────────
# Dimensionality — inferred from eval instances (all must agree)
# ───────────────────────────────────────────────
def read_dimensions(path):
    with open(path) as f:
        tokens = f.read().split()
    return int(tokens[0]), int(tokens[1])

eval_dims = [read_dimensions(p) for p in EVAL_INSTANCE_PATHS]
if len(set(eval_dims)) != 1:
    raise ValueError(
        "All eval instances must share the same (n_jobs, n_machines).\n"
        + "\n".join(f"  {p}: {d}" for p, d in zip(EVAL_INSTANCE_PATHS, eval_dims))
    )
NUM_JOBS, NUM_MACHINES = eval_dims[0]
TOTAL_OPS = NUM_JOBS * NUM_MACHINES

# ───────────────────────────────────────────────
# Resolve bc_demo_episodes
# ───────────────────────────────────────────────
if args.bc_demo_episodes is None:
    args.bc_demo_episodes = 500 if args.bc_epsilon > 0.0 else max(1, args.evo_pop // 2)

# ───────────────────────────────────────────────
# Global constants
# ───────────────────────────────────────────────
ENV_ID          = "JSSEnv:jss-v1"
BC_EPOCHS       = args.bc_epochs
BC_LR           = 1e-3
BC_BATCH        = 256
BC_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "bc_weights")
CROSSOVER_P     = 0.9
MUTATION_P      = 0.1
ALPHA           = args.alpha

# ───────────────────────────────────────────────
# BKS lookup
# ───────────────────────────────────────────────
with open(args.bks, "r") as f:
    bks_map = json.load(f)

def get_bks(instance_path):
    key = os.path.basename(instance_path).replace(".txt", "")
    return float(bks_map.get(key, 1.0))

# ───────────────────────────────────────────────
# Synthetic instance generation
# ───────────────────────────────────────────────

def generate_random_jssp_instance(n_jobs, n_machines, seed, proc_lo=1, proc_hi=99):
    """
    Generate a random JSSP instance with the Taillard file format:

        n_jobs n_machines
        <n_jobs lines, each with 2*n_machines integers: machine_id proc_time ...>

    Each job visits all machines in a unique random order.
    Processing times are drawn uniformly from [proc_lo, proc_hi].

    Parameters
    ----------
    n_jobs, n_machines : int
        Problem dimensionality — must match the eval instances.
    seed : int
        RNG seed for reproducibility.
    proc_lo, proc_hi : int
        Inclusive bounds for processing-time sampling (Taillard uses 1–99).

    Returns
    -------
    str
        File content ready to write as a Taillard-format instance.
    """
    rng = np.random.default_rng(seed)
    lines = [f"{n_jobs} {n_machines}"]
    for _ in range(n_jobs):
        machine_order = rng.permutation(n_machines).tolist()
        proc_times    = rng.integers(proc_lo, proc_hi + 1, size=n_machines).tolist()
        pairs = " ".join(f"{m} {t}" for m, t in zip(machine_order, proc_times))
        lines.append(pairs)
    return "\n".join(lines) + "\n"


def create_synthetic_training_instances(n_instances, n_jobs, n_machines,
                                        master_seed, out_dir):
    """
    Write `n_instances` synthetic JSSP files to `out_dir`.
    Instance i is seeded with (master_seed + i) for full reproducibility.

    Returns
    -------
    list[str]
        Absolute paths to the generated instance files.
    """
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

    logger.info(
        f"  Generated {n_instances} synthetic instance(s) "
        f"({n_jobs}×{n_machines}) in {synth_dir}"
    )
    return paths

# ───────────────────────────────────────────────
# Environment helper
# ───────────────────────────────────────────────
def make_wrapped_env(instance_path):
    base_env = gym.make(ENV_ID, env_config={"instance_path": instance_path})
    return DynamicJSSWrapper(base_env)

# ───────────────────────────────────────────────
# EA warm-start variants
# ───────────────────────────────────────────────

def _run_ea_warmstart_ga(jobs, pop_size, generations, seed=42, early_stop=False):
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
            child = order_crossover(p1, p2) if random.random() < CROSSOVER_P else deepcopy(p1)
            child = mutate(child, mutation_rate=MUTATION_P)
            new_pop.append(child)

        pop, fitness = new_pop, [compute_makespan(jobs, ind) for ind in new_pop]
        pop, fitness = deduplicate_population(pop, fitness, jobs)
        gen_best = min(fitness)

        if gen_best < best_val:
            best_val, best_ind, patience = gen_best, deepcopy(pop[fitness.index(gen_best)]), 0
        else:
            patience += 1

        logger.info(f"    GA | Gen {gen}/{generations} | Best: {best_val} | Patience: {patience}/3")
        if early_stop and patience >= 3:
            logger.info(f"    GA | Early stopping at gen {gen}.")
            break

    return pop, fitness, best_ind, best_val


def _run_ea_warmstart_ma(jobs, pop_size, generations, sa_iters, seed=42):
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
            child = order_crossover(p1, p2) if random.random() < CROSSOVER_P else deepcopy(p1)
            child = mutate(child, mutation_rate=MUTATION_P)
            t0    = calibrate_sa_temperature(jobs, child)
            child, _ = simulated_annealing_improve(jobs, child, iters=sa_iters, t0=t0, tend=SA_TEND)
            new_pop.append(child)

        pop, fitness = new_pop, [compute_makespan(jobs, ind) for ind in new_pop]
        pop, fitness = deduplicate_population(pop, fitness, jobs)
        gen_best = min(fitness)
        if gen_best < best_val:
            best_val, best_ind = gen_best, deepcopy(pop[fitness.index(gen_best)])

        logger.info(f"    MA | Gen {gen}/{generations} | Best: {best_val}")

    return pop, fitness, best_ind, best_val


def _run_ea_warmstart_hga(jobs, pop_size, generations, sa_iters, seed=42):
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
            child = order_crossover(p1, p2) if random.random() < CROSSOVER_P else deepcopy(p1)
            child = mutate(child, mutation_rate=MUTATION_P)
            offspring.append(child)

        ranked = sorted(zip([compute_makespan(jobs, c) for c in offspring], offspring),
                        key=lambda x: x[0])
        half   = len(ranked) // 2
        for i, (_, c) in enumerate(ranked):
            if i < half:
                t0 = calibrate_sa_temperature(jobs, c)
                c, _ = simulated_annealing_improve(jobs, c, iters=sa_iters, t0=t0, tend=SA_TEND)
            new_pop.append(c)

        pop, fitness = new_pop, [compute_makespan(jobs, ind) for ind in new_pop]
        pop, fitness = deduplicate_population(pop, fitness, jobs)
        gen_best = min(fitness)
        if gen_best < best_val:
            best_val, best_ind = gen_best, deepcopy(pop[fitness.index(gen_best)])

        logger.info(f"    HGA | Gen {gen}/{generations} | Best: {best_val}")

    return pop, fitness, best_ind, best_val


def run_ea(jobs, args, seed=42):
    """Dispatch to chosen EA variant. Returns (pop, fitness, best_ind, best_val)."""
    if args.evo_alg == "GA":
        return _run_ea_warmstart_ga(
            jobs, pop_size=args.evo_pop, generations=args.evo_gens,
            seed=seed, early_stop=args.evo_early_stop,
        )
    elif args.evo_alg == "MA":
        return _run_ea_warmstart_ma(
            jobs, pop_size=args.evo_pop, generations=args.evo_gens,
            sa_iters=args.evo_sa_iters, seed=seed,
        )
    else:
        return _run_ea_warmstart_hga(
            jobs, pop_size=args.evo_pop, generations=args.evo_gens,
            sa_iters=args.evo_sa_iters, seed=seed,
        )

# ───────────────────────────────────────────────
# Demonstration collection
# ───────────────────────────────────────────────

def direct_schedule_rollout(env, perm, n_jobs, epsilon=0.05):
    remaining = list(perm)
    obs, done, pairs = env.reset(), False, []

    while not done:
        legal      = set(np.where(obs["action_mask"] == 1)[0])
        real_legal = [a for a in legal if a < n_jobs]

        if real_legal:
            if epsilon > 0.0 and np.random.random() < epsilon:
                action = int(np.random.choice(real_legal))
            else:
                def first_pos(job):
                    for i, j in enumerate(remaining):
                        if j == job:
                            return i
                    return float("inf")
                action = min(real_legal, key=first_pos)

            for i, j in enumerate(remaining):
                if j == action:
                    remaining.pop(i)
                    break
        else:
            action = n_jobs

        pairs.append((obs, action))
        obs, _, done, _ = env.step(action)

    return pairs


def collect_demos_for_instance(instance_path, n_episodes, epsilon, label=None):
    """
    Run EA on one instance (synthetic or real) and collect demo rollouts.
    Returns a flat list of (obs, action) pairs.
    """
    tag = label or os.path.basename(instance_path)
    logger.info(f"  ── {tag} ──")
    jobs = parse_taillard(instance_path)

    t0 = time.time()
    pop, fitness, best_ind, best_val = run_ea(jobs, args, seed=42)
    logger.info(f"    EA finished in {time.time()-t0:.1f}s | best makespan={best_val}")

    ranked    = sorted(zip(fitness, pop), key=lambda x: x[0])
    n_experts = max(1, args.evo_pop // 2)
    experts   = [(f, p) for f, p in ranked[:n_experts]]

    logger.info(f"    Experts: top {n_experts} | Replaying {n_episodes} episodes (ε={epsilon})")

    env   = make_wrapped_env(instance_path)
    demos = []
    for ep in range(n_episodes):
        _, perm = experts[ep % n_experts]
        demos.extend(direct_schedule_rollout(env, perm, NUM_JOBS, epsilon=epsilon))
    env.close()

    logger.info(f"    {tag}: {len(demos):,} transitions collected")
    return demos


def _verify_demonstration_consistency(demos, n_checks=500):
    if len(demos) < n_checks:
        return
    obs_action_map = {}
    for obs, action in demos[:n_checks]:
        obs_data    = obs["real_obs"] if isinstance(obs, dict) else obs
        fingerprint = tuple(np.round(obs_data.flatten(), 1))
        obs_action_map.setdefault(fingerprint, []).append(action)

    inconsistent, total_groups = 0, 0
    for _, actions in obs_action_map.items():
        if len(actions) > 1:
            total_groups += 1
            if len(set(actions)) > 1:
                inconsistent += 1

    if total_groups > 0:
        consistency = 100 * (1 - inconsistent / total_groups)
        logger.info(
            f"  Demo consistency: {consistency:.1f}% of repeated obs → same action "
            f"({total_groups} groups)"
        )
        if consistency < 80:
            logger.warning(f"  Low consistency ({consistency:.1f}%); BC may struggle.")
    else:
        logger.info("  Demo consistency: all observations unique (good)")

# ───────────────────────────────────────────────
# Behaviour Cloning
# ───────────────────────────────────────────────

def build_bc_model(obs_shape, n_actions, hidden=(512, 512, 256)):
    shape  = obs_shape if isinstance(obs_shape, tuple) else (obs_shape,)
    inp    = keras.Input(shape=shape, name="obs")
    x      = keras.layers.Flatten()(inp)
    for i, units in enumerate(hidden):
        x = keras.layers.Dense(units, activation="relu", name=f"fc_{i+1}")(x)
    logits = keras.layers.Dense(n_actions, name="fc_out")(x)
    return keras.Model(inputs=inp, outputs=logits)


def behaviour_cloning(demos, obs_shape, n_actions, hidden=(512, 512, 256)):
    """
    Train BC model. Validation split is episode-level (last 10% of episodes)
    to prevent near-duplicate observations leaking between train and val sets.
    """
    n_episodes = max(1, len(demos) // TOTAL_OPS)
    n_val_ep   = max(1, int(n_episodes * 0.1))
    split_idx  = (n_episodes - n_val_ep) * TOTAL_OPS

    train_demos, val_demos = demos[:split_idx], demos[split_idx:]

    logger.info(
        f"  Episode-level split: ~{n_episodes - n_val_ep} train / "
        f"~{n_val_ep} val episodes "
        f"({len(train_demos):,} / {len(val_demos):,} transitions)"
    )

    def to_arrays(d):
        obs = np.array(
            [x[0]["real_obs"] if isinstance(x[0], dict) else x[0] for x in d],
            dtype=np.float32,
        )
        act = np.array([x[1] for x in d], dtype=np.int64)
        return obs, act

    obs_train, act_train = to_arrays(train_demos)
    obs_val,   act_val   = to_arrays(val_demos)

    logger.info(f"  Obs shape: {obs_shape} | Actions: {n_actions}")
    logger.info(f"  BC network: {list(hidden)} → {n_actions}")

    unique, counts = np.unique(act_train, return_counts=True)
    top5 = sorted(zip(counts, unique), reverse=True)[:5]
    logger.info(
        "  Action distribution (top 5): "
        + ", ".join(f"job {j}: {c} ({100*c/len(act_train):.1f}%)" for c, j in top5)
    )
    if counts.max() / len(act_train) > 0.5:
        logger.warning(
            f"  Dominant action: {counts.max()/len(act_train)*100:.1f}% "
            f"— BC may predict it blindly"
        )

    model = build_bc_model(obs_shape, n_actions, hidden=hidden)
    model.summary(print_fn=logger.info)
    model.compile(
        optimizer=keras.optimizers.Adam(BC_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["acc"],
    )

    history = model.fit(
        obs_train, act_train,
        epochs=BC_EPOCHS,
        batch_size=BC_BATCH,
        validation_data=(obs_val, act_val),
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=5, min_lr=1e-5),
        ],
        verbose=1,
    )

    val_acc_key    = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
    final_val_acc  = history.history[val_acc_key][-1]
    epochs_trained = len(history.history["loss"])
    random_chance  = 1.0 / n_actions

    logger.info(
        f"  BC complete ({epochs_trained}/{BC_EPOCHS} epochs) | "
        f"val acc: {final_val_acc:.4f} | random baseline: {random_chance:.4f}"
    )
    if final_val_acc <= random_chance * 1.5:
        logger.error(f"  BC FAILED: val acc near random ({random_chance:.4f})")
    else:
        logger.info("  BC accuracy >> random baseline ✓")

    model.save_weights(BC_WEIGHTS_PATH, save_format="h5")
    logger.info(f"  BC weights saved → {BC_WEIGHTS_PATH}")
    return model

# ───────────────────────────────────────────────
# Evaluation helpers
# ───────────────────────────────────────────────

def bc_greedy_rollout(model, env):
    obs, done = env.reset(), False
    while not done:
        real_obs    = obs["real_obs"] if isinstance(obs, dict) else obs
        action_mask = obs.get("action_mask") if isinstance(obs, dict) else None
        logits      = model(np.array([real_obs], dtype=np.float32), training=False).numpy()[0]
        if action_mask is not None:
            logits[action_mask == 0] = -np.inf
        obs, _, done, _ = env.step(int(np.argmax(logits)))
    return getattr(env.unwrapped, "last_time_step", float("inf"))


def evaluate_instance(instance_path, bc_model, ea_baseline, n_bc_episodes):
    """
    Evaluate BC (and optionally a fresh EA) on one eval instance.
    Returns a result dict including per-episode BC makespans for statistical tests.
    """
    inst_key = os.path.basename(instance_path).replace(".txt", "")
    inst_bks = get_bks(instance_path)

    logger.info(f"  ── Evaluating: {inst_key}  (BKS={inst_bks}) ──")

    env          = make_wrapped_env(instance_path)
    bc_makespans = [bc_greedy_rollout(bc_model, env) for _ in range(n_bc_episodes)]
    env.close()

    bc_avg  = float(np.mean(bc_makespans))
    bc_best = float(np.min(bc_makespans))
    logger.info(
        f"    BC  | avg={bc_avg:.0f} (gap {100*(bc_avg-inst_bks)/inst_bks:+.2f}%) | "
        f"best={bc_best:.0f} (gap {100*(bc_best-inst_bks)/inst_bks:+.2f}%)"
    )

    ea_best = None
    if ea_baseline:
        jobs = parse_taillard(instance_path)
        t0   = time.time()
        _, _, _, ea_best = run_ea(jobs, args, seed=0)
        ea_best = float(ea_best)
        logger.info(
            f"    EA  | best={ea_best:.0f} (gap {100*(ea_best-inst_bks)/inst_bks:+.2f}%) "
            f"in {time.time()-t0:.1f}s"
        )

    return {
        "instance":    inst_key,
        "bks":         inst_bks,
        "bc_makespans": bc_makespans,
        "bc_avg":      bc_avg,
        "bc_best":     bc_best,
        "bc_gap_avg":  100 * (bc_avg  - inst_bks) / inst_bks,
        "bc_gap_best": 100 * (bc_best - inst_bks) / inst_bks,
        "ea_best":     ea_best,
        "ea_gap_best": 100 * (ea_best - inst_bks) / inst_bks if ea_best is not None else None,
    }


def print_and_save_summary(results, label):
    logger.info("")
    logger.info("─" * 70)
    logger.info(f"  {label} Summary")
    logger.info("─" * 70)

    has_ea = any(r["ea_best"] is not None for r in results)
    header = (
        f"  {'Instance':<12} {'BKS':>7}  "
        f"{'BC avg':>8} {'gap%':>7}  "
        f"{'BC best':>8} {'gap%':>7}"
        + (f"  {'EA best':>8} {'gap%':>7}" if has_ea else "")
    )
    logger.info(header)
    logger.info("  " + "─" * (len(header) - 2))

    for r in results:
        row = (
            f"  {r['instance']:<12} {r['bks']:>7.0f}  "
            f"{r['bc_avg']:>8.0f} {r['bc_gap_avg']:>+7.2f}%  "
            f"{r['bc_best']:>8.0f} {r['bc_gap_best']:>+7.2f}%"
        )
        if has_ea and r["ea_best"] is not None:
            row += f"  {r['ea_best']:>8.0f} {r['ea_gap_best']:>+7.2f}%"
        logger.info(row)

    bc_gaps = [r["bc_gap_avg"] for r in results]
    logger.info("  " + "─" * (len(header) - 2))
    logger.info(f"  {'MEAN':<12} {'':>7}  {'':>8} {np.mean(bc_gaps):>+7.2f}%")
    if has_ea:
        ea_gaps = [r["ea_gap_best"] for r in results if r["ea_best"] is not None]
        logger.info(
            f"  {'MEAN (EA)':<12} {'':>7}  {'':>8} {'':>7}  "
            f"{'':>8} {'':>7}  {'':>8} {np.mean(ea_gaps):>+7.2f}%"
        )
    logger.info("─" * 70)

    slug     = label.lower().replace(" ", "_")
    csv_path = os.path.join(CHECKPOINT_DIR, f"bc_eval_{slug}.csv")
    fieldnames = ["instance", "bks", "bc_avg", "bc_gap_avg",
                  "bc_best", "bc_gap_best", "ea_best", "ea_gap_best"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {k: (f"{v:.4f}" if isinstance(v, float) else
                     (v if v is not None else ""))
                 for k, v in r.items() if k in fieldnames}
            )
    logger.info(f"  CSV saved → {csv_path}")

# ───────────────────────────────────────────────
# Statistical tests
# ───────────────────────────────────────────────

def _interpret(p, alpha, a_mean, b_mean, a_label, b_label):
    if p < alpha:
        direction = "BETTER" if a_mean < b_mean else "WORSE"
        return (
            f"SIGNIFICANT (p={p:.4f} < α={alpha}) — "
            f"{a_label} is {direction} than {b_label} "
            f"(mean gap: {a_mean:+.2f}% vs {b_mean:+.2f}%)"
        )
    return (
        f"NOT significant (p={p:.4f} ≥ α={alpha}) — "
        f"no reliable difference between {a_label} and {b_label} "
        f"(mean gap: {a_mean:+.2f}% vs {b_mean:+.2f}%)"
    )


def _choose_test(x, y=None):
    """
    Select and run the appropriate test based on sample size.

    n ≥ 20  → parametric  (paired/one-sample t-test)
    5–19    → non-parametric (Wilcoxon signed-rank)
    < 5     → sign test  + low-power warning

    Returns (stat, p, test_name).
    """
    n = len(x)
    if n < 5:
        diffs = np.array(x) if y is None else np.array(x) - np.array(y)
        n_pos  = int((diffs > 0).sum())
        n_ties = int((diffs == 0).sum())
        n_eff  = n - n_ties
        p      = float(2 * stats.binom.cdf(
                    min(n_pos - n_ties, n_eff - (n_pos - n_ties)), n_eff, 0.5
                 )) if n_eff > 0 else 1.0
        return float(n_pos), p, "sign test"
    elif n < 20:
        if y is None:
            stat, p = stats.wilcoxon(x, alternative="two-sided")
        else:
            stat, p = stats.wilcoxon(x, y, alternative="two-sided")
        return float(stat), float(p), "Wilcoxon signed-rank"
    else:
        if y is None:
            stat, p = stats.ttest_1samp(x, popmean=0)
        else:
            stat, p = stats.ttest_rel(x, y)
        return float(stat), float(p), "paired t-test"


def _choose_independent_test(x, y):
    """Independent (unpaired) test for comparing two separate groups."""
    n = min(len(x), len(y))
    if n < 5:
        return _choose_test(x[:n], y[:n])   # fall back to sign test on matched pairs
    elif n < 20:
        stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        return float(stat), float(p), "Mann-Whitney U"
    else:
        stat, p = stats.ttest_ind(x, y)
        return float(stat), float(p), "independent t-test"


def _log_test(title, stat, p, test_name, interpretation, n):
    logger.info("")
    logger.info(f"  ── {title} ──")
    logger.info(f"     Test:           {test_name}  (n={n})")
    logger.info(f"     Statistic:      {stat:.4f}")
    logger.info(f"     p-value:        {p:.4f}")
    logger.info(f"     Interpretation: {interpretation}")
    if n < 5:
        logger.warning(
            f"     ⚠  Only {n} instances — statistical power is very low. "
            f"Results are indicative only. Use more eval instances for reliable inference."
        )


def run_statistical_tests(eval_results, alpha):
    """
    Run three statistical tests on the out-of-sample evaluation results.

    T1  BC gap% vs 0 (one-sample)
        H₀: BC matches BKS (mean gap == 0)
        → Are BC makespans significantly above BKS?

    T2  BC gap% vs EA gap% (paired)
        H₀: BC and EA achieve the same gap
        → Is BC significantly better or worse than a fresh EA run?

    T3  BC individual episode makespans pooled across instances vs EA best (paired per instance)
        Only runs when bc_eval_episodes > 1.
        H₀: per-episode BC makespan == EA best on the same instance
        → Checks whether the gap in T2 holds at the episode level too.

    Test selection by n (number of eval instances):
        n ≥ 20  →  parametric  (t-test)
        5–19    →  non-parametric (Wilcoxon)
        < 5     →  sign test + ⚠ low-power warning
    """
    logger.info("")
    logger.info("═" * 70)
    logger.info("  Stage 4: Statistical Tests  (out-of-sample eval instances)")
    logger.info(f"  α = {alpha}  |  two-sided tests throughout")
    logger.info("═" * 70)

    bc_gaps = [r["bc_gap_avg"]  for r in eval_results]
    ea_gaps = [r["ea_gap_best"] for r in eval_results if r["ea_best"] is not None]
    n_oos   = len(bc_gaps)
    n_ea    = len(ea_gaps)

    # ── T1: BC gap vs BKS ────────────────────────────────────────────────────
    logger.info("")
    logger.info("  T1  BC (out-of-sample) gap% vs BKS")
    logger.info(f"      H₀: mean gap == 0  (BC matches BKS)")
    logger.info(f"      H₁: mean gap ≠ 0")
    logger.info(f"      Values: {[f'{g:.2f}%' for g in bc_gaps]}")

    stat, p, test_name = _choose_test(bc_gaps, y=None)
    mean_gap = float(np.mean(bc_gaps))
    if p < alpha:
        direction = "above BKS (worse)" if mean_gap > 0 else "below BKS (better — unexpected)"
        interp = (
            f"SIGNIFICANT (p={p:.4f} < α={alpha}) — "
            f"BC gap is reliably {direction}, mean={mean_gap:+.2f}%"
        )
    else:
        interp = (
            f"NOT significant (p={p:.4f} ≥ α={alpha}) — "
            f"cannot rule out BC matching BKS, mean={mean_gap:+.2f}%"
        )
    _log_test("T1: BC out-of-sample vs BKS", stat, p, test_name, interp, n_oos)

    # ── T2: BC vs EA baseline (per instance, paired) ─────────────────────────
    if n_ea > 0:
        bc_paired = bc_gaps[:n_ea]
        logger.info("")
        logger.info("  T2  BC gap% vs EA gap% (paired per instance)")
        logger.info(f"      H₀: BC gap == EA gap  (BC and EA perform equally)")
        logger.info(f"      H₁: BC gap ≠ EA gap")
        logger.info(f"      BC  gaps: {[f'{g:.2f}%' for g in bc_paired]}")
        logger.info(f"      EA  gaps: {[f'{g:.2f}%' for g in ea_gaps]}")

        stat, p, test_name = _choose_test(bc_paired, y=ea_gaps)
        interp = _interpret(p, alpha, float(np.mean(bc_paired)),
                            float(np.mean(ea_gaps)), "BC", "EA")
        _log_test("T2: BC vs EA (paired, out-of-sample)", stat, p, test_name, interp, n_ea)
    else:
        logger.info("")
        logger.info("  T2  Skipped — no EA baseline results available.")

    # ── T3: per-episode BC makespan vs EA best (pooled across instances) ──────
    if args.bc_eval_episodes > 1 and n_ea > 0:
        logger.info("")
        logger.info("  T3  Per-episode BC makespan vs EA best (pooled, paired by instance)")
        logger.info(f"      H₀: per-episode BC makespan == EA best on same instance")
        logger.info(f"      H₁: per-episode BC makespan ≠ EA best")

        # Pair each BC episode against the EA best for the same instance
        ep_bc, ep_ea = [], []
        for r in eval_results:
            if r["ea_best"] is not None:
                for ms in r["bc_makespans"]:
                    ep_bc.append(100 * (ms - r["bks"]) / r["bks"])
                    ep_ea.append(r["ea_gap_best"])

        logger.info(f"      Pooled pairs: {len(ep_bc)}")
        stat, p, test_name = _choose_test(ep_bc, y=ep_ea)
        interp = _interpret(p, alpha, float(np.mean(ep_bc)),
                            float(np.mean(ep_ea)), "BC episodes", "EA best")
        _log_test("T3: BC episodes vs EA best (pooled)", stat, p, test_name,
                  interp, len(ep_bc))
    else:
        logger.info("")
        logger.info(
            "  T3  Skipped — requires --bc-eval-episodes > 1 and EA baseline."
        )

    # ── Descriptive statistics ───────────────────────────────────────────────
    logger.info("")
    logger.info("  ── Descriptive Statistics (out-of-sample) ──")
    logger.info(f"  {'Metric':<40} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    logger.info("  " + "─" * 76)

    def _desc(label, vals):
        a = np.array(vals, dtype=float)
        logger.info(
            f"  {label:<40} {a.mean():>+8.2f} {a.std():>8.2f} "
            f"{a.min():>+8.2f} {a.max():>+8.2f}"
        )

    _desc("BC gap% (out-of-sample, avg per instance)", bc_gaps)
    if ea_gaps:
        _desc("EA gap% (out-of-sample, best per instance)", ea_gaps)
        diffs = np.array(bc_gaps[:n_ea]) - np.array(ea_gaps)
        _desc("BC − EA gap% (positive = BC worse)",       diffs.tolist())

    if args.bc_eval_episodes > 1:
        all_ep_gaps = [
            100 * (ms - r["bks"]) / r["bks"]
            for r in eval_results
            for ms in r["bc_makespans"]
        ]
        _desc("BC gap% (all individual episodes)",        all_ep_gaps)

    logger.info("═" * 70)

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info("EA Warm-Start (Synthetic Instances) → Behaviour Cloning Pipeline")
    logger.info("=" * 70)
    logger.info(f"Problem dimensionality:  {NUM_JOBS} jobs × {NUM_MACHINES} machines")
    logger.info(f"Synthetic train count:   {args.num_train_instances}  "
                f"(seed range: {args.train_instance_seed}–"
                f"{args.train_instance_seed + args.num_train_instances - 1})")
    logger.info(f"Eval instances (real):   {[os.path.basename(p) for p in EVAL_INSTANCE_PATHS]}")
    logger.info(f"EA algorithm:            {args.evo_alg}")
    logger.info(f"EA pop / gens:           {args.evo_pop} / {args.evo_gens}")
    logger.info(f"EA SA iters:             {args.evo_sa_iters}")
    logger.info(f"BC epochs:               {BC_EPOCHS}")
    logger.info(f"BC network:              {args.bc_hidden}")
    logger.info(f"BC demo eps/instance:    {args.bc_demo_episodes}  "
                f"(~{args.bc_demo_episodes * TOTAL_OPS:,} transitions each)")
    logger.info(f"BC epsilon:              {args.bc_epsilon}  "
                f"({'pure expert' if args.bc_epsilon == 0 else f'{args.bc_epsilon*100:.0f}% noise'})")
    logger.info(f"BC eval episodes:        {args.bc_eval_episodes}")
    logger.info(f"EA baseline on eval:     {args.eval_ea_baseline}")
    logger.info(f"Significance level:      α={ALPHA}")
    logger.info(f"Output dir:              {CHECKPOINT_DIR}")
    logger.info(f"Log / results file:      {RESULTS_TXT}")
    logger.info("=" * 70)

    # ── Stage 1: Generate synthetic training instances ───────────────────────
    logger.info("")
    logger.info("═══ Stage 1: Generating Synthetic Training Instances ═══")
    train_instance_paths = create_synthetic_training_instances(
        n_instances  = args.num_train_instances,
        n_jobs       = NUM_JOBS,
        n_machines   = NUM_MACHINES,
        master_seed  = args.train_instance_seed,
        out_dir      = CHECKPOINT_DIR,
    )

    # ── Stage 2: Collect EA demonstrations on synthetic instances ────────────
    logger.info("")
    logger.info("═══ Stage 2: Collecting EA Demonstrations (synthetic instances) ═══")
    all_demos = []
    for i, inst_path in enumerate(train_instance_paths):
        inst_demos = collect_demos_for_instance(
            inst_path,
            n_episodes = args.bc_demo_episodes,
            epsilon    = args.bc_epsilon,
            label      = f"synth_{i+1}/{args.num_train_instances}  ({os.path.basename(inst_path)})",
        )
        all_demos.extend(inst_demos)

    logger.info(
        f"  Pooled {len(all_demos):,} transitions from "
        f"{args.num_train_instances} synthetic instance(s)"
    )
    _verify_demonstration_consistency(all_demos)

    # ── Stage 3: Behaviour Cloning ───────────────────────────────────────────
    logger.info("")
    logger.info("═══ Stage 3: Behaviour Cloning ═══")

    env_tmp   = make_wrapped_env(train_instance_paths[0])
    obs_tmp   = env_tmp.reset()
    obs_shape = obs_tmp["real_obs"].shape if isinstance(obs_tmp, dict) else obs_tmp.shape
    n_actions = env_tmp.action_space.n
    env_tmp.close()

    bc_model = behaviour_cloning(all_demos, obs_shape, n_actions, hidden=tuple(args.bc_hidden))

    # ── Stage 4: Out-of-sample evaluation on real Taillard instances ─────────
    logger.info("")
    logger.info("═══ Stage 4: Out-of-Sample Evaluation (real Taillard instances) ═══")
    logger.info(
        "  NOTE: none of these instances were seen during training — "
        "all training used synthetically generated data."
    )
    eval_results = [
        evaluate_instance(
            p, bc_model,
            ea_baseline    = args.eval_ea_baseline,
            n_bc_episodes  = args.bc_eval_episodes,
        )
        for p in EVAL_INSTANCE_PATHS
    ]
    print_and_save_summary(eval_results, label="out_of_sample")

    # ── Stage 5: Statistical tests ───────────────────────────────────────────
    run_statistical_tests(eval_results, alpha=ALPHA)

    # ── Final summary ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("Pipeline complete.")
    logger.info(f"  BC weights:       {BC_WEIGHTS_PATH}")
    logger.info(f"  Eval CSV:         {os.path.join(CHECKPOINT_DIR, 'bc_eval_out_of_sample.csv')}")
    logger.info(f"  Full log:         {RESULTS_TXT}")
    logger.info("=" * 70)
