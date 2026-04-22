#!/usr/bin/env python3
"""
warm_start_GA.py

Three-stage warm-starting pipeline for PPO on Taillard JSSP instances:
  1. Evolutionary Algorithm (GA/MA/HGA) runs for a small number of generations
     to produce a population of diverse, high-quality schedules.
  2. Behaviour Cloning (BC) trains a neural network to imitate the EA's
     dispatching decisions via supervised learning.
  3. PPO fine-tunes the BC-initialised policy via reinforcement learning,
     correcting for distributional shift and optimising beyond EA quality.

This approach is grounded in the BC→RL paradigm (Silver et al., 2016;
Rengarajan et al., 2022) and addresses the cold-start problem in RL for
combinatorial optimisation.

Usage:
    python3 warm_start_GA.py --bks bks.json --instances instances/ta52
    python3 warm_start_GA.py --bks bks.json --instances instances/ta52 --evo-alg MA
    python3 warm_start_GA.py --bks bks.json --instances instances/ta72 --evo-gens 10
"""

import csv
import argparse
import os
import sys
import time
import json
import logging
from collections import deque
from copy import deepcopy

import numpy as np
import gym
from gym import spaces
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import LoggerCallback
from ray.tune import Stopper
from dynamic_jss_wrapper import DynamicJSSWrapper

import tensorflow as tf
from tensorflow import keras

from models import FCMaskedActionsModelTF

# ── Import EA primitives from the sibling module ──────────────────────────────
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
# Logging
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

# ───────────────────────────────────────────────
# CLI arguments
# ───────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="EA warm-start → BC → PPO training pipeline for Taillard JSSP instances.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python3 warm_start_GA.py --bks bks.json --instances instances/ta52
  python3 warm_start_GA.py --bks bks.json --instances instances/ta72 --evo-alg MA --evo-gens 10
    """,
)
parser.add_argument("--instances", type=str, default="instances/ta52",
                    help="Path to Taillard instance file (default: instances/ta52)")
parser.add_argument("--iters", type=int, default=300,
                    help="Max PPO training iterations (default: 300)")
parser.add_argument("--out", type=str, default="checkpoint_results",
                    help="Output / checkpoint directory (default: checkpoint_results)")
parser.add_argument("--bks", type=str, required=True,
                    help="Path to bks.json")
parser.add_argument("--evo-alg", type=str, default="HGA", choices=["GA", "MA", "HGA"],
                    help="EA for warm-start demonstrations (default: HGA)")
parser.add_argument("--evo-pop", type=int, default=50,
                    help="EA population size (default: 50)")
parser.add_argument("--evo-gens", type=int, default=5,
                    help="EA generations — kept small for warm-start only (default: 5)")
parser.add_argument("--evo-sa-iters", type=int, default=500,
                    help="SA iterations per offspring/elite in MA/HGA (default: 500)")
parser.add_argument("--bc-epochs", type=int, default=50,
                    help="Behaviour cloning training epochs (default: 50)")
parser.add_argument("--target-gap", type=float, default=20.0,
                    help="PPO early-stop gap target %% (default: 20.0)")
parser.add_argument("--no-warmstart", action="store_true", default=False,
                    help="Skip EA and BC stages, train PPO from scratch")
parser.add_argument("--evo-early-stop", action="store_true", default=False,
                    help="Stop EA if best makespan doesn't improve for 3 generations")
args = parser.parse_args()

ENV_ID = "JSSEnv:jss-v1"
BC_EPOCHS = args.bc_epochs
BC_LR = 1e-3
BC_BATCH = 256
CHECKPOINT_DIR = os.path.abspath(args.out)
INSTANCE_PATH = os.path.abspath(args.instances)
BC_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "bc_weights")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load BKS
with open(args.bks, "r") as f:
    bks_map = json.load(f)
instance_key = os.path.basename(INSTANCE_PATH).replace(".txt", "")
current_bks = float(bks_map.get(instance_key, 1.0))

# Instance metadata
with open(INSTANCE_PATH, "r") as f:
    tokens = f.read().split()
NUM_JOBS = int(tokens[0])
NUM_MACHINES = int(tokens[1])
TOTAL_OPS = NUM_JOBS * NUM_MACHINES

# ───────────────────────────────────────────────
# Logging / Stopping helpers (unchanged)
# ───────────────────────────────────────────────

class CustomCSVLoggerCallback(LoggerCallback):
    """Logs per-iteration metrics to both CSV and console, matching
    the format from ppo_trainer_config.py."""

    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.setup_done = False
        self.gap_window = deque(maxlen=5)

    def log_trial_result(self, iteration, trial, result):
        metrics = result.get("custom_metrics", {})
        makespan = metrics.get("make_span_mean", 0.0)
        iter_gap = metrics.get("optimality_gap_mean", None)
        ep_reward = result.get("episode_reward_mean", "N/A")
        training_iter = result.get("training_iteration", iteration)

        # ── Console logging (mirrors ppo_trainer_config.py) ──
        if iter_gap is not None:
            self.gap_window.append(iter_gap)
            rolling_avg = np.mean(self.gap_window)
            logger.info(
                f"  PPO Iter {training_iter:>4}: "
                f"reward_mean={ep_reward:>10} | "
                f"makespan={makespan:>8.2f} | "
                f"gap={iter_gap:>6.2f}% | "
                f"rolling_avg={rolling_avg:>6.2f}% "
                f"(window={len(self.gap_window)})"
            )
        else:
            logger.info(
                f"  PPO Iter {training_iter:>4}: "
                f"reward_mean={ep_reward:>10} | "
                f"makespan={makespan:>8.2f} | "
                f"gap=N/A"
            )

        # ── CSV logging ──
        mode = "a" if self.setup_done else "w"
        with open(self.csv_file_path, mode=mode, newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iteration", "makespan", "bks",
                    "optimality gap", "rolling_avg", "reward_mean",
                ],
            )
            if not self.setup_done:
                writer.writeheader()
                self.setup_done = True
            writer.writerow({
                "iteration": training_iter,
                "makespan": f"{makespan:.2f}",
                "bks": current_bks,
                "optimality gap": f"{iter_gap:.2f}" if iter_gap is not None else "N/A",
                "rolling_avg": (f"{np.mean(self.gap_window):.2f}"
                                if self.gap_window else "N/A"),
                "reward_mean": (f"{ep_reward:.4f}"
                                if isinstance(ep_reward, (int, float)) else ep_reward),
            })


class OptimalityGapStopper(Stopper):
    """
    Early stopping based on rolling-average optimality gap.

    Mirrors the logic from ppo_trainer_config.py:
      - Maintains a rolling window of gap values
      - Stops when the rolling average falls below target_gap
      - Also stops at max_iters as a safety bound
    """

    def __init__(self, window_size=5, target_gap=20.0, max_iters=200):
        self.window_size = window_size
        self.target_gap = target_gap
        self.max_iters = max_iters
        self.trial_gaps = {}

    def __call__(self, trial_id, result):
        if trial_id not in self.trial_gaps:
            self.trial_gaps[trial_id] = deque(maxlen=self.window_size)

        iteration = result.get("training_iteration", 0)
        if iteration >= self.max_iters:
            logger.info(
                f"  PPO | Reached maximum of {self.max_iters} iterations. "
                f"Stopping."
            )
            return True

        # Extract gap from custom_metrics
        iter_gap = result.get("custom_metrics", {}).get("optimality_gap_mean")

        if iter_gap is not None:
            self.trial_gaps[trial_id].append(iter_gap)
            rolling_avg = np.mean(self.trial_gaps[trial_id])

            if (len(self.trial_gaps[trial_id]) >= self.window_size
                    and rolling_avg <= self.target_gap):
                logger.info(
                    f"  PPO | EARLY STOP: Rolling avg gap {rolling_avg:.2f}% "
                    f"<= target {self.target_gap}% "
                    f"(window={self.window_size}). Saving and exiting."
                )
                return True
        else:
            # Gap not available — this happens if no episodes completed
            # in this iteration. Log but don't stop.
            logger.warning(
                f"  PPO | Iter {iteration}: optimality_gap_mean not available "
                f"in custom_metrics. No episodes completed this iteration?"
            )

        return False

    def stop_all(self):
        return False

# ───────────────────────────────────────────────
# 1. EA Warm-Start — Lightweight EA runners
#    (5 generations, no BKS gap tracking needed)
# ───────────────────────────────────────────────

CROSSOVER_P = 0.9
MUTATION_P = 0.1

def make_wrapped_env(instance_path):
    """Create a JSSEnv wrapped with dynamic progress tracking.
    Used everywhere an environment is needed to ensure BC and PPO
    see identical observation spaces."""
    base_env = gym.make(ENV_ID, env_config={"instance_path": instance_path})
    return DynamicJSSWrapper(base_env)

def _run_ea_warmstart_ga(jobs, pop_size, generations, seed=42):
    """Lightweight GA for demonstration collection. No gap tracking."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    pop = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = [compute_makespan(jobs, ind) for ind in pop]
    best_val = min(fitness)
    best_ind = deepcopy(pop[fitness.index(best_val)])

    for gen in range(1, generations + 1):
        new_pop = [deepcopy(best_ind)]
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = (order_crossover(p1, p2)
                     if random.random() < CROSSOVER_P else deepcopy(p1))
            child = mutate(child, mutation_rate=MUTATION_P)
            new_pop.append(child)

        pop = new_pop
        fitness = [compute_makespan(jobs, ind) for ind in pop]
        pop, fitness = deduplicate_population(pop, fitness, jobs)

        gen_best = min(fitness)
        if gen_best < best_val:
            best_val = gen_best
            best_ind = deepcopy(pop[fitness.index(gen_best)])

        logger.info(f"  GA warmstart | Gen {gen}/{generations} | Best: {best_val}")

    return pop, fitness, best_ind, best_val


def _run_ea_warmstart_ma(jobs, pop_size, generations, sa_iters, seed=42):
    """Lightweight MA for demonstration collection."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    pop = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = [compute_makespan(jobs, ind) for ind in pop]
    best_val = min(fitness)
    best_ind = deepcopy(pop[fitness.index(best_val)])

    for gen in range(1, generations + 1):
        new_pop = [deepcopy(best_ind)]
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = (order_crossover(p1, p2)
                     if random.random() < CROSSOVER_P else deepcopy(p1))
            child = mutate(child, mutation_rate=MUTATION_P)

            # Lamarckian local search on every offspring
            t0 = calibrate_sa_temperature(jobs, child)
            child, _ = simulated_annealing_improve(
                jobs, child, iters=sa_iters, t0=t0, tend=SA_TEND
            )
            new_pop.append(child)

        pop = new_pop
        fitness = [compute_makespan(jobs, ind) for ind in pop]
        pop, fitness = deduplicate_population(pop, fitness, jobs)

        gen_best = min(fitness)
        if gen_best < best_val:
            best_val = gen_best
            best_ind = deepcopy(pop[fitness.index(gen_best)])

        logger.info(f"  MA warmstart | Gen {gen}/{generations} | Best: {best_val}")

    return pop, fitness, best_ind, best_val


def _run_ea_warmstart_ga(jobs, pop_size, generations, seed=42, early_stop=False):
    """Lightweight GA for demonstration collection. No gap tracking."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    pop = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = [compute_makespan(jobs, ind) for ind in pop]
    best_val = min(fitness)
    best_ind = deepcopy(pop[fitness.index(best_val)])
    
    patience_counter = 0  # Track generations without improvement

    for gen in range(1, generations + 1):
        new_pop = [deepcopy(best_ind)]
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = (order_crossover(p1, p2)
                     if random.random() < CROSSOVER_P else deepcopy(p1))
            child = mutate(child, mutation_rate=MUTATION_P)
            new_pop.append(child)

        pop = new_pop
        fitness = [compute_makespan(jobs, ind) for ind in pop]
        pop, fitness = deduplicate_population(pop, fitness, jobs)

        gen_best = min(fitness)
        if gen_best < best_val:
            best_val = gen_best
            best_ind = deepcopy(pop[fitness.index(gen_best)])
            patience_counter = 0  # Reset counter if we improve
        else:
            patience_counter += 1  # Increment if no improvement

        # Added patience tracker to the standard log
        logger.info(f"  GA warmstart | Gen {gen}/{generations} | Best: {best_val} | Patience: {patience_counter}/3")

        # Break the loop if the early stop flag is active and we hit 3 iterations without improvement
        if early_stop and patience_counter >= 3:
            logger.info(f"  GA warmstart | Early stopping triggered! Ran for {gen} iterations.")
            break

    return pop, fitness, best_ind, best_val

# ───────────────────────────────────────────────
# 2. Demonstration Collection — Fixed
# ───────────────────────────────────────────────

def extract_dispatching_priorities(jobs, perm):
    """
    From a decoded EA permutation, extract a deterministic dispatching
    rule: for each job, compute a priority score based on its position
    in the permutation and its remaining processing time.

    Returns a function that, given a JSSEnv observation and legal actions,
    deterministically selects the highest-priority legal job.
    """
    n_jobs = len(jobs)
    n_machines = len(jobs[0])

    # Compute the average position of each job in the permutation.
    # Lower average position = higher priority (scheduled earlier).
    job_positions = [[] for _ in range(n_jobs)]
    for pos, j in enumerate(perm):
        job_positions[j].append(pos)
    job_priority = [np.mean(positions) for positions in job_positions]

    return job_priority


def priority_policy_rollout(env, job_priority):
    """
    Roll out a complete episode using a deterministic priority-based
    dispatching rule derived from the EA solution.

    At each step, the legal job with the lowest priority score
    (= highest priority = earliest in the EA's permutation) is chosen.

    JSSEnv uses action index n_jobs as a special no-op action.
    If the no-op is the only legal action, we select it.
    Otherwise, we filter it out and pick among real job actions.
    """
    n_jobs = len(job_priority)
    obs = env.reset()
    done = False
    pairs = []

    while not done:
        legal = np.where(obs["action_mask"] == 1)[0]

        # Filter to only real job actions (exclude no-op at index n_jobs)
        real_actions = [a for a in legal if a < n_jobs]

        if real_actions:
            best_action = min(real_actions, key=lambda a: job_priority[a])
        else:
            # Only the no-op action is legal — select it
            best_action = int(legal[0])

        pairs.append((obs, best_action))
        obs, _, done, _ = env.step(best_action)

    return pairs

def collect_evolutionary_demonstrations(n_episodes, alg, pop_size, generations, sa_iters, early_stop=False):
    """
    Run the chosen EA, extract priority-based dispatching rules from the
    best individuals, then roll out consistent demonstrations through
    the WRAPPED JSSEnv to produce enriched (obs, action) pairs.
    """
    logger.info(f"  Parsing instance: {INSTANCE_PATH}")
    jobs = parse_taillard(INSTANCE_PATH)
    logger.info(
        f"  Running {alg} (pop={pop_size}, gens={generations}, "
        f"sa_iters={sa_iters}) for warm-start demonstrations..."
    )

    t0 = time.time()

    if alg == "GA":
        pop, fitness, best_ind, best_val = _run_ea_warmstart_ga(
            jobs, pop_size=pop_size, generations=generations, seed=42, early_stop=early_stop
        )
    elif alg == "MA":
        pop, fitness, best_ind, best_val = _run_ea_warmstart_ma(
            jobs, pop_size=pop_size, generations=generations,
            sa_iters=sa_iters, seed=42
        )
    else:
        pop, fitness, best_ind, best_val = _run_ea_warmstart_hga(
            jobs, pop_size=pop_size, generations=generations,
            sa_iters=sa_iters, seed=42
        )

    elapsed = time.time() - t0
    gap = 100.0 * (best_val - current_bks) / current_bks
    logger.info(
        f"  EA finished in {elapsed:.1f}s | Best makespan: {best_val} | "
        f"Gap to BKS: {gap:.2f}%"
    )

    ranked_pairs = sorted(zip(fitness, pop), key=lambda x: x[0])
    sorted_fitness = [f for f, _ in ranked_pairs]
    ranked_pop = [p for _, p in ranked_pairs]

    logger.info(
        f"  Demo population quality — "
        f"Best: {sorted_fitness[0]}, "
        f"Median: {sorted_fitness[len(sorted_fitness)//2]}, "
        f"Worst: {sorted_fitness[-1]}"
    )

    n_experts = max(1, pop_size // 2)
    expert_priorities = []
    for i in range(n_experts):
        prio = extract_dispatching_priorities(jobs, ranked_pop[i])
        expert_priorities.append((sorted_fitness[i], prio))

    logger.info(f"  Extracted {n_experts} expert priority rules")

    # ── Use wrapped environment for demo collection ──
    env = make_wrapped_env(INSTANCE_PATH)
    demos = []
    makespans_achieved = []

    for ep in range(n_episodes):
        expert_idx = ep % n_experts
        _, priority = expert_priorities[expert_idx]

        pairs = priority_policy_rollout(env, priority)
        demos.extend(pairs)

        base_env = env.unwrapped
        makespan = base_env.last_time_step if hasattr(base_env, 'last_time_step') else 0
        makespans_achieved.append(makespan)

        if (ep + 1) % 50 == 0:
            logger.info(
                f"  Replayed {ep + 1}/{n_episodes} episodes "
                f"({len(demos)} transitions, "
                f"avg makespan: {np.mean(makespans_achieved):.0f})"
            )

    env.close()
    _verify_demonstration_consistency(demos)
    logger.info(f"  Total demonstration transitions: {len(demos)}")
    return demos


def _verify_demonstration_consistency(demos, n_checks=500):
    """
    Sample observation-action pairs and check how often the same
    (or very similar) observation maps to the same action.
    Logs a warning if consistency is low.
    """
    if len(demos) < n_checks:
        return

    # Build a rough fingerprint for each observation
    obs_action_map = {}
    for obs, action in demos[:n_checks]:
        obs_data = obs["real_obs"] if isinstance(obs, dict) else obs
        # Use a coarse fingerprint: round to 1 decimal place
        fingerprint = tuple(np.round(obs_data.flatten(), 1))

        if fingerprint not in obs_action_map:
            obs_action_map[fingerprint] = []
        obs_action_map[fingerprint].append(action)

    # Check how many fingerprints have multiple different actions
    inconsistent = 0
    total_groups = 0
    for fp, actions in obs_action_map.items():
        if len(actions) > 1:
            total_groups += 1
            if len(set(actions)) > 1:
                inconsistent += 1

    if total_groups > 0:
        consistency = 100 * (1 - inconsistent / total_groups)
        logger.info(
            f"  Demo consistency check: {consistency:.1f}% of repeated "
            f"observations map to the same action "
            f"({total_groups} groups checked)"
        )
        if consistency < 80:
            logger.warning(
                f"  Low demonstration consistency ({consistency:.1f}%). "
                f"BC may struggle to learn."
            )
    else:
        logger.info("  Demo consistency check: all observations unique (good)")

# ───────────────────────────────────────────────
# 3. Behaviour Cloning
# ───────────────────────────────────────────────

def build_bc_model(obs_shape, n_actions, hidden=(256, 256)):
    """
    Build a BC model whose architecture matches RLlib's FCMaskedActionsModelTF.
    """
    shape = obs_shape if isinstance(obs_shape, tuple) else (obs_shape,)
    inp = keras.Input(shape=shape, name="obs")
    x = keras.layers.Flatten()(inp) if len(shape) > 1 else inp
    x = keras.layers.Dense(hidden[0], activation="relu", name="fc_1")(x)
    x = keras.layers.Dense(hidden[1], activation="relu", name="fc_2")(x)
    logits = keras.layers.Dense(n_actions, name="fc_out")(x)
    return keras.Model(inputs=inp, outputs=logits)


def behaviour_cloning(demos, obs_shape, n_actions):
    """Train BC model on EA demonstrations."""
    obs_arr = np.array(
        [d[0]["real_obs"] if isinstance(d[0], dict) else d[0] for d in demos],
        dtype=np.float32,
    )
    act_arr = np.array([d[1] for d in demos], dtype=np.int64)

    logger.info(
        f"  BC dataset: {len(obs_arr)} samples | "
        f"Obs shape: {obs_shape} | Actions: {n_actions}"
    )

    unique, counts = np.unique(act_arr, return_counts=True)
    top5 = sorted(zip(counts, unique), reverse=True)[:5]
    logger.info(
        f"  BC action distribution (top 5): "
        + ", ".join(f"job {j}: {c} ({100*c/len(act_arr):.1f}%)" for c, j in top5)
    )

    max_frac = counts.max() / len(act_arr)
    if max_frac > 0.5:
        logger.warning(f"  Dominant action: {max_frac*100:.1f}%")

    obs_std = obs_arr.std(axis=0)
    n_zero_std = (obs_std < 1e-6).sum()
    logger.info(
        f"  Observation feature stats: "
        f"{obs_arr.shape[1]} features, {n_zero_std} constant, "
        f"mean std: {obs_std.mean():.4f}"
    )

    model = build_bc_model(obs_shape, n_actions)
    model.compile(
        optimizer=keras.optimizers.Adam(3e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["acc"],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    lr_reduce = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
    )

    history = model.fit(
        obs_arr, act_arr,
        epochs=BC_EPOCHS,
        batch_size=BC_BATCH,
        validation_split=0.1,
        callbacks=[early_stop, lr_reduce],
        verbose=1,
    )

    acc_key = "accuracy" if "accuracy" in history.history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
    final_val_acc = history.history[val_acc_key][-1]
    epochs_trained = len(history.history["loss"])

    logger.info(
        f"  BC complete ({epochs_trained}/{BC_EPOCHS} epochs) | "
        f"Val acc: {final_val_acc:.4f}"
    )

    random_chance = 1.0 / n_actions
    if final_val_acc <= random_chance * 1.5:
        logger.error(f"  BC FAILED: Val acc {final_val_acc:.4f} ~ random {random_chance:.4f}")
    elif final_val_acc >= 0.15:
        logger.info(f"  BC accuracy {final_val_acc:.4f} >> random {random_chance:.4f}")

    # Save ONLY the model weights, not optimizer state.
    # This prevents "Unresolved object in checkpoint" warnings
    # when the BC checkpoint is loaded into a different model.
    model.save_weights(BC_WEIGHTS_PATH, save_format='h5')
    logger.info(f"  BC weights saved (HDF5) → {BC_WEIGHTS_PATH}")
    return model

# ───────────────────────────────────────────────
# 4. Custom PPO Trainer — loads BC weights at setup
# ───────────────────────────────────────────────

class WarmStartPPOTrainer(PPOTrainer):
    """
    PPO trainer that loads BC weights during setup(), BEFORE any
    training or rollout occurs.

    setup() is a Trainer method (not a Callbacks method). It runs
    exactly once in the trainer process when the trial starts.
    """

    def setup(self, config):
        # Standard PPO setup — builds TF graph, creates workers
        super().setup(config)

        bc_weights_path = BC_WEIGHTS_PATH
        instance_path = config.get("env_config", {}).get("instance_path", None)

        if bc_weights_path is None:
            logger.warning("  No bc_weights_path in config. Skipping BC warm-start.")
            return

        # Check for HDF5 format (no .index file)
        h5_path = bc_weights_path  # HDF5 saves as the exact path
        index_path = bc_weights_path + ".index"  # TF checkpoint format

        weights_exist = os.path.exists(h5_path) or os.path.exists(index_path)
        logger.info(f"  WarmStartPPOTrainer.setup()")
        logger.info(f"    BC weights path: {bc_weights_path}")
        logger.info(f"    HDF5 exists: {os.path.exists(h5_path)}")
        logger.info(f"    TF index exists: {os.path.exists(index_path)}")

        if not weights_exist:
            bc_dir = os.path.dirname(bc_weights_path)
            if os.path.exists(bc_dir):
                logger.warning(f"    Dir contents: {os.listdir(bc_dir)}")
            logger.warning("    PPO will train from scratch.")
            return

        # Get policy and its current weights
        policy = self.workers.local_worker().get_policy()
        ppo_weights = policy.get_weights()

        # Build BC model to load saved weights
        # Use the wrapped env to get the correct enriched obs shape
        env = make_wrapped_env(instance_path)
        obs = env.reset()
        obs_shape = obs["real_obs"].shape if isinstance(obs, dict) else obs.shape
        n_actions = env.action_space.n
        env.close()

        logger.info(f"    BC model: obs_shape={obs_shape}, n_actions={n_actions}")

        bc_model = build_bc_model(obs_shape, n_actions)
        
        # Load with the format that exists
        try:
            bc_model.load_weights(bc_weights_path).expect_partial()
        except AttributeError:
            bc_model.load_weights(bc_weights_path)

        # Extract BC weights (works in both eager and graph mode)
        bc_lookup = {}
        for var in bc_model.trainable_variables:
            val = tf.keras.backend.eval(var)
            name = var.name
            if "/" in name:
                name = name.split("/", 1)[-1]
            name = name.split(":")[0]
            bc_lookup[name] = val

        logger.info(f"    BC weight keys: {list(bc_lookup.keys())}")
        logger.info(f"    PPO weight keys ({len(ppo_weights)}): "
                     f"{list(ppo_weights.keys())[:15]}...")

        # Transfer BC weights to PPO by layer name
        new_weights = {}
        matched = []
        skipped = []

        for ppo_key, ppo_val in ppo_weights.items():
            transferred = False
            for bc_key, bc_val in bc_lookup.items():
                if bc_key in ppo_key and bc_val.shape == ppo_val.shape:
                    if "value" in ppo_key.lower() or "vf" in ppo_key.lower():
                        skipped.append(f"{ppo_key} (value head)")
                        break
                    new_weights[ppo_key] = bc_val
                    matched.append(f"{ppo_key} <- {bc_key} {bc_val.shape}")
                    transferred = True
                    break
            if not transferred:
                new_weights[ppo_key] = ppo_val

        logger.info(f"    Weight transfer:")
        logger.info(f"      Matched:        {len(matched)}")
        for m in matched:
            logger.info(f"        {m}")
        logger.info(f"      Skipped (VF):   {len(skipped)}")
        for s in skipped:
            logger.info(f"        {s}")
        logger.info(f"      Kept PPO init:  {len(new_weights) - len(matched)}")

        if len(matched) == 0:
            logger.error("    NO WEIGHTS MATCHED. Check layer names.")
            return

        # Set weights and sync to all remote workers
        policy.set_weights(new_weights)
        self.workers.sync_weights()

        logger.info(
            f"    BC weights loaded and synced to "
            f"{config.get('num_workers', 0)} workers. Warm-start complete."
        )

# ───────────────────────────────────────────────
# 5. Metrics-only callback (no weight loading)
# ───────────────────────────────────────────────

class MetricsCallback(DefaultCallbacks):
    """Records makespan and optimality gap. Nothing else."""

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        env = base_env.get_unwrapped()[0]
        if env.last_time_step != float("inf"):
            makespan = env.last_time_step
            episode.custom_metrics["make_span"] = makespan
            if current_bks > 0:
                gap = ((makespan - current_bks) / current_bks) * 100
                episode.custom_metrics["optimality_gap"] = gap

# ───────────────────────────────────────────────
# 6. PPO config
# ───────────────────────────────────────────────
ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

PPO_CONFIG = {
    "env": ENV_ID,
    "env_config": {"instance_path": INSTANCE_PATH},
    "model": {
        "custom_model": "fc_masked_model_tf",
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        "vf_share_layers": False,
    },
    "gamma": 1.0,
    "lambda": 1.0,
    "clip_param": 0.3,
    "kl_coeff": 0.5,
    "entropy_coeff": 0.002,
    "entropy_coeff_schedule": [[0, 0.002], [1_000_000, 0.00025]],
    "vf_loss_coeff": 0.5,
    "use_gae": True,
    "use_critic": True,
    "num_workers": 20,
    "num_envs_per_worker": 4,
    "rollout_fragment_length": 704,
    "train_batch_size": 12000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 10,
    "batch_mode": "truncate_episodes",
    "lr": 0.00066,
    "lr_schedule": [[0, 0.00066], [1_000_000, 7.8e-05]],
    "num_gpus": 0,
    "framework": "tf",
    "callbacks": MetricsCallback,
}

# ───────────────────────────────────────────────
# 7. Main — updated for wrapped environment
# ───────────────────────────────────────────────
if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info("EA Warm-Start → Behaviour Cloning → PPO Pipeline")
    logger.info("=" * 70)
    logger.info(f"Instance:       {INSTANCE_PATH}")
    logger.info(f"BKS:            {current_bks}")
    logger.info(f"Warm-start:     {'DISABLED' if args.no_warmstart else 'ENABLED'}")
    logger.info(f"EA algorithm:   {args.evo_alg}")
    logger.info(f"EA pop size:    {args.evo_pop}")
    logger.info(f"EA generations: {args.evo_gens}")
    logger.info(f"EA SA iters:    {args.evo_sa_iters}")
    logger.info(f"BC epochs:      {BC_EPOCHS}")
    logger.info(f"PPO max iters:  {args.iters}")
    logger.info(f"PPO gap target: {args.target_gap}%")
    logger.info(f"Output dir:     {CHECKPOINT_DIR}")
    logger.info(f"Dynamic wrapper: ENABLED (progress tracking)")
    logger.info("=" * 70)

    if not args.no_warmstart:
        # ── Stage 1: EA demonstrations ──
        logger.info("")
        logger.info("═══ Stage 1: Collecting EA Demonstrations ═══")
        n_demo_episodes = args.evo_pop
        demos = collect_evolutionary_demonstrations(
            n_episodes=n_demo_episodes,
            alg=args.evo_alg,
            pop_size=args.evo_pop,
            generations=args.evo_gens,
            sa_iters=args.evo_sa_iters,
            early_stop=args.evo_early_stop,
        )

        # ── Stage 2: Behaviour Cloning ──
        logger.info("")
        logger.info("═══ Stage 2: Behaviour Cloning ═══")

        env_tmp = make_wrapped_env(INSTANCE_PATH)
        obs_tmp = env_tmp.reset()
        obs_shape = obs_tmp["real_obs"].shape if isinstance(obs_tmp, dict) else obs_tmp.shape
        n_actions = env_tmp.action_space.n
        env_tmp.close()

        bc_model = behaviour_cloning(demos, obs_shape, n_actions)
        logger.info(f"  BC obs shape: {obs_shape} | n_actions: {n_actions}")

        if os.path.exists(BC_WEIGHTS_PATH) or os.path.exists(BC_WEIGHTS_PATH + ".index") or os.path.exists(BC_WEIGHTS_PATH + ".h5"):
            bc_dir = os.path.dirname(BC_WEIGHTS_PATH)
            bc_prefix = os.path.basename(BC_WEIGHTS_PATH)
            bc_files = [f for f in os.listdir(bc_dir) if f.startswith(bc_prefix)]
            logger.info(f"  BC weights verified: {bc_files}")
        else:
            logger.error(f"  BC weights NOT FOUND at {BC_WEIGHTS_PATH}")
            logger.error(f"  Contents of {CHECKPOINT_DIR}: {os.listdir(CHECKPOINT_DIR)}")
    else:
        logger.info("")
        logger.info("═══ Skipping Stage 1 & 2 (--no-warmstart) ═══")

    # ── Stage 3: PPO fine-tuning ──
    logger.info("")
    logger.info("═══ Stage 3: PPO Fine-Tuning ═══")
    logger.info(f"  Max iterations:    {args.iters}")
    logger.info(f"  Gap target:        {args.target_gap}%")
    logger.info(f"  Stopping window:   5 iterations")
    logger.info(f"  Checkpoint freq:   every 25 iterations")
    logger.info(f"  Workers:           {PPO_CONFIG['num_workers']}")
    logger.info(f"  Envs per worker:   {PPO_CONFIG['num_envs_per_worker']}")
    logger.info(f"  Train batch size:  {PPO_CONFIG['train_batch_size']}")
    logger.info(f"  BC weights path:   {BC_WEIGHTS_PATH}")

    ray.init(ignore_reinit_error=True)

    # Register the WRAPPED environment so PPO sees enriched observations
    def _make_wrapped_env(config):
        base = gym.make(ENV_ID, env_config=config)
        return DynamicJSSWrapper(base)

    register_env("custom_jss_env", _make_wrapped_env)
    PPO_CONFIG["env"] = "custom_jss_env"

    run_name = (
        f"jss_{instance_key}_"
        f"{args.evo_alg.lower()}_warmstart_ppo"
    )

    csv_path = os.path.join(CHECKPOINT_DIR, "results.csv")
    logger.info(f"  CSV log path:      {csv_path}")
    logger.info("-" * 60)

    analysis = tune.run(
        WarmStartPPOTrainer,
        config=PPO_CONFIG,
        stop=OptimalityGapStopper(
            window_size=5,
            target_gap=args.target_gap,
            max_iters=args.iters,
        ),
        callbacks=[CustomCSVLoggerCallback(csv_path)],
        checkpoint_freq=25,
        checkpoint_at_end=True,
        keep_checkpoints_num=None,
        local_dir=CHECKPOINT_DIR,
        name=run_name,
        verbose=1,
    )

    try:
        all_trials = analysis.trials
        if all_trials:
            last_result = all_trials[0].last_result
            if last_result:
                final_metrics = last_result.get("custom_metrics", {})
                final_makespan = final_metrics.get("make_span_mean", "N/A")
                final_gap = final_metrics.get("optimality_gap_mean", "N/A")
                final_iters = last_result.get("training_iteration", "N/A")
                final_reward = last_result.get("episode_reward_mean", "N/A")

                logger.info("")
                logger.info("─" * 60)
                logger.info("PPO Training Summary")
                logger.info("─" * 60)
                logger.info(f"  Total iterations: {final_iters}")
                logger.info(f"  Final makespan:   {final_makespan}")
                logger.info(f"  Final gap:        {final_gap}")
                logger.info(f"  Final reward:     {final_reward}")
                logger.info(f"  BKS:              {current_bks}")
                logger.info(f"  Instance:         {instance_key}")
                logger.info(f"  Checkpoints:      {CHECKPOINT_DIR}/{run_name}")
            else:
                logger.warning("Trial completed but no results available.")
        else:
            logger.warning("No trials found in analysis.")
    except Exception as e:
        logger.warning(f"Could not retrieve final results: {e}")
        logger.info("Check the CSV log for per-iteration results.")

    ray.shutdown()
    logger.info("")
    logger.info("=" * 70)
    logger.info("Pipeline complete.")
    logger.info(f"  Results CSV:   {csv_path}")
    logger.info(f"  Checkpoints:   {CHECKPOINT_DIR}/{run_name}")
    logger.info("=" * 70)
