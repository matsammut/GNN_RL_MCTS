#!/usr/bin/env python3
"""
warm_start_GA.py

Two-stage pipeline for Taillard JSSP instances:
  1. Evolutionary Algorithm (GA/MA/HGA) runs to produce a population of
     diverse, high-quality schedules.
  2. Behaviour Cloning (BC) trains a neural network to imitate the EA's
     dispatching decisions via supervised learning.

After training, the BC model is evaluated by running a full episode
through JSSEnv and reporting the achieved makespan vs. BKS.

Usage:
    python3 warm_start_GA.py --bks bks.json --instances instances/ta52
    python3 warm_start_GA.py --bks bks.json --instances instances/ta52 --evo-alg MA
    python3 warm_start_GA.py --bks bks.json --instances instances/ta72 --evo-gens 10
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
    description="EA warm-start → BC pipeline for Taillard JSSP instances.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python3 warm_start_GA.py --bks bks.json --instances instances/ta52
  python3 warm_start_GA.py --bks bks.json --instances instances/ta72 --evo-alg MA --evo-gens 10
    """,
)
parser.add_argument("--instances", type=str, default="instances/ta52",
                    help="Path to Taillard instance file (default: instances/ta52)")
parser.add_argument("--out", type=str, default="checkpoint_results",
                    help="Output directory (default: checkpoint_results)")
parser.add_argument("--bks", type=str, required=True,
                    help="Path to bks.json")
parser.add_argument("--evo-alg", type=str, default="HGA", choices=["GA", "MA", "HGA"],
                    help="EA for warm-start demonstrations (default: HGA)")
parser.add_argument("--evo-pop", type=int, default=50,
                    help="EA population size (default: 50)")
parser.add_argument("--evo-gens", type=int, default=5,
                    help="EA generations (default: 5)")
parser.add_argument("--evo-sa-iters", type=int, default=500,
                    help="SA iterations per offspring/elite in MA/HGA (default: 500)")
parser.add_argument("--bc-epochs", type=int, default=50,
                    help="Behaviour cloning training epochs (default: 50)")
parser.add_argument("--bc-hidden", type=int, nargs="+", default=[512, 512, 256],
                    help="Hidden layer sizes for BC network (default: 512 512 256)")
parser.add_argument("--bc-demo-episodes", type=int, default=None,
                    help="Episodes to replay for BC dataset. "
                         "Defaults to evo_pop // 2 (one episode per unique expert). "
                         "Set higher to generate more diverse data via epsilon-greedy noise.")
parser.add_argument("--bc-epsilon", type=float, default=0.05,
                    help="Epsilon for epsilon-greedy noise during demo collection. "
                         "0.0 = pure expert, 0.05 = 5%% random actions (default: 0.05)")
parser.add_argument("--bc-eval-episodes", type=int, default=1,
                    help="Number of episodes to evaluate BC model (default: 1)")
parser.add_argument("--evo-early-stop", action="store_true", default=False,
                    help="Stop EA if best makespan doesn't improve for 3 generations")
args = parser.parse_args()

# Resolve bc_demo_episodes:
#   - If epsilon > 0, default to 500 so each episode is unique due to noise
#   - If epsilon == 0, default to evo_pop // 2 (no point repeating deterministic rollouts)
if args.bc_demo_episodes is None:
    if args.bc_epsilon > 0.0:
        args.bc_demo_episodes = 500
    else:
        args.bc_demo_episodes = max(1, args.evo_pop // 2)

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
# EA constants
# ───────────────────────────────────────────────

CROSSOVER_P = 0.9
MUTATION_P = 0.1

# ───────────────────────────────────────────────
# Environment helper
# ───────────────────────────────────────────────

def make_wrapped_env(instance_path):
    """Create a JSSEnv wrapped with dynamic progress tracking."""
    base_env = gym.make(ENV_ID, env_config={"instance_path": instance_path})
    return DynamicJSSWrapper(base_env)

# ───────────────────────────────────────────────
# 1. EA Warm-Start
# ───────────────────────────────────────────────

def _run_ea_warmstart_ga(jobs, pop_size, generations, seed=42, early_stop=False):
    """Lightweight GA for demonstration collection."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    pop = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = [compute_makespan(jobs, ind) for ind in pop]
    best_val = min(fitness)
    best_ind = deepcopy(pop[fitness.index(best_val)])
    patience_counter = 0

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
            patience_counter = 0
        else:
            patience_counter += 1

        logger.info(
            f"  GA warmstart | Gen {gen}/{generations} | "
            f"Best: {best_val} | Patience: {patience_counter}/3"
        )

        if early_stop and patience_counter >= 3:
            logger.info(f"  GA warmstart | Early stopping triggered at gen {gen}.")
            break

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


def _run_ea_warmstart_hga(jobs, pop_size, generations, sa_iters, seed=42):
    """
    Hybrid GA: only top half of population undergoes SA local search.
    Cheaper than MA (full SA on every offspring) but higher quality than GA.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    pop = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = [compute_makespan(jobs, ind) for ind in pop]
    best_val = min(fitness)
    best_ind = deepcopy(pop[fitness.index(best_val)])

    for gen in range(1, generations + 1):
        new_pop = [deepcopy(best_ind)]
        offspring = []
        while len(offspring) < pop_size - 1:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = (order_crossover(p1, p2)
                     if random.random() < CROSSOVER_P else deepcopy(p1))
            child = mutate(child, mutation_rate=MUTATION_P)
            offspring.append(child)

        offspring_fitness = [compute_makespan(jobs, c) for c in offspring]
        ranked = sorted(zip(offspring_fitness, offspring), key=lambda x: x[0])
        half = len(ranked) // 2
        for i, (f, c) in enumerate(ranked):
            if i < half:
                t0 = calibrate_sa_temperature(jobs, c)
                c, _ = simulated_annealing_improve(
                    jobs, c, iters=sa_iters, t0=t0, tend=SA_TEND
                )
            new_pop.append(c)

        pop = new_pop
        fitness = [compute_makespan(jobs, ind) for ind in pop]
        pop, fitness = deduplicate_population(pop, fitness, jobs)

        gen_best = min(fitness)
        if gen_best < best_val:
            best_val = gen_best
            best_ind = deepcopy(pop[fitness.index(gen_best)])

        logger.info(f"  HGA warmstart | Gen {gen}/{generations} | Best: {best_val}")

    return pop, fitness, best_ind, best_val

# ───────────────────────────────────────────────
# 2. Demonstration Collection
# ───────────────────────────────────────────────

def direct_schedule_rollout(env, perm, n_jobs, epsilon=0.05):
    """
    Replay a GA permutation through JSSEnv with optional epsilon-greedy noise.

    At each step:
      - With probability (1 - epsilon): pick the legal job whose next
        occurrence appears earliest in the remaining permutation
        (faithful GA replay).
      - With probability epsilon: pick a uniformly random legal job
        (exploration noise — creates unique state sequences across episodes).

    epsilon=0.0 gives pure expert replay (deterministic, no diversity).
    epsilon=0.05 randomises 5% of actions, making every episode unique
    while keeping 95% of decisions expert-guided.
    """
    remaining = list(perm)
    obs = env.reset()
    done = False
    pairs = []

    while not done:
        legal = set(np.where(obs["action_mask"] == 1)[0])
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


def collect_evolutionary_demonstrations(n_episodes, alg, pop_size, generations,
                                         sa_iters, epsilon=0.05, early_stop=False):
    """
    Run the chosen EA, then replay expert permutations through JSSEnv with
    epsilon-greedy noise to produce diverse (obs, action) pairs for BC.
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
            jobs, pop_size=pop_size, generations=generations,
            seed=42, early_stop=early_stop
        )
    elif alg == "MA":
        pop, fitness, best_ind, best_val = _run_ea_warmstart_ma(
            jobs, pop_size=pop_size, generations=generations,
            sa_iters=sa_iters, seed=42
        )
    else:  # HGA
        pop, fitness, best_ind, best_val = _run_ea_warmstart_hga(
            jobs, pop_size=pop_size, generations=generations,
            sa_iters=sa_iters, seed=42
        )

    elapsed = time.time() - t0
    gap = 100.0 * (best_val - current_bks) / current_bks
    logger.info(
        f"  EA finished in {elapsed:.1f}s | "
        f"Best makespan (compute_makespan): {best_val} | "
        f"Gap to BKS: {gap:.2f}%"
    )

    ranked_pairs = sorted(zip(fitness, pop), key=lambda x: x[0])
    sorted_fitness = [f for f, _ in ranked_pairs]
    ranked_pop     = [p for _, p in ranked_pairs]

    logger.info(
        f"  Population quality — "
        f"Best: {sorted_fitness[0]}, "
        f"Median: {sorted_fitness[len(sorted_fitness)//2]}, "
        f"Worst: {sorted_fitness[-1]}"
    )

    n_experts = max(1, pop_size // 2)
    experts = [(sorted_fitness[i], ranked_pop[i]) for i in range(n_experts)]

    logger.info(
        f"  Using top {n_experts} experts | "
        f"Replaying {n_episodes} episodes with epsilon={epsilon} | "
        f"(~{n_episodes * TOTAL_OPS:,} transitions expected)"
    )

    env = make_wrapped_env(INSTANCE_PATH)
    demos = []
    makespans_achieved = []
    ga_makespans = []

    for ep in range(n_episodes):
        expert_idx = ep % n_experts
        ga_ms, perm = experts[expert_idx]

        pairs = direct_schedule_rollout(env, perm, NUM_JOBS, epsilon=epsilon)
        demos.extend(pairs)

        base_env = env.unwrapped
        env_ms = getattr(base_env, "last_time_step", 0)
        makespans_achieved.append(env_ms)
        ga_makespans.append(ga_ms)

        if (ep + 1) % 50 == 0:
            logger.info(
                f"  Replayed {ep + 1}/{n_episodes} episodes | "
                f"{len(demos):,} transitions | "
                f"avg env makespan: {np.mean(makespans_achieved):.0f} | "
                f"avg GA makespan:  {np.mean(ga_makespans):.0f}"
            )

    env.close()

    avg_env = np.mean(makespans_achieved)
    avg_ga  = np.mean(ga_makespans)
    logger.info(
        f"  Demo fidelity — "
        f"avg GA makespan: {avg_ga:.0f} | "
        f"avg env makespan: {avg_env:.0f} | "
        f"overhead: {avg_env - avg_ga:.0f} "
        f"({100*(avg_env - avg_ga)/avg_ga:.1f}%)"
    )

    _verify_demonstration_consistency(demos)
    logger.info(f"  Total demonstration transitions: {len(demos):,}")
    return demos


def _verify_demonstration_consistency(demos, n_checks=500):
    if len(demos) < n_checks:
        return

    obs_action_map = {}
    for obs, action in demos[:n_checks]:
        obs_data = obs["real_obs"] if isinstance(obs, dict) else obs
        fingerprint = tuple(np.round(obs_data.flatten(), 1))
        obs_action_map.setdefault(fingerprint, []).append(action)

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
            f"observations map to the same action ({total_groups} groups checked)"
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

def build_bc_model(obs_shape, n_actions, hidden=(512, 512, 256)):
    """
    Build a BC model with configurable depth and width.
    Default: 3 layers [512, 512, 256] — larger than the previous 2x256
    to better capture the (100x10) observation space with 101 actions.
    """
    shape = obs_shape if isinstance(obs_shape, tuple) else (obs_shape,)
    inp = keras.Input(shape=shape, name="obs")
    x = keras.layers.Flatten()(inp)
    for i, units in enumerate(hidden):
        x = keras.layers.Dense(units, activation="relu", name=f"fc_{i+1}")(x)
    logits = keras.layers.Dense(n_actions, name="fc_out")(x)
    return keras.Model(inputs=inp, outputs=logits)


def behaviour_cloning(demos, obs_shape, n_actions, hidden=(512, 512, 256)):
    """Train BC model on EA demonstrations."""
    obs_arr = np.array(
        [d[0]["real_obs"] if isinstance(d[0], dict) else d[0] for d in demos],
        dtype=np.float32,
    )
    act_arr = np.array([d[1] for d in demos], dtype=np.int64)

    logger.info(
        f"  BC dataset: {len(obs_arr):,} samples | "
        f"Obs shape: {obs_shape} | Actions: {n_actions}"
    )
    logger.info(f"  BC network: {list(hidden)} → {n_actions}")

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

    model = build_bc_model(obs_shape, n_actions, hidden=hidden)
    model.summary(print_fn=logger.info)
    model.compile(
        optimizer=keras.optimizers.Adam(3e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["acc"],
    )

    early_stop_cb = keras.callbacks.EarlyStopping(
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
        callbacks=[early_stop_cb, lr_reduce],
        verbose=1,
    )

    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
    final_val_acc = history.history[val_acc_key][-1]
    epochs_trained = len(history.history["loss"])

    logger.info(
        f"  BC complete ({epochs_trained}/{BC_EPOCHS} epochs) | "
        f"Val acc: {final_val_acc:.4f}"
    )

    random_chance = 1.0 / n_actions
    if final_val_acc <= random_chance * 1.5:
        logger.error(
            f"  BC FAILED: Val acc {final_val_acc:.4f} ~ random {random_chance:.4f}"
        )
    else:
        logger.info(
            f"  BC accuracy {final_val_acc:.4f} >> random {random_chance:.4f}"
        )

    model.save_weights(BC_WEIGHTS_PATH, save_format="h5")
    logger.info(f"  BC weights saved (HDF5) → {BC_WEIGHTS_PATH}")
    return model

# ───────────────────────────────────────────────
# 4. BC Evaluation — greedy masked rollout
# ───────────────────────────────────────────────

def bc_greedy_rollout(model, env):
    """
    Run one episode greedily using the BC model.

    At each step:
      - Feed real_obs through the model to get logits.
      - Mask illegal actions by setting their logits to -inf.
      - Select argmax of the masked logits.

    Returns the achieved makespan.
    """
    obs = env.reset()
    done = False

    while not done:
        real_obs = obs["real_obs"] if isinstance(obs, dict) else obs
        action_mask = obs["action_mask"] if isinstance(obs, dict) else None

        logits = model(np.array([real_obs], dtype=np.float32), training=False)
        logits = logits.numpy()[0]

        if action_mask is not None:
            logits[action_mask == 0] = -np.inf

        action = int(np.argmax(logits))
        obs, _, done, _ = env.step(action)

    base_env = env.unwrapped
    makespan = getattr(base_env, "last_time_step", float("inf"))
    return makespan


def evaluate_bc_model(model, n_episodes=1):
    """
    Run greedy rollout(s) with the BC model and report the makespan.
    Results are saved to bc_eval_results.csv.
    """
    logger.info(f"  Evaluating BC model over {n_episodes} episode(s)...")
    env = make_wrapped_env(INSTANCE_PATH)
    makespans = []

    for ep in range(n_episodes):
        makespan = bc_greedy_rollout(model, env)
        gap = 100.0 * (makespan - current_bks) / current_bks if current_bks > 0 else float("nan")
        makespans.append(makespan)
        logger.info(
            f"  Episode {ep + 1:>3}/{n_episodes} | "
            f"Makespan: {makespan:.0f} | "
            f"BKS: {current_bks:.0f} | "
            f"Gap: {gap:.2f}%"
        )

    env.close()

    logger.info("")
    logger.info("─" * 50)
    logger.info("BC Evaluation Summary")
    logger.info("─" * 50)
    logger.info(f"  Episodes:     {n_episodes}")
    logger.info(f"  BKS:          {current_bks:.0f}")
    logger.info(f"  Makespan:     {makespans[0]:.0f}  "
                f"(gap: {100.0 * (makespans[0] - current_bks) / current_bks:.2f}%)")
    logger.info("─" * 50)

    csv_path = os.path.join(CHECKPOINT_DIR, "bc_eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "makespan", "bks", "gap_pct"]
        )
        writer.writeheader()
        for ep, ms in enumerate(makespans, 1):
            writer.writerow({
                "episode": ep,
                "makespan": f"{ms:.0f}",
                "bks": current_bks,
                "gap_pct": f"{100.0 * (ms - current_bks) / current_bks:.2f}",
            })
    logger.info(f"  Results saved → {csv_path}")

    return makespans

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info("EA Warm-Start → Behaviour Cloning Pipeline")
    logger.info("=" * 70)
    logger.info(f"Instance:         {INSTANCE_PATH}")
    logger.info(f"BKS:              {current_bks}")
    logger.info(f"EA algorithm:     {args.evo_alg}")
    logger.info(f"EA pop size:      {args.evo_pop}")
    logger.info(f"EA generations:   {args.evo_gens}")
    logger.info(f"EA SA iters:      {args.evo_sa_iters}")
    logger.info(f"BC epochs:        {BC_EPOCHS}")
    logger.info(f"BC network:       {args.bc_hidden}")
    logger.info(f"BC demo episodes: {args.bc_demo_episodes}  "
                f"(~{args.bc_demo_episodes * TOTAL_OPS:,} transitions)")
    logger.info(f"BC epsilon:       {args.bc_epsilon}  "
                f"({'pure expert' if args.bc_epsilon == 0 else f'{args.bc_epsilon*100:.0f}% random actions'})")
    logger.info(f"BC eval episodes: {args.bc_eval_episodes}")
    logger.info(f"Output dir:       {CHECKPOINT_DIR}")
    logger.info("=" * 70)

    # ── Stage 1: EA demonstrations ──────────────────────────────────────────
    logger.info("")
    logger.info("═══ Stage 1: Collecting EA Demonstrations ═══")
    demos = collect_evolutionary_demonstrations(
        n_episodes=args.bc_demo_episodes,
        alg=args.evo_alg,
        pop_size=args.evo_pop,
        generations=args.evo_gens,
        sa_iters=args.evo_sa_iters,
        epsilon=args.bc_epsilon,
        early_stop=args.evo_early_stop,
    )

    # ── Stage 2: Behaviour Cloning ──────────────────────────────────────────
    logger.info("")
    logger.info("═══ Stage 2: Behaviour Cloning ═══")

    env_tmp = make_wrapped_env(INSTANCE_PATH)
    obs_tmp = env_tmp.reset()
    obs_shape = obs_tmp["real_obs"].shape if isinstance(obs_tmp, dict) else obs_tmp.shape
    n_actions = env_tmp.action_space.n
    env_tmp.close()

    bc_model = behaviour_cloning(
        demos, obs_shape, n_actions, hidden=tuple(args.bc_hidden)
    )
    logger.info(f"  BC obs shape: {obs_shape} | n_actions: {n_actions}")

    # ── Stage 3: BC Evaluation ───────────────────────────────────────────────
    logger.info("")
    logger.info("═══ Stage 3: BC Model Evaluation ═══")
    makespans = evaluate_bc_model(bc_model, n_episodes=args.bc_eval_episodes)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Pipeline complete.")
    logger.info(f"  BC weights:    {BC_WEIGHTS_PATH}")
    logger.info(f"  Eval results:  {os.path.join(CHECKPOINT_DIR, 'bc_eval_results.csv')}")
    logger.info("=" * 70)
