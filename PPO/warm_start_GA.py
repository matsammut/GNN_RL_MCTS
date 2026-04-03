import csv
import argparse
import os
import time
import numpy as np
import gym
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.evaluation import SampleBatch, MultiAgentEpisode
import json
from models import *
from ray.tune.registry import register_env
from ray.rllib.agents.callbacks import DefaultCallbacks

import tensorflow as tf
from tensorflow import keras

from ray.tune.logger import LoggerCallback
from ray.tune import Stopper
from collections import deque

# ── Import reusable EA primitives from the sibling module ─────────────────────
# Safe to import: these are all pure functions with no side effects or global
# state. run_ma_hga_taillard.py must have "if __name__ == '__main__': main()"
# so its module-level constants (BKS, POP_SIZE, etc.) don't conflict here.
from run_ma_hga_taillard import (
    parse_taillard,
    compute_makespan,
    random_permutation,
    order_crossover,
    swap_mutation,
    tournament_select,
    simulated_annealing_improve,
    run_GA,
    run_MA,
    run_HGA,
)


# ─────────────────────────────────────────────
# Logging / Stopping helpers
# ─────────────────────────────────────────────

class CustomCSVLoggerCallback(LoggerCallback):
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.setup_done = False

    def log_trial_result(self, iteration, trial, result):
        metrics  = result.get("custom_metrics", {})
        makespan = metrics.get("make_span_mean", 0.0)
        iter_gap = metrics.get("optimality_gap_mean", 0.0)
        mode     = "a" if self.setup_done else "w"

        with open(self.csv_file_path, mode=mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["iteration", "makespan", "bks", "optimality gap"])
            if not self.setup_done:
                writer.writeheader()
                self.setup_done = True
            writer.writerow({
                "iteration":      result.get("training_iteration", iteration),
                "makespan":       f"{makespan:.2f}",
                "bks":            "Dynamic",
                "optimality gap": f"{iter_gap:.2f}",
            })


class OptimalityGapStopper(Stopper):
    def __init__(self, window_size=5, target_gap=20.0, max_iters=200):
        self.window_size = window_size
        self.target_gap  = target_gap
        self.max_iters   = max_iters
        self.trial_gaps  = {}

    def __call__(self, trial_id, result):
        if trial_id not in self.trial_gaps:
            self.trial_gaps[trial_id] = deque(maxlen=self.window_size)

        iteration = result.get("training_iteration", 0)
        if iteration >= self.max_iters:
            print(f"\nReached maximum of {self.max_iters} iterations. Stopping.\n")
            return True

        iter_gap = result.get("custom_metrics", {}).get("optimality_gap_mean")
        if iter_gap is not None:
            self.trial_gaps[trial_id].append(iter_gap)
            rolling_avg = np.mean(self.trial_gaps[trial_id])
            if len(self.trial_gaps[trial_id]) >= self.window_size and rolling_avg <= self.target_gap:
                print(f"\nGoal Reached! Stable gap at {rolling_avg:.2f}%. Triggering early stop.\n")
                return True
        return False

    def stop_all(self):
        return False


# ─────────────────────────────────────────────
# CLI arguments
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="EA warm-start PPO trainer for Taillard instances.")
parser.add_argument("--instances",    type=str, default="instances/ta52")
parser.add_argument("--iters",        type=int, default=200)
parser.add_argument("--out",          type=str, default="checkpoint_results")
parser.add_argument("--bks",          type=str, required=True, help="Path to bks.json")
parser.add_argument("--evo-alg",      type=str, default="HGA", choices=["GA", "MA", "HGA"],
                    help="EA for warm-start. HGA (default) best balances quality vs. speed.")
parser.add_argument("--evo-pop",      type=int, default=50,  help="EA population size.")
parser.add_argument("--evo-gens",     type=int, default=50,  help="EA generations.")
parser.add_argument("--evo-sa-iters", type=int, default=500, help="SA iters per offspring (MA/HGA).")
args = parser.parse_args()

INSTANCE_PATH   = args.instances
ENV_ID          = "JSSEnv:jss-v1"
BC_EPOCHS       = 50
BC_LR           = 1e-3
BC_BATCH        = 256
DEMO_EPISODES   = args.iters
CHECKPOINT_DIR  = args.out
BC_WEIGHTS_PATH = args.out


# ─────────────────────────────────────────────
# 1.  EA warm-start — collect demonstrations
# ─────────────────────────────────────────────

def permutation_to_env_actions(env, perm):
    """
    Replay a decoded EA permutation through JSSEnv, recording (obs, action) pairs.
    Walks the permutation in order, picking the first legal job at each step.
    Falls back to the first legal action if the scheduled job is currently blocked.
    """
    obs  = env.reset()
    done = False
    pairs        = []
    perm_cursor  = 0

    while not done:
        legal  = np.where(obs["action_mask"] == 1)[0]
        chosen = None

        # Walk the permutation to find the next schedulable job
        for scan in range(perm_cursor, len(perm)):
            if perm[scan] in legal:
                chosen      = perm[scan]
                perm_cursor = scan + 1
                break

        if chosen is None or chosen not in legal:
            chosen = int(legal[0])          # fallback

        pairs.append((obs, chosen))
        obs, _, done, _ = env.step(chosen)

    return pairs


def collect_evolutionary_demonstrations(n_episodes, alg, pop_size, generations, sa_iters):
    """
    Run the chosen EA on the parsed Taillard instance, then replay each schedule
    through JSSEnv to produce (obs, action) demo pairs for behaviour cloning.

    The EA population provides *diverse*, near-optimal demonstrations — a significant
    upgrade over SPT's single greedy trajectory.
    """
    print(f"  Parsing instance: {INSTANCE_PATH}")
    jobs = parse_taillard(INSTANCE_PATH)
    print(f"  Running {alg} (pop={pop_size}, gens={generations}) …")

    t0 = time.time()
    if alg == "GA":
        pop, fitness, best_ind, best_val, _ = run_GA(
            jobs, pop_size=pop_size, generations=generations, seed=42)
    elif alg == "MA":
        pop, fitness, best_ind, best_val, _ = run_MA(
            jobs, pop_size=pop_size, generations=generations, sa_iters=sa_iters, seed=42)
    else:  # HGA
        pop, fitness, best_ind, best_val, _ = run_HGA(
            jobs, pop_size=pop_size, generations=generations, sa_iters=sa_iters, seed=42)

    print(f"  EA finished in {time.time()-t0:.1f}s | best makespan = {best_val}")

    # Rank by quality so BC sees the best examples first
    schedules = [p for _, p in sorted(zip(fitness, pop), key=lambda x: x[0])]

    env   = gym.make(ENV_ID, env_config={"instance_path": INSTANCE_PATH})
    demos = []
    for ep in range(n_episodes):
        perm  = schedules[ep % len(schedules)]
        demos.extend(permutation_to_env_actions(env, perm))
        if (ep + 1) % 50 == 0:
            print(f"  Replayed {ep+1}/{n_episodes} schedules ({len(demos)} transitions)")
    env.close()
    return demos


# ─────────────────────────────────────────────
# 2.  Behaviour cloning
# ─────────────────────────────────────────────

def build_bc_model(obs_shape, n_actions, hidden=(256, 256)):
    inp    = keras.Input(shape=obs_shape, name="obs")
    x      = keras.layers.Flatten()(inp)
    for units in hidden:
        x  = keras.layers.Dense(units, activation="relu")(x)
    logits = keras.layers.Dense(n_actions, name="logits")(x)
    return keras.Model(inputs=inp, outputs=logits)


def behaviour_cloning(demos, obs_shape, n_actions):
    obs_arr = np.array([d[0]["real_obs"] if isinstance(d[0], dict) else d[0]
                        for d in demos], dtype=np.float32)
    act_arr = np.array([d[1] for d in demos], dtype=np.int64)

    model = build_bc_model(obs_shape, n_actions)
    model.compile(
        optimizer=keras.optimizers.Adam(BC_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(obs_arr, act_arr, epochs=BC_EPOCHS, batch_size=BC_BATCH,
              validation_split=0.1, verbose=1)
    model.save_weights(BC_WEIGHTS_PATH)
    print(f"BC weights saved → {BC_WEIGHTS_PATH}")
    return model


# ─────────────────────────────────────────────
# 3.  PPO callback — load BC weights on first iteration
# ─────────────────────────────────────────────

class WarmStartCallback(DefaultCallbacks):
    _loaded = False

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        env = base_env.get_unwrapped()[0]
        if env.last_time_step != float('inf'):
            makespan = env.last_time_step
            episode.custom_metrics['make_span'] = makespan
            if hasattr(env, 'current_bks') and env.current_bks > 0:
                gap = ((makespan - env.current_bks) / env.current_bks) * 100
                episode.custom_metrics['optimality_gap'] = gap

    def on_train_result(self, *, trainer, result, **kwargs):
        if not WarmStartCallback._loaded and os.path.exists(BC_WEIGHTS_PATH):
            print("⟳  Loading BC warm-start weights (from EA demos) into PPO policy …")
            policy  = trainer.get_policy()
            weights = policy.get_weights()

            env = gym.make(ENV_ID, env_config={"instance_path": INSTANCE_PATH})
            obs = env.reset()
            obs_shape = obs["real_obs"].shape if isinstance(obs, dict) else obs.shape
            n_actions = env.action_space.n
            env.close()

            bc_model = build_bc_model(obs_shape, n_actions)
            bc_model.load_weights(BC_WEIGHTS_PATH)

            bc_vars     = {v.name: v.numpy() for v in bc_model.trainable_variables}
            new_weights = {}
            for k, v in weights.items():
                for bc_name, bc_val in bc_vars.items():
                    if bc_val.shape == v.shape:
                        new_weights[k] = bc_val
                        break
                else:
                    new_weights[k] = v

            policy.set_weights(new_weights)
            WarmStartCallback._loaded = True
            print("✓  BC weights loaded.")


# ─────────────────────────────────────────────
# 4.  Reward function
# ─────────────────────────────────────────────
with open(args.bks, 'r') as f:
    bks_map = json.load(f)
instance_key = os.path.basename(INSTANCE_PATH).replace(".txt", "")
current_bks  = float(bks_map.get(instance_key, 1.0) if isinstance(bks_map, dict) else bks_map)
TA72_OPTIMAL = current_bks

with open(INSTANCE_PATH, "r") as f:
    tokens = f.read().split()
NUM_MACHINES = int(tokens[1])
TOTAL_OPS    = int(tokens[0]) * NUM_MACHINES


def compute_reward(env, action, done):
    reward     = 0.0
    current_lb = env._compute_makespan_lower_bound()
    delta      = env.prev_makespan_lb - current_lb
    if delta > 0:
        reward += (delta / TA72_OPTIMAL) * 2.0
    env.prev_makespan_lb = current_lb

    idle    = sum(1 for m in env.machines if not m.is_busy)
    reward -= (idle / NUM_MACHINES) * 0.02

    if env._action_on_critical_path(action):
        reward += 0.03

    jobs_done  = env._get_jobs_scheduled()
    expected_t = (jobs_done / TOTAL_OPS) * TA72_OPTIMAL
    if env._get_current_time() <= expected_t * 1.05:
        reward += 0.01

    if done:
        gap     = (env._get_makespan() - TA72_OPTIMAL) / TA72_OPTIMAL
        reward += 20.0 * max(0.0, 1.0 - gap)

    return reward


# ─────────────────────────────────────────────
# 5.  PPO config
# ─────────────────────────────────────────────
ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)
PPO_CONFIG = {
    "env":        ENV_ID,
    "env_config": {"instance_path": INSTANCE_PATH},
    "model": {
        "custom_model":    "fc_masked_model_tf",
        "fcnet_hiddens":   [256, 256],
        "fcnet_activation": "relu",
        "vf_share_layers": False,
    },
    "gamma": 1.0, "lambda": 1.0, "clip_param": 0.3, "kl_coeff": 0.5,
    "entropy_coeff": 0.002,
    "entropy_coeff_schedule": [[0, 0.002], [1_000_000, 0.00025]],
    "vf_loss_coeff": 0.5, "use_gae": True, "use_critic": True,
    "num_workers": 20, "num_envs_per_worker": 4,
    "rollout_fragment_length": 704, "train_batch_size": 12000,
    "sgd_minibatch_size": 128, "num_sgd_iter": 10,
    "batch_mode": "truncate_episodes",
    "lr": 0.00066,
    "lr_schedule": [[0, 0.00066], [1_000_000, 7.8e-05]],
    "num_gpus": 0, "framework": "tf",
    "callbacks": WarmStartCallback,
}


# ─────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print(f"\n═══ Step 1: Collecting {args.evo_alg} demonstrations ═══")
    demos = collect_evolutionary_demonstrations(
        n_episodes=DEMO_EPISODES,
        alg=args.evo_alg,
        pop_size=args.evo_pop,
        generations=args.evo_gens,
        sa_iters=args.evo_sa_iters,
    )
    print(f"Total transitions collected: {len(demos)}")

    print("\n═══ Step 2: Behaviour Cloning ═══")
    env_tmp   = gym.make(ENV_ID, env_config={"instance_path": INSTANCE_PATH})
    obs_tmp   = env_tmp.reset()
    obs_shape = obs_tmp["real_obs"].shape if isinstance(obs_tmp, dict) else obs_tmp.shape
    n_actions = env_tmp.action_space.n
    env_tmp.close()
    bc_model = behaviour_cloning(demos, obs_shape, n_actions)

    print("\n═══ Step 3: PPO fine-tuning ═══")
    ray.init(ignore_reinit_error=True)
    register_env("custom_jss_env", lambda cfg: gym.make(ENV_ID, env_config=cfg))
    PPO_CONFIG["env"] = "custom_jss_env"

    tune.run(
        PPOTrainer,
        config=PPO_CONFIG,
        stop=OptimalityGapStopper(window_size=5, target_gap=20.0, max_iters=args.iters),
        callbacks=[CustomCSVLoggerCallback(os.path.join(CHECKPOINT_DIR, "results.csv"))],
        checkpoint_freq=25,
        checkpoint_at_end=True,
        keep_checkpoints_num=None,
        local_dir=CHECKPOINT_DIR,
        name="jss_" + os.path.basename(INSTANCE_PATH) + "_ea_warmstart_ppo",
        verbose=1,
    )
    ray.shutdown()
