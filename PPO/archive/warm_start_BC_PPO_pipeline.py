

"""
Warm-start with SPT heuristic, then fine-tune with PPO on JSSEnv ta72.

Pipeline:
  1. Run SPT (Shortest Processing Time) heuristic to collect demonstration episodes
  2. Behaviour cloning (BC) to pre-train the policy network
  3. Hand off to RLlib PPO, initialised from the BC weights
"""
import csv
import argparse
import os
import numpy as np
import gym
import pickle
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.evaluation import SampleBatch,MultiAgentEpisode
import json
from models import *
from ray.tune.registry import register_env

import tensorflow as tf
from tensorflow import keras

from ray.tune.logger import LoggerCallback
from ray.tune import Stopper
from collections import deque

class CustomCSVLoggerCallback(LoggerCallback):
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.setup_done = False

    def log_trial_result(self, iteration, trial, result):
        metrics = result.get("custom_metrics", {})
        makespan = metrics.get("make_span_mean", 0.0)
        iter_gap = metrics.get("optimality_gap_mean", 0.0)
        
        # Determine if we are writing the header or appending data
        mode = "a" if self.setup_done else "w"
        
        with open(self.csv_file_path, mode=mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["iteration", "makespan", "bks", "optimality gap"])
            
            if not self.setup_done:
                writer.writeheader()
                self.setup_done = True
            
            writer.writerow({
                "iteration": result.get("training_iteration", iteration),
                "makespan": f"{makespan:.2f}",
                "bks": "Dynamic",
                "optimality gap": f"{iter_gap:.2f}"
            })

class OptimalityGapStopper(Stopper):
    def __init__(self, window_size=5, target_gap=20.0, max_iters=200):
        self.window_size = window_size
        self.target_gap = target_gap
        self.max_iters = max_iters
        self.trial_gaps = {}

    def __call__(self, trial_id, result):
        if trial_id not in self.trial_gaps:
            self.trial_gaps[trial_id] = deque(maxlen=self.window_size)

        iteration = result.get("training_iteration", 0)

        # 1. Fallback: Stop if we hit the maximum allowed iterations
        if iteration >= self.max_iters:
            print(f"\nReached maximum of {self.max_iters} iterations without hitting target gap. Stopping.\n")
            return True

        # 2. Main Condition: Stop if we hit the target optimality gap
        metrics = result.get("custom_metrics", {})
        iter_gap = metrics.get("optimality_gap_mean")

        if iter_gap is not None:
            self.trial_gaps[trial_id].append(iter_gap)
            rolling_avg = np.mean(self.trial_gaps[trial_id])
            
            # Rolling Window Stopping Condition
            if len(self.trial_gaps[trial_id]) >= self.window_size and rolling_avg <= self.target_gap:
                print(f"\nGoal Reached! Stable gap at {rolling_avg:.2f}%. Triggering early stop.\n")
                return True 
                
        return False

    def stop_all(self):
        return False

# ─────────────────────────────────────────────
# 0.  Constants
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Iteration-based PPO trainer for Taillard instances.")
parser.add_argument(
    "--instances",
    type=str,
    default="instances/ta52",
    help="Path to the Taillard instance directory or file."
)
parser.add_argument(
    "--iters",
    type=int,
    default=200,
    help="Maximum number of PPO training iterations."
)
parser.add_argument(
    "--out",
    type=str,
    default="checkpoint_results",
    help="Path to save checkpoints and the summary CSV."
)
parser.add_argument("--bks", type=str, required=True, help="Path to bks.json")
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
# 1.  SPT heuristic — collect demonstrations
# ─────────────────────────────────────────────

def spt_action(env, obs):
    """
    Shortest Processing Time dispatching rule.
    Among legal actions, pick the operation with the smallest processing time.
    Falls back to random if SPT info is unavailable.
    """
    legal_actions = np.where(obs["action_mask"] == 1)[0]
    if not hasattr(env, "job_op_dur"):          # guard for envs without direct access
        return np.random.choice(legal_actions)

    durations = [env.job_op_dur[a] for a in legal_actions]
    return legal_actions[int(np.argmin(durations))]


def collect_demonstrations(n_episodes: int):
    """Run SPT for n_episodes, return list of (obs, action) pairs."""
    env = gym.make(ENV_ID, env_config={"instance_path": INSTANCE_PATH})
    demos = []
    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        ep_data = []
        while not done:
            action          = spt_action(env, obs)
            ep_data.append((obs, action))
            obs, _, done, _ = env.step(action)
        demos.extend(ep_data)
        if (ep + 1) % 50 == 0:
            print(f"  Collected {ep+1}/{n_episodes} SPT episodes ({len(demos)} transitions)")
    env.close()
    return demos


# ─────────────────────────────────────────────
# 2.  Behaviour cloning
# ─────────────────────────────────────────────

def build_bc_model(obs_shape, n_actions, hidden=(256, 256)):
    """Simple FC network that mirrors the PPO policy head."""
    inp    = keras.Input(shape=obs_shape, name="obs")
    x      = keras.layers.Flatten()(inp)
    for units in hidden:
        x  = keras.layers.Dense(units, activation="relu")(x)
    logits = keras.layers.Dense(n_actions, name="logits")(x)
    return keras.Model(inputs=inp, outputs=logits)


def behaviour_cloning(demos, obs_shape, n_actions):
    """Train BC model on (obs, action) demonstrations."""
    obs_arr = np.array([d[0]["real_obs"] if isinstance(d[0], dict) else d[0]
                        for d in demos], dtype=np.float32)
    act_arr = np.array([d[1] for d in demos], dtype=np.int64)

    model = build_bc_model(obs_shape, n_actions)
    model.compile(
        optimizer=keras.optimizers.Adam(BC_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    model.fit(
        obs_arr, act_arr,
        epochs=BC_EPOCHS,
        batch_size=BC_BATCH,
        validation_split=0.1,
        verbose=1
    )
    model.save_weights(BC_WEIGHTS_PATH)
    print(f"BC weights saved → {BC_WEIGHTS_PATH}")
    return model


# ─────────────────────────────────────────────
# 3.  Custom PPO callback — load BC weights
#     into the RLlib policy on first iteration
# ─────────────────────────────────────────────

from ray.rllib.agents.callbacks import DefaultCallbacks

class WarmStartCallback(DefaultCallbacks):
    """Load BC weights into the RLlib policy before the first SGD update."""

    _loaded = False   # class-level flag so we only load once
    def on_episode_end(self, worker, base_env,policies,episode, **kwargs):

        # Get the unwrapped environment
        env = base_env.get_unwrapped()[0]

        if env.last_time_step != float('inf'):
            makespan = env.last_time_step
            episode.custom_metrics['make_span'] = makespan

            # Access the BKS we've attached to the environment instance
            if hasattr(env, 'current_bks') and env.current_bks > 0:
                gap = ((makespan - env.current_bks) / env.current_bks) * 100
                episode.custom_metrics['optimality_gap'] = gap

    def on_train_result(self, *, trainer, result, **kwargs):
        if not WarmStartCallback._loaded and os.path.exists(BC_WEIGHTS_PATH):
            print("⟳  Loading BC warm-start weights into PPO policy …")
            policy  = trainer.get_policy()
            weights = policy.get_weights()

            # Build a temporary BC model to load the saved weights
            env = gym.make(ENV_ID, env_config={"instance_path": INSTANCE_PATH})
            obs = env.reset()
            obs_shape = (obs["real_obs"].shape if isinstance(obs, dict)
                         else obs.shape)
            n_actions = env.action_space.n
            env.close()

            bc_model = build_bc_model(obs_shape, n_actions)
            bc_model.load_weights(BC_WEIGHTS_PATH)

            # Map BC dense layer weights → RLlib policy fc_net weights
            bc_vars    = {v.name: v.numpy() for v in bc_model.trainable_variables}
            new_weights = {}
            for k, v in weights.items():
                # Match by shape as a simple heuristic
                for bc_name, bc_val in bc_vars.items():
                    if bc_val.shape == v.shape:
                        new_weights[k] = bc_val
                        break
                else:
                    new_weights[k] = v
            if env.last_time_step != float('inf'):
                makespan = env.last_time_step
                MultiAgentEpisode.custom_metrics['make_span'] = makespan
                
                # Access the BKS we've attached to the environment instance
                if hasattr(env, 'current_bks') and env.current_bks > 0:
                    gap = ((makespan - env.current_bks) / env.current_bks) * 100
                    MultiAgentEpisode.custom_metrics['optimality_gap'] = gap
            policy.set_weights(new_weights)
            WarmStartCallback._loaded = True
            print("✓  BC weights loaded.")


# ─────────────────────────────────────────────
# 4.  Reward function (ta72-specific)
# ─────────────────────────────────────────────
with open(args.bks, 'r') as f:
        bks_map = json.load(f)
instance_key = os.path.basename(INSTANCE_PATH).replace(".txt", "")
        # Set BKS for the current instance (used by CustomCallbacks)
if isinstance(bks_map, dict):
    current_bks = bks_map.get(instance_key, 1.0)
else:
    current_bks = float(bks_map)

TA72_OPTIMAL = current_bks

with open(INSTANCE_PATH, "r") as f:
    tokens = f.read().split()
NUM_MACHINES = int(tokens[1])
TOTAL_OPS    = int(tokens[0]) * NUM_MACHINES


def compute_reward(env, action, done):
    reward = 0.0

    # Dense: makespan lower-bound improvement
    current_lb = env._compute_makespan_lower_bound()
    delta      = env.prev_makespan_lb - current_lb
    if delta > 0:
        reward += (delta / TA72_OPTIMAL) * 2.0
    env.prev_makespan_lb = current_lb

    # Dense: machine utilisation
    idle = sum(1 for m in env.machines if not m.is_busy)
    reward -= (idle / NUM_MACHINES) * 0.02

    # Dense: critical path
    if env._action_on_critical_path(action):
        reward += 0.03

    # Dense: progress pacing
    jobs_done    = env._get_jobs_scheduled()
    expected_t   = (jobs_done / TOTAL_OPS) * TA72_OPTIMAL
    current_t    = env._get_current_time()
    if current_t <= expected_t * 1.05:
        reward += 0.01

    # Terminal
    if done:
        final_makespan = env._get_makespan()
        gap    = (final_makespan - TA72_OPTIMAL) / TA72_OPTIMAL
        reward += 20.0 * max(0.0, 1.0 - gap)

    return reward


# ─────────────────────────────────────────────
# 5.  PPO config (mirrors your original config)
# ─────────────────────────────────────────────
ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)
PPO_CONFIG = {
    "env": ENV_ID,
    "env_config": {"instance_path": INSTANCE_PATH},

    # Architecture
    "model": {
        "custom_model": "fc_masked_model_tf",
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        "vf_share_layers": False,
    },

    # PPO hyperparams (from your config)
    "gamma":              1.0,
    "lambda":             1.0,
    "clip_param":         0.3,       # tightened for dense rewards
    "kl_coeff":           0.5,
    "entropy_coeff":      0.002,
    "entropy_coeff_schedule": [[0, 0.002], [1_000_000, 0.00025]],
    "vf_loss_coeff":      0.5,       # reduced for dense rewards
    "use_gae":            True,
    "use_critic":         True,

    # Sampling
    "num_workers":            20,
    "num_envs_per_worker":    4,
    "rollout_fragment_length": 704,
    "train_batch_size":       12000,
    "sgd_minibatch_size":     128,
    "num_sgd_iter":           10,
    "batch_mode":             "truncate_episodes",

    # LR schedule
    "lr": 0.00066,
    "lr_schedule": [[0, 0.00066], [1_000_000, 7.8e-05]],

    # Resources
    "num_gpus": 0,
    "framework": "tf",

    # Warm-start callback
    "callbacks": WarmStartCallback,
}


# ─────────────────────────────────────────────
# 6.  Main — BC then PPO
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: collect SPT demonstrations ──
    print("\n═══ Step 1: Collecting SPT demonstrations ═══")
    demos = collect_demonstrations(DEMO_EPISODES)
    print(f"Total transitions collected: {len(demos)}")

    # ── Step 2: behaviour cloning ──
    print("\n═══ Step 2: Behaviour Cloning ═══")
    env_tmp   = gym.make(ENV_ID, env_config={"instance_path": INSTANCE_PATH})
    obs_tmp   = env_tmp.reset()
    obs_shape = (obs_tmp["real_obs"].shape if isinstance(obs_tmp, dict)
                 else obs_tmp.shape)
    n_actions = env_tmp.action_space.n
    env_tmp.close()

    bc_model = behaviour_cloning(demos, obs_shape, n_actions)

    # ── Step 3: PPO with warm-start ──
    print("\n═══ Step 3: PPO fine-tuning ═══")
    ray.init(ignore_reinit_error=True)

    # Define how RLlib should create your environment
    def env_creator(env_config):
        # We pass the env_config dict exactly as your environment expects it
        return gym.make(ENV_ID, env_config=env_config)

    # Register this creator with a custom name
    register_env("custom_jss_env", env_creator)

    # Update your PPO config to use the new registered name
    PPO_CONFIG["env"] = "custom_jss_env"

    # Ensure your paths are set up
    csv_file_path = os.path.join(CHECKPOINT_DIR, "results.csv")

    tune.run(
        PPOTrainer,
        config=PPO_CONFIG,
        
        # 1. Use your custom rolling window stopper
        stop=OptimalityGapStopper(window_size=5, target_gap=20.0, max_iters=args.iters),
        
        # 2. Add your custom CSV logger
        callbacks=[CustomCSVLoggerCallback(csv_file_path)],
        
        # 3. Handle saving checkpoints automatically every 25 iters and at the end
        checkpoint_freq=25,
        checkpoint_at_end=True,
        keep_checkpoints_num=None, # Set to an int (e.g., 5) to save disk space
        
        local_dir=CHECKPOINT_DIR,
        name="jss_"+os.path.basename(INSTANCE_PATH)+"_warmstart_ppo",
        verbose=1,
    )
    ray.shutdown()

