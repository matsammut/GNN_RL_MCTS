#!/usr/bin/env python3
"""
train_dqn_taillard_legacy.py

Standalone script to run DQN (Ray rllib 1.1.0) on Taillard instances:
instances/ta42, instances/ta52, instances/ta62, instances/ta72

Compatible with: Python 3.8.20, ray==1.1.0, tensorflow==2.2.1, numpy==1.18.5, wandb==0.10.22

Usage:
    python3 train_dqn_taillard_legacy.py --iters 200 --eval_episodes 10

Outputs:
    dqn_legacy_results/<instance>/train_log.csv
    dqn_legacy_results/<instance>/eval_results.csv
    and saved trainer checkpoint paths printed to stdout
"""
import os
import time
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import gym
from gym import spaces
import multiprocessing as mp
# Ray 1.1.0 imports
import ray
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_CONFIG
from ray.tune.registry import register_env

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# -----------------------------
# Simple Taillard JSSP Gym Env
# -----------------------------
class TaillardJSSPEnv(gym.Env):
    """
    Minimal Taillard Job Shop Scheduling environment.

    - Action: Discrete(num_jobs) -> select a job to schedule its next operation.
    - Observation: np.float32 vector length = num_jobs + num_machines
      [ job_next_op_index_norm (0..1) for each job ; machine_end_time_norm (0..1) for each machine ]
    - Reward: 0 during episode, final step returns -makespan as reward and info['makespan'].
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config):
        super().__init__()
        # env_config should contain "instance_path"
        instance_path = env_config.get("instance_path")
        if instance_path is None:
            raise ValueError("env_config must include 'instance_path' (relative path to Taillard file)")

        # load taillard instance
        self.instance_path = Path(instance_path)
        if not self.instance_path.exists():
            raise FileNotFoundError(f"Taillard instance not found: {self.instance_path.resolve()}")

        self.jobs, self.n_jobs, self.n_machines = self._parse_taillard(self.instance_path)
        # action: choose job index [0..n_jobs-1]
        self.action_space = spaces.Discrete(self.n_jobs)
        # observation vector: n_jobs + n_machines
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_jobs + self.n_machines,), dtype=np.float32)

        # internal state
        self.reset()

    def _parse_taillard(self, path):
        """
        Expect a file where the first line contains: n_jobs n_machines
        Then each following line (one per job) contains 2*m integers: machine duration machine duration ...
        (This is consistent with some Taillard formats).
        """
        with open(path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        header = lines[0].split()
        n_jobs = int(header[0])
        n_machines = int(header[1])
        jobs = []
        for j in range(1, 1 + n_jobs):
            toks = list(map(int, lines[j].split()))
            if len(toks) != 2 * n_machines:
                # try alternative flatten reading: sometimes file is a stream of ints
                all_ints = []
                for ln in lines[1:]:
                    all_ints += list(map(int, ln.split()))
                # rebuild
                jobs = []
                idx = 0
                for jj in range(n_jobs):
                    ops = []
                    for mm in range(n_machines):
                        machine = all_ints[idx]; dur = all_ints[idx+1]; idx += 2
                        ops.append((machine, dur))
                    jobs.append(ops)
                return jobs, n_jobs, n_machines
            ops = []
            for m in range(n_machines):
                machine = toks[2*m]
                duration = toks[2*m + 1]
                ops.append((machine, duration))
            jobs.append(ops)
        return jobs, n_jobs, n_machines

    def reset(self):
        # job next operation index
        self.job_next = [0] * self.n_jobs
        # job completion times (last op end)
        self.job_end = [0] * self.n_jobs
        # machine availability times
        # Taillard machines may be 0..M-1 or 1..M; map to 0..M-1 if necessary
        self.machine_end = [0] * self.n_machines
        self.done_jobs = [False] * self.n_jobs
        self.total_duration_estimate = sum(sum(d for (_, d) in js) for js in self.jobs)
        return self._get_obs()

    def _get_obs(self):
        # normalize job_next by n_machines; machine_end by total_duration_estimate+1
        job_part = np.array([self.job_next[j] / float(self.n_machines) for j in range(self.n_jobs)], dtype=np.float32)
        denom = max(1.0, self.total_duration_estimate)
        mach_part = np.array([self.machine_end[m] / float(denom) for m in range(self.n_machines)], dtype=np.float32)
        obs = np.concatenate([job_part, mach_part], axis=0)
        return obs

    def step(self, action):
        # action = job index
        info = {}
        if action < 0 or action >= self.n_jobs:
            # invalid action
            return self._get_obs(), -10.0, False, info

        if self.done_jobs[action]:
            # picking a finished job is invalid; penalize slightly and continue
            return self._get_obs(), -1.0, False, info

        j = action
        op_idx = self.job_next[j]
        machine, duration = self.jobs[j][op_idx]
        # if machine index in file is 1-based, adjust gracefully
        if machine >= self.n_machines:
            # try adjust by -1 if values are 1..M
            if machine <= self.n_machines:
                machine = machine - 1
        start = max(self.job_end[j], self.machine_end[machine])
        finish = start + duration

        # update
        self.job_next[j] += 1
        self.job_end[j] = finish
        self.machine_end[machine] = finish
        if self.job_next[j] >= self.n_machines:
            self.done_jobs[j] = True

        done = all(self.done_jobs)
        if done:
            makespan = max(self.job_end)
            reward = -float(makespan)
            info["makespan"] = float(makespan)
            return self._get_obs(), reward, True, info
        else:
            # intermediate step reward 0 (sparse)
            return self._get_obs(), 0.0, False, info

    def render(self, mode="human"):
        print("job_next:", self.job_next)
        print("machine_end:", self.machine_end)
        print("job_end:", self.job_end)

# -----------------------------
# Training driver
# -----------------------------
DEFAULT_INSTANCES = ["instances/ta52", "instances/ta62", "instances/ta72"]
RESULTS_DIR = "dqn_legacy_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_instance(instance_path, iters=200, eval_episodes=10, use_wandb=True):
    instance_name = Path(instance_path).name
    outdir = Path(RESULTS_DIR) / instance_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Register env factory for Ray workers, use env_config to pass path
    def _creator(env_config):
        # env_config will be {'instance_path': instance_path}
        return TaillardJSSPEnv(env_config)

    register_env("TaillardJSSP", lambda cfg: TaillardJSSPEnv(cfg))

    # Build config for DQN (Ray 1.1.0 compatible)
    cfg = DQN_CONFIG.copy()
    cfg.update({
        "env": "TaillardJSSP",
        "env_config": {"instance_path": instance_path},
        "num_workers": mp.cpu_count(),   # set to 0 for simplicity; increase if you want parallelism
        "num_gpus": 1,
        "framework": "tf",
        "train_batch_size": 4000,
        "learning_starts": 1000,
        "buffer_size": 200000,
        "target_network_update_freq": 500,
        "lr": 1e-4,
        "gamma": 0.99,
        "n_step": 3,
        "dueling": True,
        "double_q": True,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.05,
            "epsilon_timesteps": 50000,
        },
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        }
    })

    # optional wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project="DQN_JSSP_legacy", name=f"DQN_{instance_name}", reinit=True, config=cfg)
        # wandb.config.update(cfg)

    # init ray and trainer
    ray.init(ignore_reinit_error=True)
    trainer = DQNTrainer(config=cfg)

    train_log = []
    print(f"Starting training DQN on {instance_name} for {iters} iterations...")
    for it in range(iters):
        res = trainer.train()
        mean_reward = res.get("episode_reward_mean", float("nan"))
        timesteps_total = res.get("timesteps_total", 0)
        print(f"[{instance_name}] iter {it+1}/{iters} reward_mean={mean_reward} timesteps={timesteps_total}")
        train_log.append({"iter": it+1, "reward_mean": mean_reward, "timesteps_total": timesteps_total})
        if WANDB_AVAILABLE and use_wandb:
            wandb.log({"iter": it+1, "reward_mean": mean_reward, "timesteps_total": timesteps_total})
        # periodic checkpointing
        if (it+1) % 50 == 0:
            ckpt = trainer.save(outdir.as_posix())
            print("Saved checkpoint:", ckpt)

    final_ckpt = trainer.save(outdir.as_posix())
    print("Final checkpoint saved:", final_ckpt)
    pd.DataFrame(train_log).to_csv(outdir / "train_log.csv", index=False)

    # Evaluation: run deterministic episodes (explore=False)
    eval_rows = []
    for ep in range(eval_episodes):
        env = TaillardJSSPEnv({"instance_path": instance_path})
        obs = env.reset()
        done = False
        while not done:
            action = trainer.compute_action(obs, explore=False)
            obs, reward, done, info = env.step(action)
        makespan = info.get("makespan", None)
        eval_rows.append({"episode": ep+1, "makespan": makespan})
        print(f"Eval ep {ep+1}: makespan={makespan}")
        if WANDB_AVAILABLE and use_wandb:
            wandb.log({"eval_makespan": makespan, "instance": instance_name})

    pd.DataFrame(eval_rows).to_csv(outdir / "eval_results.csv", index=False)
    print(f"[{instance_name}] evaluation complete.")

    # Clean up trainer and Ray processes safely
    try:
        trainer.stop()
    except Exception as e:
        print(f"Warning: trainer.stop() failed for {instance_name}: {e}")

    # Close WandB outside Ray context
    if WANDB_AVAILABLE and use_wandb:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Warning: wandb.finish() failed: {e}")

    # Graceful Ray shutdown with retry
    import gc, time
    for _ in range(5):
        try:
            ray.shutdown()
            time.sleep(1.0)
            if not ray.is_initialized():
                break
        except Exception as e:
            print("Ray shutdown retry due to:", e)
            time.sleep(1.0)
    gc.collect()
    print(f"✅ Completed {instance_name}. Moving to next instance...\n")


def main():
    parser = argparse.ArgumentParser(description="Train DQN on Taillard JSSP instances (legacy Ray 1.1.0)")
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    parser.add_argument("--iters", type=int, default=3)   # short for testing
    parser.add_argument("--eval", type=int, default=2)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    for inst in args.instances:
        if not Path(inst).exists():
            print(f"⚠️  Instance not found: {inst}, skipping.")
            continue
        try:
            train_instance(inst, iters=args.iters, eval_episodes=args.eval, use_wandb=not args.no_wandb)
        except KeyboardInterrupt:
            print("⛔ Interrupted by user.")
            break
        except Exception as e:
            print(f"❌ Error while training {inst}: {e}")
        # small delay to ensure previous Ray workers are killed
        import time; time.sleep(3)



if __name__ == "__main__":
    import argparse
    main()
