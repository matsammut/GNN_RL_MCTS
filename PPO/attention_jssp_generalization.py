#!/usr/bin/env python3
"""
attention_jssp_generalization.py

Research experiment:
  "Can a single attention-based policy learn to schedule effectively
   across diverse 6×6 instances with varying machine orders and
   processing times?"

Pipeline:
  1. Generate N_TRAIN diverse random 6×6 JSSP instances
  2. Run GA on each → collect expert demonstrations via direct rollout
  3. Train a single attention-based BC policy on the combined dataset
  4. Evaluate on 500 unseen instances vs SPT, FIFO, LWKR baselines + per-instance GA

Outputs:
  eval_results.csv   — per-instance makespan for every method
  summary.json       — mean/std/min/max + gap-to-GA statistics
  attention_policy_weights — saved Keras weights

Usage:
  python3 attention_jssp_generalization.py
  python3 attention_jssp_generalization.py --n-train-instances 500 --bc-epochs 100
"""

import argparse
import csv
import json
import logging
import os
import random
import time
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

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
    description="Attention BC generalisation experiment on random 6×6 JSSP instances."
)
parser.add_argument("--out", type=str, default="attn_results",
                    help="Output directory (default: attn_results)")
parser.add_argument("--n-train-instances", type=int, default=200,
                    help="Number of training instances (default: 200)")
parser.add_argument("--n-eval-instances", type=int, default=500,
                    help="Number of unseen evaluation instances (default: 500)")
parser.add_argument("--proc-time-low", type=int, default=1,
                    help="Lower bound for processing time sampling (default: 1)")
parser.add_argument("--proc-time-high", type=int, default=99,
                    help="Upper bound for processing time sampling (default: 99)")
parser.add_argument("--ga-pop", type=int, default=30,
                    help="GA population size per instance (default: 30)")
parser.add_argument("--ga-gens", type=int, default=20,
                    help="GA generations per instance (default: 20)")
parser.add_argument("--bc-episodes-per-instance", type=int, default=10,
                    help="Demo episodes per training instance (default: 10)")
parser.add_argument("--bc-epsilon", type=float, default=0.05,
                    help="Epsilon-greedy noise in demos (default: 0.05)")
parser.add_argument("--bc-epochs", type=int, default=50,
                    help="BC training epochs (default: 50)")
parser.add_argument("--bc-batch", type=int, default=512,
                    help="BC batch size (default: 512)")
parser.add_argument("--embed-dim", type=int, default=128,
                    help="Attention embedding dimension (default: 128)")
parser.add_argument("--n-heads", type=int, default=4,
                    help="Number of attention heads (default: 4)")
parser.add_argument("--n-attn-layers", type=int, default=2,
                    help="Number of transformer encoder blocks (default: 2)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
args = parser.parse_args()

# Constants
N_JOBS     = 6
N_MACHINES = 6
# Per-job features: remaining_ops | next_proc_time | total_remaining |
#                   machine_wait  | job_wait       | is_legal        |
#                   one-hot next machine (N_MACHINES)
FEATURE_DIM = 6 + N_MACHINES

OUT_DIR = os.path.abspath(args.out)
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. JSSP Instance Generation
# ─────────────────────────────────────────────────────────────────────────────

class JSSPInstance:
    """
    A random JSSP instance.

    machine_order[j, k] = which machine handles job j's k-th operation.
    proc_time[j, k]     = processing time for job j's k-th operation.

    Both are varied independently across instances to prevent the policy
    from memorising fixed patterns.
    """
    __slots__ = ("n_jobs", "n_machines", "machine_order", "proc_time")

    def __init__(self, n_jobs: int, n_machines: int,
                 machine_order: np.ndarray, proc_time: np.ndarray):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.machine_order = machine_order   # (n_jobs, n_machines)  int
        self.proc_time = proc_time           # (n_jobs, n_machines)  int


def generate_instance(n_jobs: int = N_JOBS,
                       n_machines: int = N_MACHINES,
                       proc_low: int = 1,
                       proc_high: int = 99,
                       rng: np.random.Generator = None) -> JSSPInstance:
    """
    Sample a random JSSP instance.

    What varies:
      - machine_order: each job gets an independent random permutation of
        machine indices → no two jobs share the same routing.
      - proc_time: drawn uniformly from [proc_low, proc_high] per operation
        → wide variation in processing times across instances.
    """
    if rng is None:
        rng = np.random.default_rng()
    machine_order = np.array(
        [rng.permutation(n_machines) for _ in range(n_jobs)], dtype=np.int32
    )
    proc_time = rng.integers(proc_low, proc_high + 1,
                              size=(n_jobs, n_machines)).astype(np.int32)
    return JSSPInstance(n_jobs, n_machines, machine_order, proc_time)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lightweight JSSP Simulator
# ─────────────────────────────────────────────────────────────────────────────

class JSSPSimulator:
    """
    Step-by-step active-scheduling JSSP simulator.

    At each decision point, one or more jobs are "legal" — their next
    operation's required machine is idle and the job itself is free.
    Choosing a job dispatches its next operation immediately.
    If no jobs are legal, time auto-advances to the earliest moment
    at least one job becomes legal (no-op is never needed externally).
    """

    def __init__(self, inst: JSSPInstance):
        self.inst = inst
        self.reset()

    def reset(self) -> dict:
        self.time = 0.0
        self.job_op       = np.zeros(self.inst.n_jobs,     dtype=int)
        self.machine_free = np.zeros(self.inst.n_machines, dtype=float)
        self.job_free     = np.zeros(self.inst.n_jobs,     dtype=float)
        self._advance_time()
        return self._obs()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _legal_mask(self) -> np.ndarray:
        mask = np.zeros(self.inst.n_jobs, dtype=np.int32)
        for j in range(self.inst.n_jobs):
            if self.job_op[j] < self.inst.n_machines:
                m = self.inst.machine_order[j, self.job_op[j]]
                if (self.machine_free[m] <= self.time and
                        self.job_free[j]     <= self.time):
                    mask[j] = 1
        return mask

    def _advance_time(self):
        """Advance self.time until at least one job is legal (or all done)."""
        while True:
            if self._done() or self._legal_mask().sum() > 0:
                break
            # Earliest time any incomplete job can next be dispatched
            candidates = []
            for j in range(self.inst.n_jobs):
                if self.job_op[j] < self.inst.n_machines:
                    m = self.inst.machine_order[j, self.job_op[j]]
                    candidates.append(
                        max(self.machine_free[m], self.job_free[j])
                    )
            if not candidates:
                break
            self.time = min(candidates)

    def _done(self) -> bool:
        return bool((self.job_op == self.inst.n_machines).all())

    def _features(self) -> np.ndarray:
        """
        Per-job feature matrix of shape (n_jobs, FEATURE_DIM).

        Index  Feature
        ─────  ──────────────────────────────────────────
        0      remaining_ops / n_machines   (progress)
        1      next_proc_time / max_pt      (urgency)
        2      total_remaining_pt / max_total
        3      machine_wait / max_pt        (contention)
        4      job_wait / max_pt            (dependency)
        5      is_legal                     (binary)
        6..    one-hot encoding of next machine needed
        """
        max_pt    = max(float(self.inst.proc_time.max()), 1.0)
        max_total = max(float(self.inst.proc_time.sum(axis=1).max()), 1.0)
        mask      = self._legal_mask()
        feats     = np.zeros((self.inst.n_jobs, FEATURE_DIM), dtype=np.float32)

        for j in range(self.inst.n_jobs):
            op = self.job_op[j]
            if op < self.inst.n_machines:
                m  = self.inst.machine_order[j, op]
                pt = float(self.inst.proc_time[j, op])
                rt = float(self.inst.proc_time[j, op:].sum())

                feats[j, 0] = (self.inst.n_machines - op) / self.inst.n_machines
                feats[j, 1] = pt / max_pt
                feats[j, 2] = rt / max_total
                feats[j, 3] = max(0.0, self.machine_free[m] - self.time) / max_pt
                feats[j, 4] = max(0.0, self.job_free[j]     - self.time) / max_pt
                feats[j, 5] = float(mask[j])
                feats[j, 6 + m] = 1.0   # one-hot next machine

        return feats

    def _obs(self) -> dict:
        return {
            "features":    self._features(),    # (n_jobs, FEATURE_DIM)
            "action_mask": self._legal_mask(),  # (n_jobs,)  int32
            "done":        self._done(),
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def step(self, job: int) -> Tuple[dict, bool]:
        """Dispatch job `job`. Returns (next_obs, done)."""
        op = self.job_op[job]
        m  = self.inst.machine_order[job, op]
        pt = float(self.inst.proc_time[job, op])

        start  = max(self.machine_free[m], self.job_free[job], self.time)
        finish = start + pt
        self.machine_free[m] = finish
        self.job_free[job]   = finish
        self.job_op[job]    += 1

        self._advance_time()
        obs = self._obs()
        return obs, self._done()

    @property
    def makespan(self) -> float:
        return float(self.job_free.max())


# ─────────────────────────────────────────────────────────────────────────────
# 3. GA for Expert Demonstrations
# ─────────────────────────────────────────────────────────────────────────────

def _eval_perm(inst: JSSPInstance, perm: list) -> float:
    """Decode a permutation into a schedule and return makespan."""
    mfree = np.zeros(inst.n_machines, dtype=float)
    jfree = np.zeros(inst.n_jobs,     dtype=float)
    jop   = np.zeros(inst.n_jobs,     dtype=int)
    for job in perm:
        op = jop[job]
        if op >= inst.n_machines:
            continue
        m  = inst.machine_order[job, op]
        pt = float(inst.proc_time[job, op])
        start  = max(mfree[m], jfree[job])
        finish = start + pt
        mfree[m]  = finish
        jfree[job] = finish
        jop[job]  += 1
    return float(jfree.max())


def _rand_perm(n_jobs: int, n_machines: int) -> list:
    p = [j for j in range(n_jobs) for _ in range(n_machines)]
    random.shuffle(p)
    return p


def _ox(p1: list, p2: list) -> list:
    """Order crossover."""
    n  = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b] = p1[a:b]
    ptr = b
    for gene in (p2[b:] + p2[:b]):
        if None in child:
            while child[ptr % n] is not None:
                ptr += 1
            child[ptr % n] = gene
        else:
            break
    return child


def _mutate_perm(perm: list, rate: float = 0.1) -> list:
    perm = list(perm)
    n = len(perm)
    for i in range(n):
        if random.random() < rate:
            j = random.randint(0, n - 1)
            perm[i], perm[j] = perm[j], perm[i]
    return perm


def run_ga_on_instance(inst: JSSPInstance,
                        pop_size: int = 30,
                        generations: int = 20) -> Tuple[list, float]:
    """Run GA on one instance. Returns (best_perm, best_makespan)."""
    pop     = [_rand_perm(inst.n_jobs, inst.n_machines) for _ in range(pop_size)]
    fitness = [_eval_perm(inst, p) for p in pop]
    bi      = int(np.argmin(fitness))
    best    = deepcopy(pop[bi])
    best_ms = fitness[bi]

    for _ in range(generations):
        new_pop = [deepcopy(best)]
        while len(new_pop) < pop_size:
            # Tournament selection (k=2)
            i1, i2 = random.sample(range(pop_size), 2)
            p1 = pop[i1] if fitness[i1] < fitness[i2] else pop[i2]
            i3, i4 = random.sample(range(pop_size), 2)
            p2 = pop[i3] if fitness[i3] < fitness[i4] else pop[i4]
            child = _ox(p1, p2) if random.random() < 0.9 else deepcopy(p1)
            child = _mutate_perm(child, 0.1)
            new_pop.append(child)
        pop     = new_pop
        fitness = [_eval_perm(inst, p) for p in pop]
        bi      = int(np.argmin(fitness))
        if fitness[bi] < best_ms:
            best_ms = fitness[bi]
            best    = deepcopy(pop[bi])

    return best, best_ms


# ─────────────────────────────────────────────────────────────────────────────
# 4. Demonstration Collection
# ─────────────────────────────────────────────────────────────────────────────

def _rollout(inst: JSSPInstance,
             perm: list,
             epsilon: float = 0.05) -> Tuple[list, float]:
    """
    Replay a GA permutation through the simulator with epsilon-greedy noise.

    At each step:
      - With prob (1 - epsilon): dispatch the legal job whose next occurrence
        appears earliest in the remaining permutation queue.
      - With prob epsilon: dispatch a uniformly random legal job.

    Returns (list of (features, mask, action), episode_makespan).
    """
    remaining = list(perm)
    sim       = JSSPSimulator(inst)
    obs       = sim.reset()
    pairs     = []

    while not obs["done"]:
        legal = [j for j in range(inst.n_jobs) if obs["action_mask"][j]]
        if not legal:
            break

        if epsilon > 0 and random.random() < epsilon:
            action = random.choice(legal)
        else:
            def first_pos(job):
                for i, g in enumerate(remaining):
                    if g == job:
                        return i
                return float("inf")
            action = min(legal, key=first_pos)

        # Remove one occurrence of chosen job from remaining queue
        for i, g in enumerate(remaining):
            if g == action:
                remaining.pop(i)
                break

        pairs.append((
            obs["features"].copy(),        # (n_jobs, FEATURE_DIM)
            obs["action_mask"].copy(),     # (n_jobs,)
            action,                        # int
        ))
        obs, _ = sim.step(action)

    return pairs, sim.makespan


def collect_all_demos(instances: List[JSSPInstance],
                       episodes_per_instance: int,
                       ga_pop: int,
                       ga_gens: int,
                       epsilon: float) -> List[tuple]:
    """
    For each training instance:
      1. Run GA to get a diverse population of expert permutations.
      2. Replay the top half as experts (with epsilon noise) for
         `episodes_per_instance` episodes.
    Returns the combined shuffled demonstration dataset.
    """
    all_demos = []
    n_experts = max(1, ga_pop // 2)

    for idx, inst in enumerate(instances):
        # ── Run GA ────────────────────────────────────────────────────────
        pop     = [_rand_perm(inst.n_jobs, inst.n_machines) for _ in range(ga_pop)]
        fitness = [_eval_perm(inst, p) for p in pop]
        bi      = int(np.argmin(fitness))
        best    = deepcopy(pop[bi])
        best_ms = fitness[bi]

        for _ in range(ga_gens):
            new_pop = [deepcopy(best)]
            while len(new_pop) < ga_pop:
                i1, i2 = random.sample(range(ga_pop), 2)
                p1 = pop[i1] if fitness[i1] < fitness[i2] else pop[i2]
                i3, i4 = random.sample(range(ga_pop), 2)
                p2 = pop[i3] if fitness[i3] < fitness[i4] else pop[i4]
                child = _ox(p1, p2) if random.random() < 0.9 else deepcopy(p1)
                child = _mutate_perm(child, 0.1)
                new_pop.append(child)
            pop     = new_pop
            fitness = [_eval_perm(inst, p) for p in pop]
            bi      = int(np.argmin(fitness))
            if fitness[bi] < best_ms:
                best_ms = fitness[bi]
                best    = deepcopy(pop[bi])

        # Top half of population as experts
        ranked  = sorted(zip(fitness, pop), key=lambda x: x[0])
        experts = [p for _, p in ranked[:n_experts]]

        # ── Collect episodes ───────────────────────────────────────────────
        for ep in range(episodes_per_instance):
            perm   = experts[ep % n_experts]
            pairs, _ = _rollout(inst, perm, epsilon=epsilon)
            all_demos.extend(pairs)

        if (idx + 1) % 20 == 0:
            logger.info(
                f"  Demos from {idx+1}/{len(instances)} instances | "
                f"Total transitions: {len(all_demos):,}"
            )

    # Shuffle across instances so batches are not instance-homogeneous
    random.shuffle(all_demos)
    return all_demos


# ─────────────────────────────────────────────────────────────────────────────
# 5. Attention-Based Policy
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoderBlock(keras.layers.Layer):
    """
    Single transformer encoder block:
      MultiHeadAttention → Add & Norm → FFN → Add & Norm

    Applied over the n_jobs axis so each job attends to every other job,
    allowing the policy to reason about inter-job machine conflicts.
    """

    def __init__(self, embed_dim: int, n_heads: int, name: str = "encoder_block"):
        super().__init__(name=name)
        self.mha   = keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=embed_dim // n_heads
        )
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn   = keras.Sequential([
            keras.layers.Dense(embed_dim * 4, activation="relu"),
            keras.layers.Dense(embed_dim),
        ], name=f"{name}_ffn")
        self.drop  = keras.layers.Dropout(0.1)

    def call(self, x, training: bool = False):
        # Self-attention across jobs
        attn = self.mha(x, x, training=training)
        x    = self.norm1(x + self.drop(attn, training=training))
        # Position-wise FFN
        ffn  = self.ffn(x, training=training)
        x    = self.norm2(x + self.drop(ffn, training=training))
        return x


def build_attention_policy(n_jobs:      int = N_JOBS,
                            feature_dim: int = FEATURE_DIM,
                            embed_dim:   int = 128,
                            n_heads:     int = 4,
                            n_layers:    int = 2) -> keras.Model:
    """
    Attention-based dispatching policy.

    Inputs:
      job_features  — (batch, n_jobs, feature_dim)
      action_mask   — (batch, n_jobs)  1 = legal, 0 = illegal

    Architecture:
      Linear projection
        → N × TransformerEncoderBlock  (jobs attend to each other)
          → per-job Dense(1)           (scalar logit per job)
            → mask illegal actions

    Output:
      masked_logits — (batch, n_jobs)

    The attention mechanism is the key design choice: it allows the policy
    to condition each job's priority on the global state of all other jobs,
    rather than scoring jobs independently. This is what we claim enables
    generalisation across instances with different machine orders.
    """
    job_input  = keras.Input(shape=(n_jobs, feature_dim), name="job_features")
    mask_input = keras.Input(shape=(n_jobs,),             name="action_mask")

    # Project raw features to embedding space
    x = keras.layers.Dense(embed_dim, name="input_proj")(job_input)

    # Stack of transformer encoder blocks
    for i in range(n_layers):
        x = TransformerEncoderBlock(embed_dim, n_heads,
                                    name=f"encoder_{i}")(x)

    # Per-job scalar logit
    logits = keras.layers.Dense(1, name="logit_proj")(x)   # (batch, n_jobs, 1)
    logits = tf.squeeze(logits, axis=-1)                    # (batch, n_jobs)

    # Mask illegal actions with large negative value
    mask_f = tf.cast(mask_input, tf.float32)
    masked = logits + (1.0 - mask_f) * (-1e9)

    return keras.Model(
        inputs=[job_input, mask_input],
        outputs=masked,
        name="attention_policy"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. BC Training
# ─────────────────────────────────────────────────────────────────────────────

def train_bc(demos:    List[tuple],
             embed_dim: int,
             n_heads:   int,
             n_layers:  int,
             epochs:    int,
             batch_size: int) -> keras.Model:
    """
    Train the attention BC policy on the collected demonstrations.

    Inputs:
      demos — list of (features, mask, action) tuples across all instances

    The model sees (features, mask) → predicts which job to dispatch.
    Training on demos from 200 diverse instances forces it to learn a
    generalizable dispatching heuristic rather than instance-specific cues.
    """
    features_arr = np.array([d[0] for d in demos], dtype=np.float32)  # (N, n_jobs, feat_dim)
    masks_arr    = np.array([d[1] for d in demos], dtype=np.float32)  # (N, n_jobs)
    actions_arr  = np.array([d[2] for d in demos], dtype=np.int64)    # (N,)

    logger.info(
        f"  BC dataset: {len(features_arr):,} transitions | "
        f"feature shape: {features_arr.shape[1:]} | "
        f"n_actions: {N_JOBS}"
    )

    unique, counts = np.unique(actions_arr, return_counts=True)
    logger.info("  Action distribution: " +
                ", ".join(f"job {j}: {100*c/len(actions_arr):.1f}%"
                          for j, c in zip(unique, counts)))

    model = build_attention_policy(
        n_jobs=N_JOBS, feature_dim=FEATURE_DIM,
        embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers
    )
    model.summary(print_fn=logger.info)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        [features_arr, masks_arr], actions_arr,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
            ),
        ],
        verbose=1,
    )

    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
    final_val_acc = history.history[val_acc_key][-1]
    epochs_trained = len(history.history["loss"])
    random_chance  = 1.0 / N_JOBS

    logger.info(
        f"  BC complete ({epochs_trained}/{epochs} epochs) | "
        f"Val acc: {final_val_acc:.4f} | "
        f"Random chance: {random_chance:.4f}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 7. Dispatching Baselines
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(inst: JSSPInstance, rule: str) -> float:
    """
    Apply a simple dispatching rule and return the makespan.

    SPT  — Shortest Processing Time (next operation's processing time)
    FIFO — First In First Out (lowest job index among legal jobs)
    LWKR — Least Work Remaining (sum of all remaining operation times)
    """
    sim = JSSPSimulator(inst)
    obs = sim.reset()

    while not obs["done"]:
        legal = [j for j in range(inst.n_jobs) if obs["action_mask"][j]]
        if not legal:
            break

        op_idx = sim.job_op  # reference, not copy — read-only
        if rule == "SPT":
            action = min(legal, key=lambda j: inst.proc_time[j, op_idx[j]])
        elif rule == "FIFO":
            action = min(legal)
        elif rule == "LWKR":
            action = min(legal, key=lambda j: inst.proc_time[j, op_idx[j]:].sum())
        else:
            raise ValueError(f"Unknown rule: {rule}")

        obs, _ = sim.step(action)

    return sim.makespan


def run_attention(model: keras.Model, inst: JSSPInstance) -> float:
    """Run the trained attention policy greedily on one instance."""
    sim = JSSPSimulator(inst)
    obs = sim.reset()

    while not obs["done"]:
        features = obs["features"][np.newaxis].astype(np.float32)         # (1, n_jobs, feat_dim)
        mask     = obs["action_mask"][np.newaxis].astype(np.float32)      # (1, n_jobs)
        logits   = model([features, mask], training=False).numpy()[0]     # (n_jobs,)
        action   = int(np.argmax(logits))
        obs, _   = sim.step(action)

    return sim.makespan


# ─────────────────────────────────────────────────────────────────────────────
# 8. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model:         keras.Model,
             eval_instances: List[JSSPInstance],
             ga_pop:         int,
             ga_gens:        int) -> dict:
    """
    Evaluate all methods on every eval instance.

    Methods:
      attn  — trained attention policy (single model, zero-shot on unseen instances)
      spt   — SPT dispatching rule
      fifo  — FIFO dispatching rule
      lwkr  — LWKR dispatching rule
      ga    — GA run specifically on each eval instance (per-instance optimal reference)
    """
    n = len(eval_instances)
    results = {m: np.zeros(n) for m in ["attn", "spt", "fifo", "lwkr", "ga"]}

    for i, inst in enumerate(eval_instances):
        results["attn"][i] = run_attention(model, inst)
        results["spt"][i]  = run_baseline(inst, "SPT")
        results["fifo"][i] = run_baseline(inst, "FIFO")
        results["lwkr"][i] = run_baseline(inst, "LWKR")
        _, results["ga"][i] = run_ga_on_instance(inst, ga_pop, ga_gens)

        if (i + 1) % 50 == 0:
            logger.info(
                f"  Evaluated {i+1}/{n} | "
                f"Attn: {results['attn'][:i+1].mean():.1f} | "
                f"SPT:  {results['spt'][:i+1].mean():.1f} | "
                f"FIFO: {results['fifo'][:i+1].mean():.1f} | "
                f"LWKR: {results['lwkr'][:i+1].mean():.1f} | "
                f"GA:   {results['ga'][:i+1].mean():.1f}"
            )

    return results


def report_results(results: dict, out_dir: str):
    """Print summary table, save CSV and JSON."""
    ga   = results["ga"]
    spt  = results["spt"]
    n    = len(ga)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Evaluation Results — 500 unseen random 6×6 JSSP instances")
    logger.info("=" * 70)
    logger.info(f"  {'Method':<12} {'Mean':>8} {'Std':>8} {'vs GA':>10} {'vs SPT':>10}")
    logger.info("  " + "-" * 52)

    labels = {"attn": "Attention", "spt": "SPT",
              "fifo": "FIFO",      "lwkr": "LWKR", "ga": "GA"}

    for method in ["attn", "spt", "fifo", "lwkr", "ga"]:
        ms     = results[method]
        vs_ga  = 100.0 * (ms.mean() - ga.mean())  / ga.mean()
        vs_spt = 100.0 * (ms.mean() - spt.mean()) / spt.mean()
        logger.info(
            f"  {labels[method]:<12} {ms.mean():>8.1f} {ms.std():>8.1f} "
            f"{vs_ga:>+9.2f}% {vs_spt:>+9.2f}%"
        )

    logger.info("=" * 70)

    # Per-instance gap to GA (primary claim metric)
    gap = 100.0 * (results["attn"] - ga) / ga
    logger.info(f"\n  Attention optimality gap vs per-instance GA:")
    logger.info(f"    mean:   {gap.mean():.2f}%")
    logger.info(f"    median: {np.median(gap):.2f}%")
    logger.info(f"    std:    {gap.std():.2f}%  ← consistency measure")
    logger.info(f"    max:    {gap.max():.2f}%")
    logger.info(f"    % within 5% of GA: {100*(gap<=5).sum()/n:.1f}%")
    logger.info(f"    % within 10% of GA: {100*(gap<=10).sum()/n:.1f}%")

    # Win rates vs baselines
    logger.info("")
    for baseline in ["spt", "fifo", "lwkr"]:
        wins = int((results["attn"] <= results[baseline]).sum())
        logger.info(f"  Attention beats {baseline.upper()}: "
                    f"{wins}/{n} ({100*wins/n:.1f}%)")

    logger.info("=" * 70)

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["instance", "attn", "spt", "fifo", "lwkr", "ga",
                           "attn_gap_to_ga_pct"]
        )
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                "instance": i,
                "attn":  f"{results['attn'][i]:.0f}",
                "spt":   f"{results['spt'][i]:.0f}",
                "fifo":  f"{results['fifo'][i]:.0f}",
                "lwkr":  f"{results['lwkr'][i]:.0f}",
                "ga":    f"{results['ga'][i]:.0f}",
                "attn_gap_to_ga_pct": f"{100*(results['attn'][i]-results['ga'][i])/results['ga'][i]:.2f}",
            })
    logger.info(f"\n  Per-instance results → {csv_path}")

    # ── Save JSON summary ─────────────────────────────────────────────────
    summary: dict = {}
    for method in ["attn", "spt", "fifo", "lwkr", "ga"]:
        ms = results[method]
        summary[method] = {
            "mean": round(float(ms.mean()), 2),
            "std":  round(float(ms.std()),  2),
            "min":  round(float(ms.min()),  2),
            "max":  round(float(ms.max()),  2),
        }
    summary["attn_gap_to_ga_pct"] = {
        "mean":            round(float(gap.mean()),        2),
        "median":          round(float(np.median(gap)),    2),
        "std":             round(float(gap.std()),         2),
        "max":             round(float(gap.max()),         2),
        "pct_within_5":   round(float(100*(gap<=5).sum()/n),  1),
        "pct_within_10":  round(float(100*(gap<=10).sum()/n), 1),
    }
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Summary → {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info("Attention BC Generalisation — Random 6×6 JSSP")
    logger.info("=" * 70)
    logger.info(f"  Training instances:      {args.n_train_instances}")
    logger.info(f"  Eval instances:          {args.n_eval_instances}")
    logger.info(f"  Processing times:        [{args.proc_time_low}, {args.proc_time_high}]")
    logger.info(f"  GA pop / gens:           {args.ga_pop} / {args.ga_gens}")
    logger.info(f"  Episodes per instance:   {args.bc_episodes_per_instance}")
    logger.info(f"  BC epsilon:              {args.bc_epsilon}")
    logger.info(f"  BC epochs:               {args.bc_epochs}")
    logger.info(f"  Attention embed_dim:     {args.embed_dim}")
    logger.info(f"  Attention heads:         {args.n_heads}")
    logger.info(f"  Attention layers:        {args.n_attn_layers}")
    logger.info(f"  Seed:                    {args.seed}")
    logger.info(f"  Output dir:              {OUT_DIR}")
    logger.info("=" * 70)

    # ── 1. Generate instances ──────────────────────────────────────────────
    logger.info("\n═══ Generating Instances ═══")
    # Separate RNG seeds ensure eval instances are truly unseen during training
    train_rng = np.random.default_rng(args.seed)
    eval_rng  = np.random.default_rng(args.seed + 10000)

    train_instances = [
        generate_instance(N_JOBS, N_MACHINES,
                          args.proc_time_low, args.proc_time_high, train_rng)
        for _ in range(args.n_train_instances)
    ]
    eval_instances = [
        generate_instance(N_JOBS, N_MACHINES,
                          args.proc_time_low, args.proc_time_high, eval_rng)
        for _ in range(args.n_eval_instances)
    ]
    logger.info(f"  {len(train_instances)} training instances (seed={args.seed})")
    logger.info(f"  {len(eval_instances)} eval instances (seed={args.seed + 10000}  → truly unseen)")

    # ── 2. Collect demonstrations ──────────────────────────────────────────
    logger.info("\n═══ Stage 1: Collecting GA Demonstrations ═══")
    t0 = time.time()
    demos = collect_all_demos(
        train_instances,
        episodes_per_instance=args.bc_episodes_per_instance,
        ga_pop=args.ga_pop,
        ga_gens=args.ga_gens,
        epsilon=args.bc_epsilon,
    )
    logger.info(
        f"  {len(demos):,} transitions from {args.n_train_instances} instances "
        f"in {time.time()-t0:.1f}s"
    )

    # ── 3. Train attention BC policy ───────────────────────────────────────
    logger.info("\n═══ Stage 2: Training Attention BC Policy ═══")
    model = train_bc(
        demos,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_attn_layers,
        epochs=args.bc_epochs,
        batch_size=args.bc_batch,
    )
    weights_path = os.path.join(OUT_DIR, "attention_policy_weights")
    model.save_weights(weights_path, save_format="h5")
    logger.info(f"  Weights saved → {weights_path}")

    # ── 4. Evaluate ────────────────────────────────────────────────────────
    logger.info("\n═══ Stage 3: Evaluation on Unseen Instances ═══")
    t0 = time.time()
    results = evaluate(
        model, eval_instances,
        ga_pop=args.ga_pop, ga_gens=args.ga_gens
    )
    logger.info(f"  Evaluation complete in {time.time()-t0:.1f}s")
    report_results(results, OUT_DIR)

    logger.info("\n" + "=" * 70)
    logger.info("Experiment complete.")
    logger.info(f"  Results: {OUT_DIR}")
    logger.info("=" * 70)
