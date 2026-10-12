#!/usr/bin/env python3
"""
Research experiment:
  "Can a single attention-based policy learn to schedule effectively
   across diverse NxN instances with varying machine orders and
   processing times?"

This script ensures training and evaluation are performed on the
same problem size, derived from the provided instance.

Outputs:
  run.log, attention_policy_weights, eval_results.csv, summary.json
"""

import argparse
import csv
import json
import logging
import os
import random
import time
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default="attn_results")
parser.add_argument("--instances", type=str, required=True,
                    help="Path to instance file to determine N_JOBS and N_MACHINES")
parser.add_argument("--n-train-instances", type=int, default=200)
parser.add_argument("--n-eval-instances", type=int, default=500)
parser.add_argument("--proc-time-low", type=int, default=1)
parser.add_argument("--proc-time-high", type=int, default=99)
parser.add_argument("--ga-pop", type=int, default=30)
parser.add_argument("--ga-gens", type=int, default=20)
parser.add_argument("--bc-episodes-per-instance", type=int, default=10)
parser.add_argument("--bc-epsilon", type=float, default=0.05)
parser.add_argument("--bc-epochs", type=int, default=50)
parser.add_argument("--bc-batch", type=int, default=512)
parser.add_argument("--embed-dim", type=int, default=128)
parser.add_argument("--n-heads", type=int, default=4)
parser.add_argument("--n-attn-layers", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Read dimensions from the instance file
with open(args.instances, "r") as f:
    tokens = f.read().split()
N_JOBS = int(tokens[0])
N_MACHINES = int(tokens[1])
FEATURE_DIM = 6 + N_MACHINES

OUT_DIR = os.path.abspath(args.out)
os.makedirs(OUT_DIR, exist_ok=True)

_log_path = os.path.join(OUT_DIR, "run.log")
logging.basicConfig(level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()
_file_handler = logging.FileHandler(_log_path, mode="w")
logger.addHandler(_file_handler)
logger.info(f"Logging to: {_log_path}")
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# Problem / generation logic as before — uses dynamic N_JOBS/N_MACHINES everywhere
class JSSPInstance:
    __slots__ = ("n_jobs", "n_machines", "machine_order", "proc_time")
    def __init__(self, n_jobs, n_machines, machine_order, proc_time):
        self.n_jobs        = n_jobs
        self.n_machines    = n_machines
        self.machine_order = machine_order
        self.proc_time     = proc_time

def generate_instance(n_jobs, n_machines, proc_low=1, proc_high=99, rng=None):
    if rng is None: rng = np.random.default_rng()
    machine_order = np.array([rng.permutation(n_machines) for _ in range(n_jobs)], dtype=np.int32)
    proc_time     = rng.integers(proc_low, proc_high + 1, size=(n_jobs, n_machines)).astype(np.int32)
    return JSSPInstance(n_jobs, n_machines, machine_order, proc_time)

class JSSPSimulator:
    def __init__(self, inst): self.inst = inst; self.reset()
    def reset(self):
        self.time = 0.0
        self.job_op  = np.zeros(self.inst.n_jobs, dtype=int)
        self.machine_free = np.zeros(self.inst.n_machines, dtype=float)
        self.job_free    = np.zeros(self.inst.n_jobs, dtype=float)
        self._advance_time()
        return self._obs()
    def _legal_mask(self):
        mask = np.zeros(self.inst.n_jobs, dtype=np.int32)
        for j in range(self.inst.n_jobs):
            if self.job_op[j] < self.inst.n_machines:
                m = self.inst.machine_order[j, self.job_op[j]]
                if (self.machine_free[m] <= self.time and self.job_free[j] <= self.time):
                    mask[j] = 1
        return mask
    def _advance_time(self):
        while True:
            if self._done() or self._legal_mask().sum() > 0:
                break
            candidates = []
            for j in range(self.inst.n_jobs):
                if self.job_op[j] < self.inst.n_machines:
                    m = self.inst.machine_order[j, self.job_op[j]]
                    candidates.append(max(self.machine_free[m], self.job_free[j]))
            if not candidates: break
            self.time = min(candidates)
    def _done(self): return bool((self.job_op == self.inst.n_machines).all())
    def _features(self):
        n_m = self.inst.n_machines
        max_pt    = max(float(self.inst.proc_time.max()), 1.0)
        max_total = max(float(self.inst.proc_time.sum(axis=1).max()), 1.0)
        mask      = self._legal_mask()
        feat_dim  = 6 + n_m
        feats     = np.zeros((self.inst.n_jobs, feat_dim), dtype=np.float32)
        for j in range(self.inst.n_jobs):
            op = self.job_op[j]
            if op < n_m:
                m  = self.inst.machine_order[j, op]
                pt = float(self.inst.proc_time[j, op])
                rt = float(self.inst.proc_time[j, op:].sum())
                feats[j, 0] = (n_m - op) / n_m
                feats[j, 1] = pt / max_pt
                feats[j, 2] = rt / max_total
                feats[j, 3] = max(0.0, self.machine_free[m] - self.time) / max_pt
                feats[j, 4] = max(0.0, self.job_free[j]     - self.time) / max_pt
                feats[j, 5] = float(mask[j])
                feats[j, 6 + m] = 1.0
        return feats
    def _obs(self):
        return dict(features=self._features(), action_mask=self._legal_mask(), done=self._done())
    def step(self, job):
        op = self.job_op[job]
        m  = self.inst.machine_order[job, op]
        pt = float(self.inst.proc_time[job, op])
        start  = max(self.machine_free[m], self.job_free[job], self.time)
        finish = start + pt
        self.machine_free[m] = finish
        self.job_free[job]   = finish
        self.job_op[job]    += 1
        self._advance_time()
        return self._obs(), self._done()
    @property
    def makespan(self): return float(self.job_free.max())

# GA for expert demos
def _eval_perm(inst, perm):
    mfree = np.zeros(inst.n_machines, dtype=float)
    jfree = np.zeros(inst.n_jobs,     dtype=float)
    jop   = np.zeros(inst.n_jobs,     dtype=int)
    for job in perm:
        op = jop[job]
        if op >= inst.n_machines: continue
        m  = inst.machine_order[job, op]
        pt = float(inst.proc_time[job, op])
        start  = max(mfree[m], jfree[job])
        finish = start + pt
        mfree[m]   = finish
        jfree[job] = finish
        jop[job]  += 1
    return float(jfree.max())

def _rand_perm(n_jobs, n_machines):
    p = [j for j in range(n_jobs) for _ in range(n_machines)]
    random.shuffle(p)
    return p

def _ox(p1, p2):
    n    = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b] = p1[a:b]
    ptr = b
    for gene in (p2[b:] + p2[:b]):
        if None in child:
            while child[ptr % n] is not None: ptr += 1
            child[ptr % n] = gene
        else: break
    return child

def _mutate_perm(perm, rate=0.1):
    perm = list(perm)
    n    = len(perm)
    for i in range(n):
        if random.random() < rate:
            j = random.randint(0, n - 1)
            perm[i], perm[j] = perm[j], perm[i]
    return perm

def run_ga_on_instance(inst, pop_size=30, generations=20):
    pop     = [_rand_perm(inst.n_jobs, inst.n_machines) for _ in range(pop_size)]
    fitness = [_eval_perm(inst, p) for p in pop]
    bi      = int(np.argmin(fitness))
    best    = deepcopy(pop[bi])
    best_ms = fitness[bi]
    for _ in range(generations):
        new_pop = [deepcopy(best)]
        while len(new_pop) < pop_size:
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
        if fitness[bi] < best_ms: best_ms = fitness[bi]; best = deepcopy(pop[bi])
    return best, best_ms

def _rollout(inst, perm, epsilon=0.05):
    remaining = list(perm)
    sim       = JSSPSimulator(inst)
    obs       = sim.reset()
    pairs     = []
    while not obs["done"]:
        legal = [j for j in range(inst.n_jobs) if obs["action_mask"][j]]
        if not legal: break
        if epsilon > 0 and random.random() < epsilon: action = random.choice(legal)
        else:
            def first_pos(job):
                for i, g in enumerate(remaining):
                    if g == job: return i
                return float("inf")
            action = min(legal, key=first_pos)
        for i, g in enumerate(remaining):
            if g == action: remaining.pop(i); break
        pairs.append((obs["features"].copy(), obs["action_mask"].copy(), action))
        obs, _ = sim.step(action)
    return pairs, sim.makespan

def collect_all_demos(instances, episodes_per_instance, ga_pop, ga_gens, epsilon):
    all_demos = []
    n_experts = max(1, ga_pop // 2)
    for idx, inst in enumerate(instances):
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
            if fitness[bi] < best_ms: best_ms = fitness[bi]; best = deepcopy(pop[bi])
        ranked  = sorted(zip(fitness, pop), key=lambda x: x[0])
        experts = [p for _, p in ranked[:n_experts]]
        for ep in range(episodes_per_instance):
            perm   = experts[ep % n_experts]
            pairs, _ = _rollout(inst, perm, epsilon=epsilon)
            all_demos.extend(pairs)
        if (idx + 1) % 20 == 0 or (idx + 1) == len(instances):
            logger.info(f"  Demos from {idx+1}/{len(instances)} | transitions: {len(all_demos):,}")
    random.shuffle(all_demos)
    return all_demos

# Transformer layers
class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, embed_dim, n_heads, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % n_heads == 0
        self.n_heads   = n_heads
        self.head_dim  = embed_dim // n_heads
        self.embed_dim = embed_dim
        self.Wq = keras.layers.Dense(embed_dim, use_bias=False, name=f"{self.name}_Wq")
        self.Wk = keras.layers.Dense(embed_dim, use_bias=False, name=f"{self.name}_Wk")
        self.Wv = keras.layers.Dense(embed_dim, use_bias=False, name=f"{self.name}_Wv")
        self.Wo = keras.layers.Dense(embed_dim, use_bias=False, name=f"{self.name}_Wo")
    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        Q = self._split_heads(self.Wq(x), batch_size)
        K = self._split_heads(self.Wk(x), batch_size)
        V = self._split_heads(self.Wv(x), batch_size)
        scale   = tf.cast(self.head_dim, tf.float32) ** 0.5
        scores  = tf.matmul(Q, K, transpose_b=True) / scale
        weights = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(weights, V)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, -1, self.embed_dim))
        return self.Wo(out)

class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, embed_dim, n_heads, name="encoder_block"):
        super().__init__(name=name)
        self.mha   = MultiHeadSelfAttention(embed_dim, n_heads, name=f"{name}_mha")
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff1   = keras.layers.Dense(embed_dim * 4, activation="relu", name=f"{name}_ff1")
        self.ff2   = keras.layers.Dense(embed_dim, name=f"{name}_ff2")
        self.drop  = keras.layers.Dropout(0.1)
    def call(self, x, training=False):
        attn = self.mha(x, training=training)
        x    = self.norm1(x + self.drop(attn, training=training))
        ffn  = self.ff2(self.ff1(x))
        x    = self.norm2(x + self.drop(ffn, training=training))
        return x

def build_attention_policy(n_jobs, feature_dim, embed_dim=128, n_heads=4, n_layers=2):
    job_input  = keras.Input(shape=(n_jobs, feature_dim), name="job_features")
    mask_input = keras.Input(shape=(n_jobs,),             name="action_mask")
    x = keras.layers.Dense(embed_dim, name="input_proj")(job_input)
    for i in range(n_layers):
        x = TransformerEncoderBlock(embed_dim, n_heads, name=f"encoder_{i}")(x)
    logits = keras.layers.Dense(1, name="logit_proj")(x)
    logits = tf.squeeze(logits, axis=-1)
    mask_f = tf.cast(mask_input, tf.float32)
    masked = logits + (1.0 - mask_f) * (-1e9)
    return keras.Model([job_input, mask_input], masked, name="attention_policy")

def train_bc(demos, n_jobs, feature_dim, embed_dim, n_heads, n_layers, epochs, batch_size):
    features_arr = np.array([d[0] for d in demos], dtype=np.float32)
    masks_arr    = np.array([d[1] for d in demos], dtype=np.float32)
    actions_arr  = np.array([d[2] for d in demos], dtype=np.int64)
    logger.info(f"  BC dataset: {len(features_arr):,} transitions | feature shape: {features_arr.shape[1:]} | n_actions: {n_jobs}")
    unique, counts = np.unique(actions_arr, return_counts=True)
    logger.info("  Action distribution: " + ", ".join(f"job {j}: {100*c/len(actions_arr):.1f}%" for j, c in zip(unique, counts)))
    model = build_attention_policy(n_jobs=n_jobs, feature_dim=feature_dim, embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers)
    model.summary(print_fn=logger.info)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit([features_arr, masks_arr], actions_arr, epochs=epochs, batch_size=batch_size, validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)],
        verbose=1,
    )
    val_acc_key    = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
    final_val_acc  = history.history[val_acc_key][-1]
    epochs_trained = len(history.history["loss"])
    random_chance  = 1.0 / n_jobs
    logger.info(f"  BC complete ({epochs_trained}/{epochs} epochs) | Val acc: {final_val_acc:.4f} | Random chance: {random_chance:.4f}")
    return model

def run_baseline(inst, rule):
    sim = JSSPSimulator(inst)
    obs = sim.reset()
    while not obs["done"]:
        legal  = [j for j in range(inst.n_jobs) if obs["action_mask"][j]]
        if not legal: break
        op_idx = sim.job_op
        if rule == "SPT": action = min(legal, key=lambda j: inst.proc_time[j, op_idx[j]])
        elif rule == "FIFO": action = min(legal)
        elif rule == "LWKR": action = min(legal, key=lambda j: inst.proc_time[j, op_idx[j]:].sum())
        else: raise ValueError(f"Unknown rule: {rule}")
        obs, _ = sim.step(action)
    return sim.makespan

def run_attention(model, inst):
    sim = JSSPSimulator(inst)
    obs = sim.reset()
    while not obs["done"]:
        features = obs["features"][np.newaxis].astype(np.float32)
        mask     = obs["action_mask"][np.newaxis].astype(np.float32)
        logits   = model([features, mask], training=False).numpy()[0]
        action   = int(np.argmax(logits))
        obs, _   = sim.step(action)
    return sim.makespan

def evaluate(model, eval_instances, ga_pop, ga_gens):
    n = len(eval_instances)
    results = {m: np.zeros(n) for m in ["attn", "spt", "fifo", "lwkr", "ga"]}
    for i, inst in enumerate(eval_instances):
        results["attn"][i] = run_attention(model, inst)
        results["spt"][i]  = run_baseline(inst, "SPT")
        results["fifo"][i] = run_baseline(inst, "FIFO")
        results["lwkr"][i] = run_baseline(inst, "LWKR")
        _, results["ga"][i] = run_ga_on_instance(inst, ga_pop, ga_gens)
        if (i + 1) % 50 == 0 or (i + 1) == n:
            logger.info(f"  Evaluated {i+1}/{n} | "
                        f"Attention: {results['attn'][:i+1].mean():.1f} | "
                        f"SPT: {results['spt'][:i+1].mean():.1f} | "
                        f"FIFO: {results['fifo'][:i+1].mean():.1f} | "
                        f"LWKR: {results['lwkr'][:i+1].mean():.1f} | "
                        f"GA: {results['ga'][:i+1].mean():.1f}")
    return results

def report_results(results, out_dir):
    ga  = results["ga"]
    spt = results["spt"]
    n   = len(ga)
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Evaluation Results — {n} unseen random {N_JOBS}×{N_MACHINES} JSSP instances")
    logger.info("=" * 70)
    logger.info(f"{'Method':<14} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}  {'Gap to GA'}")
    logger.info("-" * 80)
    def method_row(name, arr):
        gap_to_ga = 100.0 * ((arr.mean() - ga.mean()) / ga.mean())
        logger.info(f"{name:<14} {arr.mean():8.2f} {np.median(arr):8.2f} {arr.std():8.2f} {arr.min():8.2f} {arr.max():8.2f}   {gap_to_ga:+7.2f}%")
    method_row('Attention (ours)', results['attn'])
    method_row('SPT', results['spt'])
    method_row('FIFO', results['fifo'])
    method_row('LWKR', results['lwkr'])
    method_row('GA',   results['ga'])
    logger.info("-" * 80)
    gap = 100.0 * (results["attn"] - ga) / ga
    logger.info(f"\n  Attention optimality gap to per-instance GA (lower is better):")
    logger.info(f"    mean:   {gap.mean():.2f}% | median: {np.median(gap):.2f}% | std: {gap.std():.2f}% | min: {gap.min():.2f}% | max: {gap.max():.2f}%")
    logger.info(f"    % within 5% of GA: {100*(gap<=5).sum()/n:.1f}%")
    logger.info(f"    % within 10% of GA: {100*(gap<=10).sum()/n:.1f}%")
    logger.info("")
    for baseline in ["spt", "fifo", "lwkr"]:
        wins = int((results["attn"] <= results[baseline]).sum())
        logger.info(f"  Attention beats {baseline.upper()}: {wins}/{n} ({100*wins/n:.1f}%)")
    logger.info("=" * 70)
    csv_path = os.path.join(out_dir, "eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance", "attn", "spt", "fifo", "lwkr", "ga", "attn_gap_to_ga_pct"])
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
    json_path = os.path.join(out_dir, "summary.json")
    summary: dict = {
        "instance_size": f"{N_JOBS}x{N_MACHINES}",
        "n_eval_instances": n,
    }
    for method in ["attn", "spt", "fifo", "lwkr", "ga"]:
        arr = results[method]
        summary[method] = { "mean": float(np.mean(arr)), "std": float(np.std(arr)), "min": float(np.min(arr)), "max": float(np.max(arr)), "median": float(np.median(arr)) }
    summary["attn_gap_to_ga_pct"] = { "mean": float(gap.mean()), "median": float(np.median(gap)), "std": float(gap.std()), "min": float(gap.min()), "max": float(gap.max()) }
    with open(json_path, "w") as f: json.dump(summary, f, indent=2)
    logger.info(f"eval_results.csv and summary.json saved to {out_dir}")

if __name__ == "__main__":
    logger.info(f"N_JOBS={N_JOBS}, N_MACHINES={N_MACHINES}, FEAT_DIM={FEATURE_DIM}.")
    logger.info(f"Train/Eval will always use these dimensions.")
    logger.info(f"Generating random training and evaluation instances ...")
    train_rng = np.random.default_rng(args.seed)
    eval_rng  = np.random.default_rng(args.seed + 10000)
    train_instances = [generate_instance(N_JOBS, N_MACHINES, args.proc_time_low, args.proc_time_high, train_rng)
                       for _ in range(args.n_train_instances)]
    eval_instances = [generate_instance(N_JOBS, N_MACHINES, args.proc_time_low, args.proc_time_high, eval_rng)
                      for _ in range(args.n_eval_instances)]
    logger.info(f"{len(train_instances)} train and {len(eval_instances)} eval instances generated.")

    logger.info("║══ Stage 1: Collecting Demonstrations ═══════")
    demos = collect_all_demos(train_instances, args.bc_episodes_per_instance, args.ga_pop, args.ga_gens, args.bc_epsilon)
    logger.info(f"{len(demos):,} demonstration transitions collected.")

    logger.info("║══ Stage 2: Behaviour Cloning Training ══════")
    model = train_bc(demos, N_JOBS, FEATURE_DIM, args.embed_dim, args.n_heads, args.n_attn_layers, args.bc_epochs, args.bc_batch)
    weights_path = os.path.join(OUT_DIR, "attention_policy_weights")
    model.save_weights(weights_path, save_format="h5"); logger.info(f"Weights saved: {weights_path}")

    logger.info("║══ Stage 3: Evaluation on Unseen Instances ══")
    results = evaluate(model, eval_instances, args.ga_pop, args.ga_gens)
    report_results(results, OUT_DIR)

    logger.info("")
    logger.info("="*70)
    logger.info("Experiment complete.")
    logger.info(f"  Output directory: {OUT_DIR}")
    logger.info("  Files: run.log | attention_policy_weights | eval_results.csv | summary.json")
    logger.info("="*70)
    for handler in logging.getLogger().handlers:
        handler.flush()
        handler.close()
