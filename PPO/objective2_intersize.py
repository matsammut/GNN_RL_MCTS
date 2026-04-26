"""
objective2_intersize.py
=======================
Objective 2: Extend the pretrained 6x6 policy (Objective 1) to
generalize across multiple problem sizes (6x6, 8x8, 10x10, 12x12).

Strategy:
  - Load pretrained weights from Objective 1
  - Continue PPO training with domain randomization over all sizes
  - Evaluate: per-size performance, transfer efficiency, zero-shot
    performance on 12x12 (if trained only up to 10x10)

Key insight: the attention architecture is already size-agnostic.
Only the training distribution changes between Objective 1 and 2.

Dependencies:
  pip install torch numpy gymnasium
  (objective1_intrasize.py must be in the same directory)
"""

import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# Re-use all building blocks from Objective 1
from objective1_intrasize import (
    JSSPInstance,
    JSSPEnv,
    AttentionPolicy,
    RolloutBuffer,
    generate_instance,
    run_expert_episode,
    run_heuristic,
    ppo_update,
    FEATURE_DIM,
    HIDDEN_DIM,
)


# ─────────────────────────────────────────────
# SECTION 1: MULTI-SIZE INSTANCE SAMPLER
# ─────────────────────────────────────────────

# Target size configurations for Objective 2.
# Add or remove sizes as needed for your thesis experiments.
SIZE_CONFIGS = [
    (6,  6),
    (8,  8),
    (10, 10),
    (12, 12),
]

# Sizes used during training (exclude 12x12 to test zero-shot transfer)
TRAIN_SIZES = [(6, 6), (8, 8), (10, 10)]
ZEROSHOT_SIZES = [(12, 12)]


def sample_random_size(sizes: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Sample uniformly from available size configs."""
    return random.choice(sizes)


def generate_cloning_dataset_multisize(
    n_instances: int = 3000,
    sizes: List[Tuple[int, int]] = TRAIN_SIZES,
) -> List[Dict]:
    """
    Generate BC dataset across multiple problem sizes.
    Instances are sampled uniformly across sizes.
    """
    from objective1_intrasize import spt_heuristic

    dataset = []
    for i in range(n_instances):
        n_jobs, n_machines = sample_random_size(sizes)
        inst = generate_instance(n_jobs, n_machines)
        traj = run_expert_episode(inst, heuristic=spt_heuristic)
        dataset.extend(traj)
        if (i + 1) % 200 == 0:
            print(f"  [BC dataset] {i+1}/{n_instances} instances, "
                  f"{len(dataset)} decisions so far")
    return dataset


# ─────────────────────────────────────────────
# SECTION 2: VARIABLE-SIZE BATCH COLLATION
# ─────────────────────────────────────────────

def collate_batch(samples: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a batch of decisions with potentially different n_jobs.
    Pads shorter job lists to match the largest in the batch.

    Args:
        samples: list of {features: (J, F), mask: (J,), action: int}

    Returns:
        feat_batch: (B, J_max, F)  — padded
        mask_batch: (B, J_max)     — padded with False
        action_batch: (B,)
    """
    max_jobs = max(s["features"].shape[0] for s in samples)
    feat_dim = samples[0]["features"].shape[1]

    B = len(samples)
    feat_batch   = torch.zeros(B, max_jobs, feat_dim)
    mask_batch   = torch.zeros(B, max_jobs, dtype=torch.bool)
    action_batch = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(samples):
        J = s["features"].shape[0]
        feat_batch[i, :J, :]  = torch.FloatTensor(s["features"])
        mask_batch[i, :J]     = torch.BoolTensor(s["mask"])
        action_batch[i]       = s["action"]

    return feat_batch, mask_batch, action_batch


def behavioral_cloning_multisize(
    policy: AttentionPolicy,
    dataset: List[Dict],
    n_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> AttentionPolicy:
    """
    BC training with variable-size batches.
    Uses collate_batch to handle different n_jobs per sample.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(dataset)

    print(f"\n[BC Multi-size] Training on {n} decisions, {n_epochs} epochs")

    for epoch in range(n_epochs):
        indices = np.random.permutation(n)
        total_loss = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch = [dataset[i] for i in batch_idx]

            feat, mask, target = collate_batch(batch)

            dist, _ = policy(feat, mask)
            loss = F.cross_entropy(dist.logits, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        print(f"  Epoch {epoch+1:02d}/{n_epochs} | Loss: {total_loss/batches:.4f}")

    return policy


# ─────────────────────────────────────────────
# SECTION 3: VARIABLE-SIZE PPO ROLLOUT
# ─────────────────────────────────────────────

def collect_rollout_multisize(
    policy: AttentionPolicy,
    buffer: RolloutBuffer,
    n_episodes: int,
    train_sizes: List[Tuple[int, int]],
) -> List[float]:
    """
    Collect rollout episodes across random sizes.
    Returns list of makespans for logging.
    """
    makespans = []

    for _ in range(n_episodes):
        n_jobs, n_machines = sample_random_size(train_sizes)
        inst = generate_instance(n_jobs, n_machines)
        env  = JSSPEnv(inst)
        feat, mask = env.reset()

        while not env.done:
            action, log_prob, value = policy.act(feat, mask)
            next_feat, next_mask, reward, done = env.step(action)
            buffer.add(feat, mask, action,
                        log_prob.item(), value.item(),
                        reward, done)
            feat, mask = next_feat, next_mask

        makespans.append(env.makespan)

    return makespans


def ppo_update_multisize(
    policy: AttentionPolicy,
    buffer: RolloutBuffer,
    optimizer: torch.optim.Optimizer,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    n_epochs: int = 4,
    batch_size: int = 64,
):
    """
    PPO update that handles variable n_jobs via per-step processing.
    Each step in the buffer may have a different number of jobs,
    so we cannot stack them naively — process one at a time.
    """
    returns = torch.FloatTensor(buffer.compute_returns())
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    old_log_probs = torch.FloatTensor(buffer.log_probs)

    n = len(buffer.actions)

    for _ in range(n_epochs):
        idx = np.random.permutation(n)

        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]

            # Collate variable-size steps
            samples = [{
                "features": buffer.features[i],
                "mask":     buffer.masks[i],
                "action":   buffer.actions[i],
            } for i in b]

            feat, mask, acts = collate_batch(samples)
            ret = returns[b]
            olp = old_log_probs[b]

            dist, values = policy(feat, mask)
            new_log_probs = dist.log_prob(acts)
            entropy = dist.entropy().mean()

            adv = ret - values.squeeze(-1).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            ratio = torch.exp(new_log_probs - olp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(-1), ret)

            loss = actor_loss + value_coef * critic_loss \
                             - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


# ─────────────────────────────────────────────
# SECTION 4: EVALUATION
# ─────────────────────────────────────────────

def evaluate_per_size(
    policy: AttentionPolicy,
    sizes: List[Tuple[int, int]],
    n_eval: int = 300,
    deterministic: bool = True,
) -> Dict[Tuple, Dict]:
    """
    Evaluate policy and baselines separately for each size.

    Returns:
        {(n_jobs, n_machines): {method: mean_makespan}}
    """
    policy.eval()
    all_results = {}

    for (n_jobs, n_machines) in sizes:
        results = {"Policy": [], "SPT": [], "FIFO": [], "LWKR": []}

        for _ in range(n_eval):
            inst = generate_instance(n_jobs, n_machines)

            # Policy rollout
            env = JSSPEnv(inst)
            feat, mask = env.reset()
            while not env.done:
                action, _, _ = policy.act(feat, mask, deterministic=deterministic)
                feat, mask, _, _ = env.step(action)
            results["Policy"].append(env.makespan)

            # Baselines
            for rule in ["SPT", "FIFO", "LWKR"]:
                results[rule].append(run_heuristic(inst, rule))

        all_results[(n_jobs, n_machines)] = {
            k: np.mean(v) for k, v in results.items()
        }

    policy.train()
    return all_results


def print_eval_table(results: Dict[Tuple, Dict]):
    """Pretty-print per-size evaluation table."""
    print(f"\n  {'Size':<10} {'Policy':>10} {'SPT':>10} "
          f"{'FIFO':>10} {'LWKR':>10} {'Gap vs SPT':>12}")
    print("  " + "-" * 64)

    for (nj, nm), scores in sorted(results.items()):
        spt = scores["SPT"]
        pol = scores["Policy"]
        gap = (pol - spt) / spt * 100
        print(f"  {nj}x{nm:<6} "
              f"{pol:>10.2f} "
              f"{spt:>10.2f} "
              f"{scores['FIFO']:>10.2f} "
              f"{scores['LWKR']:>10.2f} "
              f"{gap:>+11.1f}%")


# ─────────────────────────────────────────────
# SECTION 5: TRANSFER EFFICIENCY ANALYSIS
# ─────────────────────────────────────────────

def measure_transfer_efficiency(
    pretrained_path: str,
    target_size: Tuple[int, int] = (10, 10),
    n_eval: int = 200,
    ppo_steps: int = 2000,
    rollout_episodes: int = 16,
) -> Dict:
    """
    Compare:
      A) Policy pretrained on 6x6, fine-tuned on target size
      B) Policy trained from scratch on target size

    Measures how many PPO steps each needs to reach a threshold
    (e.g. within 10% of SPT baseline on target size).

    Returns dict with convergence step counts.
    """
    results = {}

    for label, load_pretrained in [("Pretrained", True), ("Scratch", False)]:
        policy = AttentionPolicy(FEATURE_DIM, HIDDEN_DIM)

        if load_pretrained and pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            policy.load_state_dict(ckpt["policy_state"])
            print(f"\n[Transfer] Loaded pretrained weights from {pretrained_path}")

        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        buffer    = RolloutBuffer()
        n_jobs, n_machines = target_size

        # Compute SPT baseline on target size
        spt_baseline = np.mean([
            run_heuristic(generate_instance(n_jobs, n_machines), "SPT")
            for _ in range(n_eval)
        ])
        threshold = spt_baseline * 1.10  # within 10% of SPT

        convergence_step = None
        episode = 0
        history = []

        while episode < ppo_steps:
            buffer.clear()
            makespans = collect_rollout_multisize(
                policy, buffer, rollout_episodes, [target_size])
            ppo_update_multisize(policy, buffer, optimizer)
            episode += rollout_episodes

            if episode % 200 == 0:
                eval_res = evaluate_per_size(policy, [target_size], n_eval=100)
                pol_ms   = eval_res[target_size]["Policy"]
                history.append((episode, pol_ms))

                if convergence_step is None and pol_ms <= threshold:
                    convergence_step = episode
                    print(f"  [{label}] Converged at episode {episode} "
                          f"(makespan={pol_ms:.2f}, threshold={threshold:.2f})")

        results[label] = {
            "convergence_step": convergence_step,
            "history": history,
            "spt_baseline": spt_baseline,
        }

    return results


# ─────────────────────────────────────────────
# SECTION 6: MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train_objective2(
    # Pretrained weights from Objective 1
    pretrained_path: str = "policy_obj1.pt",
    # Training size distribution
    train_sizes: List[Tuple[int, int]] = TRAIN_SIZES,
    # Zero-shot test sizes (not seen during training)
    zeroshot_sizes: List[Tuple[int, int]] = ZEROSHOT_SIZES,
    # Optional: re-run BC on multi-size data before PPO
    run_multisize_bc: bool = True,
    bc_instances: int = 3000,
    bc_epochs: int = 5,
    # PPO config
    ppo_episodes: int = 8000,
    ppo_lr: float = 3e-4,
    ppo_clip: float = 0.2,
    ppo_value_coef: float = 0.5,
    ppo_entropy_coef: float = 0.01,
    ppo_update_epochs: int = 4,
    rollout_episodes: int = 16,
    # Eval config
    eval_every: int = 1000,
    eval_instances: int = 200,
    # Save
    save_path: str = "policy_obj2.pt",
):
    """
    Full Objective 2 training:
      1. Load pretrained 6x6 policy from Objective 1
      2. Optional: BC warm-up on multi-size instances
      3. PPO fine-tuning with domain randomization over all sizes
      4. Evaluate per size + zero-shot transfer
    """
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    policy = AttentionPolicy(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM)

    # ── Load Pretrained Weights ──────────────────────────────
    print("=" * 60)
    print("LOADING PRETRAINED POLICY (Objective 1)")
    print("=" * 60)

    if pretrained_path:
        try:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            policy.load_state_dict(ckpt["policy_state"])
            print(f"  Loaded: {pretrained_path}")
            print(f"  Trained at episode: {ckpt.get('episode', '?')}")
            print(f"  Best makespan (6x6): {ckpt.get('makespan', '?'):.2f}")
        except FileNotFoundError:
            print(f"  WARNING: {pretrained_path} not found. "
                  f"Starting from random weights.")

    # ── Baseline before fine-tuning ──────────────────────────
    print("\n[Pre-training evaluation across all sizes]")
    pre_results = evaluate_per_size(policy, train_sizes + zeroshot_sizes,
                                     n_eval=eval_instances)
    print_eval_table(pre_results)

    # ── Optional: Multi-size BC Warm-up ──────────────────────
    if run_multisize_bc:
        print("\n" + "=" * 60)
        print("PHASE 1: Multi-size Behavioral Cloning Warm-up")
        print("=" * 60)

        print(f"Generating multi-size BC dataset "
              f"({bc_instances} instances across {train_sizes})...")
        dataset = generate_cloning_dataset_multisize(bc_instances, train_sizes)

        optimizer_bc = torch.optim.Adam(policy.parameters(), lr=1e-3)
        policy = behavioral_cloning_multisize(policy, dataset,
                                               n_epochs=bc_epochs)

        print("\n[Post-BC evaluation]")
        post_bc = evaluate_per_size(policy, train_sizes + zeroshot_sizes,
                                     n_eval=eval_instances)
        print_eval_table(post_bc)

    # ── PPO Fine-tuning ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PHASE 2: PPO Fine-tuning across sizes {train_sizes}")
    print("=" * 60)

    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_lr)
    buffer    = RolloutBuffer()
    best_avg_makespan = np.inf
    episode   = 0
    t0        = time.time()

    while episode < ppo_episodes:
        buffer.clear()

        # Collect rollout across random sizes
        makespans = collect_rollout_multisize(
            policy, buffer, rollout_episodes, train_sizes)

        # PPO update with variable-size batch handling
        ppo_update_multisize(
            policy, buffer, optimizer,
            clip_eps=ppo_clip,
            value_coef=ppo_value_coef,
            entropy_coef=ppo_entropy_coef,
            n_epochs=ppo_update_epochs,
        )

        episode += rollout_episodes
        avg_ms = np.mean(makespans)

        if episode % 400 == 0:
            elapsed = time.time() - t0
            print(f"  Episode {episode:5d}/{ppo_episodes} | "
                  f"Avg makespan: {avg_ms:.2f} | "
                  f"Elapsed: {elapsed:.0f}s")

        # Periodic full evaluation across all sizes
        if episode % eval_every == 0:
            eval_sizes = train_sizes + zeroshot_sizes
            results = evaluate_per_size(policy, eval_sizes,
                                         n_eval=eval_instances)
            print(f"\n  [Eval @ episode {episode}]")
            print_eval_table(results)

            # Track average across training sizes only
            avg_train_ms = np.mean([
                results[s]["Policy"] for s in train_sizes
            ])

            if avg_train_ms < best_avg_makespan:
                best_avg_makespan = avg_train_ms
                torch.save({
                    "episode": episode,
                    "policy_state": policy.state_dict(),
                    "avg_makespan": best_avg_makespan,
                    "train_sizes": train_sizes,
                    "config": {
                        "feature_dim": FEATURE_DIM,
                        "hidden_dim": HIDDEN_DIM,
                    }
                }, save_path)
                print(f"\n  ✓ Saved best policy → {save_path} "
                      f"(avg makespan={best_avg_makespan:.2f})")

    # ── Final Evaluation ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    all_sizes = train_sizes + zeroshot_sizes
    final_results = evaluate_per_size(policy, all_sizes, n_eval=500)

    print("\nTraining sizes:")
    print_eval_table({s: final_results[s] for s in train_sizes})

    print("\nZero-shot transfer (never seen during training):")
    print_eval_table({s: final_results[s] for s in zeroshot_sizes})

    # ── Transfer Efficiency Analysis ─────────────────────────
    print("\n" + "=" * 60)
    print("TRANSFER EFFICIENCY ANALYSIS (10x10 target)")
    print("=" * 60)
    print("Comparing: pretrained init vs random init on 10x10...\n")

    transfer = measure_transfer_efficiency(
        pretrained_path=pretrained_path,
        target_size=(10, 10),
        n_eval=200,
        ppo_steps=2000,
    )

    for label, res in transfer.items():
        conv = res["convergence_step"]
        conv_str = str(conv) if conv else "Did not converge"
        print(f"  {label:<15}: convergence at episode {conv_str}")

    return policy, final_results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    policy, final_results = train_objective2(
        pretrained_path="policy_obj1.pt",   # output from objective1_intrasize.py
        train_sizes=[(6, 6), (8, 8), (10, 10)],
        zeroshot_sizes=[(12, 12)],
        run_multisize_bc=True,
        bc_instances=3000,
        bc_epochs=5,
        ppo_episodes=8000,
        rollout_episodes=16,
        eval_every=1000,
        save_path="policy_obj2.pt",
    )
