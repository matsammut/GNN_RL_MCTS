"""
objective1_intrasize.py
=======================
Objective 1: Train a single attention-based policy to generalize
across diverse 6x6 JSSP instances with varying machine orders
and processing times.

Pipeline:
  1. Instance Generator     — random 6x6 JSSP matrices
  2. Feature Extractor      — fixed-size per-job feature vector
  3. Attention Policy + Critic
  4. GA Expert              — generates (state, action) dataset
  5. Behavioral Cloning     — pretrain policy on GA decisions
  6. PPO Fine-tuning        — domain randomization over 6x6 instances
  7. Evaluation             — vs SPT / FIFO / LWKR baselines

Dependencies:
  pip install torch numpy gymnasium
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


# ─────────────────────────────────────────────
# SECTION 1: JSSP INSTANCE & ENVIRONMENT
# ─────────────────────────────────────────────

@dataclass
class JSSPInstance:
    """
    A single JSSP problem instance.

    Attributes:
        n_jobs:      Number of jobs.
        n_machines:  Number of machines.
        processing:  (n_jobs, n_machines) processing times.
        machine_order: (n_jobs, n_machines) machine visit order per job.
    """
    n_jobs: int
    n_machines: int
    processing: np.ndarray    # shape: (n_jobs, n_machines)
    machine_order: np.ndarray # shape: (n_jobs, n_machines)


def generate_instance(n_jobs: int = 6,
                       n_machines: int = 6,
                       min_time: int = 1,
                       max_time: int = 20) -> JSSPInstance:
    """Generate a random JSSP instance."""
    processing = np.random.randint(min_time, max_time + 1,
                                   size=(n_jobs, n_machines))
    machine_order = np.array([
        np.random.permutation(n_machines) for _ in range(n_jobs)
    ])
    return JSSPInstance(n_jobs, n_machines, processing, machine_order)


class JSSPEnv:
    """
    Lightweight JSSP environment (no external dependency).

    State:
        - next_op[j]:       index of next unscheduled operation for job j
        - job_available[j]: earliest time job j can be scheduled
        - machine_free[m]:  earliest time machine m is free

    Action:
        Integer j in [0, n_jobs-1]. Must be an eligible job
        (next operation exists and its machine is not busy).
    """

    def __init__(self, instance: JSSPInstance):
        self.inst = instance
        self.n_jobs = instance.n_jobs
        self.n_machines = instance.n_machines
        self.reset()

    def reset(self):
        self.next_op       = np.zeros(self.n_jobs, dtype=int)
        self.job_available = np.zeros(self.n_jobs, dtype=float)
        self.machine_free  = np.zeros(self.n_machines, dtype=float)
        self.current_time  = 0.0
        self.done          = False
        self.makespan      = 0.0
        return self._get_features(), self._get_mask()

    def _get_mask(self) -> np.ndarray:
        """True = job is eligible to be dispatched now."""
        mask = np.zeros(self.n_jobs, dtype=bool)
        for j in range(self.n_jobs):
            if self.next_op[j] < self.n_machines:
                m = self.inst.machine_order[j, self.next_op[j]]
                if self.job_available[j] <= self.current_time and \
                   self.machine_free[m] <= self.current_time:
                    mask[j] = True
        return mask

    def _get_features(self) -> np.ndarray:
        """
        Build fixed-size feature matrix: (n_jobs, 7).

        Features per job:
          0: remaining operations (normalised)
          1: next operation processing time (normalised)
          2: next machine index (normalised)
          3: job available time (normalised)
          4: next machine free time (normalised)
          5: total remaining work (normalised)
          6: slack = machine_free - job_available (normalised, clipped)
        """
        max_t = float(self.inst.processing.max() * self.n_machines + 1)
        features = np.zeros((self.n_jobs, 7), dtype=np.float32)

        for j in range(self.n_jobs):
            op = self.next_op[j]
            remaining_ops = self.n_machines - op

            if op < self.n_machines:
                m = self.inst.machine_order[j, op]
                proc_time = self.inst.processing[j, op]
                mfree = self.machine_free[m]
                javail = self.job_available[j]
                total_remaining = self.inst.processing[j, op:].sum()
                slack = mfree - javail
            else:
                # Job completed
                m, proc_time, mfree, javail, total_remaining, slack = 0,0,0,0,0,0

            features[j] = [
                remaining_ops / self.n_machines,
                proc_time / max_t,
                m / max(self.n_machines - 1, 1),
                javail / max_t,
                mfree / max_t,
                total_remaining / max_t,
                np.clip(slack, -max_t, max_t) / max_t,
            ]

        return features

    def step(self, job: int) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Dispatch job. Returns (features, mask, reward, done).
        Reward is 0 at each step; final reward = -makespan.
        """
        assert not self.done, "Episode already finished."
        op = self.next_op[job]
        m  = self.inst.machine_order[job, op]
        pt = self.inst.processing[job, op]

        start = max(self.job_available[job], self.machine_free[m])
        end   = start + pt

        self.job_available[job] = end
        self.machine_free[m]    = end
        self.next_op[job]      += 1
        self.makespan           = max(self.makespan, end)

        # Advance time if no job is currently eligible
        mask = self._get_mask()
        if not mask.any() and self.next_op.max() < self.n_machines:
            # Jump to earliest next availability
            eligible_times = []
            for j in range(self.n_jobs):
                if self.next_op[j] < self.n_machines:
                    mm = self.inst.machine_order[j, self.next_op[j]]
                    eligible_times.append(
                        max(self.job_available[j], self.machine_free[mm])
                    )
            if eligible_times:
                self.current_time = min(eligible_times)
            mask = self._get_mask()

        self.done = (self.next_op >= self.n_machines).all()
        reward = -self.makespan if self.done else 0.0

        return self._get_features(), mask, reward, self.done


# ─────────────────────────────────────────────
# SECTION 2: ATTENTION POLICY + CRITIC
# ─────────────────────────────────────────────

FEATURE_DIM = 7   # must match _get_features() output width
HIDDEN_DIM  = 128


class AttentionPolicy(nn.Module):
    """
    Attention-based actor-critic for variable-size JSSP.

    Actor:
      job_encoder(job_features) → embeddings h (B, J, H)
      mean_pool(h) → global context g (B, 1, H)
      scorer(h + context_proj(g)) → logits (B, J)
      masked softmax → action distribution

    Critic:
      V(s) = critic_head(g) → scalar
    """

    def __init__(self, feature_dim: int = FEATURE_DIM,
                 hidden_dim: int = HIDDEN_DIM):
        super().__init__()

        self.job_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self,
                job_features: torch.Tensor,
                mask: torch.Tensor
                ) -> Tuple[Categorical, torch.Tensor]:
        """
        Args:
            job_features: (B, J, feature_dim)
            mask:         (B, J) bool — True = eligible

        Returns:
            dist:  Categorical distribution over jobs
            value: (B, 1) critic estimate
        """
        h = self.job_encoder(job_features)            # (B, J, H)
        g = h.mean(dim=1, keepdim=True)               # (B, 1, H)
        combined = h + self.context_proj(g)            # (B, J, H)
        logits = self.scorer(combined).squeeze(-1)     # (B, J)

        # Mask ineligible jobs
        logits = logits.masked_fill(~mask, float('-inf'))
        dist   = Categorical(logits=logits)
        value  = self.critic_head(g.squeeze(1))        # (B, H) → (B, 1)

        return dist, value

    def act(self,
            job_features: np.ndarray,
            mask: np.ndarray,
            deterministic: bool = False
            ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Single-step action selection (no batch dimension).

        Returns:
            action:   integer job index
            log_prob: log probability of chosen action
            value:    critic estimate
        """
        feat_t = torch.FloatTensor(job_features).unsqueeze(0)  # (1, J, F)
        mask_t = torch.BoolTensor(mask).unsqueeze(0)            # (1, J)

        with torch.no_grad():
            dist, value = self.forward(feat_t, mask_t)
            action = dist.probs.argmax(dim=-1) if deterministic \
                     else dist.sample()

        return action.item(), dist.log_prob(action), value


# ─────────────────────────────────────────────
# SECTION 3: GA EXPERT + BEHAVIORAL CLONING
# ─────────────────────────────────────────────

def spt_heuristic(features: np.ndarray, mask: np.ndarray) -> int:
    """
    Shortest Processing Time dispatching rule.
    Feature index 1 = next operation processing time (normalised).
    Returns job index with shortest next processing time among eligible jobs.
    """
    times = features[:, 1].copy()
    times[~mask] = np.inf
    return int(np.argmin(times))


def run_expert_episode(instance: JSSPInstance,
                       heuristic=spt_heuristic
                       ) -> List[Dict]:
    """
    Run one episode using the expert heuristic.
    Returns list of {features, mask, action} dicts.
    """
    env = JSSPEnv(instance)
    features, mask = env.reset()
    trajectory = []

    while not env.done:
        action = heuristic(features, mask)
        trajectory.append({
            "features": features.copy(),
            "mask":     mask.copy(),
            "action":   action,
        })
        features, mask, _, _ = env.step(action)

    return trajectory


def generate_cloning_dataset(n_instances: int = 1000,
                              n_jobs: int = 6,
                              n_machines: int = 6
                              ) -> List[Dict]:
    """Generate BC dataset by running expert on many random instances."""
    dataset = []
    for i in range(n_instances):
        inst = generate_instance(n_jobs, n_machines)
        traj = run_expert_episode(inst)
        dataset.extend(traj)
        if (i + 1) % 100 == 0:
            print(f"  [BC dataset] {i+1}/{n_instances} instances, "
                  f"{len(dataset)} decisions so far")
    return dataset


def behavioral_cloning(policy: AttentionPolicy,
                        dataset: List[Dict],
                        n_epochs: int = 10,
                        batch_size: int = 256,
                        lr: float = 1e-3) -> AttentionPolicy:
    """
    Pretrain policy via supervised imitation of expert decisions.
    Loss: cross-entropy between policy logits and expert action.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(dataset)

    print(f"\n[BC] Training on {n} decisions, {n_epochs} epochs")

    for epoch in range(n_epochs):
        indices = np.random.permutation(n)
        total_loss = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch = [dataset[i] for i in batch_idx]

            # Stack — all same size (6x6) so no padding needed here
            feat   = torch.FloatTensor(
                         np.stack([b["features"] for b in batch]))  # (B, J, F)
            mask   = torch.BoolTensor(
                         np.stack([b["mask"] for b in batch]))       # (B, J)
            target = torch.LongTensor([b["action"] for b in batch]) # (B,)

            dist, _ = policy(feat, mask)
            loss = F.cross_entropy(dist.logits, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        print(f"  Epoch {epoch+1:02d}/{n_epochs} | Loss: {avg_loss:.4f}")

    return policy


# ─────────────────────────────────────────────
# SECTION 4: PPO ROLLOUT BUFFER
# ─────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    features:   List[np.ndarray] = field(default_factory=list)
    masks:      List[np.ndarray] = field(default_factory=list)
    actions:    List[int]        = field(default_factory=list)
    log_probs:  List[float]      = field(default_factory=list)
    values:     List[float]      = field(default_factory=list)
    rewards:    List[float]      = field(default_factory=list)
    dones:      List[bool]       = field(default_factory=list)

    def clear(self):
        for f in self.__dataclass_fields__:
            setattr(self, f, [])

    def add(self, feat, mask, action, log_prob, value, reward, done):
        self.features.append(feat)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns(self, gamma: float = 0.99) -> np.ndarray:
        """Monte-Carlo returns (episode ends = done flag)."""
        returns = np.zeros(len(self.rewards))
        G = 0.0
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                G = 0.0
            G = self.rewards[t] + gamma * G
            returns[t] = G
        return returns


# ─────────────────────────────────────────────
# SECTION 5: PPO UPDATE
# ─────────────────────────────────────────────

def ppo_update(policy: AttentionPolicy,
               buffer: RolloutBuffer,
               optimizer: torch.optim.Optimizer,
               clip_eps: float = 0.2,
               value_coef: float = 0.5,
               entropy_coef: float = 0.01,
               n_epochs: int = 4,
               batch_size: int = 64):
    """
    PPO clipped surrogate update.

    Since episodes have variable-length step counts but all steps
    within an episode share the same J (jobs), we process one
    step at a time (batch_size steps sampled randomly).
    """
    returns = torch.FloatTensor(buffer.compute_returns())
    old_log_probs = torch.FloatTensor(buffer.log_probs)

    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    n = len(buffer.actions)
    total_loss = 0.0

    for _ in range(n_epochs):
        idx = np.random.permutation(n)

        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]

            # Build batch tensors — same n_jobs for all (6x6 objective 1)
            feat = torch.FloatTensor(
                       np.stack([buffer.features[i] for i in b]))
            mask = torch.BoolTensor(
                       np.stack([buffer.masks[i] for i in b]))
            acts = torch.LongTensor([buffer.actions[i] for i in b])
            ret  = returns[b]
            olp  = old_log_probs[b]

            dist, values = policy(feat, mask)
            new_log_probs = dist.log_prob(acts)
            entropy = dist.entropy().mean()

            # Advantage
            adv = ret - values.squeeze(-1).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Clipped surrogate
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

            total_loss += loss.item()

    return total_loss


# ─────────────────────────────────────────────
# SECTION 6: BASELINES
# ─────────────────────────────────────────────

def run_heuristic(instance: JSSPInstance,
                  rule: str = "SPT") -> float:
    """
    Run a classical dispatching rule and return makespan.
    Rules: SPT, FIFO, LWKR
    """
    env = JSSPEnv(instance)
    features, mask = env.reset()
    job_arrival = np.zeros(instance.n_jobs)  # FIFO: job index as proxy

    while not env.done:
        eligible = np.where(mask)[0]

        if rule == "SPT":
            times = features[:, 1]  # next processing time (normalised)
            times_masked = np.where(mask, times, np.inf)
            action = int(np.argmin(times_masked))

        elif rule == "FIFO":
            action = int(eligible[0])  # lowest job index first

        elif rule == "LWKR":
            remaining = features[:, 5]  # total remaining work (normalised)
            remaining_masked = np.where(mask, remaining, np.inf)
            action = int(np.argmin(remaining_masked))

        else:
            raise ValueError(f"Unknown rule: {rule}")

        features, mask, _, _ = env.step(action)

    return env.makespan


# ─────────────────────────────────────────────
# SECTION 7: EVALUATION
# ─────────────────────────────────────────────

def evaluate_policy(policy: AttentionPolicy,
                    n_eval: int = 500,
                    n_jobs: int = 6,
                    n_machines: int = 6,
                    deterministic: bool = True) -> Dict:
    """
    Evaluate policy and baselines on n_eval fresh 6x6 instances.
    Returns dict of {method: mean_makespan}.
    """
    policy.eval()
    results = {"Policy": [], "SPT": [], "FIFO": [], "LWKR": []}

    for i in range(n_eval):
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

    summary = {k: np.mean(v) for k, v in results.items()}
    policy.train()
    return summary, results


# ─────────────────────────────────────────────
# SECTION 8: MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train_objective1(
    # Instance config
    n_jobs: int = 6,
    n_machines: int = 6,
    # BC config
    bc_instances: int = 1000,
    bc_epochs: int = 10,
    bc_lr: float = 1e-3,
    # PPO config
    ppo_episodes: int = 5000,
    ppo_lr: float = 3e-4,
    ppo_clip: float = 0.2,
    ppo_value_coef: float = 0.5,
    ppo_entropy_coef: float = 0.01,
    ppo_update_epochs: int = 4,
    rollout_episodes: int = 16,   # collect N episodes before each update
    # Eval config
    eval_every: int = 500,
    eval_instances: int = 200,
    # Save
    save_path: str = "policy_obj1.pt",
):
    """
    Full Objective 1 training:
      1. Behavioral cloning on 6x6 instances
      2. PPO fine-tuning with domain randomization over 6x6 instances
    """
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    policy = AttentionPolicy(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM)

    # ── Phase 1: Behavioral Cloning ──────────────────────────
    print("=" * 60)
    print("PHASE 1: Behavioral Cloning")
    print("=" * 60)

    print(f"Generating dataset: {bc_instances} instances ({n_jobs}x{n_machines})...")
    dataset = generate_cloning_dataset(bc_instances, n_jobs, n_machines)

    policy = behavioral_cloning(policy, dataset,
                                 n_epochs=bc_epochs, lr=bc_lr)

    print("\n[BC] Evaluation after cloning:")
    summary, _ = evaluate_policy(policy, n_eval=eval_instances,
                                  n_jobs=n_jobs, n_machines=n_machines)
    for k, v in summary.items():
        print(f"  {k:<10}: {v:.2f}")

    # ── Phase 2: PPO Fine-tuning ──────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: PPO Fine-tuning (domain randomization over 6x6)")
    print("=" * 60)

    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_lr)
    buffer    = RolloutBuffer()
    best_makespan = np.inf
    episode   = 0
    t0        = time.time()

    while episode < ppo_episodes:

        # Collect rollout_episodes episodes before updating
        buffer.clear()
        episode_makespans = []

        for _ in range(rollout_episodes):
            # Domain randomization: fresh random 6x6 instance each episode
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

            episode_makespans.append(env.makespan)
            episode += 1

        # PPO update
        ppo_update(policy, buffer, optimizer,
                   clip_eps=ppo_clip,
                   value_coef=ppo_value_coef,
                   entropy_coef=ppo_entropy_coef,
                   n_epochs=ppo_update_epochs)

        avg_ms = np.mean(episode_makespans)

        if episode % 200 == 0:
            elapsed = time.time() - t0
            print(f"  Episode {episode:5d}/{ppo_episodes} | "
                  f"Avg makespan: {avg_ms:.2f} | "
                  f"Elapsed: {elapsed:.0f}s")

        # Periodic full evaluation
        if episode % eval_every == 0:
            summary, _ = evaluate_policy(policy, n_eval=eval_instances,
                                          n_jobs=n_jobs, n_machines=n_machines)
            policy_ms = summary["Policy"]
            spt_ms    = summary["SPT"]
            gap       = (policy_ms - spt_ms) / spt_ms * 100

            print(f"\n  [Eval @ ep {episode}]")
            for k, v in summary.items():
                print(f"    {k:<10}: {v:.2f}")
            print(f"    Gap vs SPT: {gap:+.1f}%\n")

            if policy_ms < best_makespan:
                best_makespan = policy_ms
                torch.save({
                    "episode": episode,
                    "policy_state": policy.state_dict(),
                    "makespan": best_makespan,
                    "config": {
                        "n_jobs": n_jobs,
                        "n_machines": n_machines,
                        "feature_dim": FEATURE_DIM,
                        "hidden_dim": HIDDEN_DIM,
                    }
                }, save_path)
                print(f"  ✓ Saved best policy → {save_path} "
                      f"(makespan={best_makespan:.2f})")

    # ── Final Evaluation ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (500 unseen 6x6 instances)")
    print("=" * 60)

    summary, results = evaluate_policy(policy, n_eval=500,
                                        n_jobs=n_jobs, n_machines=n_machines)
    spt_ms = summary["SPT"]
    for k, v in summary.items():
        gap = (v - spt_ms) / spt_ms * 100
        print(f"  {k:<10}: {v:.2f}  ({gap:+.1f}% vs SPT)")

    return policy, summary


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    policy, final_results = train_objective1(
        n_jobs=6,
        n_machines=6,
        bc_instances=1000,
        bc_epochs=10,
        ppo_episodes=5000,
        rollout_episodes=16,
        eval_every=500,
        save_path="policy_obj1.pt",
    )
