import gym
import numpy as np
from gym import spaces


class DynamicJSSWrapper(gym.Wrapper):
    """
    Wraps JSSEnv to inject dynamic scheduling-progress features into
    the observation, giving the neural network visibility into:

    1. How far each job has progressed (normalised operation count)
    2. Remaining processing time per job (normalised)
    3. Machine load balance (how unevenly work is distributed)

    These features break observation stationarity and provide the
    dispatching-relevant signals that the base observation lacks.
    """

    def __init__(self, env):
        super().__init__(env)

        # JSSEnv observation space is a Dict with "action_mask" and "real_obs"
        orig_obs_space = env.observation_space

        if isinstance(orig_obs_space, spaces.Dict):
            real_obs_space = orig_obs_space["real_obs"]
            mask_space = orig_obs_space["action_mask"]
            self.num_jobs = real_obs_space.shape[0]
            self.orig_features = real_obs_space.shape[1]
        else:
            self.num_jobs = orig_obs_space.shape[0]
            self.orig_features = orig_obs_space.shape[1]
            real_obs_space = orig_obs_space
            mask_space = None

        self.n_machines = env.machines if hasattr(env, 'machines') else self.orig_features

        # Parse instance to get processing times for remaining-work calculation
        self._job_durations = None
        if hasattr(env, 'instance_path'):
            self._parse_job_durations(env.instance_path)
        elif hasattr(env, 'env_config') and 'instance_path' in env.env_config:
            self._parse_job_durations(env.env_config['instance_path'])

        # New features: progress (1) + remaining_work_norm (1) + global_progress (1) = 3
        self.n_extra_features = 3
        new_real_shape = (self.num_jobs, self.orig_features + self.n_extra_features)
        new_real_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=new_real_shape,
            dtype=np.float32,
        )

        if mask_space is not None:
            self.observation_space = spaces.Dict({
                "action_mask": mask_space,
                "real_obs": new_real_space,
            })
        else:
            self.observation_space = new_real_space

        # Internal tracking
        self.job_step_counts = np.zeros(self.num_jobs, dtype=np.float32)
        self.total_steps = 0

    def _parse_job_durations(self, instance_path):
        """Parse instance file to get per-job, per-operation durations."""
        try:
            with open(instance_path, "r") as f:
                toks = f.read().strip().split()
            n_jobs = int(toks[0])
            n_machines = int(toks[1])
            nums = list(map(int, toks[2:]))

            self._job_durations = []
            for j in range(n_jobs):
                ops = []
                for m in range(n_machines):
                    dur = nums[(j * n_machines + m) * 2 + 1]
                    ops.append(dur)
                self._job_durations.append(ops)

            # Total processing time per job (for normalisation)
            self._total_job_time = np.array(
                [sum(ops) for ops in self._job_durations], dtype=np.float32
            )
            # Max total for normalisation
            self._max_total_time = max(self._total_job_time.max(), 1.0)

        except Exception:
            self._job_durations = None

    def reset(self, **kwargs):
        self.job_step_counts = np.zeros(self.num_jobs, dtype=np.float32)
        self.total_steps = 0
        obs = self.env.reset(**kwargs)
        return self._enrich_observation(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if 0 <= action < self.num_jobs:
            self.job_step_counts[action] += 1.0
        self.total_steps += 1

        return self._enrich_observation(obs), reward, done, info

    def _enrich_observation(self, obs):
        """Append dynamic features to the observation."""
        if isinstance(obs, dict) and "real_obs" in obs:
            raw_obs = obs["real_obs"]
        else:
            raw_obs = obs

        # Feature 1: Job progress (0.0 = not started, 1.0 = all ops scheduled)
        progress = (self.job_step_counts / max(self.n_machines, 1)).reshape(-1, 1)

        # Feature 2: Remaining processing time (normalised)
        if self._job_durations is not None:
            remaining = np.zeros(self.num_jobs, dtype=np.float32)
            for j in range(self.num_jobs):
                op_idx = int(self.job_step_counts[j])
                if op_idx < self.n_machines:
                    remaining[j] = sum(self._job_durations[j][op_idx:])
            remaining_norm = (remaining / self._max_total_time).reshape(-1, 1)
        else:
            # Fallback: use inverse progress as proxy
            remaining_norm = (1.0 - progress)

        # Feature 3: Global schedule progress (same for all jobs, but
        # gives the network a sense of "how far into the episode are we")
        total_ops = self.num_jobs * self.n_machines
        global_progress = np.full(
            (self.num_jobs, 1),
            self.total_steps / max(total_ops, 1),
            dtype=np.float32,
        )

        enriched = np.hstack((
            raw_obs,
            progress,
            remaining_norm,
            global_progress,
        )).astype(np.float32)

        if isinstance(obs, dict):
            obs = dict(obs)
            obs["real_obs"] = enriched
            return obs
        return enriched
