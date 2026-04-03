import gym
from gym.spaces import Dict, Box, Discrete
import numpy as np
import os
import re
import random


class GraphJSSPEnv(gym.Env):
    """
    Graph-based JSSP Environment for GNN + PPO.
    
    Key improvements over the original:
    - Dense step-wise reward (negative step duration) instead of sparse terminal reward
    - Richer node features (8-dim) including temporal information
    - Self-loops and symmetric adjacency with normalization
    - Proper padding mask for graph pooling
    """

    def __init__(self, env_config):
        instance_path_arg = env_config["instance_path"]
        self.bks_map = env_config.get("bks_map", {})
        base_dir, range_str = os.path.split(instance_path_arg)

        match = re.match(r"ta(\d+)-ta(\d+)", range_str)
        if not match:
            raise ValueError(f"Instance range format must be taXX-taYY, got {range_str}")
        start_idx, end_idx = int(match.group(1)), int(match.group(2))

        self.valid_files = sorted([
            os.path.join(base_dir, f)
            for f in os.listdir(base_dir)
            if f.startswith("ta")
            and re.findall(r"\d+", f)
            and start_idx <= int(re.findall(r"\d+", f)[0]) <= end_idx
        ])
        if not self.valid_files:
            raise FileNotFoundError(
                f"No ta files found in {base_dir} for range {start_idx}-{end_idx}"
            )

        # Probe first file for dimensions
        with open(self.valid_files[0], "r") as f:
            for line in f:
                if line.strip():
                    meta = line.strip().split()
                    self.max_jobs, self.max_machines = int(meta[0]), int(meta[1])
                    break

        self.max_nodes = self.max_jobs * self.max_machines
        self.node_features_dim = 8  # Richer features (see _get_state)

        # ---------- Observation & Action Spaces ----------
        self.observation_space = Dict({
            "node_features": Box(
                low=-1.0, high=1.0,
                shape=(self.max_nodes, self.node_features_dim),
                dtype=np.float32,
            ),
            "adj_matrix": Box(
                low=0.0, high=1.0,
                shape=(self.max_nodes, self.max_nodes),
                dtype=np.float32,
            ),
            "node_mask": Box(
                low=0.0, high=1.0,
                shape=(self.max_nodes,),
                dtype=np.float32,
            ),
            "action_mask": Box(
                low=0.0, high=1.0,
                shape=(self.max_jobs,),
                dtype=np.float32,
            ),
        })
        self.action_space = Discrete(self.max_jobs)

        self.reset()

    # ------------------------------------------------------------------ #
    #  Instance Loading (auto-detects OR-Library vs Standard Taillard)    #
    # ------------------------------------------------------------------ #
    def _load_instance(self, filepath):
        with open(filepath, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        meta = lines[0].split()
        self.n_jobs, self.n_machines = int(meta[0]), int(meta[1])

        self.proc_times = np.zeros((self.n_jobs, self.n_machines), dtype=np.int32)
        self.machine_seq = np.zeros((self.n_jobs, self.n_machines), dtype=np.int32)

        data_lines = lines[1:]
        first_vals = [int(x) for x in data_lines[0].split()]

        if len(first_vals) == self.n_machines * 2:
            # OR-Library interleaved format
            for i in range(self.n_jobs):
                vals = [int(x) for x in data_lines[i].split()]
                for op in range(self.n_machines):
                    self.machine_seq[i, op] = vals[op * 2]
                    self.proc_times[i, op] = vals[op * 2 + 1]
        else:
            # Standard Taillard (separate blocks, 1-indexed machines)
            for i in range(self.n_jobs):
                self.proc_times[i] = [int(x) for x in data_lines[i].split()]
            for i in range(self.n_jobs):
                self.machine_seq[i] = [
                    int(x) - 1 for x in data_lines[self.n_jobs + i].split()
                ]

        # Pre-compute useful derived quantities
        self.total_proc = self.proc_times.sum()
        self.max_proc = self.proc_times.max()
        # Remaining processing time per job (from each operation onward)
        self.remaining_work = np.zeros_like(self.proc_times, dtype=np.float32)
        for j in range(self.n_jobs):
            self.remaining_work[j] = np.cumsum(self.proc_times[j][::-1])[::-1]

    # ------------------------------------------------------------------ #
    #  Reset                                                              #
    # ------------------------------------------------------------------ #
    def reset(self):
        self.current_path = random.choice(self.valid_files)
        instance_key = os.path.basename(self.current_path).replace(".txt", "")
        self.current_bks = (
            self.bks_map.get(instance_key, 1.0)
            if isinstance(self.bks_map, dict)
            else float(self.bks_map)
        )
        self._load_instance(self.current_path)

        self.job_next_op = np.zeros(self.n_jobs, dtype=np.int32)
        self.machine_free_time = np.zeros(self.n_machines, dtype=np.int32)
        self.job_free_time = np.zeros(self.n_jobs, dtype=np.int32)
        self.prev_makespan = 0
        self.done = False
        self.num_steps = 0

        return self._get_state()

    # ------------------------------------------------------------------ #
    #  Step — Dense Reward                                                #
    # ------------------------------------------------------------------ #
    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, {}

        # Invalid action → small penalty, no state change
        if action >= self.n_jobs or self.job_next_op[action] >= self.n_machines:
            return self._get_state(), -0.01, self.done, {}

        op_idx = self.job_next_op[action]
        m_idx = self.machine_seq[action, op_idx]
        p_time = self.proc_times[action, op_idx]

        start_time = max(self.job_free_time[action], self.machine_free_time[m_idx])
        end_time = start_time + p_time

        self.job_free_time[action] = end_time
        self.machine_free_time[m_idx] = end_time
        self.job_next_op[action] += 1
        self.num_steps += 1

        self.done = np.all(self.job_next_op >= self.n_machines)

        # ---- Dense reward: negative *incremental* makespan increase ----
        # This gives the agent immediate feedback for every action.
        # Normalized by total processing time so rewards are in a stable range.
        current_makespan = int(self.machine_free_time.max())
        reward = -(current_makespan - self.prev_makespan) / self.max_proc
        self.prev_makespan = current_makespan

        info = {}
        if self.done:
            info["makespan"] = current_makespan
            info["bks"] = self.current_bks

        return self._get_state(), reward, self.done, info

    # ------------------------------------------------------------------ #
    #  Compatibility property for CustomCallbacks                        #
    # ------------------------------------------------------------------ #
    @property
    def last_time_step(self):
        return int(self.machine_free_time.max())

    # ------------------------------------------------------------------ #
    #  Build GNN Observation                                              #
    # ------------------------------------------------------------------ #
    def _get_state(self):
        node_features = np.zeros(
            (self.max_nodes, self.node_features_dim), dtype=np.float32
        )
        adj_matrix = np.zeros(
            (self.max_nodes, self.max_nodes), dtype=np.float32
        )
        node_mask = np.zeros(self.max_nodes, dtype=np.float32)
        action_mask = np.zeros(self.max_jobs, dtype=np.float32)

        current_makespan = max(1, int(self.machine_free_time.max()))
        ops_by_machine = {m: [] for m in range(self.n_machines)}

        for j in range(self.n_jobs):
            for op in range(self.n_machines):
                nid = j * self.n_machines + op
                node_mask[nid] = 1.0  # Real node (not padding)

                m_idx = self.machine_seq[j, op]
                p_time = self.proc_times[j, op]
                is_completed = float(op < self.job_next_op[j])
                is_ready = float(op == self.job_next_op[j])
                is_future = float(op > self.job_next_op[j])

                # 8-dimensional node features
                node_features[nid] = [
                    is_completed,
                    is_ready,
                    is_future,
                    p_time / self.max_proc,                              # normalised proc time
                    m_idx / max(1, self.n_machines - 1),                 # machine id (normalised)
                    self.machine_free_time[m_idx] / max(1, current_makespan),  # machine load
                    self.job_free_time[j] / max(1, current_makespan),          # job availability
                    self.remaining_work[j, op] / max(1, self.total_proc),      # remaining work ratio
                ]

                # Self-loop (critical for GCN)
                adj_matrix[nid, nid] = 1.0

                # Conjunctive (precedence) edges — BIDIRECTIONAL
                if op < self.n_machines - 1:
                    next_nid = nid + 1
                    adj_matrix[nid, next_nid] = 1.0
                    adj_matrix[next_nid, nid] = 1.0

                # Collect uncompleted ops per machine for disjunctive edges
                if not is_completed:
                    ops_by_machine[m_idx].append(nid)

        # Disjunctive edges (same machine, both uncompleted)
        for m, ops in ops_by_machine.items():
            for i in range(len(ops)):
                for k in range(i + 1, len(ops)):
                    adj_matrix[ops[i], ops[k]] = 1.0
                    adj_matrix[ops[k], ops[i]] = 1.0

        # ---- Symmetric normalisation: D^{-1/2} A D^{-1/2} ----
        degree = adj_matrix.sum(axis=1)
        degree_inv_sqrt = np.zeros_like(degree)
        nonzero = degree > 0
        degree_inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
        # D^{-1/2} A D^{-1/2}  (element-wise: adj[i,j] * d[i] * d[j])
        adj_matrix = adj_matrix * degree_inv_sqrt[:, None] * degree_inv_sqrt[None, :]

        # Action mask
        for j in range(self.n_jobs):
            if j < self.n_jobs and self.job_next_op[j] < self.n_machines:
                action_mask[j] = 1.0

        return {
            "node_features": node_features,
            "adj_matrix": adj_matrix,
            "node_mask": node_mask,
            "action_mask": action_mask,
        }
