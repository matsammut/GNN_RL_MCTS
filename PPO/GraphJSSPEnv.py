import gym
from gym.spaces import Dict, Box, Discrete
import numpy as np
import os
import re
import random

class GraphJSSPEnv(gym.Env):
    def __init__(self, env_config):
        # 1. Parsing Multiple Instances
        instance_path_arg = env_config['instance_path']
        self.bks_map = env_config.get('bks_map', {})
        base_dir, range_str = os.path.split(instance_path_arg)
        
        match = re.match(r"ta(\d+)-ta(\d+)", range_str)
        start_idx, end_idx = int(match.group(1)), int(match.group(2))
        
        self.valid_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) 
                            if f.startswith("ta") and start_idx <= int(re.findall(r'\d+', f)[0]) <= end_idx]
        
        first_file = self.valid_files[0]
        with open(first_file, 'r') as f:
            for line in f:
                if line.strip():
                    meta = line.strip().split()
                    self.max_jobs, self.max_machines = int(meta[0]), int(meta[1])
                    break
        
        self.max_nodes = self.max_jobs * self.max_machines
        self.node_features_dim = 4 #        
        # 3. Define the GNN Observation Space
        self.observation_space = Dict({
            "node_features": Box(low=0.0, high=1.0, shape=(self.max_nodes, self.node_features_dim), dtype=np.float32),
            "adj_matrix": Box(low=0.0, high=1.0, shape=(self.max_nodes, self.max_nodes), dtype=np.float32),
            "action_mask": Box(low=0.0, high=1.0, shape=(self.max_jobs,), dtype=np.float32)
        })
        self.action_space = Discrete(self.max_jobs)
        
        self.reset()

    def _load_instance(self, filepath):
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # First line contains n_jobs, n_machines
        meta = lines[0].split()
        self.n_jobs, self.n_machines = int(meta[0]), int(meta[1])
        
        self.proc_times = np.zeros((self.n_jobs, self.n_machines), dtype=int)
        self.machine_seq = np.zeros((self.n_jobs, self.n_machines), dtype=int)
        
        data_lines = lines[1:]
        first_line_vals = [int(x) for x in data_lines[0].split()]
        
        # 4. AUTO-DETECT FILE FORMAT
        if len(first_line_vals) == self.n_machines * 2:
            # Format A (OR-Library): Interleaved machine/time pairs (e.g. your ta42 file)
            for i in range(self.n_jobs):
                vals = [int(x) for x in data_lines[i].split()]
                for op in range(self.n_machines):
                    self.machine_seq[i, op] = vals[op * 2]       # Even indices: machine
                    self.proc_times[i, op] = vals[op * 2 + 1]    # Odd indices: time
        else:
            # Format B (Standard Taillard): Separated blocks
            for i in range(self.n_jobs):
                self.proc_times[i] = [int(x) for x in data_lines[i].split()]
            for i in range(self.n_jobs):
                # Standard Taillard is 1-indexed for machines, so we subtract 1
                self.machine_seq[i] = [int(x) - 1 for x in data_lines[self.n_jobs + i].split()]        

    def reset(self):
        # Pick random instance
        self.current_path = random.choice(self.valid_files)
        instance_key = os.path.basename(self.current_path).replace(".txt", "")
        self.current_bks = self.bks_map.get(instance_key, 1.0) if isinstance(self.bks_map, dict) else float(self.bks_map)
        
        self._load_instance(self.current_path)
        
        # Reset State Variables
        self.job_next_op = np.zeros(self.n_jobs, dtype=int)
        self.machine_free_time = np.zeros(self.n_machines, dtype=int)
        self.job_free_time = np.zeros(self.n_jobs, dtype=int)
        
        self.done = False
        return self._get_state()

    def step(self, action):
        if self.done or self.job_next_op[action] >= self.n_machines:
            return self._get_state(), 0.0, self.done, {} # Illegal action or already done
            
        op_idx = self.job_next_op[action]
        m_idx = self.machine_seq[action, op_idx]
        p_time = self.proc_times[action, op_idx]
        
        # Calculate start and completion times
        start_time = max(self.job_free_time[action], self.machine_free_time[m_idx])
        end_time = start_time + p_time
        
        # Update states
        self.job_free_time[action] = end_time
        self.machine_free_time[m_idx] = end_time
        self.job_next_op[action] += 1
        
        # Check completion
        self.done = np.all(self.job_next_op == self.n_machines)
        
        # Sparse reward: 0 during episode, negative makespan at the end (minimizes makespan)
        reward = -float(max(self.machine_free_time)) if self.done else 0.0
        
        return self._get_state(), reward, self.done, {}

    @property
    def last_time_step(self):
        # Compatibility property for CustomCallbacks.py
        return max(self.machine_free_time)

    def _get_state(self):
        node_features = np.zeros((self.max_nodes, self.node_features_dim), dtype=np.float32)
        adj_matrix = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        action_mask = np.zeros(self.max_jobs, dtype=np.float32)
        
        node_idx = 0
        ops_by_machine = {m: [] for m in range(self.n_machines)}
        
        # 1. Build Node Features & Conjunctive Edges
        for j in range(self.n_jobs):
            for op in range(self.n_machines):
                m_idx = self.machine_seq[j, op]
                p_time = self.proc_times[j, op]
                
                is_completed = 1.0 if op < self.job_next_op[j] else 0.0
                is_ready = 1.0 if op == self.job_next_op[j] else 0.0
                
                node_features[node_idx] = [is_completed, is_ready, p_time / 100.0, m_idx / self.n_machines]
                
                if not is_completed:
                    ops_by_machine[m_idx].append(node_idx)
                
                if op < self.n_machines - 1:
                    adj_matrix[node_idx, node_idx + 1] = 1.0
                    
                node_idx += 1

        # 2. Build Disjunctive Edges (Competing operations on same machine)
        for m, ops in ops_by_machine.items():
            for i in range(len(ops)):
                for k in range(i + 1, len(ops)):
                    n1, n2 = ops[i], ops[k]
                    adj_matrix[n1, n2] = 1.0
                    adj_matrix[n2, n1] = 1.0

        # 3. Action Mask
        for j in range(self.n_jobs):
            if self.job_next_op[j] < self.n_machines:
                action_mask[j] = 1.0

        return {
            "node_features": node_features,
            "adj_matrix": adj_matrix,
            "action_mask": action_mask
        }
