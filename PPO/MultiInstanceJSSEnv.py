import gym
import os
import random
import re
import numpy as np

class MultiInstanceJSSEnv(gym.Env):
    def __init__(self, env_config):
        # input format: "instances/ta41-ta50"
        instance_path_arg = env_config['instance_path']
        self.bks_map = env_config.get('bks_map', {})
        
        # Parse Directory and Range
        # Split by the last slash to separate directory from the range string
        base_dir, range_str = os.path.split(instance_path_arg)
        
        match = re.match(r"ta(\d+)-ta(\d+)", range_str)
        if not match:
            raise ValueError(f"Instance range format must be taXX-taYY, got {range_str}")
        
        start_idx, end_idx = int(match.group(1)), int(match.group(2))
        
        # Filter files in the folder that match the numeric range
        all_files = os.listdir(base_dir)
        self.valid_files = []
        for f in all_files:
            if f.startswith("ta"):
                # Extract digits to find the instance number
                nums = re.findall(r'\d+', f)
                if nums:
                    f_num = int(nums[0])
                    if start_idx <= f_num <= end_idx:
                        self.valid_files.append(os.path.join(base_dir, f))
        
        if not self.valid_files:
            raise FileNotFoundError(f"No ta files found in {base_dir} for range {start_idx}-{end_idx}")

        # Initialize the first instance to define spaces
        self.current_path = self.valid_files[0]
        self.inner_env = gym.make('JSSEnv:jss-v1', env_config={'instance_path': self.current_path})
        
        self.observation_space = self.inner_env.observation_space
        self.action_space = self.inner_env.action_space
        self.current_bks = 1.0

    def reset(self):
        # Pick a random instance from our filtered list
        self.current_path = random.choice(self.valid_files)
        instance_key = os.path.basename(self.current_path).replace(".txt", "")
        
        # Set BKS for the current instance (used by CustomCallbacks)
        if isinstance(self.bks_map, dict):
            self.current_bks = self.bks_map.get(instance_key, 1.0)
        else:
            self.current_bks = float(self.bks_map)

        # Re-make the environment for the new instance
        self.inner_env = gym.make('JSSEnv:jss-v1', env_config={'instance_path': self.current_path})
        return self.inner_env.reset()

    def step(self, action):
        obs, reward, done, info = self.inner_env.step(action)
        return obs, reward, done, info

    @property
    def last_time_step(self):
        return self.inner_env.last_time_step
