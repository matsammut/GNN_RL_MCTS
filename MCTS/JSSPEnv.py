import numpy as np
import gym
import copy

class JSSPEnv:
    def __init__(self, instance_path):
        """
        Wraps the Gym environment (JSSEnv) to be compatible with MCTS.
        instance_path: path to the taillard or jss instance file.
        """
        self.instance_path = instance_path
        # Standardize initialization with the training config
        self.env = gym.make('JSSEnv:jss-v1', env_config={'instance_path': instance_path})
        self.n_jobs = self.env.jobs
        self.n_machines = self.env.machines
        self.reset()

    def reset(self):
        # Gym returns a dict: {"action_mask": ..., "real_obs": ...}
        self.curr_obs = self.env.reset()
        self.done = False
        self.history = [] 
        return self.curr_obs

    def get_legal_actions(self):
        """Returns the indices of valid actions from the Gym mask."""
        # The key is usually 'action_mask' in JSS gym environments
        mask = self.curr_obs.get('action_mask', None)
        if mask is None:
            # Fallback if the env doesn't use a dict for reset
            return np.where(self.env.get_legal_actions() == 1)[0]
        return np.where(mask == 1)[0]

    def step(self, action):
        # Gym step returns: obs (dict), reward, done, info
        self.curr_obs, reward, self.done, info = self.env.step(action)
        self.history.append(action)
        return self.curr_obs, reward

    def _get_state(self):
        """Returns the current dict observation (obs + mask)."""
        return self.curr_obs
    
    def copy(self):
        """Creates a deep copy of the environment state for MCTS branching."""
        # Create a new shell
        new_env = JSSPEnv(self.instance_path)
        # Deepcopy the underlying Gym state
        new_env.env = copy.deepcopy(self.env)
        new_env.curr_obs = copy.deepcopy(self.curr_obs)
        new_env.done = self.done
        new_env.history = list(self.history)
        return new_env

    def get_makespan(self):
        """Retrieves the makespan (last time step) from the environment."""
        return self.env.last_time_step
