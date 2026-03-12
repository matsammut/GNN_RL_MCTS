import gym
import numpy as np
from gym.spaces import Box, Discrete

class ContinuousToDiscreteEnvWrapper(gym.Env):
    """
    Wraps a Discrete-action environment so that SAC (which expects continuous actions)
    can act on it. Converts continuous Box actions -> discrete argmax index.
    """
    def __init__(self, env):
        super().__init__()
        assert isinstance(env.action_space, Discrete), "Inner env must have Discrete action space."
        self.env = env
        self.n = env.action_space.n
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.n,), dtype=np.float32)
        self.observation_space = env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if isinstance(action, np.ndarray):
            discrete_action = int(np.argmax(action))
        else:
            discrete_action = int(np.argmax(np.array(action)))
        obs, reward, done, info = self.env.step(discrete_action)
        return obs, reward, done, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()
