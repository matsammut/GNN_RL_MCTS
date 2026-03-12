from typing import Dict
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

class CustomCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker: "RolloutWorker", base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode, **kwargs):
        
        # Get the unwrapped environment
        env = base_env.get_unwrapped()[0]
        
        if env.last_time_step != float('inf'):
            makespan = env.last_time_step
            episode.custom_metrics['make_span'] = makespan
            
            # Access the BKS we've attached to the environment instance
            if hasattr(env, 'current_bks') and env.current_bks > 0:
                gap = ((makespan - env.current_bks) / env.current_bks) * 100
                episode.custom_metrics['optimality_gap'] = gap
