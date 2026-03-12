from JSSPEnv import JSSPEnv
from MCTS import MCTS
from JSSPNet import JSSPNet # Ensure your model class is imported
import torch

def solve_with_neural_mcts(instance_path, model, n_mcts_sims=100):
    # 1. Initialize the aligned environment
    env = JSSPEnv(instance_path)
    mcts = MCTS(model, env)
    
    current_state = env.reset()
    while not env.done:
        # Run search to find the best next job
        root = mcts.search(current_state, n_iterations=n_mcts_sims)
        
        # Action selection: Pick the child with the most visits (robust)
        best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        
        # Apply to the real environment
        current_state, _ = env.step(best_action)
    
    return env

# Example Usage
if __name__ == "__main__":
    instance = "instances/ta40"
    
    # Load your trained model (Ensure weights are loaded if using PyTorch)
    # If using the RLlib TF model, you'll need to use the RLlib Policy API instead.
    model = JSSPNet(input_dim=..., n_jobs=...) 
    model.eval()

    final_env = solve_with_neural_mcts(instance, model, n_mcts_sims=50)
    print(f"Search Complete. Final Makespan: {final_env.get_makespan()}")
