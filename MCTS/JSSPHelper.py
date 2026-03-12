#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
from MCTS import MCTS
from JSSPEnv import JSSPEnv
from default_config import load_taillard_instance


#def plot_gantt(env):
#    fig, ax = plt.subplots(figsize=(12, 6))
#    
#    # Generate distinct colors for each job
#    colors = list(mcolors.TABLEAU_COLORS.values())
#    if env.n_jobs > len(colors):
#        colors = list(mcolors.CSS4_COLORS.values()) # Fallback for large instances
#
#    for op in env.history:
#        job_id = op['Job']
#        machine_id = op['Machine']
#        start = op['Start']
#        duration = op['End'] - op['Start']
#        
#        ax.broken_barh([(start, duration)], (machine_id - 0.4, 0.8), 
#                       facecolors=colors[job_id % len(colors)], 
#                       edgecolor='black', alpha=0.8)
#        
#        # Add text label for the job ID inside the bar
#        ax.text(start + duration/2, machine_id, f'J{job_id}', 
#                ha='center', va='center', color='white', fontweight='bold')
#
#    ax.set_xlabel('Time')
#    ax.set_ylabel('Machine ID')
#    ax.set_title(f'JSSP Schedule (Makespan: {max(env.machine_free_time)})')
#    
#    # Set y-ticks to match machine IDs
#    machines = sorted(list(set(op['Machine'] for op in env.history)))
#    ax.set_yticks(machines)
#    
#    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
#    plt.tight_layout()
#    plt.show()


def solve_taillard(instance_path, model, n_mcts_sims):
    proc_times, machine_seq, n_jobs, n_m = load_taillard_instance(instance_path)
    env = JSSPEnv(instance_path)
    mcts = MCTS(model, env)
    
    current_state = env.reset()
    while not env.done:
        root = mcts.search(current_state, n_mcts_sims)
        best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        current_state, _ = env.step(best_action)
    
    return env # Return the env object instead of just the makespan
