import os
import time
import pandas as pd
import numpy as np
from load import load_taillard_instance
from run_ma_hga_taillard import (
    parse_taillard, compute_makespan, random_permutation, 
    tournament_select, order_crossover, swap_mutation, 
    simulated_annealing_improve, SA_T0, SA_TEND, TOURNAMENT_K, run_HGA
)
from copy import deepcopy

def run_ea_until_target(alg_type, jobs, target, pop_size, generations, **kwargs):
    """
    Modified EA loop that exits early if the target makespan is hit.
    """
    start_time = time.time()
    pop = [random_permutation(jobs) for _ in range(pop_size)]
    fitness = [compute_makespan(jobs, ind) for ind in pop]
    
    best_val = min(fitness)
    best_ind = deepcopy(pop[fitness.index(best_val)])
    
    # Check if initial population already hits target
    if best_val <= target:
        return best_val, time.time() - start_time

    for gen in range(1, generations + 1):
        new_pop = [deepcopy(best_ind)] # Elitism
        
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(pop, fitness, TOURNAMENT_K)
            child = order_crossover(p1, p2) if np.random.random() < 0.9 else deepcopy(p1)
            child = swap_mutation(child, mutation_rate=0.1)
            
            if alg_type == "MA":
                # Fixed: passing SA_T0/SA_TEND as positional or ensuring names match
                child, _ = simulated_annealing_improve(jobs, child, iters=kwargs.get('sa_iters', 100))
            
            new_pop.append(child)
        
        pop = new_pop
        fitness = [compute_makespan(jobs, ind) for ind in pop]

        if alg_type == "HGA":
            idx_sorted = sorted(range(len(pop)), key=lambda i: fitness[i])
            topk = max(1, int(kwargs.get('sa_fraction', 0.2) * pop_size))
            for i in idx_sorted[:topk]:
                improved, val = simulated_annealing_improve(jobs, pop[i], iters=kwargs.get('sa_iters', 50))
                pop[i] = improved
                fitness[i] = val

        gen_best = min(fitness)
        if gen_best < best_val:
            best_val = gen_best
            best_ind = deepcopy(pop[fitness.index(gen_best)])
        
        # Exit if we've reached the PPO target
        if best_val <= target:
            break
            
    return best_val, time.time() - start_time

# Load PPO data
ppo_results = pd.read_csv("tabulated_ppo_results.csv")
comparison_data = []

# EA Parameters
common_params = {"pop_size": 100, "generations": 500}


for _, row in ppo_results.iterrows():
    inst_id = row['Instance']
    ppo_target = row['Achieved Result']
    inst_path = os.path.join("instances", inst_id)
    print(f"Benchmarking {inst_id} (Target: {ppo_target})...")
    # Correctly parse the instance into the required list structure
    jobs_data = parse_taillard(inst_path) 
    # Run algorithms
    ga_val, ga_time = run_ea_until_target("GA", jobs_data, ppo_target, **common_params)
    print(f"GA Makespan: {ga_val}\n",f"GA Time (s): {round(ga_time, 4)}\n")
    ma_val, ma_time = run_ea_until_target("MA", jobs_data, ppo_target, sa_iters=50, **common_params)
    print(f"MA Makespan: {ma_val}\n"f"MA Time (s): {round(ma_time, 4)}\n")
    #hga_val, hga_time = run_ea_until_target("HGA", jobs_data, ppo_target, sa_fraction=0.2, sa_iters=25, **common_params)
    res = run_HGA(jobs_data, ppo_target)
    hga_val = res['best_makespan']
    # If it stagnated and didn't hit the target, we record "DNF" for time
    hga_time = "DNF" if res['status'] == "DNF" else round(res['time'], 4)
    print(f"HGA Time (s): {hga_time}\n"f"HGA Makespan: {hga_val}\n")
    comparison_data.append({
        "Instance": inst_id,
        "PPO Result": ppo_target,
        "GA Makespan": ga_val,
        "GA Time (s)": round(ga_time, 4),
        "MA Makespan": ma_val,
        "MA Time (s)": round(ma_time, 4),
        "HGA Makespan": hga_val,
        "HGA Time (s)": hga_time
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv("ppo_vs_evolutionary_comparison.csv", index=False)
print("\n--- Final Comparison Table ---")
print(comparison_df.to_string(index=False))
