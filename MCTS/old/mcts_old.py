import math
import random
from env import JSSPEnvironment
from default_config import load_taillard_instance


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}     # action → MCTSNode
        self.N = 0             # visit count
        self.W = 0.0           # total value

    def Q(self):
        return self.W / (self.N + 1e-8)

    def select_child(node, env, c=1.4):
        best_score = -float("inf")
        best_action = None
        best_child = None

        for action, child in node.children.items():
            ucb = (
                child.Q()
                + c * math.sqrt(math.log(node.N + 1) / (child.N + 1))
            )
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(node, env):
        legal = env.legal_actions(node.state)

        for action in legal:
            if action not in node.children:
                next_state = env.step(node.state, action)
                node.children[action] = MCTSNode(
                    state=next_state,
                    parent=node
                )

    def rollout(state, env):
        current_state = state

        while not env.is_terminal(current_state):
            action = random.choice(env.legal_actions(current_state))
            current_state = env.step(current_state, action)

        return -env.makespan(current_state)

    def backup(node, value):
        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent

def mcts_simulation(root, env):
    node = root

    # Selection
    while node.children:
        _, node = select_child(node, env)

    # Expansion
    if not env.is_terminal(node.state):
        expand(node, env)
        # pick one child to rollout
        node = next(iter(node.children.values()))

    # Simulation
    value = rollout(node.state, env)

    # Backup
    backup(node, value)

def mcts_search(env, root_state, n_simulations=500):
    root = MCTSNode(root_state)

    for _ in range(n_simulations):
        mcts_simulation(root, env)

    return root

def extract_schedule(env, root, n_simulations=500):
    schedule = []
    state = root.state

    while not env.is_terminal(state):
        root = mcts_search(env, state, n_simulations)

        action = max(
            root.children.items(),
            key=lambda kv: kv[1].N
        )[0]

        schedule.append(action)
        state = env.step(state, action)

    return schedule

proc, machines, nj, nm = load_taillard_instance("ta40")
env = JSSPEnvironment(proc, machines)

state = env.initial_state()
schedule = extract_schedule(env, state, n_simulations=200)

final_state = state
for job in schedule:
    final_state = env.step(final_state, job)

print("Final makespan:", env.makespan(final_state))