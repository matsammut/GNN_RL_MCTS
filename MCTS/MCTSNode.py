import math

class MCTSNode:
    def __init__(self, state, parent=None, prior=0):
        self.state = state  # Dictionary containing observation and mask
        self.parent = parent
        self.children = {} # action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        
    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, legal_actions, priors):
        """Expands the node using policy probabilities from the model."""
        for action in legal_actions:
            prob = priors[action]
            self.children[action] = MCTSNode(state=None, parent=self, prior=prob)

    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def select_child(self, c_puct):
        """Selects child with highest UCT score."""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            # AlphaZero UCT variant
            u_score = child.value + c_puct * child.prior * (math.sqrt(self.visit_count) / (1 + child.visit_count))
            if u_score > best_score:
                best_score = u_score
                best_action = action
                best_child = child
        return best_action, best_child
