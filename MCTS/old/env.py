import numpy as np

import numpy as np


class JSSPEnvironment:
    """
    Job-Shop Scheduling environment for Taillard instances.
    Designed for Neural Monte Carlo Tree Search.
    """

    def __init__(self, proc_times, machine_seq):
        """
        proc_times: np.ndarray [n_jobs, n_machines]
        machine_seq: np.ndarray [n_jobs, n_machines]
        """
        self.proc_times = proc_times
        self.machine_seq = machine_seq

        self.n_jobs, self.n_machines = proc_times.shape
        self.total_ops = self.n_jobs * self.n_machines

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def initial_state(self):
        """
        Returns the initial empty schedule state.
        """
        return {
            "job_op": np.zeros(self.n_jobs, dtype=np.int32),
            "job_time": np.zeros(self.n_jobs, dtype=np.float32),
            "machine_time": np.zeros(self.n_machines, dtype=np.float32),
            "scheduled": 0
        }

    def is_terminal(self, state):
        """
        Terminal when all operations are scheduled.
        """
        return state["scheduled"] == self.total_ops

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------

    def legal_actions(self, state):
        """
        Legal actions = jobs whose next operation is unscheduled.
        """
        return [
            job for job in range(self.n_jobs)
            if state["job_op"][job] < self.n_machines
        ]

    # ------------------------------------------------------------------
    # Transition function (discrete-event simulation)
    # ------------------------------------------------------------------

    def step(self, state, job):
        """
        Schedule the next operation of `job`.
        Returns a NEW state (does not mutate input).
        """
        op = state["job_op"][job]
        machine = self.machine_seq[job, op]
        duration = self.proc_times[job, op]

        start_time = max(
            state["job_time"][job],
            state["machine_time"][machine]
        )

        finish_time = start_time + duration

        next_state = {
            "job_op": state["job_op"].copy(),
            "job_time": state["job_time"].copy(),
            "machine_time": state["machine_time"].copy(),
            "scheduled": state["scheduled"] + 1
        }

        next_state["job_op"][job] += 1
        next_state["job_time"][job] = finish_time
        next_state["machine_time"][machine] = finish_time

        return next_state

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def makespan(self, state):
        """
        Makespan of the (partial or complete) schedule.
        """
        return float(np.max(state["machine_time"]))

    # ------------------------------------------------------------------
    # Neural encoding
    # ------------------------------------------------------------------

    def encode_state(self, state):
        """
        Fixed-size numeric encoding for neural networks.
        """
        return np.concatenate([
            state["job_op"] / self.n_machines,
            state["job_time"] / (np.max(state["job_time"]) + 1e-8),
            state["machine_time"] / (np.max(state["machine_time"]) + 1e-8)
        ]).astype(np.float32)


# state = {
#     "job_op": np.zeros(n_jobs, dtype=int),     # next operation index per job
#     "job_time": np.zeros(n_jobs),               # earliest start time per job
#     "machine_time": np.zeros(n_machines),       # earliest start time per machine
#     "scheduled": 0                              # number of scheduled ops
# }

# def get_legal_actions(state, n_jobs, n_machines):
#     actions = []
#     for job in range(n_jobs):
#         if state["job_op"][job] < n_machines:
#             actions.append(job)
#     return actions

# def apply_action(state, job, proc_times, machine_seq):
#     op = state["job_op"][job]
#     machine = machine_seq[job, op]
#     duration = proc_times[job, op]

#     start = max(
#         state["job_time"][job],
#         state["machine_time"][machine]
#     )

#     finish = start + duration

#     next_state = {
#         "job_op": state["job_op"].copy(),
#         "job_time": state["job_time"].copy(),
#         "machine_time": state["machine_time"].copy(),
#         "scheduled": state["scheduled"] + 1
#     }

#     next_state["job_op"][job] += 1
#     next_state["job_time"][job] = finish
#     next_state["machine_time"][machine] = finish

#     return next_state


# def is_terminal(state, n_jobs, n_machines):
#     return state["scheduled"] == n_jobs * n_machines


# def makespan(state):
#     return max(state["machine_time"])
