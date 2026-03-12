import numpy as np

def load_taillard_instance(path):
    """
    Loader for Taillard Job-Shop Scheduling Problem (JSSP) instances.

    Returns:
        proc_times: np.ndarray [n_jobs, n_machines]
        machine_seq: np.ndarray [n_jobs, n_machines]
        n_jobs: int
        n_machines: int
    """
    with open(path, "r") as f:
        tokens = f.read().split()

    if len(tokens) < 2:
        raise ValueError("Invalid Taillard instance")

    n_jobs = int(tokens[0])
    n_machines = int(tokens[1])

    data = list(map(int, tokens[2:]))

    expected_len = n_jobs * n_machines * 2
    if len(data) != expected_len:
        raise ValueError(
            f"Expected {expected_len} values, got {len(data)}"
        )

    proc_times = np.zeros((n_jobs, n_machines), dtype=np.int32)
    machine_seq = np.zeros((n_jobs, n_machines), dtype=np.int32)

    idx = 0
    for job in range(n_jobs):
        for op in range(n_machines):
            machine = data[idx]
            time = data[idx + 1]
            idx += 2

            machine_seq[job, op] = machine
            proc_times[job, op] = time

    return proc_times, machine_seq, n_jobs, n_machines
