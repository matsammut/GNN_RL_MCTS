def encode_state(state):
    return np.concatenate([
        state["job_op"],
        state["job_time"],
        state["machine_time"]
    ]).astype(np.float32)
