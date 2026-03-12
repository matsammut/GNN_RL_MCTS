import json
from pathlib import Path
import math

RUN_DIR = Path("ray_results/PPO_ta72")
RESULTS = RUN_DIR / "result.json"

# Best-known makespan for Taillard ta72
BKS = 5181

def gap(makespan, bks=BKS):
    return (makespan - bks) / bks * 100.0

# Storage
global_min = math.inf
global_min_it = None

last_iter_metrics = None

# Helper to pull any variant of makespan keys
def extract_ms_from_record(rec):
    # RLlib often nests under "custom_metrics"
    cm = rec.get("custom_metrics", {})
    # accept both "make_span_*" and "makespan_*"
    candidates = {}
    for k, v in cm.items():
        lk = k.lower()
        if ("make_span" in lk or "makespan" in lk) and isinstance(v, (int, float)):
            candidates[lk] = float(v)
    return candidates

# Stream result.json
with open(RESULTS, "r") as f:
    for i, line in enumerate(f):
        rec = json.loads(line)
        ms = extract_ms_from_record(rec)
        if ms:
            # track global min
            for k, v in ms.items():
                if ("min" in k) and v < global_min:
                    global_min = v
                    global_min_it = rec.get("training_iteration", i)
            # remember last iteration's snapshot
            last_iter_metrics = {
                "iteration": rec.get("training_iteration", i),
                "make_span_min": ms.get("make_span_min", ms.get("makespan_min")),
                "make_span_mean": ms.get("make_span_mean", ms.get("makespan_mean")),
                "make_span_max": ms.get("make_span_max", ms.get("makespan_max")),
            }

report = {}

# Global best (across all iterations seen)
if global_min is not None and global_min < math.inf:
    report["global_best_makespan"] = global_min
    report["global_best_gap_%"] = gap(global_min)
    report["global_best_at_iteration"] = global_min_it

# Last-iteration snapshot
if last_iter_metrics:
    for k, v in last_iter_metrics.items():
        if k == "iteration":
            continue
        if v is not None:
            report[f"last_iter_{k}"] = v
            report[f"last_iter_{k}_gap_%"] = gap(v)

# Always include BKS
report["BKS_ta72"] = BKS

# Pretty print
from pprint import pprint
pprint(report)

