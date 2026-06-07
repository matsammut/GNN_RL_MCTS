  #!/usr/bin/env bash
# ── run_detached.sh ───────────────────────────────────────────────────────────
# Runs the CQL pipeline detached, capturing both stdout+stderr so nothing
# is lost when the terminal closes or the process is OOM-killed.
#
# Usage:
#   chmod +x run_detached.sh
#   ./run_detached.sh
#
# Monitor:
#   tail -f logs/run_20260510_1234/combined.log     # everything
#   tail -f logs/run_20260510_1234/stderr.log        # errors only
#   cat  logs/run_20260510_1234/pid                  # process id
#   kill $(cat logs/run_20260510_1234/pid)           # stop it
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
PYTHON=/jss_project/venv/bin/python3
SCRIPT=/jss_project/RL-Job-Shop-Scheduling/PPO/warm_obj2_20260529.py
LOG_BASE=/jss_project/RL-Job-Shop-Scheduling/PPO/checkpoint_results
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_DIR="${LOG_BASE}/${TIMESTAMP}_obj2"
mkdir -p "${LOG_DIR}"

# ── arguments ─────────────────────────────────────────────────────────────────
ARGS=(
    --bks       /jss_project/RL-Job-Shop-Scheduling/PPO/bks.json
    --eval-instances
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta51
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta52
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta53
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta54
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta55
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta56
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta57
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta58
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta59
    --num-train-instances 125
    --evo-alg GA
    --n-workers 33
    --evo-gens 300
    --evo-pop 250
    --evo-sa-iters 100
    --evo-target-gap 5.0
    --eval-target-gap 15.0
    --nstep-returns 15
    --expert-episodes 35
    --random-episodes 15
    --priority-alpha 0.6
    --priority-beta 0.4
    --cql-epochs 500
    --cql-target-gap 12.0
    --cql-alpha 1.0
    --cql-lr 3e-4
    --cql-batch 48000
    --cql-tau 0.005
    --cql-target-update-every 100
    --mlp-hidden 1024 1024 512 256
    --mlp-dropout 0.25
    --out "${LOG_DIR}/checkpoint"
)

# ── launch ────────────────────────────────────────────────────────────────────
echo "Launching — logs in ${LOG_DIR}"
echo "  combined : ${LOG_DIR}/combined.log"
echo "  stderr   : ${LOG_DIR}/stderr.log"

# -u = unbuffered python output so lines appear immediately in the log
nohup "${PYTHON}" -u "${SCRIPT}" "${ARGS[@]}" \
    > >(tee -a "${LOG_DIR}/combined.log")       \
    2> >(tee -a "${LOG_DIR}/stderr.log" >&2)    \
    &

PID=$!
echo "${PID}" > "${LOG_DIR}/pid"
echo "PID ${PID} — to follow: tail -f ${LOG_DIR}/combined.log"

# Write a small status script next to the log
cat > "${LOG_DIR}/status.sh" <<EOF
#!/usr/bin/env bash
PID=\$(cat "${LOG_DIR}/pid" 2>/dev/null)
if kill -0 "\${PID}" 2>/dev/null; then
    echo "RUNNING  (pid=\${PID})"
    ps -p "\${PID}" -o pid,etime,%cpu,%mem,vsz --no-headers
else
    echo "STOPPED"
fi
tail -n 30 "${LOG_DIR}/combined.log"
EOF
chmod +x "${LOG_DIR}/status.sh"
echo "Status  : ${LOG_DIR}/status.sh"
