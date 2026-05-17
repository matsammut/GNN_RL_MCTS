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
SCRIPT=/jss_project/RL-Job-Shop-Scheduling/PPO/warm_obj2_mlp_20260516.py
LOG_BASE=/jss_project/RL-Job-Shop-Scheduling/PPO/checkpoint_results
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_DIR="${LOG_BASE}/${TIMESTAMP}_obj2"
mkdir -p "${LOG_DIR}"

# ── arguments ─────────────────────────────────────────────────────────────────
ARGS=(
    --bks       /jss_project/RL-Job-Shop-Scheduling/PPO/bks.json
    --eval-instances
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta61
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta62
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta63
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta64
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta65
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta66
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta67
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta68
        /jss_project/RL-Job-Shop-Scheduling/PPO/instances/ta69
    --num-train-instances 200
    --evo-alg GA
    --evo-gens 25
    --evo-early-stop
    --cql-epochs 2000
    --cql-alpha 0.5
    --cql-lr 5e-5
    --cql-batch 1024
    --cql-tau 0.001
    --cql-target-update-every 100
    --mlp-hidden 512 512 256
    --cql-batch 2048
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
