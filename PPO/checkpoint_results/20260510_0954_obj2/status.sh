#!/usr/bin/env bash
PID=$(cat "/jss_project/RL-Job-Shop-Scheduling/PPO/checkpoint_results/20260510_0954_obj2/pid" 2>/dev/null)
if kill -0 "${PID}" 2>/dev/null; then
    echo "RUNNING  (pid=${PID})"
    ps -p "${PID}" -o pid,etime,%cpu,%mem,vsz --no-headers
else
    echo "STOPPED"
fi
tail -n 30 "/jss_project/RL-Job-Shop-Scheduling/PPO/checkpoint_results/20260510_0954_obj2/combined.log"
