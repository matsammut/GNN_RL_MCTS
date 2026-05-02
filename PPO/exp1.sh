#!/bin/bash
set -e
PY=python3

${PY} warm_obj1_29042026.py   --bks bks.json   --eval-instances instances/ta41 instances/ta42 instances/ta43 instances/ta44 instances/ta45 instances/ta46 instances/ta47 instances/ta48    instances/ta49 instances/ta50   --evo-alg GA --evo-gens 15 --out checkpoint_results/20260430_ta4_exp1 --num-train-instances 20 --evo-early-stop --bc-epochs 50
${PY} warm_obj1_29042026.py   --bks bks.json   --eval-instances instances/ta51 instances/ta52 instances/ta53 instances/ta54 instances/ta55 instances/ta56 instances/ta57 instances/ta58    instances/ta59 instances/ta60   --evo-alg GA --evo-gens 15 --out checkpoint_results/20260430_ta5_exp1 --num-train-instances 20 --evo-early-stop --bc-epochs 50
${PY} warm_obj1_29042026.py   --bks bks.json   --eval-instances instances/ta61 instances/ta62 instances/ta63 instances/ta64 instances/ta65 instances/ta66 instances/ta67 instances/ta68    instances/ta69 instances/ta70   --evo-alg GA --evo-gens 15 --out checkpoint_results/20260430_ta6_exp1 --num-train-instances 20 --evo-early-stop --bc-epochs 50

echo "All runs complete."
