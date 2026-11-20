#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhemamba_env_ab
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# No `#$ -M` / `#$ -m` on purpose -- see job_preflight.sh. Submit with:
#
#     qsub -M you@nd.edu -m abe cluster/job_env_ab.sh
#
# ENV A/B -- does moving from DP-GRPO's `pdpo` to our own `fhemamba` clone change
# a number? Runs preflight step 5 (the short official-backend evaluation) with the
# SAME code on the SAME card, once per env. The first preflight in the new env gave
# 18.9548 against 18.9538 recorded in pdpo on 2026-09-19, but real_mamba/patch.py
# (official backend only) has changed since, so that comparison mixed two changes.
# Here the code is held fixed and only the env differs.

set -uo pipefail
FAIL=0

echo "=============================================================="
echo "job      : ${JOB_NAME:-?} (${JOB_ID:-?})   queue: ${QUEUE:-?}"
echo "host     : $(hostname)"
echo "started  : $(date)"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

for ENV in "$HOME/.conda/envs/pdpo" "/groups/tjung/$USER/envs/fhemamba"; do
  NAME=$(basename "$ENV")
  printf '\n\033[1m=== %s ===\033[0m\n' "$NAME"
  (
    export FHEMAMBA_CONDA_ENV="$ENV"
    source cluster/env.sh || exit 1
    "$PY" eval_poly_exp.py --transition poly4 --interval-mode per-head --pin-zero \
        --backend official --device cuda --blocks 12 --seq-len 1024 \
        --stats-json runs/part6_transition_stats/transition_stats.json \
        --outdir "runs/env_ab/$NAME" 2>&1 | tail -20
    exit "${PIPESTATUS[0]}"
  ) || { echo ">>> $NAME FAILED"; FAIL=1; }
done

echo
echo "finished : $(date)"
exit $FAIL
