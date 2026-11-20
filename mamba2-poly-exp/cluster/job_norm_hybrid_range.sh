#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_norm_hrange
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# No `#$ -M` / `#$ -m` on purpose (see job_norm_pathA.sh):
#
#     qsub -M you@nd.edu -m abe -v PYTHONUNBUFFERED=1 cluster/job_norm_hybrid_range.sh
#
# Range-aware hybrid: 24 pre-norms -> constant divisor (0 levels), 24 gated + norm_f
# -> clamped CKKS Chebyshev 1/sqrt. job_norm_hybrid.sh (job 1756344) showed the hybrid
# is exact at L = 512 (24.43 = A_A1) but inf at L = 2048: 0.1-0.2% of gated-norm
# inputs land above the fitted range, the clamp stops normalising them, and the
# overflow compounds. Calibrating wider cannot fix it (margin 1.0 is unfittable).
# So teach the backbone to stay inside the circuit's range:
#   - start from job 1756344's trained A_A1 (runs/norm_hybrid/A_A1_20000000tok)
#   - calibrate each gated/f range on it ONCE and keep it fixed (the FHE circuit)
#   - hinge penalty on log v outside the range shrunk by RANGE_INNER of its log width
#     per side, summed over positions like the KD term (--lambda-range)
#   - while training, an out-of-range input gets the exact 1/sqrt, not the clamp, so
#     a stray token cannot send the run to NaN as it did at step 37 (--train-oob exact)
#   - train at L = 2048, where the tail appeared
#   - evaluate with the REAL clamp; the log prints the clamp-hit rate
# Runs: margin 0.05 (depth 7-12 + clamp) and 0.5 (8-19 + clamp).
# Compare: control_20000000tok 23.54 / 20.11, A_A1 24.43 / 20.89 (L = 512 / 2048).
: "${FHEMAMBA_CONDA_ENV:=/groups/tjung/$USER/envs/fhemamba}"
: "${HF_HOME:=/groups/tjung/$USER/FHEMAMBA/hf-cache}"
export FHEMAMBA_CONDA_ENV HF_HOME

source cluster/env.sh
source cluster/job_common.sh

A_STATE=runs/norm_hybrid/A_A1_20000000tok/trained_state.pt
if [ ! -s "$A_STATE" ]; then
  echo "$A_STATE missing: run cluster/job_norm_hybrid.sh phase 1 first"; exit 1
fi
: "${R_BUDGET:=5000000}"
: "${R_MARGINS:=0.05 0.5}"
: "${R_LAMBDA:=1.0}"
: "${RANGE_INNER:=0.1}"

COMMON="--backend official --device cuda --dtype float32 \
        --norm-stats norm/norm_stats_train.json --stats-length 512 \
        --train-data wikitext103 --train-split train \
        --eval-data wikitext2 --eval-split validation \
        --eval-lengths 512 2048 --eval-blocks 60 \
        --train-seq-len 2048 --micro-batch 1 --grad-accum 4 --precision-bits 20 \
        --unfreeze adjacent --backbone-lr 2e-5 --lambda-aux 1.0 --save-weights \
        --path H --init-from $A_STATE --calib-seq-len 8192 --calib-blocks 6 \
        --cheb-target 12 --ramp norm_pre:0.0-0.0 --train-oob exact \
        --lambda-range $R_LAMBDA --range-inner $RANGE_INNER --token-budget $R_BUDGET"

for M in $R_MARGINS; do
  tag="R_m${M}_lam${R_LAMBDA}_in${RANGE_INNER}_${R_BUDGET}tok"
  echo; echo "############ $tag, $(date) ############"
  python norm/distill_norm.py $COMMON --calib-margin "$M" --outdir "runs/norm_hybrid/$tag" \
      || echo "=== $tag FAILED (exit $?) ==="
done
echo "range-aware hybrid job finished at $(date)"
