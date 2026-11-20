#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_norm_hrobust
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# No `#$ -M` / `#$ -m` on purpose (see job_norm_pathA.sh):
#
#     qsub -M you@nd.edu -m abe -v PYTHONUNBUFFERED=1 cluster/job_norm_hybrid_robust.sh
#
# Robustness of the asymmetric-range hybrid before claiming it. Job 1770027 had
# b = max x 30, a = q1e-3 at 24.36 / 20.82 (L = 512 / 2048), zero clamps above b, but on
# only 60 blocks per length, and one overflowing token makes the ppl inf. Here:
#   - EVERY block of wikitext2 validation and test (~120 x 2048 and ~30 x 8192 each)
#   - L = 8192, 4x the training length: the out-of-range tail grew with L before
#   - b = max x 10 and x 30 (a = q1e-3); the log prints the highest v/b anywhere, i.e.
#     how much of the headroom above the calibrated max was actually used
# Same weights (range-trained R_m0.05) and calibration (6 x 8192 train tokens) as 1770027.
# Eval only; the student-before pass is skipped (it equals after with no training).
#
# Job 1771112, all finite (teacher -> hybrid; highest v/b anywhere in brackets):
#            valid L=2048        valid L=8192        test L=2048         test L=8192
#   x30   21.50 -> 22.15 (.36)  23.26 -> 24.23 (.32) 20.51 -> 21.12 (.78) 23.01 -> 24.36 (.29)
#   x10   same ppl, 1.6e-7 above b (v/b 1.06)        same ppl, 4.2e-7 above b (v/b 2.35)
#   depth  x30 9-13 + clamp, x10 8-12 + clamp; below-a clamps ~6e-4 (harmless)
: "${FHEMAMBA_CONDA_ENV:=/groups/tjung/$USER/envs/fhemamba}"
: "${HF_HOME:=/groups/tjung/$USER/FHEMAMBA/hf-cache}"
export FHEMAMBA_CONDA_ENV HF_HOME

source cluster/env.sh
source cluster/job_common.sh

: "${INIT:=runs/norm_hybrid/R_m0.05_lam1.0_in0.1_5000000tok/trained_state.pt}"
if [ ! -s "$INIT" ]; then
  echo "$INIT missing: run cluster/job_norm_hybrid_range.sh first"; exit 1
fi
: "${HIS:=30 10}"
: "${Q:=1e-3}"
: "${SPLITS:=validation test}"

COMMON="--backend official --device cuda --dtype float32 \
        --norm-stats norm/norm_stats_train.json --stats-length 512 \
        --train-data wikitext103 --train-split train --eval-data wikitext2 \
        --eval-lengths 2048 8192 --eval-blocks 100000 \
        --micro-batch 1 --grad-accum 4 --unfreeze adjacent \
        --path H --init-from $INIT --calib-seq-len 8192 --calib-blocks 6 \
        --cheb-target 12 --ramp norm_pre:0.0-0.0 --token-budget 0 \
        --calib-lo-quantile $Q"

for S in $SPLITS; do
  for HI in $HIS; do
    tag="robust_hi${HI}_q${Q}_${S}"
    echo; echo "############ $tag: b = max x $HI, a = q$Q, wikitext2 $S, all blocks, $(date) ############"
    python norm/distill_norm.py $COMMON --eval-split "$S" --calib-hi-factor "$HI" \
        --outdir "runs/norm_hybrid/$tag" || echo "=== $tag FAILED (exit $?) ==="
  done
done
echo "robustness job finished at $(date)"
