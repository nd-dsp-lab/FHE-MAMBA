#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_norm_hybrid
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# No `#$ -M` / `#$ -m` on purpose (see job_norm_pathA.sh):
#
#     qsub -M you@nd.edu -m abe -v PYTHONUNBUFFERED=1 cluster/job_norm_hybrid.sh
#
# The hybrid, all 49 norms replaced in ONE model:
#   24 pre-norms        -> Path A constant divisor (0 levels)
#   24 gated + norm_f   -> CKKS Chebyshev 1/sqrt, ideal clamp, >= 12 bits
#
# Two phases, because the CKKS ranges must be calibrated on the model they run in.
# Ranges taken from the EXACT model (ckks_t12_m05_g.json, +0.0005 ppl there) sent the
# untrained hybrid to inf in a CPU smoke test: constant pre-norms move the gated
# norms' v out of [a, b], and a clamped 1/sqrt stops normalising above b.
#   1. A_A1, 20M tokens, exactly job 1724727's run (+0.78 vs control at L=2048), now
#      saving its trained tensors
#   2. H: load them, calibrate each gated/f norm's v range on THIS model (6 x 8192
#      train tokens), install the clamped Chebyshev, evaluate with no further
#      training, at log margins 0.05 / 0.5 / 1.0 (wider = higher degree, fewer clamp
#      hits). A clamp hit is not a small error: above b the norm stops normalising,
#      and in the smoke test one site clamping 70% of a batch's tokens sent every
#      later block to inf. The log prints each margin's clamp-hit rate.
#   3. a short fine-tune with the hybrid in place, at H_MARGIN (default 0.5)
# Compare with control_20000000tok (20.11 at L=2048) and A_A1_20000000tok (20.89).
#
# Job 1756344: phase 1 reproduced A_A1 (24.43 / 20.89). Hybrid, no training: L=512
# 24.43 at margin 0.05 and 0.5 (= A_A1, the Chebyshev is exact there), L=2048 inf at
# both (clamp hit on 2e-3 / 9e-4 of inputs, worst norm_gated_6 2.7% / norm_gated_10
# 0.8%). Margin 1.0 cannot be fitted (norm_gated.0 R = 3.3e8). Fine-tune at 0.5 hit
# NaN at step 37: norm_gated_6 clamped 455/1024 tokens of one batch. Constant
# pre-norms leave the gated norms' v too heavy-tailed for any fittable clamped range.
: "${FHEMAMBA_CONDA_ENV:=/groups/tjung/$USER/envs/fhemamba}"
: "${HF_HOME:=/groups/tjung/$USER/FHEMAMBA/hf-cache}"
export FHEMAMBA_CONDA_ENV HF_HOME

source cluster/env.sh
source cluster/job_common.sh

if [ ! -s norm/norm_stats_train.json ]; then
  echo "norm/norm_stats_train.json missing: run cluster/job_norm_stats_train.sh first"; exit 1
fi
: "${A_BUDGET:=20000000}"
: "${H_BUDGET:=2000000}"

COMMON="--backend official --device cuda --dtype float32 \
        --norm-stats norm/norm_stats_train.json --stats-length 512 \
        --train-data wikitext103 --train-split train \
        --eval-data wikitext2 --eval-split validation \
        --eval-lengths 512 2048 --eval-blocks 60 \
        --micro-batch 2 --grad-accum 8 --precision-bits 20 \
        --unfreeze adjacent --backbone-lr 2e-5 --lambda-aux 1.0 --save-weights"

A_DIR="runs/norm_hybrid/A_A1_${A_BUDGET}tok"

if [ -s "$A_DIR/trained_state.pt" ]; then
  echo "=== phase 1: reusing $A_DIR/trained_state.pt ==="
else
  echo; echo "############ phase 1: A_A1, $A_BUDGET tokens, $(date) ############"
  python norm/distill_norm.py $COMMON --path A --stage A1 --ramp norm_pre:0.0-0.3 \
      --token-budget "$A_BUDGET" --outdir "$A_DIR" || { echo "=== phase 1 FAILED ==="; exit 1; }
fi

HYB="$COMMON --path H --init-from $A_DIR/trained_state.pt \
     --calib-seq-len 8192 --calib-blocks 6 --cheb-target 12"
: "${H_MARGINS:=0.05 0.5 1.0}"
: "${H_MARGIN:=0.5}"
for M in $H_MARGINS; do
  echo; echo "############ phase 2: hybrid, margin $M, no training, $(date) ############"
  python norm/distill_norm.py $HYB --calib-margin "$M" --ramp none --token-budget 0 \
      --outdir "runs/norm_hybrid/H_eval_m${M}" || echo "=== phase 2 margin $M FAILED (exit $?) ==="
done

# ramp 'norm_pre:0.0-0.0' keeps every beta at 1 but turns on the blend wrapper, which
# is what supplies the aux loss its targets (the exact norm of each site's own input)
echo; echo "############ phase 3: hybrid fine-tune, margin $H_MARGIN, $H_BUDGET tokens, $(date) ############"
python norm/distill_norm.py $HYB --calib-margin "$H_MARGIN" --ramp norm_pre:0.0-0.0 \
    --token-budget "$H_BUDGET" --outdir "runs/norm_hybrid/H_ft_m${H_MARGIN}_${H_BUDGET}tok" \
    || echo "=== phase 3 FAILED (exit $?) ==="
echo "hybrid job finished at $(date)"
