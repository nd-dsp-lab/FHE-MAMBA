#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_norm_hasym
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# No `#$ -M` / `#$ -m` on purpose (see job_norm_pathA.sh):
#
#     qsub -M you@nd.edu -m abe -v PYTHONUNBUFFERED=1 cluster/job_norm_hybrid_asym.sh
#
# Hybrid with an ASYMMETRIC CKKS range, eval only, from the range-trained weights.
# Job 1768266 (job_norm_hybrid_range.sh, margin 0.05): L = 512 24.35 (better than A_A1's
# 24.43), L = 2048 still inf, clamp hits 1.8e-3 at eval vs ~1e-5 out of range in
# training. Suspected cascade: above b the clamp leaves the output's rms at
# sqrt(v / b) > 1, so later blocks see inflated v and clamp too. Below a the rms is
# sqrt(v / a) < 1: it shrinks and does not compound. So:
#   D. diagnostic: symmetric margin 0.05, eval with the exact fallback (an oracle):
#      the PRIMARY out-of-range rate per site and how far above b it goes (max v/b)
#   1-3. b = calibrated max x HI, a = calibration quantile Q of v (raised a keeps b / a
#      fittable), real clamp at eval. The calib table prints each site's a, b, depth.
# Compare: control 23.54 / 20.11, A_A1 24.43 / 20.89, R_m0.05 24.35 / inf (L = 512 / 2048).
#
# Job 1770027 (L = 512 / 2048, gated/f depth excl. clamp):
#   D  oracle, sym 0.05      24.35 / 20.82   primary out-of-range 5.5e-5 at 2048 (clamped
#                                            eval: 1.8e-3, so ~30x was cascade); max v/b 17
#   hi4  q1e-2               24.42 / 20.87   depth 7-12, above b 3e-6 (v/b <= 1.9), below a 6.4e-3
#   hi30 q1e-2               24.42 / 20.87   depth 9-13, above b 0
#   hi30 q1e-3               24.36 / 20.82   depth 9-13, above b 0, below a 6e-4  (= oracle)
# Job 1768266 R_m0.5 (symmetric 0.5, depth 8-19) also came out finite: 24.35 / 20.82.
: "${FHEMAMBA_CONDA_ENV:=/groups/tjung/$USER/envs/fhemamba}"
: "${HF_HOME:=/groups/tjung/$USER/FHEMAMBA/hf-cache}"
export FHEMAMBA_CONDA_ENV HF_HOME

source cluster/env.sh
source cluster/job_common.sh

: "${INIT:=runs/norm_hybrid/R_m0.05_lam1.0_in0.1_5000000tok/trained_state.pt}"
if [ ! -s "$INIT" ]; then
  echo "$INIT missing: run cluster/job_norm_hybrid_range.sh first"; exit 1
fi
# "HI:Q" pairs
: "${ASYM:=4:1e-2 30:1e-2 30:1e-3}"
: "${DIAG:=1}"

COMMON="--backend official --device cuda --dtype float32 \
        --norm-stats norm/norm_stats_train.json --stats-length 512 \
        --train-data wikitext103 --train-split train \
        --eval-data wikitext2 --eval-split validation \
        --eval-lengths 512 2048 --eval-blocks 60 \
        --micro-batch 1 --grad-accum 4 --precision-bits 20 --unfreeze adjacent \
        --path H --init-from $INIT --calib-seq-len 8192 --calib-blocks 6 \
        --cheb-target 12 --ramp norm_pre:0.0-0.0 --token-budget 0"

if [ "$DIAG" = 1 ]; then
  echo; echo "############ D: symmetric 0.05, exact fallback at eval (oracle), $(date) ############"
  python norm/distill_norm.py $COMMON --calib-margin 0.05 --eval-oob exact \
      --outdir runs/norm_hybrid/asym_diag_m0.05 || echo "=== D FAILED (exit $?) ==="
fi
for p in $ASYM; do
  HI=${p%%:*}; Q=${p##*:}
  tag="asym_hi${HI}_q${Q}"
  echo; echo "############ $tag: b = max x $HI, a = q$Q, real clamp, $(date) ############"
  python norm/distill_norm.py $COMMON --calib-hi-factor "$HI" --calib-lo-quantile "$Q" \
      --outdir "runs/norm_hybrid/$tag" || echo "=== $tag FAILED (exit $?) ==="
done
echo "asymmetric hybrid job finished at $(date)"
