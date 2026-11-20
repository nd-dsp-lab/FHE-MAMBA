#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_norm_opt
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# No `#$ -M` / `#$ -m` on purpose (see job_norm_pathA.sh). Only gpu0 of qa-rtx6k-019
# creates CUDA contexts (Nov 2026), so pin it:
#
#     qsub -M you@nd.edu -m abe -l 'gpu_card=1(gpu0)' cluster/job_norm_opt.sh
#
# The optimised round of Paths A and P (norm/distill_norm.py --unfreeze/--ramp). The
# first round (job_norm_native.sh) trained only the replacements, 1M tokens, cold:
# +20 ppl or worse, and every all-49 stage went to NaN. Changes here:
#   - each mixer's in_proj and out_proj train too, against a separate frozen teacher
#   - the family ramps in over the first 30% of training (norm/blend.py)
#   - per-norm loss against the exact norm of the replacement's own input
#   - 5M tokens; P starts from Path A's constant (--poly-init const), degree 1
# ONE FAMILY PER RUN, the other kept exact. A CPU smoke test showed why all 49 cannot
# work at any setting: with pre AND gated norms both non-normalising, a Mamba block is
# ~cubic in its input, the degree compounds over 24 blocks, and outlier tokens reach
# inf as soon as the second family blends in at even 4%, in either order. In FHE the
# kept-exact family would be the CKKS Chebyshev 1/sqrt (job_ckks_invsqrt.sh).
#   control   every norm exact, same training: what the training alone does
#   A_G1/A_A1 constant divisor on the 24 gated norms / the 24 pre-norms
#   P_G1/P_A1 1/sqrt(v) ~ (a0 + a1 v/s)/sqrt(s), 3 levels with v
: "${FHEMAMBA_CONDA_ENV:=/groups/tjung/$USER/envs/fhemamba}"
: "${HF_HOME:=/groups/tjung/$USER/FHEMAMBA/hf-cache}"
export FHEMAMBA_CONDA_ENV HF_HOME

source cluster/env.sh
source cluster/job_common.sh

if [ ! -s norm/norm_stats_train.json ]; then
  echo "norm/norm_stats_train.json missing: run cluster/job_norm_stats_train.sh first"; exit 1
fi
: "${OPT_BUDGET:=5000000}"

COMMON="--backend official --device cuda --dtype float32 \
        --norm-stats norm/norm_stats_train.json --stats-length 512 \
        --train-data wikitext103 --train-split train \
        --eval-data wikitext2 --eval-split validation \
        --eval-lengths 512 2048 --eval-blocks 60 \
        --micro-batch 2 --grad-accum 8 --precision-bits 20 \
        --unfreeze adjacent --backbone-lr 2e-5 --lambda-aux 1.0 \
        --token-budget $OPT_BUDGET"

run() {
  local tag=$1; shift
  echo; echo "############ $tag, $OPT_BUDGET tokens, $(date) ############"
  python norm/distill_norm.py $COMMON "$@" --outdir "runs/norm_opt/${tag}_${OPT_BUDGET}tok" \
      || echo "=== $tag FAILED (exit $?) ==="
}

# job 1721515 (5M): control -0.10, A_A1 +1.94, A_G1 +3.57 at L=2048; P_G1 +486 and
# P_A1 +3443, so P is dropped by default. Pick runs with -v OPT_RUNS="control A_A1 ...".
: "${OPT_RUNS:=A_A1 control A_G1}"
for R in $OPT_RUNS; do
  case $R in
    control) run control --path A --stage A3 --control --ramp none ;;
    A_G1)    run A_G1    --path A --stage G1 --ramp norm_gated:0.0-0.3 ;;
    A_A1)    run A_A1    --path A --stage A1 --ramp norm_pre:0.0-0.3 ;;
    P_G1)    run P_G1    --path P --stage G1 --ramp norm_gated:0.0-0.3 --poly-degree 1 --poly-init const ;;
    P_A1)    run P_A1    --path P --stage A1 --ramp norm_pre:0.0-0.3 --poly-degree 1 --poly-init const ;;
    *)       echo "=== unknown run $R ===" ;;
  esac
done
echo "norm opt job finished at $(date)"
