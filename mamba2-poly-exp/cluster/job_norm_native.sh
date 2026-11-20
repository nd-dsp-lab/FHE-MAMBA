#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_norm_native
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# No `#$ -M` / `#$ -m` on purpose (see job_norm_pathA.sh). One path per job, after
# job_norm_stats_train.sh:
#
#     qsub -M you@nd.edu -m abe -hold_jid <stats job> -v NATIVE_PATH=P cluster/job_norm_native.sh
#
# Quick evaluation of the FHE-native RMSNorm replacements, each distilled from the
# pretrained model with only the replacement's scalars and gammas trainable:
#   A  learned constant divisor, 0 levels           (norm/const_norm.py)
#   P  learned quadratic in v, ~3 levels            (norm/native_norm.py)
#   D  DyT, gamma * tanh(alpha x), tanh exact here  (norm/native_norm.py)
# Stages: A1 = 24 pre-norms, G1 = 24 gated norms, A3 = all 49. A CPU smoke test
# had A and P overflow to NaN before step 0 with pre + gated replaced together,
# so G1 judges the gated site on its own. A failed stage is reported, not fatal.
#
# fp32, not fp16: sm_75 has no bf16 and these trainers have no GradScaler.
: "${FHEMAMBA_CONDA_ENV:=/groups/tjung/$USER/envs/fhemamba}"
: "${HF_HOME:=/groups/tjung/$USER/FHEMAMBA/hf-cache}"
export FHEMAMBA_CONDA_ENV HF_HOME

source cluster/env.sh
source cluster/job_common.sh

: "${NATIVE_PATH:=A}"
: "${NATIVE_STAGES:=A1 G1 A3}"
: "${NATIVE_BUDGET:=1000000}"
echo "[native] path=$NATIVE_PATH stages='$NATIVE_STAGES' budget=$NATIVE_BUDGET"
if [ ! -s norm/norm_stats_train.json ]; then
  echo "norm/norm_stats_train.json missing: run cluster/job_norm_stats_train.sh first"; exit 1
fi

COMMON="--backend official --device cuda --dtype float32 \
        --norm-stats norm/norm_stats_train.json --stats-length 512 \
        --train-data wikitext103 --train-split train \
        --eval-data wikitext2 --eval-split validation \
        --eval-lengths 512 2048 --eval-blocks 60 \
        --micro-batch 2 --grad-accum 8 --precision-bits 20"

for STAGE in $NATIVE_STAGES; do
  echo; echo "############ path $NATIVE_PATH $STAGE, $NATIVE_BUDGET tokens, $(date) ############"
  python norm/distill_norm.py --path "$NATIVE_PATH" --stage "$STAGE" \
      --token-budget "$NATIVE_BUDGET" $COMMON \
      --outdir "runs/norm_native/${NATIVE_PATH}_${STAGE}_${NATIVE_BUDGET}tok" \
      || echo "=== path $NATIVE_PATH $STAGE FAILED (exit $?) ==="
done
echo "native norm job finished at $(date)"
