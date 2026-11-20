#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_ckks_invsqrt
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# No `#$ -M` / `#$ -m` on purpose (see job_norm_floor.sh). Only gpu0 of qa-rtx6k-019
# creates CUDA contexts (Nov 2026), so pin it:
#
#     qsub -M you@nd.edu -m abe -l 'gpu_card=1(gpu0)' cluster/job_ckks_invsqrt.sh
#
# The CKKS-only 1/sqrt (fhe/ckks_invsqrt, real_mamba/ckks_invsqrt.py) in all 49 norms,
# paired against exact, with the same calibration as cal8k_m05_g (GATES.md 9d) so
# the rows sit next to the TFHE route's. No range guard exists for this route:
# out-of-range inputs extrapolate, and `out-of-domain` reports how many.
# Job 1717851 without a guard: 1e-4 of inputs escaped and blew up 10-16 of 16 shards
# with all 49 replaced (eval/ckks_invsqrt/ckks_t{12,8}_m05.json); margin 0.5 cannot be
# fitted at all (norm_gated.0 spans 2e7). So now the IDEAL clamp, --range-guard: if it
# recovers the model, a CKKS comparison (sign polynomial) is worth benchmarking.
#   ckks_t12_m05_g   >= 12 bits, margin 0.05, clamped: the configuration bench.cpp timed
#   ckks_t8_m05_g    >= 8 bits, clamped, one level cheaper for most norms
: "${FHEMAMBA_CONDA_ENV:=/groups/tjung/$USER/envs/fhemamba}"
: "${HF_HOME:=/groups/tjung/$USER/FHEMAMBA/hf-cache}"
export FHEMAMBA_CONDA_ENV HF_HOME

source cluster/env.sh
source cluster/job_common.sh

# 24 MB, excluded by the root .gitignore's `*.json`; regenerate if absent
if [ ! -s runs/gate_stats/gate_stats.json ]; then
  echo "=== runs/gate_stats/gate_stats.json missing -- collecting it first ==="
  python collect_gate_stats.py --backend official --device cuda --dtype float16 \
      --blocks 16 --outdir runs/gate_stats
fi

mkdir -p eval/ckks_invsqrt
GATES="invsqrt_pre invsqrt_gated invsqrt_f invsqrt"
COMMON="--backend official --device cuda --shards 16 --lengths 512 2048 4096 8192 \
--calib-split train --calib-seq-len 8192 --invsqrt-impl ckks --gate $GATES"

run() {
  local tag=$1; shift
  echo "=== ckks_invsqrt: $tag ($*) started $(date) ==="
  python eval/seglut_paired.py $COMMON "$@" --out "eval/ckks_invsqrt/${tag}.json" \
      || echo "=== ckks_invsqrt: $tag FAILED (exit $?) ==="
}

run ckks_t12_m05_g  --cheb-target 12 --margin 0.05 --range-guard
run ckks_t8_m05_g   --cheb-target 8  --margin 0.05 --range-guard
echo "ckks_invsqrt job finished at $(date)"
