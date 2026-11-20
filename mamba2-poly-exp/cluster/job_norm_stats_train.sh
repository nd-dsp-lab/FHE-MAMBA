#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N fhe_norm_stats
#$ -cwd
#
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err
#
# No `#$ -M` / `#$ -m` on purpose (see job_norm_pathA.sh); submit with
#
#     qsub -M you@nd.edu -m abe cluster/job_norm_stats_train.sh
#
# Per-instance statistics of v = mean(x^2) at all 49 RMSNorms, on wikitext2 TRAIN:
# job_norm_phase1.sh collected them on validation, which is also the evaluation
# set. Paths A / P / D initialise from these (cluster/job_norm_native.sh).
: "${FHEMAMBA_CONDA_ENV:=/groups/tjung/$USER/envs/fhemamba}"
: "${HF_HOME:=/groups/tjung/$USER/FHEMAMBA/hf-cache}"
export FHEMAMBA_CONDA_ENV HF_HOME

source cluster/env.sh
source cluster/job_common.sh

python cluster/stackdump.py 300 norm/collect_norm_stats.py --backend official --device cuda \
    --data wikitext2 --split train --lengths 512 2048 --blocks 48 \
    --out norm/norm_stats_train.json
echo "norm stats finished at $(date)"
