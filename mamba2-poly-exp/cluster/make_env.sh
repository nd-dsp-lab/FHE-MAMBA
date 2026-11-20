#!/bin/bash
# Build FHEMAMBA's OWN conda env on CRC. Run it once, by hand, on the login node:
#
#   cd /groups/tjung/$USER/FHEMAMBA && bash cluster/make_env.sh
#
# Until now every job borrowed `pdpo`, the DP-GRPO project's env. That coupled the
# two projects: we could not add causal_conv1d (PROGRESS.md, the skipped
# whole-mixer parity tests) without changing which conv path DP-GRPO's running
# jobs take, and an upgrade over there could silently move our numbers.
#
# The new env is a CLONE of pdpo, so it starts byte-for-byte on the versions every
# existing result was produced with (python 3.10.18, torch 2.3.1+cu121, triton
# 2.3.1, mamba_ssm 2.2.2). From then on it is ours to change.
#
# It lives in the GROUP space next to the repo, never in $HOME (100 G, has hit 96%
# twice), and so do conda's package cache and pip's cache. NEVER run this from a
# batch job -- see the top of env.sh for what happened the last time.

#
# `bash cluster/make_env.sh --verify` re-runs only the checks and lock files on an
# env that already exists, without rebuilding it.

set -euo pipefail

VERIFY_ONLY=0
if [ "${1:-}" = "--verify" ]; then VERIFY_ONLY=1; shift; fi

if [ -n "${JOB_ID:-}" ]; then
  echo "FATAL: refusing to build an env inside a batch job (JOB_ID=$JOB_ID)."
  echo "       Run this by hand on crcfe01."
  exit 1
fi

_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FHEMAMBA_ROOT="${FHEMAMBA_ROOT:-$(dirname "$_DIR")}"
DEST="${1:-$(dirname "$FHEMAMBA_ROOT")/envs/fhemamba}"   # /groups/tjung/$USER/envs/fhemamba
SRC="${FHEMAMBA_CLONE_FROM:-$HOME/.conda/envs/pdpo}"
ENVS_DIR="$(dirname "$DEST")"

echo "[make_env] source : $SRC"
echo "[make_env] dest   : $DEST"

if [ "$VERIFY_ONLY" -eq 1 ]; then
  if [ ! -x "$DEST/bin/python" ]; then echo "FATAL: --verify, but no env at $DEST."; exit 1; fi
elif [ -e "$DEST" ]; then
  echo "FATAL: $DEST already exists. Remove it yourself first if you mean to rebuild,"
  echo "       or pass --verify to re-check it."
  exit 1
fi
if [ ! -x "$SRC/bin/python" ]; then
  echo "FATAL: no python at $SRC/bin/python -- set FHEMAMBA_CLONE_FROM to the env to clone."
  exit 1
fi
case "$DEST" in
  "$HOME"/*) echo "FATAL: $DEST is under \$HOME. Put the env in the group space."; exit 1 ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  if type module >/dev/null 2>&1; then module load conda 2>/dev/null || true; fi
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "FATAL: conda is not on PATH (tried \`module load conda\`)."
  exit 1
fi

mkdir -p "$ENVS_DIR"

# Space: the clone is about the size of the source; ask for double, for safety.
SRC_GB=$(du -sBG "$SRC" 2>/dev/null | cut -f1 | tr -dc '0-9')
NEED_GB=$(( ${SRC_GB:-8} * 2 ))
DEST_AVAIL=$(df -BG --output=avail "$ENVS_DIR" | tail -1 | tr -dc '0-9')
HOME_AVAIL=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "[make_env] source is ${SRC_GB:-?} GB; free: ${DEST_AVAIL} GB on $ENVS_DIR, ${HOME_AVAIL} GB on \$HOME"
if [ "$DEST_AVAIL" -lt "$NEED_GB" ]; then
  echo "FATAL: need ${NEED_GB} GB free on $ENVS_DIR."
  exit 1
fi
if [ "$HOME_AVAIL" -lt 2 ]; then
  echo "FATAL: \$HOME has under 2 GB free; conda still writes small files there."
  exit 1
fi

# Writes go to the group space; the existing $HOME cache is only READ from, so the
# clone does not re-download what pdpo already pulled.
export CONDA_PKGS_DIRS="$ENVS_DIR/pkgs,$HOME/.conda/pkgs"
export PIP_CACHE_DIR="$ENVS_DIR/pip-cache"
export PYTHONNOUSERSITE=1

if [ "$VERIFY_ONLY" -eq 0 ]; then
  conda create -y -p "$DEST" --clone "$SRC"
fi

# Verify: same packages at the same versions, and the imports the jobs need.
# pip/setuptools/wheel are left out of the comparison: pdpo had pip upgraded to
# 25.3 by pip itself, and the clone comes back with conda's own pip build
# (`pip @ file:///home/conda/feedstock_root/...`). That changes the installer,
# not anything an experiment imports. conda-explicit.txt still records them.
LOCK_DIR="$FHEMAMBA_ROOT/cluster/env-lock"
mkdir -p "$LOCK_DIR"
_freeze() { "$1/bin/python" -m pip freeze --all | grep -vE '^(pip|setuptools|wheel)( |==|@)' | sort; }
_freeze "$SRC"  > "$LOCK_DIR/.pdpo-freeze.tmp"
_freeze "$DEST" > "$LOCK_DIR/pip-freeze.txt"
if ! diff -u "$LOCK_DIR/.pdpo-freeze.tmp" "$LOCK_DIR/pip-freeze.txt"; then
  echo "FATAL: the clone's packages differ from $SRC (diff above). $DEST is NOT ready."
  exit 1
fi
rm -f "$LOCK_DIR/.pdpo-freeze.tmp"
conda list -p "$DEST" --explicit > "$LOCK_DIR/conda-explicit.txt"

"$DEST/bin/python" - <<'PYEOF'
import importlib.util as u, sys
missing = [m for m in ("torch", "triton", "mamba_ssm", "transformers", "einops", "pyarrow", "numpy")
           if u.find_spec(m) is None]
if missing:
    sys.exit("FATAL: missing in the clone: " + ", ".join(missing))
import torch, triton, mamba_ssm
print(f"[make_env] python {sys.version.split()[0]}  torch {torch.__version__}  "
      f"triton {triton.__version__}  mamba_ssm {mamba_ssm.__version__}")
PYEOF

echo "[make_env] done. Lock files: $LOCK_DIR/{pip-freeze.txt,conda-explicit.txt}"
echo "[make_env] cluster/env.sh now picks $DEST up automatically."
echo "[make_env] Run cluster/job_preflight.sh once before trusting it with real jobs."
