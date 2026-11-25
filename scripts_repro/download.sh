#!/usr/bin/env bash
# Pre-download all models/data used by the SpatialStack reproduction into the
# same HF cache the repo's eval script uses (repo/cache/{hf-home,hub}).
set -uo pipefail
ROOT=/home/anujaya/ransadi/SpatialStack
REPO=$ROOT/repo
export HF_HOME=$REPO/cache/hf-home
export HUGGINGFACE_HUB_CACHE=$REPO/cache/hub
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1   # cached token is expired; all repos are public
export HF_TOKEN=
HF=${HF:-/home/anujaya/venumi/gaussian_cl4d_refactored/.venv/bin/hf}
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

dl() { echo "=== $(date +%T) $*"; for i in 1 2 3; do "$HF" download "$@" && return 0; echo "retry $i"; sleep 10; done; echo "FAILED: $*"; }

# 1. Models
dl Journey9ni/SpatialStack-Qwen3.5-4B
dl facebook/VGGT-1B --include config.json model.safetensors README.md
dl Qwen/Qwen3.5-4B                      # base model (baseline comparison + training init)

# 2. Training annotations
dl Journey9ni/SpatialStackData --repo-type dataset --include "*.json" --local-dir $REPO/data/annotations

# 3. Eval benchmarks (Video-MME, 101 GB, skipped: not spatial, does not fit on disk)
dl nyu-visionx/VSI-Bench   --repo-type dataset
dl nyu-visionx/CV-Bench    --repo-type dataset
dl BLINK-Benchmark/BLINK   --repo-type dataset
dl jasonzhango/SPAR-Bench  --repo-type dataset
dl RunsenXu/MMSI-Bench     --repo-type dataset
echo "=== $(date +%T) ALL DONE"
du -sh $REPO/cache/hub/* $REPO/data/annotations
