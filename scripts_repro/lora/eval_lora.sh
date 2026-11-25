#!/usr/bin/env bash
# Evaluate a (merged) SpatialStack-Qwen3.5 checkpoint with the repo's eval.sh, using the same settings
# as the baseline reproduction (32 frames, max_length 12800, thinking disabled, VGGT-1B).
#
# Usage: bash repo/src/lora/eval_lora.sh <model_dir> <bench1,bench2,...> [TAG] [LIMIT]
#   e.g. bash repo/src/lora/eval_lora.sh /mnt/4dgs/spatialstack/lora_runs/scannetpp_lora5k/merged vsibench_scannetpp
#        bash repo/src/lora/eval_lora.sh <model_dir> vsibench,cvbench,blink_spatial,sparbench,mmsibench
# Results: results/<TAG>/<bench>/   (TAG defaults to "lora"; with LIMIT -> results/<TAG>_limit<LIMIT>/)
set -euo pipefail

ROOT=/home/anujaya/ransadi/SpatialStack
REPO=$ROOT/repo
ENV=$ROOT/env
MODEL=${1:?model_dir required}
BENCH=${2:?benchmark list required}
TAG=${3:-lora}
LIMIT=${4:-}
VGGT=/mnt/4dgs/spatialstack/models/VGGT-1B

export PATH=$ENV/bin:$PATH CUDA_HOME=$ENV QWEN35_ENV_ROOT=$ENV
export HF_HOME=$REPO/cache/hf-home HUGGINGFACE_HUB_CACHE=$REPO/cache/hub
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} HF_DATASETS_OFFLINE=${HF_HUB_OFFLINE:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PROCESSES_PER_MACHINE=1 LMMS_EVAL_LAUNCHER=torchrun
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_PATH=$MODEL MODEL_IMPL=qwen3_5 BENCHMARKS=$BENCH LIMIT
export MODEL_ARGS_BASE="pretrained=$MODEL,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=$VGGT,disable_thinking=true"
export OUTPUT_PATH=$ROOT/results/$TAG${LIMIT:+_limit$LIMIT}

cd "$REPO"
bash scripts/evaluation/eval.sh
