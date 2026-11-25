#!/usr/bin/env bash
# Evaluate a model on spatial benchmarks with the repo's eval.sh (paper settings).
# Usage: run_eval.sh <spatialstack|qwen25|base> "<bench1,bench2>" [LIMIT]
set -euo pipefail
ROOT=/home/anujaya/ransadi/SpatialStack
WHICH=${1:-spatialstack}; BENCH=${2:-vsibench}; LIMIT=${3:-}
E=$ROOT/env
[[ $WHICH == qwen25 ]] && { E=$ROOT/env_qwen25; export PYTHONPATH=$E/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}; export HF_DATASETS_CACHE=$ROOT/repo/cache/datasets_qwen25; }   # transformers 4.50 overlay (eval.sh puts env/bin first, so use PYTHONPATH)
export PATH=$E/bin:$PATH CUDA_HOME=$ROOT/env QWEN35_ENV_ROOT=$E
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN= HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} HF_DATASETS_OFFLINE=${HF_HUB_OFFLINE:-0}
export CUDA_VISIBLE_DEVICES=0 PROCESSES_PER_MACHINE=1 LMMS_EVAL_LAUNCHER=torchrun
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIMIT
cd $ROOT/repo
if [[ $WHICH == spatialstack ]]; then
  # Models live as plain files on /mnt/4dgs (exFAT, outside the home quota) once copied + verified.
  LM=/mnt/4dgs/spatialstack/models
  if [[ -f $LM/.verified ]]; then M=$LM/SpatialStack-Qwen3.5-4B; G=$LM/VGGT-1B
  else M=Journey9ni/SpatialStack-Qwen3.5-4B; G=facebook/VGGT-1B; fi
  export MODEL_ARGS_BASE="pretrained=$M,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=$G,disable_thinking=true"
elif [[ $WHICH == qwen25 ]]; then
  # Paper's SpatialStack-4B (Qwen2.5-VL-3B base); repo eval.sh defaults = this model's settings
  M=/mnt/4dgs/spatialstack/models/SpatialStack-Qwen2.5-4B
  export MODEL_ARGS_BASE="pretrained=$M,use_flash_attention_2=true,max_num_frames=32,max_length=12800"
  export MODEL_PATH=$M MODEL_IMPL=spatialstack BENCHMARKS=$BENCH
else
  M=Qwen/Qwen3.5-4B
  export MODEL_ARGS_BASE="pretrained=$M,use_flash_attention_2=true,max_num_frames=32,max_length=12800,disable_thinking=true"
fi
[[ $WHICH == qwen25 ]] || export MODEL_PATH=$M MODEL_IMPL=qwen3_5 BENCHMARKS=$BENCH
export OUTPUT_ROOT=$ROOT/results/$WHICH OUTPUT_PATH=$ROOT/results/$WHICH${LIMIT:+_limit$LIMIT}
bash scripts/evaluation/eval.sh
