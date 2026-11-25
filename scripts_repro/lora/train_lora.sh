#!/usr/bin/env bash
# LoRA fine-tune of SpatialStack-Qwen3.5-4B on the ScanNet++ 5k subset (single GPU).
#
# Usage:
#   RUN_NAME=pilot MAX_STEPS=30 bash repo/src/lora/train_lora.sh     # pilot (memory / speed check)
#   bash repo/src/lora/train_lora.sh                                  # full run (1 epoch, ~313 steps)
# Env overrides: RUN_NAME, MAX_STEPS (-1 = full epoch), LR, GRAD_ACCUM, FRAMES, CUDA_VISIBLE_DEVICES
set -euo pipefail

ROOT=/home/anujaya/ransadi/SpatialStack
REPO=$ROOT/repo
ENV=$ROOT/env
LORA=$REPO/src/lora

RUN_NAME=${RUN_NAME:-scannetpp_lora5k}
MAX_STEPS=${MAX_STEPS:--1}
LR=${LR:-1e-4}
GRAD_ACCUM=${GRAD_ACCUM:-16}
FRAMES=${FRAMES:-8}
OUT=/mnt/4dgs/spatialstack/lora_runs/$RUN_NAME
mkdir -p "$OUT"

export PATH=$ENV/bin:$PATH CUDA_HOME=$ENV
export HF_HOME=$REPO/cache/hf-home HUGGINGFACE_HUB_CACHE=$REPO/cache/hub
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$REPO"
echo "[train_lora] $(date) run=$RUN_NAME max_steps=$MAX_STEPS lr=$LR grad_accum=$GRAD_ACCUM frames=$FRAMES out=$OUT"

args=(
  # model (released SpatialStack-Qwen3.5-4B + frozen VGGT, same geometry settings as the authors)
  --model_name_or_path /mnt/4dgs/spatialstack/models/SpatialStack-Qwen3.5-4B
  --use_geometry_encoder true --geometry_encoder_type vggt
  --geometry_encoder_path /mnt/4dgs/spatialstack/models/VGGT-1B
  --feature_fusion_method deepstack_language_add
  --geometry_encoder_layers 11 17 23 --geometry_fusion_layers 0 1 2
  --tune_mm_llm True --tune_mm_vision False --tune_mm_mlp False   # ignored: LoRA decides what trains
  # data
  --dataset_use scannetpp_lora5k
  --data_flatten False --model_max_length 12800
  --max_pixels $((576*28*28)) --min_pixels $((16*28*28))
  --base_interval 2 --video_max_frames "$FRAMES" --video_min_frames "$FRAMES"
  --video_max_frame_pixels $((1664*28*28)) --video_min_frame_pixels $((256*28*28))
  --group_by_modality_length true --dataloader_num_workers 4
  # optimisation
  --bf16 --optim adamw_torch
  --per_device_train_batch_size 1 --gradient_accumulation_steps "$GRAD_ACCUM"
  --learning_rate "$LR" --mm_projector_lr 1e-5 --vision_tower_lr 1e-6
  --num_train_epochs 1 --max_steps "$MAX_STEPS"
  --warmup_ratio 0.03 --lr_scheduler_type cosine --weight_decay 0.01
  --gradient_checkpointing True --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  # logging / saving (adapter-only checkpoints; merged model written at the end)
  --output_dir "$OUT" --cache_dir ./cache
  --logging_steps 1 --save_strategy steps --save_steps 100
  --seed 0 --report_to none
)

# Plain python (not torchrun): one GPU, no process group, so the Trainer never makes NCCL calls
# (a torchrun world-size-1 run crashed in an NCCL all_gather when the shared GPU was momentarily full).
python "$LORA/train_lora.py" "${args[@]}" 2>&1 | tee -a "$OUT/train.log"
