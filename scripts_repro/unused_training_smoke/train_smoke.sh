#!/usr/bin/env bash
# 1-GPU training smoke test of the SpatialStack recipe (same hyper-params as scripts/train/train.sh
# except: ZeRO-2 CPU optimizer offload, smoke dataset, global batch 8, no checkpoint saving).
set -euo pipefail
ROOT=/home/anujaya/ransadi/SpatialStack; E=$ROOT/env; cd $ROOT/repo
export PATH=$E/bin:$PATH CUDA_HOME=$E HF_HOME=$PWD/cache/hf-home HUGGINGFACE_HUB_CACHE=$PWD/cache/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL_PATH=${MODEL_PATH:-Journey9ni/SpatialStack-Qwen3.5-4B}; GA=${GA:-8}; MAX_STEPS=${MAX_STEPS:-30}
OUT=$ROOT/train_smoke_out; mkdir -p $OUT
torchrun --nproc_per_node=1 --master_port=22233 $ROOT/scripts_repro/train_nosave.py \
  --model_name_or_path "$MODEL_PATH" --tune_mm_llm True --tune_mm_vision False --tune_mm_mlp False \
  --dataset_use "vlm3r_scannet%100" --output_dir $OUT --cache_dir ./cache --bf16 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps $GA --max_steps $MAX_STEPS \
  --learning_rate 1e-5 --mm_projector_lr 1e-5 --vision_tower_lr 1e-6 --optim adamw_torch \
  --model_max_length 12800 --data_flatten False --max_pixels $((576*28*28)) --min_pixels $((16*28*28)) \
  --base_interval 2 --video_max_frames 8 --video_min_frames 4 \
  --video_max_frame_pixels $((1664*28*28)) --video_min_frame_pixels $((256*28*28)) \
  --num_train_epochs 1 --warmup_ratio 0.03 --lr_scheduler_type cosine --weight_decay 0.01 \
  --logging_steps 1 --save_strategy no --deepspeed $ROOT/scripts_repro/zero2_offload_smoke.json \
  --gradient_checkpointing --dataloader_num_workers 2 --group_by_modality_length true --seed 0 --report_to none \
  --use_geometry_encoder True --geometry_encoder_type vggt --geometry_encoder_path facebook/VGGT-1B \
  --feature_fusion_method deepstack_language_add --geometry_fusion_layers 0 1 2 --geometry_encoder_layers 11 17 23 \
  2>&1 | tee $OUT/train.log
