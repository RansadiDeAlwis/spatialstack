#!/usr/bin/env bash
# Wait for the image-benchmark eval to finish and for >=30 GB free GPU memory, then run VSI-Bench.
# Retries (up to 20x) if it OOMs because another user grabbed the GPU.
ROOT=/home/anujaya/ransadi/SpatialStack; L=$ROOT/logs/eval_full_vsibench.log
while pgrep -f "lmms_eval.*sparbench\|lmms_eval.*mmsibench\|run_eval.sh spatialstack cvbench" >/dev/null || pgrep -f "eval_full_image" >/dev/null || pgrep -f "tasks (sparbench|mmsibench)" > /dev/null || pgrep -f -- "--tasks mmsibench" >/dev/null || pgrep -f -- "--tasks sparbench" >/dev/null; do sleep 60; done
for try in $(seq 1 20); do
  until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -ge 30000 ]; do sleep 120; done
  echo "[queue] try $try $(date)" >> $ROOT/logs/vsibench_queue.log
  $ROOT/scripts_repro/run_eval.sh spatialstack vsibench > $L 2>&1
  if ! grep -q OutOfMemoryError $L && grep -q "^|vsibench" $L; then echo "[queue] done $(date)" >> $ROOT/logs/vsibench_queue.log; exit 0; fi
  echo "[queue] failed (OOM?) $(date)" >> $ROOT/logs/vsibench_queue.log; sleep 300
done
