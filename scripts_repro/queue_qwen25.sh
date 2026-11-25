#!/usr/bin/env bash
# After the Qwen3.5 evals (image run + VSI queue) finish, evaluate SpatialStack-Qwen2.5-4B on all
# benchmarks one at a time: waits for >=30 GB free GPU memory, retries on OOM, stops on other errors.
ROOT=/home/anujaya/ransadi/SpatialStack; Q=$ROOT/logs/qwen25_queue.log
gpu_wait() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 30000 ]; do sleep 120; done; }
while pgrep -f "queue_vsibench.sh|run_eval.sh spatialstack" >/dev/null; do sleep 60; done
# sanity check: 4 samples each on an image and a video benchmark
gpu_wait; L=$ROOT/logs/eval_qwen25_sanity.log
$ROOT/scripts_repro/run_eval.sh qwen25 cvbench,vsibench 4 > $L 2>&1
if ! grep -q "^|vsibench" $L; then echo "[qwen25] SANITY FAILED, see $L $(date)" >> $Q; exit 1; fi
echo "[qwen25] sanity ok $(date)" >> $Q
for b in cvbench blink_spatial sparbench mmsibench vsibench; do
  L=$ROOT/logs/eval_qwen25_$b.log
  for try in $(seq 1 20); do
    gpu_wait; echo "[qwen25] $b try $try $(date)" >> $Q
    $ROOT/scripts_repro/run_eval.sh qwen25 $b > $L 2>&1
    if grep -qE "^\|(${b}|blink_spatial)" $L && ! grep -q OutOfMemoryError $L; then echo "[qwen25] $b done $(date)" >> $Q; break; fi
    if ! grep -q OutOfMemoryError $L; then echo "[qwen25] $b non-OOM failure, see $L $(date)" >> $Q; break; fi
    echo "[qwen25] $b OOM, retry in 5 min $(date)" >> $Q; sleep 300
  done
done
echo "[qwen25] ALL DONE $(date)" >> $Q
