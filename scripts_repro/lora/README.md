# ScanNet++ LoRA fine-tune of SpatialStack-Qwen3.5-4B

**Goal:** a small improvement on top of the released checkpoint `Journey9ni/SpatialStack-Qwen3.5-4B`
(the paper's "SpatialStack-5B (Qwen3.5)" row). The model is fine-tuned with LoRA on ScanNet++ VSI-style video QA.
The repo defines this data (`VLM3R_SCANNETPP`) but the paper's training mix never used it.
ScanNet++ is the model's weakest VSI-Bench scene source (61.8 vs 67–69 for the other two).

## What is trained
| Component | Paper | Here |
|---|---|---|
| Language model (32 decoder layers) | full fine-tune, lr 1e-5 | LoRA r=16, α=32 on attention (q/k/v/o or in_proj_qkv/in_proj_z/out_proj) + MLP (gate/up/down) of all 32 layers |
| Geometry fusion MLPs (VGGT layers 11/17/23 → LLM layers 0/1/2) | full, lr 1e-5 | **full**, lr 1e-5 (fp32 master weights, saved as bf16) |
| ViT, vision merger, VGGT, embeddings, LM head | frozen | frozen |

200 LoRA layers (30.5M params, lr 1e-4) + fusion MLPs (132.2M params, lr 1e-5) = 162.6M trainable (2.60%).
Data: 5,000 ScanNet++ QAs covering all 856 training scenes (≥1 question per scene; 0 overlap with VSI-Bench test), 8 frames per video (what the authors' 4–8 setting gives for every ScanNet++ video), 1 epoch,
batch 1 × grad-accum 16 (≈313 steps), lr 1e-4 cosine, warmup 3%, gradient checkpointing on.
All other settings match `scripts/train/train.sh`.

## Steps (run from `/home/anujaya/ransadi/SpatialStack`)
```bash
env/bin/python repo/src/lora/make_subset.py                    # 1. build 5k subset (+ checks)
RUN_NAME=pilot MAX_STEPS=30 bash repo/src/lora/train_lora.sh   # 2. pilot: memory, s/step, loss
env/bin/python repo/src/lora/merge_check.py /mnt/4dgs/spatialstack/lora_runs/pilot
bash repo/src/lora/train_lora.sh                               # 3. full run -> lora_runs/scannetpp_lora5k
M=/mnt/4dgs/spatialstack/lora_runs/scannetpp_lora5k/merged
bash repo/src/lora/eval_lora.sh $M vsibench_scannetpp          # 4. quick check (1,458 Qs)
env/bin/python repo/src/lora/compare_lora.py --run lora=results/lora --scannetpp
bash repo/src/lora/eval_lora.sh $M vsibench,cvbench,blink_spatial,sparbench,mmsibench   # 5. full eval
env/bin/python repo/src/lora/compare_lora.py --run lora=results/lora --out results/lora/comparison.md
```

## Files
| File | Purpose |
|---|---|
| `config.py` | paths, subset composition, LoRA hyperparameters |
| `make_subset.py` | stratified 5k ScanNet++ subset; checks videos exist and no test-scene overlap |
| `lora_utils.py` | LoRA target regex, inject (in place), save/load adapter, merge into base weights |
| `train_lora.py` | runs the repo's `train_qwen.train()` with LoRA patched in from outside (repo code untouched) |
| `train_lora.sh` | single-GPU launcher; waits for free GPU memory; output in `/mnt/4dgs/spatialstack/lora_runs/<RUN_NAME>` |
| `merge_check.py` | checks the merged checkpoint against base + adapter |
| `eval_lora.sh` | repo `eval.sh` with the baseline's eval settings |
| `compare_lora.py` | baseline vs LoRA in the paper's Table 3 / Table 4 / Table 2 layouts |

Run outputs: `adapter-step-N/` (adapter-only checkpoints), `adapter-final/`, `merged/` (full checkpoint,
released layout, directly evaluable), `lora_stats.jsonl` (s/step, peak memory, loss), `train.log`.

## Repo additions outside this folder
- `src/qwen_vl/data/__init__.py`: registered `vlm3r_scannetpp` (it was defined but missing from `data_dict`).
- `src/lmms_eval/tasks/vsibench/`: `vsibench_scannetpp.yaml` + `utils.process_docs_scannetpp` (VSI-Bench ScanNet++ subset).
- `data/vlm3r/`: symlinks to the ScanNet++ QAs/videos on `/mnt/4dgs/spatialstack/aweb`.
