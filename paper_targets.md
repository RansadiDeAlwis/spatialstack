# SpatialStack: reported numbers (reproduction targets)

Sources: arXiv 2603.27437 (v3 HTML), HF card Journey9ni/SpatialStack-Qwen3.5-4B, spatial-stack.github.io.
The released HF checkpoint = paper's "SpatialStack-5B (Qwen3.5)" row (HF card quotes the same 67.5 / 85.5).

## VSI-Bench
| Model | Avg | ObjCount | AbsDist | ObjSize | RoomSize | RelDist | RelDir | RoutePlan | ApprOrder |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B (base) | 53.6 | 56.5 | 36.5 | 67.5 | 53.8 | 60.3 | 57.5 | 34.0 | 62.3 |
| SpatialStack (Qwen3.5) | 67.5 | 71.0 | 55.6 | 69.1 | 68.2 | 67.3 | 84.1 | 41.2 | 83.5 |

## CV-Bench (2D / 3D / Avg)
- Qwen3.5-4B base: 79.7 / 90.2 / 85.0
- SpatialStack (Qwen3.5): 78.9 / 92.2 / 85.5

## Table 2/6 (final SpatialStack row): VSI | SPAR | BLINK-Spatial | CV | Overall
- SpatialStack final: 67.52 | 71.39 | 52.12 | 85.53 | 69.14
- Qwen3.5 fine-tuned, no geometry: 64.76 | 68.75 | 56.10 | 84.49 | 68.52

## Table 4 (general): MMBench | Video-MME | BLINK | TempCompass
- Qwen3.5-4B: 83.25 | 62.44 | 61.12 | 66.84
- SpatialStack: 83.42 | 63.74 | 55.46 | 69.37

## Training recipe (paper 5.1 + appendix)
SPAR 140k + LLaVA-Hound 38.3k + VLM3R-ScanNet 31.1k + VSI appearance-order 1.9k (≈212k), 1 epoch,
global batch 64, lr 1e-5, AdamW, cosine, warmup 0.03, ZeRO-2, 4–8 frames/video, VGGT + ViT frozen.
Paper: 32×A100-80GB; repo reference: 64×H200.
