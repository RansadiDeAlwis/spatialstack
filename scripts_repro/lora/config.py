"""Shared paths and constants for the SpatialStack-Qwen3.5-4B ScanNet++ LoRA fine-tune."""
import os

ROOT = "/home/anujaya/ransadi/SpatialStack"
REPO = os.path.join(ROOT, "repo")
ENV = os.path.join(ROOT, "env")

# Models (local copies of Journey9ni/SpatialStack-Qwen3.5-4B and facebook/VGGT-1B)
MODELS = "/mnt/4dgs/spatialstack/models"
BASE_MODEL = os.path.join(MODELS, "SpatialStack-Qwen3.5-4B")
VGGT = os.path.join(MODELS, "VGGT-1B")

# Data: ScanNet++ VSI-style training QAs from Journey9ni/aweb (train split; 0 overlap with VSI-Bench test)
FULL_QA = os.path.join(REPO, "data/vlm3r/annotations/vsibench_train/merged_qa_scannetpp_train.json")
MEDIA_ROOT = os.path.join(REPO, "data/vlm3r/media")  # contains ScanNetpp/videos/train/*.mp4
SUBSET_NAME = "scannetpp_lora5k"
SUBSET_QA = "/mnt/4dgs/spatialstack/aweb/ScanNetpp/scannetpp_lora5k.json"

# Outputs (home quota is nearly full -> /mnt)
RUNS = "/mnt/4dgs/spatialstack/lora_runs"
RESULTS = os.path.join(ROOT, "results")
BASELINE_RESULTS = os.path.join(RESULTS, "spatialstack")

# Training subset composition (5,000 total), weighted toward ScanNet++'s weak VSI-Bench categories
SEED = 0
SUBSET_COUNTS = {
    "object_abs_distance": 1250,
    "object_rel_direction_v1": 417,
    "object_rel_direction_v2": 417,
    "object_rel_direction_v3": 416,
    "object_counting": 1000,
    "object_rel_distance": 600,
    "object_size_estimation": 600,
    "room_size_estimation": 300,
}

# LoRA hyper-parameters (language model)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Expected number of LoRA-wrapped Linear layers:
#   8 full-attn layers x {q,k,v,o}  + 24 linear-attn layers x {in_proj_qkv,in_proj_z,out_proj}
#   + 32 layers x MLP {gate,up,down}
EXPECTED_LORA_LAYERS = 8 * 4 + 24 * 3 + 32 * 3  # = 200

# Geometry fusion MLPs (VGGT layers 11/17/23 -> LLM layers 0/1/2): FULLY fine-tuned, not LoRA.
# Same learning rate as the authors used for them (base lr 1e-5); trained in fp32 master weights.
FUSION_KEY = "language_feature_fusion"
FUSION_LR = 1e-5
