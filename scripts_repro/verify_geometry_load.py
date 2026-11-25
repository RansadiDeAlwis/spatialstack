"""Check that every geometry-related tensor in the checkpoint is loaded into the model unchanged."""
import torch, glob, os
from safetensors import safe_open
from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry as M  # noqa
from huggingface_hub import snapshot_download
p = snapshot_download("Journey9ni/SpatialStack-Qwen3.5-4B")
model = M.from_pretrained(p, torch_dtype=torch.bfloat16, geometry_encoder_path="facebook/VGGT-1B")
sd = model.state_dict()
f = safe_open(glob.glob(os.path.join(p, "*.safetensors"))[0], "pt")
geo = [k for k in f.keys() if any(s in k for s in ("geometry", "fusion", "merger"))]
ok = bad = miss = 0
for k in geo:
    cands = [k, k.removeprefix("model."), "model." + k]
    tgt = next((c for c in cands if c in sd), None)
    if tgt is None: miss += 1; continue
    if torch.equal(sd[tgt].cpu(), f.get_tensor(k).to(sd[tgt].dtype)): ok += 1
    else: bad += 1; print("MISMATCH", k)
print(f"geometry/fusion tensors in ckpt: {len(geo)} | equal: {ok} | mismatch: {bad} | no-target: {miss}")
print("config:", {k: getattr(model.config, k, None) for k in ["use_geometry_encoder","feature_fusion_method","geometry_encoder_layers","geometry_fusion_layers"]})
