"""Check every tensor of the SpatialStack-Qwen2.5-4B checkpoint loads into the model unchanged (CPU)."""
import json, os, sys, torch
from safetensors import safe_open
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT as M
p = sys.argv[1]
model = M.from_pretrained(p, torch_dtype=torch.bfloat16, device_map="cpu")
sd = model.state_dict()
idx = json.load(open(os.path.join(p, "model.safetensors.index.json")))["weight_map"]
ok = bad = miss = 0; geo = 0
for shard in sorted(set(idx.values())):
    f = safe_open(os.path.join(p, shard), "pt")
    for k in f.keys():
        geo += any(s in k for s in ("geometry", "fusion", "merger_list", "vggt"))
        t = next((c for c in (k, k.removeprefix("model."), "model." + k) if c in sd), None)
        if t is None: miss += 1; print("NO TARGET", k) if miss < 5 else None; continue
        if torch.equal(sd[t], f.get_tensor(k).to(sd[t].dtype)): ok += 1
        else: bad += 1; print("MISMATCH", k)
print(f"ckpt tensors equal: {ok} | mismatch: {bad} | no-target: {miss} | (geometry-related: {geo})")
print("model tensors:", len(sd))
