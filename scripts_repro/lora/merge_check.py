"""Verify a merged LoRA checkpoint against the base checkpoint and the saved adapter.

Usage: python repo/src/lora/merge_check.py <run_dir>      (expects <run_dir>/merged and <run_dir>/adapter-final)
Checks:
  1. merged and base checkpoints have the same tensor names and shapes (released layout preserved);
  2. every tensor NOT trained is bit-identical to the base (ViT, VGGT, embeddings, norms, ...);
  3. every LoRA-targeted weight equals base + (alpha/r) * B @ A, up to bf16 rounding;
  4. every fusion-MLP tensor (fully trained) equals the saved trained value (cast to bf16).
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file  # noqa: E402


class Checkpoint:
    """Lazy tensor access over a single-file or sharded safetensors checkpoint."""

    def __init__(self, d):
        idx = os.path.join(d, "model.safetensors.index.json")
        files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else \
            [os.path.basename(p) for p in glob.glob(os.path.join(d, "*.safetensors"))]
        self.handles = {}
        for f in files:
            h = safe_open(os.path.join(d, f), "pt")
            for k in h.keys():
                self.handles[k] = h

    def keys(self):
        return set(self.handles)

    def get(self, k):
        return self.handles[k].get_tensor(k)

    def shape(self, k):
        return tuple(self.handles[k].get_slice(k).get_shape())


def main(run_dir):
    base = Checkpoint(C.BASE_MODEL)
    merged = Checkpoint(os.path.join(run_dir, "merged"))
    adapter = load_file(os.path.join(run_dir, "adapter-final", "adapter.safetensors"))
    scale = C.LORA_ALPHA / C.LORA_R

    # module name -> (A, B); adapter keys look like '<module>.lora_A.default.weight'.
    # Keys without 'lora_' are the fully trained fusion-MLP tensors.
    lora, full = {}, {}
    for k, v in adapter.items():
        if ".lora_" in k:
            mod, ab = k.split(".lora_")[0], k.split(".lora_")[1][0]
            lora.setdefault(mod, {})[ab] = v.float()
        else:
            full[k] = v.to(torch.bfloat16)
    print(f"adapter: {len(lora)} LoRA modules, {len(full)} fully trained fusion tensors")

    # 1. same names / shapes
    kb, km = base.keys(), merged.keys()
    print(f"tensors: base {len(kb)}  merged {len(km)}  only-in-base {len(kb - km)}  only-in-merged {len(km - kb)}")
    shape_bad = [k for k in kb & km if base.shape(k) != merged.shape(k)]
    print(f"shape mismatches: {len(shape_bad)}")
    dtype_bad = [k for k in kb & km if base.handles[k].get_slice(k).get_dtype() != merged.handles[k].get_slice(k).get_dtype()]
    print(f"dtype mismatches: {len(dtype_bad)}" + (f"  e.g. {dtype_bad[0]}" if dtype_bad else ""))
    shape_bad += dtype_bad

    # 2 + 3
    def lora_for(key):
        if not key.endswith(".weight"):
            return None
        mod = key[: -len(".weight")]
        for cand in (mod, "model." + mod, mod.removeprefix("model.")):
            if cand in lora:
                return lora[cand]
        return None

    def full_for(key):
        for cand in (key, "model." + key, key.removeprefix("model.")):
            if cand in full:
                return full[cand]
        return None

    same = changed_ok = fusion_ok = fusion_changed = 0
    worst = 0.0
    errors = []
    for k in sorted(kb & km):
        ab = lora_for(k)
        ft = full_for(k)
        wb, wm = base.get(k), merged.get(k)
        if ft is not None:
            if torch.equal(wm, ft):
                fusion_ok += 1
                fusion_changed += not torch.equal(wm, wb)
            else:
                errors.append(f"fusion tensor differs from trained value: {k}")
        elif ab is None:
            if torch.equal(wb, wm):
                same += 1
            else:
                errors.append(f"non-LoRA tensor changed: {k}")
        else:
            expected = wb.float() + scale * (ab["B"] @ ab["A"])
            err = (wm.float() - expected).abs().max().item()
            tol = 2 * torch.finfo(torch.bfloat16).eps * expected.abs().max().item()  # bf16 rounding
            worst = max(worst, err / max(tol, 1e-12))
            if err <= tol:
                changed_ok += 1
            else:
                errors.append(f"merge mismatch {k}: max|err|={err:.3e} tol={tol:.3e}")

    print(f"non-LoRA tensors identical: {same}")
    print(f"LoRA-merged weights correct: {changed_ok}  (worst error / tolerance = {worst:.2f})")
    print(f"fusion-MLP tensors equal to trained values: {fusion_ok}  (of which changed vs base: {fusion_changed})")
    ok = not (kb ^ km) and not shape_bad and not errors
    for e in errors[:10]:
        print("  ", e)
    print("MERGE CHECK PASSED" if ok else "MERGE CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1])
