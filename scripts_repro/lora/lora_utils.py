"""LoRA + fusion-MLP fine-tuning helpers for SpatialStack-Qwen3.5-4B: inject, save, load, merge.

Trainable parameters:
  * LoRA adapters on every decoder layer of the language model (attention + MLP linears);
  * the geometry fusion MLPs (`language_feature_fusion`), FULLY fine-tuned, as in the authors' recipe.
Everything else (ViT, vision merger, VGGT, embeddings, LM head, base LLM weights) is frozen.

LoRA is injected *in place* (peft.inject_adapter_in_model) so the model keeps its original class
(Qwen3_5ForConditionalGenerationWithGeometry) and the repo's training code keeps working unchanged.
Trainable weights are kept in fp32 during training (no fp32 master copy otherwise: small updates would round
away in bf16) and cast back to bf16 before the final save so the checkpoint matches the released layout.
"""
import json
import os

import torch
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file, save_file

import config as C

# Module names verified against the released checkpoint. Anchored to the language model so the ViT
# (model.visual.*), VGGT (geometry_encoder.*) and fusion MLPs are never LoRA-wrapped. The tiny linear-attention
# gates in_proj_a/in_proj_b (32 x 2560) are skipped on purpose.
TARGET_REGEX = (
    r"model\.language_model\.layers\.\d+\.("
    r"self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj))"
)


def lora_config():
    return LoraConfig(
        r=C.LORA_R,
        lora_alpha=C.LORA_ALPHA,
        lora_dropout=C.LORA_DROPOUT,
        target_modules=TARGET_REGEX,
        bias="none",
    )


def lora_layer_names(model):
    return [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]


def is_lora_param(name):
    return "lora_" in name


def is_fusion_param(name):
    return C.FUSION_KEY in name and "lora_" not in name


def _cast_input_to_weight(module, inputs):
    w = next(module.parameters())
    return tuple(x.to(w.dtype) if torch.is_tensor(x) and x.is_floating_point() else x for x in inputs)


def _cast_output_bf16(module, inputs, output):
    return output.to(torch.bfloat16)


def apply_lora(model, verbose=True):
    """Freeze everything; inject LoRA into the language model; make LoRA + fusion MLPs trainable (fp32)."""
    for p in model.parameters():
        p.requires_grad = False
    inject_adapter_in_model(lora_config(), model)

    names = lora_layer_names(model)
    bad = [n for n in names if "visual" in n or "geometry_encoder" in n or C.FUSION_KEY in n]
    assert not bad, f"LoRA injected outside the language model: {bad[:5]}"
    assert len(names) == C.EXPECTED_LORA_LAYERS, (
        f"Expected {C.EXPECTED_LORA_LAYERS} LoRA layers, got {len(names)}: check TARGET_REGEX"
    )

    n_lora = n_fusion = 0
    for n, p in model.named_parameters():  # named_parameters() de-duplicates aliased modules
        if is_lora_param(n) or is_fusion_param(n):
            p.data = p.data.float()
            p.requires_grad = True
            if is_lora_param(n):
                n_lora += p.numel()
            else:
                n_fusion += p.numel()
        else:
            p.requires_grad = False
    assert n_fusion > 0, "No fusion parameters found"

    # fp32 fusion MLPs inside a bf16 model: feed them inputs in their own dtype, and hand their output back
    # in bf16 so the fused vision tokens stay bf16 (works with or without autocast)
    hooks = 0
    for n, m in model.named_modules():
        if C.FUSION_KEY in n and n.endswith("geo_mlp"):
            m.register_forward_pre_hook(_cast_input_to_weight)
            m.register_forward_hook(_cast_output_bf16)
            hooks += 1

    if verbose:
        total = sum(p.numel() for p in model.parameters())
        groups = {
            "self_attn": sum("self_attn" in n for n in names),
            "linear_attn": sum("linear_attn" in n for n in names),
            "mlp": sum(".mlp." in n for n in names),
        }
        print(f"[lora] LoRA on {len(names)} layers {groups}: {n_lora/1e6:.1f}M params")
        print(f"[lora] fusion MLPs fully trained ({hooks} modules): {n_fusion/1e6:.1f}M params")
        print(f"[lora] trainable total: {(n_lora+n_fusion)/1e6:.1f}M / {total/1e9:.2f}B "
              f"({100*(n_lora+n_fusion)/total:.2f}%)")
    return model


def trainable_state_dict(model):
    """LoRA adapter weights + full fusion-MLP weights (everything that differs from the base checkpoint)."""
    return {n: p.detach().cpu().contiguous() for n, p in model.named_parameters()
            if is_lora_param(n) or is_fusion_param(n)}


def save_adapter(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    save_file(trainable_state_dict(model), os.path.join(out_dir, "adapter.safetensors"))
    meta = {"r": C.LORA_R, "alpha": C.LORA_ALPHA, "dropout": C.LORA_DROPOUT,
            "target_regex": TARGET_REGEX, "fully_trained": C.FUSION_KEY, "base_model": C.BASE_MODEL}
    json.dump(meta, open(os.path.join(out_dir, "adapter_config.json"), "w"), indent=1)
    print(f"[lora] saved adapter (+ fusion MLPs) -> {out_dir}")


def load_adapter(model, adapter_dir):
    """Inject LoRA into a fresh base model and load saved adapter + fusion weights."""
    apply_lora(model, verbose=False)
    sd = load_file(os.path.join(adapter_dir, "adapter.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing_trainable = [k for k in missing if is_lora_param(k)]
    assert not missing_trainable and not unexpected, (missing_trainable[:5], unexpected[:5])
    return model


@torch.no_grad()
def merge_lora(model):
    """Fold every LoRA update into its base Linear, swap the base Linear back in, and cast the fusion MLPs
    back to bf16. Afterwards module tree, dtypes and state_dict keys match the released checkpoint."""
    names = lora_layer_names(model)
    for name in names:
        layer = model.get_submodule(name)
        layer.merge(safe_merge=True)
        parent_name, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child, layer.get_base_layer())
    for n, p in model.named_parameters():
        if is_fusion_param(n):
            p.data = p.data.to(torch.bfloat16)
    assert not lora_layer_names(model)
    assert not any("lora_" in k for k in model.state_dict())
    print(f"[lora] merged {len(names)} LoRA layers into base weights; fusion MLPs cast to bf16")
    return model
