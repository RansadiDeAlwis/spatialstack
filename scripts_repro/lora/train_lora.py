"""LoRA fine-tuning entry point for SpatialStack-Qwen3.5-4B.

Runs the repo's own training loop (qwen_vl.train.train_qwen.train) unchanged, patching it from outside:
  1. registers the ScanNet++ 5k subset as a dataset name,
  2. set_model           -> inject LoRA into all decoder layers, fully train the geometry fusion MLPs,
                            freeze everything else,
  3. Trainer             -> subclass with a 2-lr optimizer (LoRA / fusion), adapter-only checkpoints,
                            and s/step + peak GPU memory logging,
  4. final save          -> save adapter, merge LoRA into the base weights, save a full checkpoint
                            in the released layout to <output_dir>/merged (directly usable by eval).
Launch via train_lora.sh (all CLI flags are the repo's normal training arguments).
"""
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402

sys.path.insert(0, os.path.join(C.REPO, "src"))

import torch  # noqa: E402
import transformers  # noqa: E402

import lora_utils as L  # noqa: E402
from qwen_vl.data import data_dict  # noqa: E402
from qwen_vl.train import train_qwen as TQ  # noqa: E402  (also installs the repo's optimizer/sampler patches)

# 1. dataset registration (runtime only; no repo edit)
data_dict[C.SUBSET_NAME] = {"annotation_path": C.SUBSET_QA, "data_path": C.MEDIA_ROOT, "tag": "3d"}

# 2. trainable parameters = LoRA only
TQ.set_model = lambda model_args, model: L.apply_lora(model)


# 3. Trainer: adapter-only checkpoints + speed/memory logging
class StatsCallback(transformers.TrainerCallback):
    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, "lora_stats.jsonl")
        self.t0 = None

    def on_step_begin(self, args, state, control, **kw):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kw):
        dt = time.time() - self.t0
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        rec = {"step": state.global_step, "sec_per_step": round(dt, 2), "peak_mem_gb": round(peak, 2)}
        if state.log_history and "loss" in state.log_history[-1]:
            rec["loss"] = state.log_history[-1]["loss"]
        if args.local_rank in (-1, 0):
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[lora] step {state.global_step}/{state.max_steps}  {dt:.1f}s/step  peak_mem {peak:.1f}GB")


class LoraTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(StatsCallback(self.args.output_dir))

    def create_optimizer(self):
        """AdamW with two learning rates: LoRA at --learning_rate, fusion MLPs at config.FUSION_LR (authors' 1e-5).
        Weight decay on matrices only (not biases / norm weights). The LR schedule scales both groups."""
        if self.optimizer is None:
            groups = {"lora": [], "fusion_decay": [], "fusion_no_decay": []}
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if L.is_lora_param(n):
                    groups["lora"].append(p)
                elif L.is_fusion_param(n):
                    groups["fusion_decay" if p.ndim >= 2 else "fusion_no_decay"].append(p)
                else:
                    raise RuntimeError(f"Unexpected trainable parameter: {n}")
            wd = self.args.weight_decay
            param_groups = [
                {"params": groups["lora"], "lr": self.args.learning_rate, "weight_decay": wd},
                {"params": groups["fusion_decay"], "lr": C.FUSION_LR, "weight_decay": wd},
                {"params": groups["fusion_no_decay"], "lr": C.FUSION_LR, "weight_decay": 0.0},
            ]
            for g, (k, v) in zip(param_groups, groups.items()):
                print(f"[lora] optimizer group {k}: {sum(p.numel() for p in v)/1e6:.1f}M params, lr={g['lr']}")
            self.optimizer = torch.optim.AdamW(
                param_groups, betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon)
        return self.optimizer

    def _save_checkpoint(self, model, trial=None, *args, **kwargs):
        # 'adapter-*' (not 'checkpoint-*') so train_qwen's auto-resume never triggers
        d = os.path.join(self.args.output_dir, f"adapter-step-{self.state.global_step}")
        L.save_adapter(self.model, d)
        self.state.save_to_json(os.path.join(d, "trainer_state.json"))


TQ.Trainer = LoraTrainer

# 4. final save: adapter -> merge -> full checkpoint in released layout
_original_save = TQ.safe_save_model_for_hf_trainer


def save_merged(trainer, output_dir):
    L.save_adapter(trainer.model, os.path.join(output_dir, "adapter-final"))
    L.merge_lora(trainer.model)
    # The model up-casts the frozen VGGT to fp32 at load time; the released checkpoint stores it in bf16.
    # Cast back (lossless: values are bf16 weights stored in fp32) so the merged checkpoint matches exactly.
    with torch.no_grad():
        n_cast = 0
        for n, p in trainer.model.named_parameters():
            if "geometry_encoder" in n and p.dtype == torch.float32:
                b = p.data.to(torch.bfloat16)
                assert torch.equal(b.float(), p.data), f"VGGT tensor not exactly representable in bf16: {n}"
                p.data = b
                n_cast += 1
        print(f"[lora] cast {n_cast} frozen VGGT tensors back to bf16 (lossless)")
    merged = os.path.join(output_dir, "merged")
    os.makedirs(merged, exist_ok=True)
    _original_save(trainer=trainer, output_dir=merged)
    # processor / tokenizer / chat template: from the run dir (written by train()), else from the base model
    skip = {"trainer_state.json", "config.json", "lora_stats.jsonl"}
    for src in (output_dir, C.BASE_MODEL):
        for f in os.listdir(src):
            p = os.path.join(src, f)
            if (os.path.isfile(p) and f not in skip and not f.startswith("model")
                    and f.endswith((".json", ".jinja", ".txt", ".model"))
                    and not os.path.exists(os.path.join(merged, f))):
                shutil.copy2(p, merged)
    print(f"[lora] merged checkpoint -> {merged}")


TQ.safe_save_model_for_hf_trainer = save_merged

if __name__ == "__main__":
    TQ.train(attn_implementation="flash_attention_2")
