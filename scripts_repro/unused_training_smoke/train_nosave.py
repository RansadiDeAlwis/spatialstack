"""Run the repo's train_qwen.train() with checkpoint saving disabled (smoke test; disk quota)."""
import sys, os
sys.path.insert(0, "/home/anujaya/ransadi/SpatialStack/repo/src/qwen_vl/train")
sys.path.insert(0, "/home/anujaya/ransadi/SpatialStack/repo/src")
import train_qwen
train_qwen.safe_save_model_for_hf_trainer = lambda trainer, output_dir: print("[smoke] final save skipped")
import transformers
transformers.Trainer.save_state = lambda self: None
transformers.Trainer._save_checkpoint = lambda self, *a, **k: None
if __name__ == "__main__":
    train_qwen.train(attn_implementation="flash_attention_2")
