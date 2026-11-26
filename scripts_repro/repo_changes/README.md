# Changes to the authors' code (`repo/`)

`repo/` is a clone of the official SpatialStack repository and is not committed here (it is gitignored,
has its own git history, and holds ~17 GB of caches). Our changes to it are kept in this folder instead.

- **Upstream:** https://github.com/jzh15/SpatialStack, commit `b84ebaa5451df42a275077772e7a0a6c66e74453`
- **`repo_changes.patch`:** all changes below as one patch (verified to apply cleanly to that commit)
- **`files/`:** the full changed files, at the same paths as in `repo/`

## Recreate `repo/` with our changes
```bash
git clone https://github.com/jzh15/SpatialStack repo
cd repo
git checkout b84ebaa5451df42a275077772e7a0a6c66e74453
git apply ../scripts_repro/repo_changes/repo_changes.patch
```
The LoRA code itself (`repo/src/lora/`) lives in `scripts_repro/lora/`; copy it to `repo/src/lora/` if needed.

## What changed
| File | Change | Purpose |
|---|---|---|
| `src/lmms_eval/tasks/blink/blink_default_yaml` | `token: True` → `False` | Reproduction: load the public dataset without a Hugging Face login token |
| `src/lmms_eval/tasks/mmsibench/mmsibench.yaml` | `token: true` → `False` | Same |
| `src/lmms_eval/tasks/sparbench/sparbench.yaml` | `token: True` → `False` | Same |
| `src/qwen_vl/data/__init__.py` | +1 line: register `vlm3r_scannetpp` | LoRA fine-tune: the ScanNet++ dataset was defined upstream but missing from `data_dict` |
| `src/lmms_eval/tasks/vsibench/utils.py` | + `process_docs_scannetpp` | LoRA eval: VSI-Bench restricted to ScanNet++ scenes (1,458 Qs) |
| `src/lmms_eval/tasks/vsibench/vsibench_scannetpp.yaml` | new task `vsibench_scannetpp` | LoRA eval: uses the filter above (copy of `vsibench_random200.yaml`) |

To regenerate the patch after further edits (run inside `repo/`):
```bash
{ git diff; git diff --no-index -- /dev/null src/lmms_eval/tasks/vsibench/vsibench_scannetpp.yaml; } > ../scripts_repro/repo_changes/repo_changes.patch
```
