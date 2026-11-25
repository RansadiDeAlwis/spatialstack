#!/usr/bin/env bash
# Second pass: build causal-conv1d (patched to sm_89) and flash_attn 2.8.3 (sm_80 kernels, run on Ada).
set -eo pipefail
E=/home/anujaya/ransadi/SpatialStack/env
export PATH=$E/bin:$PATH CUDA_HOME=$E MAX_JOBS=10 FLASH_ATTN_CUDA_ARCHS="80" TORCH_CUDA_ARCH_LIST="8.9"
export TMPDIR=/tmp/claude-1004/-home-anujaya-venumi/99cc7aff-d6ac-4404-b693-f05073d0f3f5/scratchpad
echo "=== causal-conv1d $(date +%T)"
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-cache-dir --no-build-isolation $TMPDIR/b/causal-conv1d 2>&1 | tail -2
echo "=== flash_attn $(date +%T)"
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install --no-cache-dir --no-build-isolation flash_attn==2.8.3 2>&1 | tail -2
cd /home/anujaya/ransadi/SpatialStack/repo && pip install --no-cache-dir -e . --no-deps -q
rm -rf $TMPDIR/b
echo "=== done $(date +%T)"
python - <<'PY'
import torch, transformers, qwen_vl_utils, causal_conv1d, fla, decord, flash_attn
print("torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__, "| flash_attn", flash_attn.__version__, "| causal_conv1d", causal_conv1d.__version__, "| fla", fla.__version__)
PY
