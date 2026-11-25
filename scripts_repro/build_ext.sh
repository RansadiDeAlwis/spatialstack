#!/usr/bin/env bash
# Build CUDA extensions (flash_attn 2.8.3, causal-conv1d) + install remaining deps.
set -eo pipefail
E=/home/anujaya/ransadi/SpatialStack/env
export PATH=$E/bin:$PATH CUDA_HOME=$E
export TORCH_CUDA_ARCH_LIST="8.9" FLASH_ATTN_CUDA_ARCHS="80" MAX_JOBS=12
pip install -q psutil ninja wheel setuptools packaging
pip install --upgrade -q transformers==5.3.0 accelerate==1.13.0 qwen_vl_utils==0.0.14 decord
pip install -q -U git+https://github.com/fla-org/flash-linear-attention
pip install -q deepspeed datasets pyarrow evaluate pytablewriter pandas \
  loguru jsonlines sqlitedict sacrebleu terminaltables zss tenacity==8.3.0 \
  wandb openai tiktoken scipy openpyxl numexpr sympy nltk sentencepiece ftfy \
  timm opencv-python-headless av tqdm-multiprocess transformers-stream-generator hf_transfer einops
echo "=== building causal-conv1d $(date +%T)"
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install -v -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation 2>&1 | tail -5
echo "=== building flash_attn $(date +%T)"
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install -v flash_attn==2.8.3 --no-build-isolation 2>&1 | tail -5
cd /home/anujaya/ransadi/SpatialStack/repo && pip install -e . --no-deps -q
echo "=== done $(date +%T)"
python - <<'PY'
import torch, transformers, qwen_vl_utils, causal_conv1d, fla, decord, flash_attn
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__, "flash_attn", flash_attn.__version__)
print("causal_conv1d", causal_conv1d.__version__, "fla", fla.__version__, "decord", decord.__version__)
PY
