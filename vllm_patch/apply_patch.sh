#!/usr/bin/env bash
# Clone vLLM at the release this work forked from, and apply the in-engine
# confidence-head patch.
#
#   ./vllm_patch/apply_patch.sh [dest-dir]
#
# The patch is pure Python (11 files, no C++/CUDA), so an editable install over
# a matching prebuilt wheel is enough -- no recompilation.
set -euo pipefail

BASE_TAG="v0.19.1"
UPSTREAM="https://github.com/vllm-project/vllm.git"
DEST="${1:-vllm}"
PATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/in_engine_head.patch"

if [ ! -f "$PATCH" ]; then
  echo "error: patch not found at $PATCH" >&2
  exit 1
fi

if [ ! -d "$DEST/.git" ]; then
  echo "==> cloning vLLM into $DEST"
  git clone --filter=blob:none "$UPSTREAM" "$DEST"
fi

cd "$DEST"
echo "==> checking out $BASE_TAG"
git fetch --tags --quiet origin
git checkout --quiet "$BASE_TAG"

echo "==> verifying the patch applies cleanly"
git apply --check "$PATCH"

echo "==> applying"
git apply "$PATCH"

echo
echo "Done. vLLM $BASE_TAG + in-engine confidence head is in: $(pwd)"
echo
echo "Install (choose one):"
echo "  pip install -e .                      # from source"
echo "  pip install vllm==0.19.1 && \\"
echo "    cp -r vllm/ \$(python -c 'import site;print(site.getsitepackages()[0])')/"
echo
echo "Enable at request time by setting these environment variables:"
echo "  export VLLM_CASCADE_ATTN_POOL_CKPT=/path/to/head.pt"
echo "  export VLLM_CASCADE_ATTN_POOL_TAU=/path/to/tau.json"
