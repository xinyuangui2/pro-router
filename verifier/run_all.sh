#!/usr/bin/env bash
# End-to-end reproduction of the verifier-accuracy result.
#
#   export HF_TOKEN=...        # dataset download
#   export OPENAI_API_KEY=...  # judge labels
#   ./run_all.sh [output-dir]
#
# Stage 1 downloads datasets (CPU + network). Stage 2 needs a GPU -- developed
# on 4x A10G at tensor-parallel 4, but it auto-detects and runs on fewer.
# Stages 3-5 are CPU-only.
#
# SMOKE=1 caps stage 2 at the first 32 records of each split, and stages 3-5
# follow from its output. Stage 1 is not capped: it resolves the manifest
# against the full upstream splits either way.
set -euo pipefail

OUT="${1:-runs}"
SMOKE_ARG=""
if [ "${SMOKE:-0}" = "1" ]; then
  SMOKE_ARG="--limit 32"
  echo "### SMOKE MODE: 32 records per split ###"
fi

export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/src:${PYTHONPATH:-}"

echo "=== 1/5  build records ==="
python src/build_records.py --manifest bench_mix_manifest.json \
  --out-dir "$OUT/records"

echo "=== 2/5  small-model generation + per-token features ==="
for split in val test; do
  python src/extract_features.py \
    --records "$OUT/records/bench_${split}.jsonl" \
    --split "$split" --out-dir "$OUT/features" $SMOKE_ARG
done

echo "=== 3/5  judge labels ==="
for split in val test; do
  python src/judge_labels.py \
    --answers "$OUT/features/bench_${split}_answers.jsonl" \
    --out "$OUT/labels/${split}.json"
done

echo "=== 4/5  train the confidence head ==="
python src/train_head.py --features "$OUT/features" \
  --labels "$OUT/labels" --out-dir "$OUT/head"

echo "=== 5/5  evaluate ==="
python src/eval_head.py --scores "$OUT/head" --out "$OUT/results"

echo
echo "Done. Results: $OUT/results/verifier_accuracy.json"
