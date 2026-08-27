"""Stage 2 -- run the small model and record its per-token sampling distribution.

For every record this generates one greedy answer and emits the [T, 4] feature
sequence the head consumes: [chosen_logprob, max_prob, neg_entropy, pos_frac].

Entropy is computed over the returned top-k entries of the sampling
distribution, renormalised. This is the driver-side path used for the offline
results and it runs on an unmodified vLLM; `vllm_patch/` contains the in-engine
variant, which computes the same features over the full vocabulary inside the
sampler (see the top-level README).

    python src/extract_features.py --records runs/records/bench_test.jsonl \
        --split test --out-dir runs/features
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hardware  # noqa: E402

DEFAULT_SMALL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def _load_records(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def _build_request(tokenizer, prompt: str, image_paths: list[str]) -> dict:
    from PIL import Image

    content = [{"type": "image"} for _ in image_paths]
    content.append({"type": "text", "text": prompt})
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )
    req: dict = {"prompt": text}
    if image_paths:
        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        req["multi_modal_data"] = {"image": imgs if len(imgs) > 1 else imgs[0]}
    return req


def _features(completion) -> list[list[float]]:
    """[chosen_logprob, max_prob, neg_entropy, position_fraction] per token."""
    rows: list[list[float]] = []
    if completion.logprobs is None:
        return rows
    n = len(completion.token_ids)
    for idx, (tok_id, lp_dict) in enumerate(zip(completion.token_ids,
                                                completion.logprobs)):
        if not lp_dict:
            continue
        lp = lp_dict.get(tok_id)
        chosen = float(lp.logprob) if lp is not None else 0.0
        probs = [math.exp(float(o.logprob)) for o in lp_dict.values()]
        if not probs:
            continue
        z = sum(probs)
        ent = (-sum((p / z) * math.log(p / z) for p in probs if p > 0)
               if z > 0 and len(probs) > 1 else 0.0)
        rows.append([chosen, max(probs), -ent, (idx + 1) / max(1, n)])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--small-model", default=DEFAULT_SMALL_MODEL)
    ap.add_argument("--tensor-parallel-size", type=int, default=0,
                    help="0 = derive from visible GPUs.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--logprobs", type=int, default=20)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-images", type=int, default=2)
    ap.add_argument("--max-pixels", type=int, default=1003520)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N records (smoke test).")
    args = ap.parse_args()

    import numpy as np
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    records = _load_records(args.records)
    if args.limit:
        records = records[: args.limit]
    print(f"[extract] {len(records)} records from {args.records}", flush=True)

    tp = hardware.pick_tensor_parallel(args.tensor_parallel_size, label="small model")
    llm = LLM(
        model=args.small_model,
        tensor_parallel_size=tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=hardware.pick_dtype(args.dtype),
        **hardware.mm_kwargs(args.small_model, args.max_pixels, args.max_images),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.small_model)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0,
                        logprobs=args.logprobs)

    requests = [_build_request(tokenizer, r["prompt"], r.get("images") or [])
                for r in records]
    outputs = llm.generate(requests, sp, use_tqdm=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_token, answers = [], []
    for rec, out in zip(records, outputs):
        completion = out.outputs[0]
        feats = _features(completion)
        per_token.append({
            "id": rec["id"], "source": rec["source"],
            "n_tokens": len(completion.token_ids),
            "features": np.asarray(feats, dtype=np.float16).reshape(-1, 4),
        })
        answers.append({
            "id": rec["id"], "source": rec["source"], "prompt": rec["prompt"],
            # carried through so pipeline/train_prescorer.py can use the image
            # count as a feature: it reads this file, not the records, and
            # without the field every training row would look image-free while
            # every serving request has images.
            "images": rec.get("images") or [],
            "gold": rec.get("gold", ""), "answer": completion.text,
        })

    pkl = out_dir / f"bench_{args.split}_per_token.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(per_token, fh)
    jsonl = out_dir / f"bench_{args.split}_answers.jsonl"
    with open(jsonl, "w") as fh:
        for d in answers:
            fh.write(json.dumps(d) + "\n")
    print(f"[extract] wrote {pkl} and {jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
