"""Stage 3 -- label each small-model answer correct/incorrect with an LLM judge.

The head's training target is whether the small model's answer is correct, so
the labels must come from something stronger than a string match: the answers
are free-form and a programmatic extractor mis-scores prose and LaTeX badly.
The judge sees only text (query, gold, small-model answer) -- never the image -- and
records that abstain are dropped rather than guessed.

    OPENAI_API_KEY=... python src/judge_labels.py \
        --answers runs/features/bench_test_answers.jsonl --out runs/labels/test.json

Works against any OpenAI-compatible chat endpoint via --base-url.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

JUDGE_TEMPLATE = """You are a strict, impartial judge. Decide whether the response is CORRECT vs. the gold answer for the user's query.

The gold answer is: {gold}

[User query]
{prompt}

[Response]
{response}

Output exactly one of these three tokens (no other text, no explanation):
  CORRECT
  INCORRECT
  UNSURE

Rules:
  - If the response semantically conveys the gold answer (paraphrase, equivalent answer, same meaning, contains the gold letter for MCQ) -> CORRECT.
  - If the response clearly contradicts the gold answer or is unrelated -> INCORRECT.
  - If you genuinely cannot tell from the response alone -> UNSURE.

Verdict:"""


def _parse_verdict(text: str) -> str:
    up = (text or "").strip().upper()
    if "INCORRECT" in up:
        return "INCORRECT"
    if "CORRECT" in up:
        return "CORRECT"
    return "UNSURE"


async def _judge_one(session, sem, url, key, model, rec, retries=4):
    payload = {
        "model": model, "temperature": 0.0, "max_tokens": 8,
        "messages": [{"role": "user", "content": JUDGE_TEMPLATE.format(
            gold=rec.get("gold", ""), prompt=rec.get("prompt", ""),
            response=rec.get("answer", ""))}],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            async with sem, session.post(url, json=payload,
                                         headers=headers) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
                body = await resp.json()
                text = body["choices"][0]["message"]["content"]
                return rec["id"], _parse_verdict(text)
        except Exception as exc:
            if attempt == retries - 1:
                print(f"[judge] giving up on {rec['id']}: {exc}", flush=True)
                return rec["id"], "UNSURE"
            await asyncio.sleep(2 ** attempt)


async def _run(records, url, key, model, concurrency):
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [_judge_one(session, sem, url, key, model, r) for r in records]
        return await asyncio.gather(*tasks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-2024-11-20")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    records = [json.loads(l) for l in open(args.answers) if l.strip()]
    print(f"[judge] labelling {len(records)} answers with {args.model}",
          flush=True)
    results = asyncio.run(_run(records, f"{args.base_url}/chat/completions",
                               key, args.model, args.concurrency))

    labels, dropped = {}, 0
    for rid, verdict in results:
        if verdict == "UNSURE":
            dropped += 1
            continue
        labels[rid] = 1 if verdict == "CORRECT" else 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "labels": labels}, indent=1))
    pos = sum(labels.values())
    print(f"[judge] wrote {out}: {len(labels)} labelled "
          f"({pos} correct, {len(labels) - pos} incorrect), {dropped} UNSURE dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
