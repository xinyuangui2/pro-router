"""Train the pre-generation scorer used to order the dispatch buffer.

The scorer reads only the prompt -- never the answer, never the image, never
the source label -- and predicts whether the small model will get this request
right. The cascade uses it to order a shared buffer so the small tier pulls the
requests it is most likely to be able to ship, which is why a query-only signal
is useful even though it is much weaker than the post-generation head.

Trained on the same validation split as the head, from artifacts the accuracy
pipeline already produces:

    python train_prescorer.py \
        --answers verifier/runs/features/bench_val_answers.jsonl \
        --labels verifier/runs/labels/val.json \
        --out weights/prescorer.pkl

Features (see FEATURE_NAMES): the prompt's TF-IDF vector, plus three scalars
describing the request's shape. The scalars are z-scored and the statistics are
persisted alongside the model -- a scorer that normalises at train time but not
at inference sees raw character counts, saturates the sigmoid, and silently
degenerates to FIFO. `prorouter/pre_router.py` recomputes the same features at
request time and must stay in step with `_meta` below.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


# The non-text features, in the order they are concatenated after the TF-IDF
# block. `prorouter/pre_router.py` builds this same vector at serving time; changing
# the order or the contents here invalidates every pickle already trained.
META_FEATURE_NAMES = ["n_images", "prompt_chars", "prompt_words"]

# Everything the scorer sees. It reads the prompt only -- never the generated
# answer, never the image pixels, never the source label -- because it runs
# before generation and exists to order the dispatch buffer, not to decide.
FEATURE_NAMES = ["tfidf(prompt) 1-2gram"] + META_FEATURE_NAMES


def _meta(prompt: str, n_images: int) -> list[float]:
    """The META_FEATURE_NAMES scalars for one request, in that order."""
    return [float(n_images), float(len(prompt)), float(len(prompt.split()))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", required=True,
                    help="bench_<split>_answers.jsonl (supplies the prompts).")
    ap.add_argument("--labels", required=True,
                    help="<split>.json from the judge (supplies correctness).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-features", type=int, default=20000)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    labels = json.load(open(args.labels))["labels"]
    prompts, metas, ys = [], [], []
    for line in open(args.answers):
        if not line.strip():
            continue
        r = json.loads(line)
        y = labels.get(r["id"])
        if y is None:
            continue
        prompts.append(r.get("prompt") or "")
        metas.append(_meta(prompts[-1], len(r.get("images") or [])))
        ys.append(int(y))

    if len(set(ys)) < 2:
        raise SystemExit("need both correct and incorrect examples to train")
    y = np.asarray(ys)
    print(f"[prescorer] {len(y)} examples, {y.mean():.3f} positive")
    print(f"[prescorer] features: {', '.join(FEATURE_NAMES)}")

    vec = TfidfVectorizer(max_features=args.max_features,
                          ngram_range=(1, args.ngram_max), sublinear_tf=True)
    X_text = vec.fit_transform(prompts)

    meta = np.asarray(metas, dtype=np.float32)
    mu, sd = meta.mean(axis=0), meta.std(axis=0)
    meta_z = (meta - mu) / np.maximum(sd, 1e-6)
    X = hstack([X_text, csr_matrix(meta_z)]).tocsr()

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed)
    clf.fit(X, y)
    auroc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    print(f"[prescorer] in-sample AUROC {auroc:.4f} "
          f"(optimistic -- it is a ranking prior, not a held-out claim)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"vec": vec, "clf": clf,
                     "meta_mu": mu.tolist(), "meta_sd": sd.tolist()}, fh)
    print(f"[prescorer] meta z-score stats persisted "
          f"({dict(zip(META_FEATURE_NAMES, mu.round(2).tolist()))} "
          f"+/- {dict(zip(META_FEATURE_NAMES, sd.round(2).tolist()))})")
    print(f"[prescorer] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
