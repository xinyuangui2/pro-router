"""Stage 5 -- the verifier-accuracy table.

Reports discrimination (AUROC), the accuracy/coverage trade-off the router
actually operates on (AUACC and the risk-coverage curve), and calibration
(ECE, Brier). The shipping threshold is taken as a quantile of the *validation*
scores, so the operating point never sees a test label.

    python src/eval_head.py --scores runs/head --out runs/results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

COVERAGES = np.arange(0.05, 1.0001, 0.05)


def _read(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return (np.array([r["score"] for r in rows], dtype=float),
            np.array([r["label"] for r in rows], dtype=float),
            [r["source"] for r in rows])


def risk_coverage(scores: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Accuracy among the highest-confidence `c` fraction, for each coverage."""
    order = np.argsort(-scores)
    lab = labels[order]
    out = []
    for c in COVERAGES:
        k = max(1, int(round(c * len(lab))))
        out.append({"coverage": round(float(c), 4),
                    "selective_accuracy": float(lab[:k].mean()),
                    "n": int(k)})
    return out


# np.trapz was renamed np.trapezoid in NumPy 2.0 and removed under the old
# name; bind whichever this install has.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def auacc(curve: list[dict]) -> float:
    """Area under the accuracy-vs-coverage curve (trapezoid, normalised)."""
    cov = np.array([p["coverage"] for p in curve])
    acc = np.array([p["selective_accuracy"] for p in curve])
    return float(_trapezoid(acc, cov) / (cov[-1] - cov[0]))


def ece(scores: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores > lo) & (scores <= hi) if lo > 0 else (scores >= lo) & (scores <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(labels[m].mean() - scores[m].mean())
    return float(total)


def threshold_at_coverage(val_scores: np.ndarray, coverage: float) -> float:
    """Ship the top `coverage` fraction: tau is the (1-coverage) val quantile."""
    return float(np.quantile(val_scores, 1.0 - coverage))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True,
                    help="Directory holding val_scores.jsonl / test_scores.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-coverage", type=float, default=0.5,
                    help="Validation ship rate used to pick the threshold.")
    args = ap.parse_args()

    sdir = Path(args.scores)
    val_s, val_y, _ = _read(sdir / "val_scores.jsonl")
    test_s, test_y, test_src = _read(sdir / "test_scores.jsonl")

    tau = threshold_at_coverage(val_s, args.target_coverage)
    shipped = test_s >= tau
    curve = risk_coverage(test_s, test_y)

    res = {
        "n_val": int(len(val_y)), "n_test": int(len(test_y)),
        "base_rate_test": float(test_y.mean()),
        "auroc": float(roc_auc_score(test_y, test_s)),
        "auacc": auacc(curve),
        "ece_15bin": ece(test_s, test_y),
        "brier": float(np.mean((test_s - test_y) ** 2)),
        "operating_point": {
            "target_val_coverage": args.target_coverage,
            "tau": tau,
            "test_ship_rate": float(shipped.mean()),
            "accuracy_on_shipped": float(test_y[shipped].mean()) if shipped.any() else None,
            "accuracy_on_escalated": float(test_y[~shipped].mean()) if (~shipped).any() else None,
        },
        "risk_coverage": curve,
        "per_source": {},
    }
    for src in sorted(set(test_src)):
        m = np.array([s == src for s in test_src])
        if m.sum() < 10 or len(set(test_y[m])) < 2:
            continue
        res["per_source"][src] = {
            "n": int(m.sum()),
            "auroc": float(roc_auc_score(test_y[m], test_s[m])),
            "base_rate": float(test_y[m].mean()),
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verifier_accuracy.json").write_text(json.dumps(res, indent=1))

    op = res["operating_point"]
    print("\n=== verifier accuracy (test split) ===")
    print(f"  records            {res['n_test']}")
    print(f"  small-model correct rate {res['base_rate_test']:.4f}")
    print(f"  AUROC              {res['auroc']:.4f}")
    print(f"  AUACC              {res['auacc']:.4f}")
    print(f"  ECE (15-bin)       {res['ece_15bin']:.4f}")
    print(f"  Brier              {res['brier']:.4f}")
    print(f"\n  operating point (tau from val @ coverage "
          f"{args.target_coverage:.2f}): tau={op['tau']:.4f}")
    print(f"    ship rate           {op['test_ship_rate']:.4f}")
    print(f"    accuracy shipped    {op['accuracy_on_shipped']}")
    print(f"    accuracy escalated  {op['accuracy_on_escalated']}")
    if res["per_source"]:
        print("\n  per-source AUROC:")
        for src, d in res["per_source"].items():
            print(f"    {src:<16} n={d['n']:>4}  auroc={d['auroc']:.4f}")
    print(f"\nwrote {out / 'verifier_accuracy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
