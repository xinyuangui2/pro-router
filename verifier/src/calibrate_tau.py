"""Fit the ship/escalate thresholds for *your* setup.

The threshold table that ships with this package was fit for one model pair on
one record set. Scores are not comparable across setups -- change the model, the
prompt style or the answer length and the same numeric threshold lands at a
different quantile, so the cascade ships (or escalates) nearly everything. Any
serving run on a new pair should start here.

Two criteria, both computed on the validation split only:

  coverage  pick tau so a target fraction of requests ship. Simple, needs no
            labels beyond the ones already used for training, and gives a
            predictable operating point.
  loss      pick the highest-coverage tau whose shipped set stays within a
            budget of wrong answers. Needs labels; this is the criterion the
            shipped table was built with.

    python src/calibrate_tau.py --scores runs/head/val_scores.jsonl \\
        --criterion coverage --target-coverage 0.5 --out runs/weights/tau.json

The output slots straight into the serving pipeline:

    python ../run_pipeline.py --tau runs/weights/tau.json ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MIN_SOURCE_N = 20


def _read(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def tau_for_coverage(scores: np.ndarray, coverage: float) -> float:
    """Ship the top `coverage` fraction: the (1 - coverage) quantile."""
    return float(np.quantile(scores, 1.0 - coverage))


def tau_for_loss(scores: np.ndarray, labels: np.ndarray,
                 max_loss: float) -> tuple[float, dict]:
    """Highest coverage whose shipped set holds wrong answers under `max_loss`.

    Sweeps every score as a candidate threshold and keeps the most permissive
    one that satisfies the budget, so the result is the best coverage available
    at that quality bar rather than an arbitrary quantile.
    """
    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    best = (None, 0.0)
    for k in range(1, len(s) + 1):
        loss = float((1.0 - y[:k]).sum() / len(s))
        if loss <= max_loss:
            best = (float(s[k - 1]), k / len(s))
    if best[0] is None:
        return float(s[0]) + 1e-6, {"ship_rate": 0.0, "note": "budget unreachable"}
    tau, cov = best
    ship = s >= tau
    return tau, {"ship_rate": round(float(ship.mean()), 4),
                 "accuracy_on_shipped": round(float(y[ship].mean()), 4) if ship.any() else None,
                 "gross_loss": round(float((1.0 - y[ship]).sum() / len(s)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True,
                    help="val_scores.jsonl from train_head.py "
                         "({id, source, label, score}).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--criterion", choices=["coverage", "loss"], default="coverage")
    ap.add_argument("--target-coverage", type=float, default=0.5,
                    help="criterion=coverage: fraction of requests to ship.")
    ap.add_argument("--max-loss", type=float, default=0.05,
                    help="criterion=loss: budget of wrong shipped answers, "
                         "as a fraction of all validation requests.")
    ap.add_argument("--per-source", action="store_true", default=True,
                    help="Fit one threshold per source (default).")
    ap.add_argument("--global-only", dest="per_source", action="store_false",
                    help="Fit a single threshold for every source.")
    args = ap.parse_args()

    rows = _read(args.scores)
    scores = np.array([r["score"] for r in rows], dtype=float)
    sources = [r.get("source") or "" for r in rows]

    # A missing label must not become 0. Defaulting it silently marks the
    # record as a wrong answer, so the loss budget is spent on records that
    # were never judged and tau is driven far above where it belongs.
    missing = [r.get("id") for r in rows if r.get("label") is None]
    if args.criterion == "loss" and missing:
        raise SystemExit(
            f"[calibrate] --criterion loss needs a label on every record, but "
            f"{len(missing)} of {len(rows)} have none (e.g. {missing[:3]}).\n"
            f"  Attach judge labels (score_head.py --labels, or use "
            f"train_head.py's own val_scores.jsonl), or fit with\n"
            f"  --criterion coverage, which does not read labels.")
    labels = np.array([r.get("label", 0) for r in rows], dtype=float)
    base = ("unlabelled" if missing
            else f"base rate {labels.mean():.4f}")
    print(f"[calibrate] {len(rows)} validation records, "
          f"{len(set(sources))} source(s), {base}")

    if args.criterion == "coverage":
        g_tau = tau_for_coverage(scores, args.target_coverage)
        g_info = {"ship_rate": round(float((scores >= g_tau).mean()), 4)}
    else:
        g_tau, g_info = tau_for_loss(scores, labels, args.max_loss)
    table: dict = {"global": {"tau": g_tau}, "per_source": {}}
    print(f"[calibrate] global tau {g_tau:.4f}  {g_info}")

    if args.per_source:
        for src in sorted(set(sources)):
            m = np.array([s == src for s in sources])
            if m.sum() < MIN_SOURCE_N:
                print(f"[calibrate] {src}: only {m.sum()} records -- "
                      f"using the global threshold")
                continue
            if args.criterion == "coverage":
                t = tau_for_coverage(scores[m], args.target_coverage)
                info = {"ship_rate": round(float((scores[m] >= t).mean()), 4)}
            else:
                t = tau_for_loss(scores[m], labels[m], args.max_loss)
                t, info = t
            table["per_source"][src] = {"best": {"tau": t}}
            print(f"[calibrate] {src:<16} tau {t:.4f}  {info}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1))
    print(f"\n[calibrate] wrote {out}")
    print("[calibrate] thresholds are fit on validation only; the test split "
          "is untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
