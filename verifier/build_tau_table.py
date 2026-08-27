"""T P18 — derive per-source τ table from a saved TransformerSeq
classifier checkpoint.

For each source, find τ such that ship-when-score-ge-τ achieves a target
precision (e.g., 95% of shipped requests are actually v0_correct). Falls
back to the median score if no τ meets the target with enough ship count.

Inputs:
  --ckpt   trained TransformerSeq .pt (state_dict + hparams)
  --val-pkl   per-token sequences for the calibration split (typically c18_val)

Output:
  --out  tau_table.json — same schema as
         gatekeeper_tau_per_source.json so the scheduler's gate can
         consume it identically.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the model class definition
spec = importlib.util.spec_from_file_location(
    "p16b", str(Path(__file__).resolve().parent /
                "sys22t_p16b_train_seq_models.py"),
)
p16b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p16b)

ARCH_MAP = {
    "TransformerSeq": p16b.TransformerSeq,
    "BiLSTMSeq": p16b.BiLSTMSeq,
    "CNN1D": p16b.CNN1D,
    "AttnPool": p16b.AttnPool,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="path to trained .pt (state_dict + hparams)")
    ap.add_argument("--val-pkl", required=True,
                    help="per-token sequences for tau-tuning (uses val).")
    ap.add_argument("--target-precision", type=float, default=0.9,
                    help="ship only when projected precision >= this.")
    ap.add_argument("--min-ship-n", type=int, default=10,
                    help="τ must produce at least this many shipped "
                         "candidates on val to be valid. Else fall back "
                         "to median score (and emit a warning).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    hp = ckpt["hparams"]
    arch = hp["arch"]
    kwargs = {k: v for k, v in hp.items() if k != "arch"}
    model = ARCH_MAP[arch](**kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with open(args.val_pkl, "rb") as f:
        val_recs = pickle.load(f)

    # Run model on all val records, gather scores per source
    from torch.utils.data import DataLoader
    SeqDataset = p16b.SeqDataset
    collate_pad = p16b.collate_pad
    dl = DataLoader(SeqDataset(val_recs), batch_size=64, shuffle=False,
                     collate_fn=collate_pad)
    by_src: dict[str, dict[str, list]] = {}
    with torch.inference_mode():
        for X, lens, y, srcs in dl:
            scores = torch.sigmoid(model(X, lens)).numpy().tolist()
            labels = y.numpy().tolist()
            for s, lab, src in zip(scores, labels, srcs):
                if src not in by_src:
                    by_src[src] = {"scores": [], "labels": []}
                by_src[src]["scores"].append(float(s))
                by_src[src]["labels"].append(int(lab))

    out: dict = {
        "ckpt_path": args.ckpt,
        "ckpt_arch": arch,
        "ckpt_hparams": hp,
        "val_pkl": args.val_pkl,
        "target_precision": args.target_precision,
        "min_ship_n": args.min_ship_n,
        "rule": "ship iff transformer_score >= tau",
        "per_source": {},
    }
    for src, data in sorted(by_src.items()):
        scores = np.array(data["scores"])
        labels = np.array(data["labels"])
        n = len(scores)
        v0_acc = float(labels.mean())
        # Sort by descending score
        order = np.argsort(-scores)
        scores_sorted = scores[order]
        labels_sorted = labels[order]
        # cumulative shipped count + correct count
        cum_correct = np.cumsum(labels_sorted)
        ranks = np.arange(1, n + 1)
        cum_precision = cum_correct / ranks
        # Find largest k such that ship count >= min_ship_n AND
        # precision[k-1] >= target_precision. Pick τ just below
        # scores_sorted[k-1] so all k items get shipped.
        best = None
        for k in range(args.min_ship_n, n + 1):
            if cum_precision[k - 1] >= args.target_precision:
                # τ is the score at rank k. ship iff score >= τ.
                tau = float(scores_sorted[k - 1])
                ship_n = int(k)
                precision = float(cum_precision[k - 1])
                best = {"tau": tau, "ship_n": ship_n,
                        "ship_rate": ship_n / n, "precision": precision}
        if best is None:
            # Fallback: ship none (τ above max). Conservative — escalates
            # everything in this source.
            best = {
                "tau": float(scores_sorted[0]) + 1e-3,  # > max → ship none
                "ship_n": 0, "ship_rate": 0.0,
                "precision": float("nan"),
                "fallback": "no_tau_meets_target_precision",
            }
        out["per_source"][src] = {
            "n": n, "v0_acc": v0_acc,
            "best": best,
            "score_range": [float(scores.min()), float(scores.max())],
            "score_median": float(np.median(scores)),
        }
    # Also derive a global fallback tau: pool everything, target same precision.
    all_scores = np.concatenate([np.array(d["scores"]) for d in by_src.values()])
    all_labels = np.concatenate([np.array(d["labels"]) for d in by_src.values()])
    order = np.argsort(-all_scores)
    scores_sorted = all_scores[order]
    labels_sorted = all_labels[order]
    cum_correct = np.cumsum(labels_sorted)
    ranks = np.arange(1, len(all_scores) + 1)
    cum_precision = cum_correct / ranks
    global_best = None
    for k in range(args.min_ship_n, len(all_scores) + 1):
        if cum_precision[k - 1] >= args.target_precision:
            global_best = {
                "tau": float(scores_sorted[k - 1]),
                "ship_n": int(k),
                "ship_rate": float(k / len(all_scores)),
                "precision": float(cum_precision[k - 1]),
            }
    if global_best is None:
        global_best = {"tau": 0.5, "ship_n": 0, "ship_rate": 0.0,
                       "precision": float("nan"),
                       "fallback": "no_tau_meets_target_precision"}
    out["global"] = global_best

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[p18:tau] wrote {args.out}")
    print("=== per-source ===")
    for src, v in out["per_source"].items():
        b = v["best"]
        print(f"  {src:<20} n={v['n']:>3} v0_acc={v['v0_acc']:.2f}  "
              f"τ={b['tau']:.4f}  ship={b['ship_n']:>3}/{v['n']:<3} "
              f"({b['ship_rate']:.2f})  prec={b.get('precision', 0):.3f}")
    g = out["global"]
    print(f"=== global ===")
    print(f"  τ={g['tau']:.4f}  ship={g['ship_n']}/{len(all_scores)} "
          f"({g['ship_rate']:.2f})  prec={g.get('precision', 0):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
