"""Score extracted features with an existing head checkpoint.

`train_head.py` fits a new head and writes the scores *that* head produces.
When you serve the bundled `weights/head.pt` instead, its thresholds have to be
fit against its own score distribution: two heads trained on different corpora
put their scores in different places, so a threshold taken from one and applied
to the other lands at the wrong quantile. That is the same failure as reusing a
threshold across model pairs, one level down.

So the calibration path for a head you did not just train is:

    extract_features.py  ->  score_head.py  ->  calibrate_tau.py

    python src/score_head.py --features runs/features --split val \\
        --head ../weights/head.pt --out runs/head/val_scores.jsonl

Labels are optional and only affect `calibrate_tau.py --criterion loss`. Pass
`--labels runs/labels` to attach the judge verdicts if you have them; without
them the output carries scores only, which is all `--criterion coverage` needs.

Extract the features at the SAME `--max-tokens` you will serve at. Answer
length changes the trajectory the head sees, so a threshold fit at one
generation length does not transfer to another.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from head_model import TransformerSeq

_ARCH = {"TransformerSeq": TransformerSeq}


def load_head(path: str):
    """Rebuild the module from the checkpoint's own hparams."""
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    hp = dict(ckpt.get("hparams") or {})
    arch = hp.pop("arch", None) or ckpt.get("arch") or "TransformerSeq"
    if arch not in _ARCH:
        raise SystemExit(f"[score] unknown head arch {arch!r}")
    model = _ARCH[arch](**hp)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[score] loaded {arch} from {path} ({n:,} trainable parameters)")
    return model


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True,
                    help="Directory holding bench_<split>_per_token.pkl.")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--head", required=True, help="Head checkpoint to score with.")
    ap.add_argument("--out", required=True, help="Destination .jsonl.")
    ap.add_argument("--labels", default=None,
                    help="Optional directory holding <split>.json judge "
                         "labels. Needed only for calibrate_tau.py "
                         "--criterion loss.")
    args = ap.parse_args()

    pkl = Path(args.features) / f"bench_{args.split}_per_token.pkl"
    with open(pkl, "rb") as fh:
        feats = pickle.load(fh)

    labels = {}
    if args.labels:
        labels = json.load(
            open(Path(args.labels) / f"{args.split}.json"))["labels"]

    model = load_head(args.head)
    rows, empty, unlabelled = [], 0, 0
    with torch.inference_mode():
        for rec in feats:
            f = rec["features"]
            if f.shape[0] == 0:
                empty += 1
                continue
            x = torch.as_tensor(f, dtype=torch.float32).unsqueeze(0)
            lens = torch.tensor([f.shape[0]])
            score = float(torch.sigmoid(model(x, lens)).item())
            row = {"id": rec["id"], "source": rec["source"],
                   "score": score, "n_tokens": int(f.shape[0])}
            if args.labels:
                lab = labels.get(rec["id"])
                if lab is None:
                    unlabelled += 1
                else:
                    row["label"] = int(lab)
            rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"[score] wrote {len(rows)} records to {out}"
          + (f" ({empty} skipped: no tokens)" if empty else ""))
    if args.labels and unlabelled:
        print(f"[score] {unlabelled} record(s) had no label (judge abstained); "
              f"they carry a score only and are unusable for "
              f"--criterion loss")
    if not args.labels:
        print("[score] no labels attached -- use calibrate_tau.py "
              "--criterion coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
