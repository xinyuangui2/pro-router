"""Stage 4 -- train the confidence head on the validation split.

The head is fit on the val split only (an internal 80/20 slice supplies the
early-stopping signal) and the test split is untouched until evaluation, so no
test information reaches either the weights or the threshold.

    python src/train_head.py --features runs/features --labels runs/labels \
        --out-dir runs/head
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from head_model import (SeqDataset, TransformerSeq, collate_pad,
                        count_parameters)


def load_split(features_dir: Path, labels_dir: Path, split: str) -> list[dict]:
    with open(features_dir / f"bench_{split}_per_token.pkl", "rb") as fh:
        feats = pickle.load(fh)
    labels = json.load(open(labels_dir / f"{split}.json"))["labels"]
    out = []
    for rec in feats:
        lab = labels.get(rec["id"])
        if lab is None or rec["features"].shape[0] == 0:
            continue
        out.append({"id": rec["id"], "source": rec["source"],
                    "features": rec["features"], "label": int(lab)})
    print(f"[train] {split}: {len(out)} usable records "
          f"({len(feats) - len(out)} dropped: unlabelled or empty)")
    return out


def train(train_recs, val_recs, epochs, batch_size, lr, wd, early_stop, seed):
    torch.manual_seed(seed)
    model = TransformerSeq()
    print(f"[train] head parameters: {count_parameters(model):,}")
    tr = DataLoader(SeqDataset(train_recs), batch_size=batch_size,
                    shuffle=True, collate_fn=collate_pad)
    va = DataLoader(SeqDataset(val_recs), batch_size=batch_size,
                    shuffle=False, collate_fn=collate_pad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        total, seen = 0.0, 0
        for X, lens, y, _ in tr:
            opt.zero_grad()
            loss = bce(model(X, lens), y)
            loss.backward()
            opt.step()
            total += float(loss) * X.size(0)
            seen += X.size(0)
        model.eval()
        scores, ys = [], []
        with torch.inference_mode():
            for X, lens, y, _ in va:
                scores.append(torch.sigmoid(model(X, lens)).numpy())
                ys.append(y.numpy())
        auroc = float(roc_auc_score(np.concatenate(ys),
                                    np.concatenate(scores)))
        if auroc > best:
            best, bad = auroc, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= early_stop:
                break
        if ep % 5 == 0 or ep < 3:
            print(f"  epoch {ep:>2d}  loss={total / max(1, seen):.4f}  "
                  f"heldout_auroc={auroc:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, best


@torch.inference_mode()
def score(model, recs) -> list[dict]:
    model.eval()
    dl = DataLoader(SeqDataset(recs), batch_size=64, shuffle=False,
                    collate_fn=collate_pad)
    out, i = [], 0
    for X, lens, y, _ in dl:
        probs = torch.sigmoid(model(X, lens)).numpy()
        for p in probs:
            r = recs[i]
            out.append({"id": r["id"], "source": r["source"],
                        "label": int(r["label"]), "score": float(p)})
            i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--early-stop", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    features_dir, labels_dir = Path(args.features), Path(args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val = load_split(features_dir, labels_dir, "val")
    test = load_split(features_dir, labels_dir, "test")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(val))
    cut = int(0.8 * len(val))
    train_recs = [val[i] for i in order[:cut]]
    heldout = [val[i] for i in order[cut:]]

    model, best = train(train_recs, heldout, args.epochs, args.batch_size,
                        args.lr, args.weight_decay, args.early_stop, args.seed)
    print(f"[train] best held-out AUROC: {best:.4f}")

    # "hparams" is what rebuilds the module: prorouter/gate.py and
    # score_head.py both instantiate from it, so a head trained here can be
    # served without knowing which defaults it was built with. "arch" is
    # duplicated at the top level for readers of older checkpoints.
    torch.save({"state_dict": model.state_dict(),
                "arch": "TransformerSeq",
                "hparams": {"arch": "TransformerSeq",
                            "n_features": model.proj.in_features,
                            "d_model": model.proj.out_features,
                            "n_heads": model.encoder.layers[0].self_attn.num_heads,
                            "n_layers": len(model.encoder.layers),
                            "d_ff": model.encoder.layers[0].linear1.out_features,
                            "max_len": model.pe.pe.shape[0]},
                "train_meta": {"n_train": len(train_recs),
                               "n_internal_val": len(heldout),
                               "seed": args.seed, "epochs": args.epochs,
                               "best_internal_val_auroc": best}},
               out_dir / "head.pt")
    for name, recs in (("val", val), ("test", test)):
        rows = score(model, recs)
        with open(out_dir / f"{name}_scores.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print(f"[train] wrote {out_dir}/head.pt and per-record scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
