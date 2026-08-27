"""T P16B — train sequence models on per-token logprob sequences.

Inputs: pickled per-token sequences from P16A. Each record has:
  features [T, 4] = per-token [chosen_lp, max_p, neg_entropy, pos_frac]
  label    = v0_correct (binary)
  source   = task name

Models tested:
  1. quantile_mlp:   aggregate to quantile features (p10/25/50/75/90 of
     each of 3 stats) -> 15 features -> small MLP. NO sequence model;
     just richer aggregation than the 4 mean/min stats.
  2. transformer:    Linear(4 -> d_model=64) + positional encoding +
     2 transformer encoder layers + mean pool -> Linear(64 -> 1).
  3. bilstm:         BiLSTM(4 -> 32 hidden) -> mean pool -> Linear -> 1.
  4. cnn1d:          Conv1d(4 -> 32, k=3) + Conv1d(32 -> 32, k=3) +
                       global max pool -> Linear -> 1.
  5. attn_pool:      Linear proj + learnable query attending to
     sequence -> Linear -> 1.

All on the same val set (calibration) -> test (eval), with
val_AUROC used for early-stop. Baseline lp_only (4 scalars + tiny MLP)
val=0.852 / test=0.841 from P11 for comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


ROOT = os.getenv("PROROUTER_TRAIN_ROOT", "runs/train_routers")


def _load_pkl(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


class SeqDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return torch.from_numpy(np.asarray(r["features"],
                                            dtype=np.float32)), \
               r["label"], r["source"]


def collate_pad(batch):
    """Pad to max length in batch, return (X [B,T,4], lens [B],
    labels [B])."""
    feats, labels, sources = zip(*batch)
    lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    T = int(lens.max().item())
    B = len(batch)
    X = torch.zeros((B, T, 4), dtype=torch.float32)
    for i, f in enumerate(feats):
        if f.shape[0] > 0:
            X[i, :f.shape[0], :] = f
    return X, lens, torch.tensor(labels, dtype=torch.float32), list(sources)


# ----------------------------- models -----------------------------

def _quantile_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Aggregate to 15 features: q10/q25/q50/q75/q90 of each of
    [chosen_lp, max_p, neg_entropy] (skip pos_frac for now)."""
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    X = np.zeros((len(records), 15), dtype=np.float32)
    y = np.zeros((len(records),), dtype=np.float32)
    sources = []
    for i, r in enumerate(records):
        f = np.asarray(r["features"], dtype=np.float32)
        if f.shape[0] == 0:
            continue
        cols = []
        for c in range(3):
            cols.extend(np.quantile(f[:, c], qs).tolist())
        X[i] = cols
        y[i] = r["label"]
        sources.append(r["source"])
    return X, y, sources


class QuantileMLP(nn.Module):
    def __init__(self, in_dim=15, hidden=(64, 32)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.01)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class TransformerSeq(nn.Module):
    def __init__(self, n_features=4, d_model=64, n_heads=4,
                 n_layers=2, d_ff=128):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pe = PositionalEncoding(d_model, max_len=600)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x, lens):
        # x: [B, T, 4]. Build padding mask (True = pad).
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= \
               lens.to(x.device).unsqueeze(1)
        h = self.proj(x)
        h = self.pe(h)
        h = self.encoder(h, src_key_padding_mask=mask)
        # mean pool over valid positions
        not_mask = (~mask).float().unsqueeze(-1)
        pooled = (h * not_mask).sum(dim=1) / not_mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled).squeeze(-1)


class BiLSTMSeq(nn.Module):
    def __init__(self, n_features=4, hidden=32, n_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, num_layers=n_layers,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * hidden, 1)

    def forward(self, x, lens):
        # Pack to honor variable lengths.
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False,
        )
        out, _ = self.rnn(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # mean over valid positions
        T_unp = unpacked.shape[1]
        mask = (torch.arange(T_unp, device=x.device).unsqueeze(0)
                < lens.to(x.device).unsqueeze(1)).float().unsqueeze(-1)
        pooled = (unpacked * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, n_features=4, channels=32, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, channels, kernel,
                                padding=kernel // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel,
                                padding=kernel // 2)
        self.act = nn.LeakyReLU(0.01)
        self.classifier = nn.Linear(channels, 1)

    def forward(self, x, lens):
        # x: [B, T, 4] -> [B, 4, T]
        h = x.transpose(1, 2)
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        # Mask positions beyond length
        T = h.shape[2]
        mask = (torch.arange(T, device=x.device).unsqueeze(0)
                < lens.to(x.device).unsqueeze(1)).unsqueeze(1)
        h = h.masked_fill(~mask, -1e9)
        pooled, _ = h.max(dim=2)  # global max pool
        return self.classifier(pooled).squeeze(-1)


class AttnPool(nn.Module):
    def __init__(self, n_features=4, d_model=64):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.query = nn.Parameter(torch.randn(d_model))
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x, lens):
        h = self.proj(x)  # [B, T, d_model]
        # Compute attention scores via dot with query.
        scores = (h * self.query).sum(dim=-1)  # [B, T]
        # Mask
        T = scores.shape[1]
        mask = (torch.arange(T, device=x.device).unsqueeze(0)
                < lens.to(x.device).unsqueeze(1))
        scores = scores.masked_fill(~mask, -1e9)
        attn = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (h * attn).sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


# ----------------------------- train + eval -----------------------------

def _train_seq(name: str, model, train_recs, val_recs,
                epochs=40, batch_size=64, lr=5e-4, wd=0.01,
                early_stop=8):
    model = model.to("cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[p16b] === {name} (n_params={n_params:,}) ===", flush=True)
    tr_dl = DataLoader(SeqDataset(train_recs), batch_size=batch_size,
                        shuffle=True, collate_fn=collate_pad)
    va_dl = DataLoader(SeqDataset(val_recs), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_pad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()
    best = -1.0; best_state = None; bad = 0
    for ep in range(epochs):
        model.train()
        tloss = 0.0; n_seen = 0
        for X, lens, y, _ in tr_dl:
            opt.zero_grad()
            loss = bce(model(X, lens), y)
            loss.backward(); opt.step()
            tloss += float(loss) * X.size(0); n_seen += X.size(0)
        tloss /= max(1, n_seen)
        model.eval()
        scores, ys = [], []
        with torch.inference_mode():
            for X, lens, y, _ in va_dl:
                scores.append(torch.sigmoid(model(X, lens)).numpy())
                ys.append(y.numpy())
        scores = np.concatenate(scores); ys = np.concatenate(ys)
        try:
            au = float(roc_auc_score(ys, scores))
        except ValueError:
            au = float("nan")
        if au > best:
            best = au
            best_state = {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= early_stop:
                break
        if ep % 5 == 0 or ep < 3:
            print(f"  ep {ep:>2d} loss={tloss:.4f} val_auroc={au:.4f}",
                  flush=True)
    model.load_state_dict(best_state)
    return model, best, n_params


def _per_source_scores(model, recs):
    """Dump per-source (score, label) lists for tau-table derivation.

    Output schema:
      {source_name: {"scores": [float, ...], "labels": [0/1, ...]}}
    The bench's CpuTransformerRouter pipeline applies a per-source
    threshold τ_src to model output (sigmoid(logit) >= τ → SHIP).
    Quantiles over each source's score distribution at known accuracy
    operating points produce that table.
    """
    dl = DataLoader(SeqDataset(recs), batch_size=64, shuffle=False,
                     collate_fn=collate_pad)
    model.eval()
    by_src: dict[str, dict[str, list]] = {}
    with torch.inference_mode():
        for X, lens, y, srcs in dl:
            sc = torch.sigmoid(model(X, lens)).numpy().tolist()
            yl = y.numpy().tolist()
            for s, label, src in zip(sc, yl, srcs):
                if src not in by_src:
                    by_src[src] = {"scores": [], "labels": []}
                by_src[src]["scores"].append(float(s))
                by_src[src]["labels"].append(int(label))
    return by_src


def _eval_seq(model, recs):
    dl = DataLoader(SeqDataset(recs), batch_size=64, shuffle=False,
                     collate_fn=collate_pad)
    model.eval()
    scores, ys, srcs = [], [], []
    with torch.inference_mode():
        for X, lens, y, s in dl:
            scores.append(torch.sigmoid(model(X, lens)).numpy())
            ys.append(y.numpy()); srcs.extend(s)
    scores = np.concatenate(scores); ys = np.concatenate(ys)
    srcs = np.array(srcs)
    per_src = {}
    for src in sorted(set(srcs.tolist())):
        m = srcs == src
        if m.sum() < 5 or len(set(ys[m].tolist())) < 2:
            per_src[src] = {"n": int(m.sum()), "auroc": float("nan")}
        else:
            per_src[src] = {"n": int(m.sum()),
                             "auroc": float(roc_auc_score(ys[m], scores[m]))}
    try:
        pool = float(roc_auc_score(ys, scores))
    except ValueError:
        pool = float("nan")
    return {"pooled_auroc": pool, "per_source": per_src}


def _train_quantile_mlp(train_recs, val_recs, test_recs):
    Xtr, ytr, _ = _quantile_features(train_recs)
    Xva, yva, _ = _quantile_features(val_recs)
    Xte, yte, src_te = _quantile_features(test_recs)
    src_te = np.array(src_te)
    Xtr = torch.from_numpy(Xtr); Xva = torch.from_numpy(Xva)
    Xte = torch.from_numpy(Xte); ytr = torch.from_numpy(ytr)
    yva = torch.from_numpy(yva); yte = torch.from_numpy(yte)
    model = QuantileMLP(in_dim=15, hidden=(64, 32))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[p16b] === quantile_mlp (n_params={n_params:,}) ===")
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    bce = nn.BCEWithLogitsLoss()
    best = -1.0; best_state = None; bad = 0
    for ep in range(40):
        model.train()
        # full batch since data is small
        opt.zero_grad()
        loss = bce(model(Xtr), ytr)
        loss.backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            va_sc = torch.sigmoid(model(Xva)).numpy()
        try:
            au = float(roc_auc_score(yva.numpy(), va_sc))
        except ValueError:
            au = float("nan")
        if au > best:
            best = au
            best_state = {k: v.detach().clone()
                           for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 10:
                break
        if ep % 5 == 0 or ep < 3:
            print(f"  ep {ep:>2d} loss={float(loss):.4f} "
                  f"val_auroc={au:.4f}")
    model.load_state_dict(best_state)
    # Eval
    with torch.inference_mode():
        te_sc = torch.sigmoid(model(Xte)).numpy()
    per_src = {}
    for src in sorted(set(src_te.tolist())):
        m = src_te == src
        if m.sum() < 5 or len(set(yte.numpy()[m].tolist())) < 2:
            per_src[src] = {"n": int(m.sum()), "auroc": float("nan")}
        else:
            per_src[src] = {"n": int(m.sum()),
                             "auroc": float(roc_auc_score(
                                 yte.numpy()[m], te_sc[m]))}
    try:
        pool = float(roc_auc_score(yte.numpy(), te_sc))
    except ValueError:
        pool = float("nan")
    return {"best_val_auroc": best, "n_params": n_params,
            "test_pooled_auroc": pool, "per_source": per_src}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-pkl",
                    default=f"{ROOT}/c18_per_token_seqs/c18_val_per_token.pkl")
    ap.add_argument("--test-pkl",
                    default=f"{ROOT}/c18_per_token_seqs/c18_test_per_token.pkl")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--transformer-layers", default="1,2,3,4,6",
                    help="comma-separated layer counts to sweep for "
                         "TransformerSeq (each spawns one model_specs entry)")
    ap.add_argument("--out", default=f"{ROOT}/p16b_seq_models_eval.json")
    ap.add_argument("--save-ckpt-dir", default=None,
                    help="Directory to save each trained sequence model's "
                         "state_dict + per-source test scores. Files named "
                         "<modelname>.pt + <modelname>_scores.json. Used by "
                         "P18 production cascade to load the chosen "
                         "(transformer_L2 by default) decider.")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    val = _load_pkl(args.val_pkl)
    test = _load_pkl(args.test_pkl)
    # Train on val (since we don't have train per-token yet — small set),
    # eval on test. Wait — that's not ideal. Better: 80/20 split val.
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(val))
    n_train = int(0.8 * len(val))
    tr_recs = [val[i] for i in idx[:n_train]]
    va_recs = [val[i] for i in idx[n_train:]]
    print(f"[p16b] val n={len(val)} -> train {len(tr_recs)} / val "
          f"{len(va_recs)}; test n={len(test)}")
    lens = [r["n_tokens"] for r in val + test]
    print(f"[p16b] sequence lengths: min={min(lens)} med="
          f"{sorted(lens)[len(lens)//2]} max={max(lens)}")

    results = {}

    # Quantile MLP (baseline +)
    results["quantile_mlp"] = _train_quantile_mlp(tr_recs, va_recs, test)

    # Sequence models
    transformer_layer_sweep = [int(x) for x in args.transformer_layers.split(",")]
    model_specs: list[tuple[str, nn.Module, dict]] = []
    for nl in transformer_layer_sweep:
        m = TransformerSeq(n_layers=nl)
        hp = dict(arch="TransformerSeq", n_features=4, d_model=64,
                  n_heads=4, n_layers=nl, d_ff=128)
        model_specs.append((f"transformer_L{nl}", m, hp))
    model_specs += [
        ("bilstm", BiLSTMSeq(),
         dict(arch="BiLSTMSeq", n_features=4, hidden=32, n_layers=1)),
        ("cnn1d", CNN1D(),
         dict(arch="CNN1D", n_features=4, channels=32, kernel=3)),
        ("attn_pool", AttnPool(),
         dict(arch="AttnPool", n_features=4, d_model=64)),
    ]
    if args.save_ckpt_dir:
        os.makedirs(args.save_ckpt_dir, exist_ok=True)
    for name, model, hp in model_specs:
        m, val_au, n_p = _train_seq(name, model, tr_recs, va_recs,
                                      epochs=args.epochs)
        r = _eval_seq(m, test)
        results[name] = {
            "best_val_auroc": val_au,
            "n_params": n_p,
            "test_pooled_auroc": r["pooled_auroc"],
            "per_source": r["per_source"],
        }
        if args.save_ckpt_dir:
            ckpt_path = os.path.join(args.save_ckpt_dir, f"{name}.pt")
            torch.save({"state_dict": m.state_dict(), "hparams": hp,
                        "best_val_auroc": val_au,
                        "test_pooled_auroc": r["pooled_auroc"]},
                       ckpt_path)
            # Also dump per-source scores on test for tau-table derivation.
            scores_path = os.path.join(args.save_ckpt_dir,
                                         f"{name}_scores.json")
            test_scores = _per_source_scores(m, test)
            with open(scores_path, "w") as f:
                json.dump(test_scores, f, indent=2)
            print(f"  [ckpt] saved {ckpt_path} + {scores_path}")

    print("\n=== P16B summary (single-image VQA test pooled AUROC) ===")
    print(f"{'model':<20} {'n_params':>10} {'val_auroc':>10} "
          f"{'test_pooled':>12}")
    for name, r in results.items():
        print(f"{name:<20} {r['n_params']:>10,} "
              f"{r['best_val_auroc']:>10.4f} "
              f"{r['test_pooled_auroc']:>12.4f}")
    print(f"\nBaseline lp_only_pure (4 mean/min/max/entropy scalars): "
          f"val=0.852 test=0.841")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[p16b] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
