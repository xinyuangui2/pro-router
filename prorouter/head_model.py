"""The confidence head: a 2-layer transformer over a generation's per-token
sampling-distribution trajectory.

The head consumes a [T, 4] sequence -- one row per generated token, columns
[chosen_logprob, max_softmax_prob, neg_entropy, position_fraction] -- and emits
a single logit for "the small model's answer is correct". Padding is masked so
that batching cannot leak information across sequences of different lengths.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

N_FEATURES = 4
FEATURE_NAMES = ["chosen_logprob", "max_prob", "neg_entropy", "position_frac"]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 600):
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
    """The paper's head. The defaults are exactly the bundled
    `weights/head.pt` geometry, so `TransformerSeq()` loads that checkpoint's
    state_dict directly; they give the reported 67,329 parameters.

    `max_len` sizes the positional-encoding buffer, not the parameter count,
    but it does bound the answer length the head can score: a generation
    longer than `max_len` tokens raises in `PositionalEncoding.forward`.
    `gate.py` rebuilds from the checkpoint's own `hparams` and overrides these.
    """

    def __init__(self, n_features: int = N_FEATURES, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, d_ff: int = 128,
                 max_len: int = 1024):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pe = PositionalEncoding(d_model, max_len=max_len)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x, lens):
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= \
            lens.to(x.device).unsqueeze(1)
        h = self.proj(x)
        h = self.pe(h)
        h = self.encoder(h, src_key_padding_mask=mask)
        not_mask = (~mask).float().unsqueeze(-1)
        pooled = (h * not_mask).sum(dim=1) / not_mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled).squeeze(-1)


class SeqDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        feats = torch.from_numpy(np.asarray(r["features"], dtype=np.float32))
        return feats, float(r["label"]), r.get("source", "")


def collate_pad(batch):
    feats, labels, sources = zip(*batch)
    lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    T = max(1, int(lens.max().item()))
    X = torch.zeros((len(batch), T, N_FEATURES), dtype=torch.float32)
    for i, f in enumerate(feats):
        if f.shape[0] > 0:
            X[i, :f.shape[0], :] = f
    return X, lens, torch.tensor(labels, dtype=torch.float32), list(sources)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
