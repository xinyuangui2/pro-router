"""Train the classifier head on cached Qwen2.5-VL-7B hidden states.

Reads .npz feature files produced by extract_hidden_states_for_classifier.py
(features [N, n_cuts, H], position_fractions [N, n_cuts], labels [N], sources [N]).

Flattens to per-(sample, cut) examples, trains the chosen head architecture
(mlp / mlp_resnet / multitask / moe / film), BCE loss, AdamW. Tracks per-cut
and per-source AUROC on val. Early-stops on val_auroc_100pct. Fits a
temperature scalar after.

Single-GPU. Run on cuda:0.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# sklearn is only used by training/eval here; the vLLM worker imports
# this module via eval_classifier_head for the head builders + forward
# fn alone and never touches sklearn. Stub at module load so a worker
# image without sklearn can still import.
try:
    from sklearn.metrics import brier_score_loss, roc_auc_score
except ImportError:
    def _missing_sklearn(_name):
        def _raise(*_a, **_kw):
            raise ImportError(
                f"{_name} requires scikit-learn; install it to use "
                "train_classifier_head's training/eval paths."
            )
        return _raise
    brier_score_loss = _missing_sklearn("brier_score_loss")
    roc_auc_score = _missing_sklearn("roc_auc_score")
from torch.utils.data import DataLoader, TensorDataset


# Source → coarse domain grouping for multitask / per-source MoE analysis.
DOMAIN_GROUPS = {
    "lmsys-chat-1m": "chat",
    "lmsys": "chat",
    "mixinstruct": "chat",
    "oasst": "chat",
    "wildchat": "chat",
    "dolly": "chat",
    "mmlu": "aug-mc",
    "gsm8k": "aug-math",
    "mbpp": "aug-code",
    "magicoder": "aug-code",
    "humaneval": "aug-code",
}


def domain_for_source(s: str) -> str:
    s = s.lower()
    if s in DOMAIN_GROUPS:
        return DOMAIN_GROUPS[s]
    if "chat" in s or "oasst" in s or "wild" in s or "dolly" in s or "mix" in s:
        return "chat"
    if "math" in s or "gsm" in s:
        return "aug-math"
    if "code" in s or "mbpp" in s or "humaneval" in s or "magicoder" in s:
        return "aug-code"
    if "mmlu" in s:
        return "aug-mc"
    return "other"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--val-features", required=True)
    ap.add_argument("--hidden-dim", type=int, default=3584)
    ap.add_argument("--pos-dim", type=int, default=1)
    ap.add_argument("--layers", type=int, nargs="+", default=[1024, 512, 256])
    ap.add_argument("--activation", choices=["leaky_relu", "gelu", "relu"],
                    default="leaky_relu")
    ap.add_argument("--architecture",
                    choices=["mlp", "mlp_resnet", "multitask", "moe", "film"],
                    default="mlp")
    ap.add_argument("--moe-experts", type=int, default=4)
    ap.add_argument("--moe-load-balance-loss", type=float, default=0.01)
    ap.add_argument("--multitask-domain-loss", type=float, default=0.5)
    ap.add_argument("--multitask-domain-mode", choices=["coarse", "source"], default="source",
                    help="`source` uses raw source strings as labels; `coarse` "
                         "uses chat/aug-* groupings (degenerate when train is chat-only).")
    ap.add_argument("--multitask-body-layers", type=int, nargs="+", default=None,
                    help="Hidden widths for the shared multitask body. "
                         "Default None → [512, 256] (legacy). "
                         "Pass e.g. 1024 512 256 for a wider body matching the "
                         "unified MLP.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout probability between hidden layers. 0 disables.")
    ap.add_argument("--loss", choices=["bce"], default="bce")
    ap.add_argument("--uniform-cut-weight", action="store_true")
    ap.add_argument("--downweight-100pct", type=float, default=None,
                    help="If set, multiply the 100%% cut's BCE by this factor.")
    ap.add_argument("--cut-positions-train", type=float, nargs="+", default=None,
                    help="If set, only train + early-stop on these cut positions. "
                         "Filters the cached features to the matching cut indices "
                         "after loading. Useful for L1-only (0.00) or L2-only "
                         "(1.00) head ablations.")
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--temperature-scale-on-val", action="store_true")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--metrics-out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


ACT_MAP = {
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def make_mlp(in_dim: int, hiddens: list[int], activation: str,
             dropout: float = 0.0) -> nn.Sequential:
    act = ACT_MAP[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hiddens:
        layers.append(nn.Linear(prev, h))
        layers.append(act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class MlpResNet(nn.Module):
    """MLP with residual connections every 2 layers (skip from block input to block output,
    with linear projection when dims differ).
    """
    def __init__(self, in_dim: int, hiddens: list[int], activation: str, dropout: float = 0.0):
        super().__init__()
        self.act = ACT_MAP[activation]()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linears = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        prev = in_dim
        block_in = in_dim
        for i, h in enumerate(hiddens):
            self.linears.append(nn.Linear(prev, h))
            if i % 2 == 1:
                self.skip_projs.append(
                    nn.Linear(block_in, h, bias=False) if block_in != h else nn.Identity()
                )
                block_in = h
            prev = h
        self.out = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block_in = x
        sp_idx = 0
        for i, lin in enumerate(self.linears):
            y = self.act(lin(x))
            y = self.dropout(y)
            if i % 2 == 1:
                y = y + self.skip_projs[sp_idx](block_in)
                sp_idx += 1
                block_in = y
            x = y
        return self.out(x)


class MultitaskHead(nn.Module):
    def __init__(self, in_dim: int, n_domains: int,
                 body_layers: list | None = None, dropout: float = 0.0):
        super().__init__()
        # Backward-compat default: old hardcoded body [512, 256]
        if body_layers is None:
            body_layers = [512, 256]
        body: list[nn.Module] = []
        prev = in_dim
        for i, w in enumerate(body_layers):
            body.append(nn.Linear(prev, w))
            body.append(nn.LeakyReLU(0.01))
            if i < len(body_layers) - 1:
                body.append(nn.Dropout(dropout))
            prev = w
        self.body = nn.Sequential(*body)
        last = body_layers[-1]
        self.confidence = nn.Sequential(
            nn.Linear(last, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
        )
        self.domain = nn.Sequential(
            nn.Linear(last, 64), nn.LeakyReLU(0.01),
            nn.Linear(64, n_domains),
        )

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        return self.confidence(h), self.domain(h)


class MoEHead(nn.Module):
    def __init__(self, in_dim: int, n_experts: int, dropout: float = 0.0):
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 64), nn.LeakyReLU(0.01),
            nn.Linear(64, n_experts),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 256), nn.LeakyReLU(0.01),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(256, 1),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor):
        gate_logits = self.gate(x)
        gate = F.softmax(gate_logits, dim=-1)  # [B, K]
        expert_logits = torch.cat([e(x) for e in self.experts], dim=-1)  # [B, K]
        logit = (gate * expert_logits).sum(dim=-1, keepdim=True)
        return logit, gate


class FilmHead(nn.Module):
    """4-layer MLP with FiLM modulation conditioned on a learned embedding of the input itself
    (since we don't get explicit domain labels at inference). The modulator can amortize a small
    domain-classifier internally.
    """
    def __init__(self, in_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, 16), nn.LeakyReLU(0.01),
            nn.Linear(16, 16),
        )
        widths = [1024, 512, 256, 128]
        self.layers = nn.ModuleList()
        self.gammas = nn.ModuleList()
        self.betas = nn.ModuleList()
        prev = in_dim
        for w in widths:
            self.layers.append(nn.Linear(prev, w))
            self.gammas.append(nn.Linear(16, w))
            self.betas.append(nn.Linear(16, w))
            prev = w
        self.act = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embed(x)
        for lin, g, b in zip(self.layers, self.gammas, self.betas):
            y = lin(x)
            y = y * g(e) + b(e)
            y = self.act(y)
            y = self.dropout(y)
            x = y
        return self.out(x)


def build_model(args: argparse.Namespace, in_dim: int, n_domains: int) -> nn.Module:
    if args.architecture == "mlp":
        return make_mlp(in_dim, args.layers, args.activation, dropout=args.dropout)
    if args.architecture == "mlp_resnet":
        return MlpResNet(in_dim, args.layers, args.activation, dropout=args.dropout)
    if args.architecture == "multitask":
        body_layers = getattr(args, "multitask_body_layers", None)
        return MultitaskHead(in_dim, n_domains=n_domains,
                              body_layers=body_layers, dropout=args.dropout)
    if args.architecture == "moe":
        return MoEHead(in_dim, n_experts=args.moe_experts, dropout=args.dropout)
    if args.architecture == "film":
        return FilmHead(in_dim, dropout=args.dropout)
    raise ValueError(f"unknown architecture {args.architecture}")


def forward_logits(model: nn.Module, args: argparse.Namespace, x: torch.Tensor):
    """Returns (logits[B], aux). aux is dict that may contain 'gate' or 'domain_logits'."""
    if args.architecture == "mlp" or args.architecture == "mlp_resnet" or args.architecture == "film":
        return model(x).squeeze(-1), {}
    if args.architecture == "multitask":
        conf, dom = model(x)
        return conf.squeeze(-1), {"domain_logits": dom}
    if args.architecture == "moe":
        logit, gate = model(x)
        return logit.squeeze(-1), {"gate": gate}
    if args.architecture == "c17_moe":
        # C17MoEHead.forward returns (logit_squeezed, aux_dict)
        logit, aux = model(x)
        return logit, aux
    raise ValueError(f"unknown architecture {args.architecture}")


def flatten(features: np.ndarray, pos: np.ndarray, labels: np.ndarray):
    """(N, C, H), (N, C), (N) → (N*C, H+1), (N*C), (N*C cut_index)."""
    N, C, H = features.shape
    feats = features.reshape(N * C, H).astype(np.float32)
    posflat = pos.reshape(N * C, 1).astype(np.float32)
    x = np.concatenate([feats, posflat], axis=1)
    y = np.repeat(labels, C).astype(np.float32)
    cut_index = np.tile(np.arange(C), N)
    return x, y, cut_index


def per_cut_auroc(scores: np.ndarray, labels: np.ndarray, cut_index: np.ndarray,
                  cut_positions: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for ci, c in enumerate(cut_positions):
        mask = cut_index == ci
        ys = labels[mask]
        ss = scores[mask]
        if len(set(ys.tolist())) < 2:
            out[f"{c:.2f}"] = float("nan")
        else:
            out[f"{c:.2f}"] = float(roc_auc_score(ys, ss))
    return out


def per_source_auroc_at_cut(scores: np.ndarray, labels: np.ndarray, cut_index: np.ndarray,
                            sources: np.ndarray, cut_positions: np.ndarray,
                            target_cut: float = 1.0) -> dict[str, float]:
    """sources is per-record (length N); we expand by tiling × C and select the target cut."""
    N = len(sources)
    C = len(cut_positions)
    src_tiled = np.repeat(sources, C)
    target_ci = int(np.argmin(np.abs(cut_positions - target_cut)))
    mask_target = cut_index == target_ci
    out: dict[str, float] = {}
    for src in sorted(set(sources.tolist())):
        m = mask_target & (src_tiled == src)
        ys = labels[m]
        ss = scores[m]
        if len(set(ys.tolist())) < 2 or len(ys) < 10:
            out[src] = float("nan")
        else:
            out[src] = float(roc_auc_score(ys, ss))
    return out


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = probs[m].mean()
        acc = labels[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def fit_temperature(logits_at_100pct: np.ndarray, labels_at_100pct: np.ndarray,
                    device: torch.device, max_iter: int = 200) -> float:
    """LBFGS on scalar T to minimize NLL of (logit / T)."""
    logits = torch.tensor(logits_at_100pct, dtype=torch.float32, device=device)
    labels = torch.tensor(labels_at_100pct, dtype=torch.float32, device=device)
    log_T = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T)
        scaled = logits / T
        loss = F.binary_cross_entropy_with_logits(scaled, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_T.detach()).item())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[load] train features: {args.train_features}")
    tr = np.load(args.train_features, allow_pickle=True)
    print(f"  features={tr['features'].shape} labels={tr['labels'].shape} "
          f"ACCEPT%={tr['labels'].mean()*100:.1f}")
    print(f"[load] val features:   {args.val_features}")
    va = np.load(args.val_features, allow_pickle=True)
    print(f"  features={va['features'].shape} labels={va['labels'].shape} "
          f"ACCEPT%={va['labels'].mean()*100:.1f}")

    cut_positions = tr["cut_positions"] if "cut_positions" in tr.files else \
                    np.array([0.10, 0.25, 0.50, 0.75, 1.00], dtype=np.float32)
    cut_positions = np.asarray(cut_positions, dtype=np.float32)

    tr_features = tr["features"]
    tr_pos = tr["position_fractions"]
    va_features = va["features"]
    va_pos = va["position_fractions"]
    if args.cut_positions_train:
        keep_idxs = [i for i, p in enumerate(cut_positions)
                     if any(abs(float(p) - float(q)) < 1e-3 for q in args.cut_positions_train)]
        if not keep_idxs:
            raise SystemExit(
                f"--cut-positions-train {args.cut_positions_train} did not match "
                f"any of the cached cuts {cut_positions.tolist()}"
            )
        cut_positions = cut_positions[keep_idxs]
        tr_features = tr_features[:, keep_idxs, :]
        tr_pos = tr_pos[:, keep_idxs]
        va_features = va_features[:, keep_idxs, :]
        va_pos = va_pos[:, keep_idxs]
        print(f"[filter] kept cut idxs {keep_idxs} → cuts {cut_positions.tolist()}")

    x_tr, y_tr, ci_tr = flatten(tr_features, tr_pos, tr["labels"])
    x_va, y_va, ci_va = flatten(va_features, va_pos, va["labels"])
    print(f"[flatten] train (N*C,H+1)={x_tr.shape} val={x_va.shape}")

    # Domain labels for multitask, expanded over cuts.
    sources_tr = np.asarray(tr["sources"])
    sources_va = np.asarray(va["sources"])
    if args.multitask_domain_mode == "coarse":
        domains_tr_record = np.array([domain_for_source(str(s)) for s in sources_tr])
        domains_va_record = np.array([domain_for_source(str(s)) for s in sources_va])
    else:
        domains_tr_record = np.array([str(s) for s in sources_tr])
        domains_va_record = np.array([str(s) for s in sources_va])
    domain_vocab = sorted(set(domains_tr_record.tolist()) | set(domains_va_record.tolist()))
    domain_to_id = {d: i for i, d in enumerate(domain_vocab)}
    n_domains = len(domain_vocab)
    d_tr = np.repeat(np.array([domain_to_id[d] for d in domains_tr_record]),
                     len(cut_positions))
    d_va = np.repeat(np.array([domain_to_id[d] for d in domains_va_record]),
                     len(cut_positions))
    print(f"[domains] vocab={domain_vocab} n_domains={n_domains}")

    in_dim = args.hidden_dim + args.pos_dim
    if x_tr.shape[1] != in_dim:
        raise SystemExit(f"feature dim {x_tr.shape[1]} != hidden+pos {in_dim}")

    model = build_model(args, in_dim, n_domains).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model arch={args.architecture}] params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_ds = TensorDataset(
        torch.from_numpy(x_tr), torch.from_numpy(y_tr), torch.from_numpy(ci_tr),
        torch.from_numpy(d_tr).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_va), torch.from_numpy(y_va), torch.from_numpy(ci_va),
        torch.from_numpy(d_va).long(),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4096, shuffle=False, num_workers=2, pin_memory=True,
    )

    per_epoch: list[dict] = []
    best_val_auroc_100 = -1.0
    best_epoch = -1
    best_train_auroc_100 = float("nan")
    best_per_cut: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        train_n = 0
        for xb, yb, cib, dib in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            cib = cib.to(device, non_blocking=True)
            dib = dib.to(device, non_blocking=True)
            logits, aux = forward_logits(model, args, xb)
            per_sample = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            if args.downweight_100pct is not None:
                w = torch.where(
                    cib == (len(cut_positions) - 1),
                    torch.full_like(per_sample, args.downweight_100pct),
                    torch.ones_like(per_sample),
                )
                bce = (per_sample * w).mean()
            else:
                bce = per_sample.mean()
            loss = bce
            if args.architecture == "multitask":
                loss = loss + args.multitask_domain_loss * F.cross_entropy(aux["domain_logits"], dib)
            elif args.architecture == "moe":
                gate = aux["gate"]                    # [B, K]
                gate_mean = gate.mean(dim=0)          # [K] mean assignment per expert
                cv = gate_mean.std() / (gate_mean.mean() + 1e-8)
                loss = loss + args.moe_load_balance_loss * (cv ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * xb.size(0)
            train_n += xb.size(0)

        # Eval on train (sample) and val.
        model.eval()
        with torch.inference_mode():
            sub_n = min(20000, x_tr.shape[0])
            idx_sub = np.random.choice(x_tr.shape[0], sub_n, replace=False)
            xs = torch.from_numpy(x_tr[idx_sub]).to(device)
            train_logits_sub, _ = forward_logits(model, args, xs)
            train_logits_sub = train_logits_sub.cpu().numpy()
            train_labels_sub = y_tr[idx_sub]
            train_ci_sub = ci_tr[idx_sub]
            target_ci = len(cut_positions) - 1
            mask = train_ci_sub == target_ci
            if mask.sum() > 10 and len(set(train_labels_sub[mask].tolist())) > 1:
                train_auroc_100 = float(
                    roc_auc_score(train_labels_sub[mask], train_logits_sub[mask])
                )
            else:
                train_auroc_100 = float("nan")

            val_logits_chunks: list[np.ndarray] = []
            val_loss_sum = 0.0
            val_n = 0
            for xb, yb, cib, _dib in val_loader:
                xb_d = xb.to(device, non_blocking=True)
                yb_d = yb.to(device, non_blocking=True)
                logits, _aux = forward_logits(model, args, xb_d)
                vloss = F.binary_cross_entropy_with_logits(logits, yb_d).item()
                val_loss_sum += vloss * xb.size(0)
                val_n += xb.size(0)
                val_logits_chunks.append(logits.cpu().numpy())
            val_logits = np.concatenate(val_logits_chunks)
            val_scores = 1.0 / (1.0 + np.exp(-val_logits))

        per_cut = per_cut_auroc(val_scores, y_va, ci_va, cut_positions)
        val_auroc_100 = per_cut[f"{cut_positions[-1]:.2f}"]
        if not np.isfinite(val_auroc_100):
            val_auroc_100 = -1.0

        epoch_dt = time.time() - t0
        per_epoch.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_n, 1),
            "val_loss": val_loss_sum / max(val_n, 1),
            "train_auroc_100pct": train_auroc_100,
            "val_auroc_100pct": val_auroc_100,
            "val_auroc_per_cut": per_cut,
            "wall_s": epoch_dt,
        })
        print(
            f"[epoch {epoch:>2}] train_loss={per_epoch[-1]['train_loss']:.4f} "
            f"val_loss={per_epoch[-1]['val_loss']:.4f} "
            f"val_AUROC(100%)={val_auroc_100:.4f} "
            f"per_cut={ {k: round(v,3) for k,v in per_cut.items()} } "
            f"({epoch_dt:.1f}s)", flush=True,
        )

        if val_auroc_100 > best_val_auroc_100:
            best_val_auroc_100 = val_auroc_100
            best_epoch = epoch
            best_train_auroc_100 = train_auroc_100
            best_per_cut = per_cut
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"[early-stop] no improvement for {no_improve} epochs at epoch {epoch}")
                break

    if best_state is None:
        raise SystemExit("training produced no usable epoch")
    model.load_state_dict(best_state)

    # Final val pass: full logits at all cuts → per-source @ 100%, calibration, aux stats.
    model.eval()
    val_aux_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.inference_mode():
        chunks: list[np.ndarray] = []
        for xb, _, _, _dib in val_loader:
            logits, aux = forward_logits(model, args, xb.to(device))
            chunks.append(logits.cpu().numpy())
            for k, v in aux.items():
                val_aux_chunks[k].append(v.cpu().numpy())
        val_logits_all = np.concatenate(chunks)
    val_aux_all = {k: np.concatenate(v) for k, v in val_aux_chunks.items()}

    val_scores_all = 1.0 / (1.0 + np.exp(-val_logits_all))
    val_per_source = per_source_auroc_at_cut(
        val_scores_all, y_va, ci_va, va["sources"], cut_positions, target_cut=1.0
    )

    # Calibration on the 100% cut only.
    target_ci = len(cut_positions) - 1
    mask_100 = ci_va == target_ci
    val_logits_100 = val_logits_all[mask_100]
    val_labels_100 = y_va[mask_100]
    val_scores_100 = val_scores_all[mask_100]
    ece_pre = expected_calibration_error(val_scores_100, val_labels_100)
    brier_pre = float(brier_score_loss(val_labels_100, val_scores_100))

    temperature = 1.0
    if args.temperature_scale_on_val:
        temperature = fit_temperature(val_logits_100, val_labels_100, device)
    val_logits_100_T = val_logits_100 / temperature
    val_scores_100_T = 1.0 / (1.0 + np.exp(-val_logits_100_T))
    ece_post = expected_calibration_error(val_scores_100_T, val_labels_100)
    brier_post = float(brier_score_loss(val_labels_100, val_scores_100_T))

    metrics: dict = {
        "args": vars(args),
        "architecture": args.architecture,
        "n_params": int(n_params),
        "cut_positions": cut_positions.tolist(),
        "per_epoch": per_epoch,
        "best_epoch": best_epoch,
        "best_val_auroc_100pct": best_val_auroc_100,
        "best_val_auroc_per_cut": best_per_cut,
        "best_train_auroc_100pct": best_train_auroc_100,
        "val_per_source_auroc_100pct": val_per_source,
        "domain_vocab": domain_vocab,
        "temperature": temperature,
        "val_brier": brier_post,
        "val_ece": ece_post,
        "val_brier_pre_temp": brier_pre,
        "val_ece_pre_temp": ece_pre,
    }

    # MoE: per-source gate distribution at 100% cut.
    if args.architecture == "moe" and "gate" in val_aux_all:
        gate_top = val_aux_all["gate"][mask_100]  # [N_top, K]
        # sources at 100% cut, one per record
        srcs_top = np.asarray(va["sources"])
        per_src_gate: dict[str, dict] = {}
        for src in sorted(set(srcs_top.tolist())):
            m = srcs_top == src
            if m.sum() == 0:
                continue
            mean = gate_top[m].mean(axis=0)
            per_src_gate[str(src)] = {f"e{k}": float(mean[k]) for k in range(gate_top.shape[1])}
        metrics["per_source_gate"] = per_src_gate
        metrics["overall_gate_mean"] = {f"e{k}": float(gate_top.mean(axis=0)[k])
                                        for k in range(gate_top.shape[1])}

    # Multitask: domain accuracy on val.
    if args.architecture == "multitask" and "domain_logits" in val_aux_all:
        dom_top = val_aux_all["domain_logits"][mask_100]
        true_dom = d_va[mask_100]
        pred_dom = dom_top.argmax(axis=-1)
        metrics["val_domain_accuracy"] = float((pred_dom == true_dom).mean())

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "args": vars(args),
        "cut_positions": cut_positions.tolist(),
        "temperature": temperature,
        "best_val_auroc_100pct": best_val_auroc_100,
        "best_epoch": best_epoch,
        "domain_vocab": domain_vocab,
    }, args.checkpoint)
    print(f"[save] checkpoint → {args.checkpoint}")

    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save] metrics → {args.metrics_out}")


if __name__ == "__main__":
    main()
