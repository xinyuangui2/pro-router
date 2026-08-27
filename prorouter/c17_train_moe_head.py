"""Track B — Train a MoE-k4 head on VLM features.

Mirrors the earlier MoE recipe but exposes gate_logits and adds an optional
source-classification auxiliary cross-entropy term:

    L = BCE(logit, y) + λ_lb * (cv(gate_mean) ** 2)
                       + λ_src * CE(gate_logits, source_id)

When --source-ce-weight > 0, the gate is forced to predict the record's
source. Reveals whether the cut-0/cut-1 hidden state contains source-
separable structure on VLM data (it did not on chat data).

Cuts trained on: 0.00 + 1.00 (the cuts the cascade fires at). Saves
checkpoint with full args + domain_vocab so eval scripts can rebuild
the model.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


class C17MoEHead(nn.Module):
    """MoE head that exposes gate logits (for source-CE) alongside the
    weighted-sum binary logit.
    """
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
        gate = F.softmax(gate_logits, dim=-1)
        expert_logits = torch.cat([e(x) for e in self.experts], dim=-1)
        logit = (gate * expert_logits).sum(dim=-1, keepdim=True)
        return logit.squeeze(-1), {"gate": gate, "gate_logits": gate_logits}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--val-features", required=True)
    ap.add_argument("--hidden-dim", type=int, default=3584)
    ap.add_argument("--pos-dim", type=int, default=1)
    ap.add_argument("--n-experts", type=int, default=4)
    ap.add_argument("--lb-loss-weight", type=float, default=0.01)
    ap.add_argument("--source-ce-weight", type=float, default=0.0)
    ap.add_argument("--cut-positions-train", type=float, nargs="+",
                    default=[0.0, 1.0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--temperature-scale-on-val", action="store_true",
                    default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def flatten(features, pos, labels, sources, cut_keep):
    N, C, H = features.shape
    feats = features.reshape(N * C, H).astype(np.float32)
    posflat = pos.reshape(N * C, 1).astype(np.float32)
    x = np.concatenate([feats, posflat], axis=1)
    y = np.repeat(labels, C).astype(np.float32)
    cut_idx = np.tile(np.arange(C), N)
    src = np.repeat(sources, C)
    keep = np.isin(cut_idx, cut_keep)
    return x[keep], y[keep], cut_idx[keep], src[keep]


def fit_temperature(logits, labels, device):
    log_T = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=200)
    L = torch.tensor(logits, dtype=torch.float32, device=device)
    Y = torch.tensor(labels, dtype=torch.float32, device=device)

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T)
        loss = F.binary_cross_entropy_with_logits(L / T, Y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(log_T.detach()).item())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[load] train {args.train_features}")
    tr = np.load(args.train_features, allow_pickle=True)
    print(f"[load] val   {args.val_features}")
    va = np.load(args.val_features, allow_pickle=True)

    cut_positions = np.asarray(tr["cut_positions"], dtype=np.float32)
    cut_keep = [i for i, p in enumerate(cut_positions)
                if any(abs(float(p) - float(q)) < 1e-3 for q in args.cut_positions_train)]
    if not cut_keep:
        raise SystemExit(f"no cut positions match {args.cut_positions_train}")
    cut_positions_used = cut_positions[cut_keep]
    print(f"[cuts] using {cut_positions_used.tolist()}  (idxs {cut_keep})")

    x_tr, y_tr, ci_tr, src_tr = flatten(
        tr["features"], tr["position_fractions"], tr["labels"], tr["sources"], cut_keep,
    )
    x_va, y_va, ci_va, src_va = flatten(
        va["features"], va["position_fractions"], va["labels"], va["sources"], cut_keep,
    )

    domain_vocab = sorted(set([str(s) for s in tr["sources"].tolist()]))
    src_to_id = {s: i for i, s in enumerate(domain_vocab)}
    s_tr = np.array([src_to_id[str(s)] for s in src_tr], dtype=np.int64)
    s_va = np.array([src_to_id[str(s)] for s in src_va], dtype=np.int64)
    n_domains = len(domain_vocab)
    print(f"[domains] vocab={domain_vocab} n={n_domains}")

    n_experts = args.n_experts
    if args.source_ce_weight > 0 and n_experts != n_domains:
        print(f"[warn] source-CE active and n_experts={n_experts} != n_domains={n_domains}; "
              f"overriding n_experts → {n_domains} so gate logits map 1:1 to source labels")
        n_experts = n_domains

    in_dim = args.hidden_dim + args.pos_dim
    if x_tr.shape[1] != in_dim:
        raise SystemExit(f"feature dim {x_tr.shape[1]} != hidden+pos {in_dim}")

    model = C17MoEHead(in_dim, n_experts=n_experts, dropout=args.dropout).to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_ds = TensorDataset(
        torch.from_numpy(x_tr), torch.from_numpy(y_tr),
        torch.from_numpy(ci_tr), torch.from_numpy(s_tr),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_va), torch.from_numpy(y_va),
        torch.from_numpy(ci_va), torch.from_numpy(s_va),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False,
                              num_workers=2, pin_memory=True)

    per_epoch = []
    best = {"val_auroc_100": -1.0, "epoch": -1, "state": None}
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sum_loss = 0.0
        sum_n = 0
        sum_bce = 0.0
        sum_lb = 0.0
        sum_src = 0.0
        for xb, yb, _cib, sb in train_loader:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            logits, aux = model(xb)
            bce = F.binary_cross_entropy_with_logits(logits, yb)
            gate_mean = aux["gate"].mean(dim=0)
            cv = gate_mean.std() / (gate_mean.mean() + 1e-8)
            lb = args.lb_loss_weight * (cv ** 2)
            loss = bce + lb
            if args.source_ce_weight > 0:
                src_loss = F.cross_entropy(aux["gate_logits"], sb)
                loss = loss + args.source_ce_weight * src_loss
                sum_src += src_loss.item() * xb.size(0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sum_loss += loss.item() * xb.size(0)
            sum_bce += bce.item() * xb.size(0)
            sum_lb += lb.item() * xb.size(0)
            sum_n += xb.size(0)

        # Eval on val
        model.eval()
        with torch.inference_mode():
            chunks_logits = []
            chunks_gate = []
            for xb, _, _, _ in val_loader:
                logits, aux = model(xb.to(device))
                chunks_logits.append(logits.cpu().numpy())
                chunks_gate.append(aux["gate"].cpu().numpy())
            val_logits = np.concatenate(chunks_logits)
            val_gate = np.concatenate(chunks_gate)
        val_scores = 1.0 / (1.0 + np.exp(-val_logits))

        # AUROC at last cut
        target_ci = max(cut_keep)
        m_top = ci_va == target_ci
        if m_top.sum() > 10 and len(set(y_va[m_top].tolist())) > 1:
            val_auroc_100 = float(roc_auc_score(y_va[m_top], val_scores[m_top]))
        else:
            val_auroc_100 = float("nan")
        per_cut_auroc = {}
        for ci in cut_keep:
            m = ci_va == ci
            if m.sum() > 10 and len(set(y_va[m].tolist())) > 1:
                per_cut_auroc[f"{cut_positions[ci]:.2f}"] = float(
                    roc_auc_score(y_va[m], val_scores[m])
                )

        # Source-classification accuracy from gate-argmax (cut 1.0)
        gate_argmax = val_gate.argmax(axis=1)
        src_acc = float((gate_argmax[m_top] == s_va[m_top]).mean()) if m_top.sum() else 0.0

        epoch_dt = time.time() - t0
        rec = {
            "epoch": epoch,
            "loss": sum_loss / max(sum_n, 1),
            "bce": sum_bce / max(sum_n, 1),
            "lb": sum_lb / max(sum_n, 1),
            "src_loss": sum_src / max(sum_n, 1) if args.source_ce_weight > 0 else 0.0,
            "val_auroc_100": val_auroc_100,
            "val_per_cut_auroc": per_cut_auroc,
            "val_gate_src_acc_at_1.0": src_acc,
            "wall_s": epoch_dt,
        }
        per_epoch.append(rec)
        print(f"[ep {epoch:>2}] loss={rec['loss']:.4f} bce={rec['bce']:.4f} "
              f"lb={rec['lb']:.4f} "
              f"{'src=' + str(round(rec['src_loss'], 3)) if args.source_ce_weight > 0 else ''} "
              f"AUROC(1.0)={val_auroc_100:.4f} src_acc={src_acc:.3f} ({epoch_dt:.1f}s)",
              flush=True)

        if val_auroc_100 > best["val_auroc_100"]:
            best = {"val_auroc_100": val_auroc_100, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "per_cut": per_cut_auroc, "src_acc": src_acc}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"[early-stop] no improvement {no_improve} epochs")
                break

    # Reload best, fit T
    model.load_state_dict(best["state"])
    model.eval()
    with torch.inference_mode():
        all_logits = []
        for xb, _, _, _ in val_loader:
            logits, _ = model(xb.to(device))
            all_logits.append(logits.cpu().numpy())
        val_logits_all = np.concatenate(all_logits)

    target_ci = max(cut_keep)
    m_top = ci_va == target_ci
    T = fit_temperature(val_logits_all[m_top], y_va[m_top], device) if args.temperature_scale_on_val else 1.0
    print(f"[temp] T={T:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "args": {
            "architecture": "c17_moe",
            "hidden_dim": args.hidden_dim,
            "pos_dim": args.pos_dim,
            "n_experts": n_experts,
            "n_experts_requested": args.n_experts,
            "dropout": args.dropout,
            "lb_loss_weight": args.lb_loss_weight,
            "source_ce_weight": args.source_ce_weight,
        },
        "domain_vocab": domain_vocab,
        "cut_positions": cut_positions_used.tolist(),
        "temperature": T,
        "best_val_auroc_100": best["val_auroc_100"],
        "best_per_cut": best["per_cut"],
        "best_src_acc": best["src_acc"],
        "per_epoch": per_epoch,
    }
    torch.save(ckpt, args.output)
    print(f"[save] → {args.output}")
    metrics_out = args.output.replace(".pt", "_metrics.json")
    json.dump({k: v for k, v in ckpt.items() if k != "state_dict"},
              open(metrics_out, "w"), indent=2,
              default=lambda o: list(o) if hasattr(o, "tolist") else str(o))
    print(f"[save] → {metrics_out}")


if __name__ == "__main__":
    main()
