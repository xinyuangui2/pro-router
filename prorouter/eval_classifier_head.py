"""Apply a trained head to one or more eval splits.

Loads a head checkpoint, scores each (sample, cut), reports overall AUROC /
Brier / ECE / recall_at_p99 per split, per-cut AUROC, per-source AUROC.
Writes a single eval JSON plus optional ROC plots.

Supports head architectures from train_classifier_head.py:
  mlp / mlp_resnet / multitask / moe / film.

Usage:
  python -m prorouter.eval_classifier_head \
    --checkpoint /path/to/head.pt \
    --eval-features val=/path/val.npz indist_test=/path/indist_test.npz ... \
    --output /path/eval.json \
    [--plots-dir /path/plots]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# sklearn is only used by the eval/diagnostic paths in this module. The
# vLLM worker imports build_model_from_ckpt + _forward_logits and never
# touches sklearn, so a worker image without it must still import this
# module cleanly. Stub at module load; raise on first eval call.
try:
    from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
except ImportError:
    def _missing_sklearn(_name):
        def _raise(*_a, **_kw):
            raise ImportError(
                f"{_name} requires scikit-learn; install it to use "
                "eval_classifier_head's diagnostic paths."
            )
        return _raise
    brier_score_loss = _missing_sklearn("brier_score_loss")
    roc_auc_score = _missing_sklearn("roc_auc_score")
    roc_curve = _missing_sklearn("roc_curve")

# Architecture builders mirror train_classifier_head.py. Imported via Python module path.
from prorouter.train_classifier_head import (
    MlpResNet,
    MultitaskHead,
    MoEHead,
    FilmHead,
    forward_logits as _forward_logits,
    make_mlp,
)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(probs[m].mean() - labels[m].mean())
    return float(ece)


def parse_kv_list(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"expected name=path, got {it}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    return (d["features"], d["position_fractions"], d["labels"], d["sources"],
            d["cut_positions"] if "cut_positions" in d.files else
            np.array([0.10, 0.25, 0.50, 0.75, 1.00], dtype=np.float32))


class Args:
    """Lightweight stand-in to feed forward_logits the architecture flag from a ckpt."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def build_model_from_ckpt(ckpt: dict, in_dim: int, n_domains: int) -> tuple[nn.Module, Args]:
    cargs = ckpt.get("args", {})
    arch = cargs.get("architecture", "mlp")
    layers = list(cargs.get("layers", [1024, 512, 256]))
    activation = cargs.get("activation", "leaky_relu")
    dropout = float(cargs.get("dropout", 0.0))

    if arch == "mlp":
        model = make_mlp(in_dim, layers, activation, dropout=dropout)
    elif arch == "mlp_resnet":
        model = MlpResNet(in_dim, layers, activation, dropout=dropout)
    elif arch == "multitask":
        body_layers = cargs.get("multitask_body_layers", None)
        model = MultitaskHead(in_dim, n_domains=n_domains,
                                body_layers=body_layers, dropout=dropout)
    elif arch == "moe":
        model = MoEHead(in_dim, n_experts=int(cargs.get("moe_experts", 4)), dropout=dropout)
    elif arch == "film":
        model = FilmHead(in_dim, dropout=dropout)
    elif arch == "c17_moe":
        from prorouter.c17_train_moe_head import C17MoEHead
        model = C17MoEHead(in_dim, n_experts=int(cargs.get("n_experts", 4)), dropout=dropout)
    else:
        raise SystemExit(f"unknown architecture in ckpt: {arch}")

    args_obj = Args(architecture=arch)
    return model, args_obj


def score_split(model: nn.Module, fwd_args: Args, device: torch.device,
                features: np.ndarray, position_fractions: np.ndarray,
                batch: int = 4096) -> np.ndarray:
    """Returns logits flattened to (N*C,) following row-major (sample, cut)."""
    N, C, H = features.shape
    feats = features.reshape(N * C, H).astype(np.float32)
    pos = position_fractions.reshape(N * C, 1).astype(np.float32)
    x = np.concatenate([feats, pos], axis=1)
    out = np.empty(x.shape[0], dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for s in range(0, x.shape[0], batch):
            xb = torch.from_numpy(x[s:s+batch]).to(device)
            logits, _aux = _forward_logits(model, fwd_args, xb)
            out[s:s+batch] = logits.cpu().numpy()
    return out


def per_cut_auroc(scores: np.ndarray, labels_per_pair: np.ndarray, cut_index: np.ndarray,
                  cut_positions: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for ci, c in enumerate(cut_positions):
        m = cut_index == ci
        ys = labels_per_pair[m]
        ss = scores[m]
        out[f"{c:.2f}"] = float(roc_auc_score(ys, ss)) if len(set(ys.tolist())) > 1 else float("nan")
    return out


def per_source_auroc_at_cut(scores: np.ndarray, labels_per_pair: np.ndarray,
                            cut_index: np.ndarray, sources_per_pair: np.ndarray,
                            cut_positions: np.ndarray, target_cut: float = 1.0) -> dict[str, dict]:
    target_ci = int(np.argmin(np.abs(cut_positions - target_cut)))
    m_target = cut_index == target_ci
    out: dict[str, dict] = {}
    for src in sorted(set(sources_per_pair[m_target].tolist())):
        m = m_target & (sources_per_pair == src)
        ys = labels_per_pair[m]
        ss = scores[m]
        if len(ys) < 10 or len(set(ys.tolist())) < 2:
            out[src] = {"n": int(len(ys)), "auroc": float("nan")}
        else:
            out[src] = {"n": int(len(ys)), "auroc": float(roc_auc_score(ys, ss))}
    return out


def recall_at_precision(probs: np.ndarray, labels: np.ndarray, target_precision: float):
    """Pick the lowest-prob threshold whose precision_skip on this split is >= target,
    return recall (skip_rate among ACCEPTs) and the threshold.
    """
    if len(set(labels.tolist())) < 2:
        return float("nan"), float("nan")
    order = np.argsort(-probs)
    p = probs[order]; y = labels[order]
    cum_pos = np.cumsum(y)
    cum_n = np.arange(1, len(y) + 1)
    precision = cum_pos / cum_n
    valid = precision >= target_precision
    if not valid.any():
        return 0.0, float("inf")
    last = int(np.where(valid)[0].max())
    tau = float(p[last])
    n_pos_total = float(y.sum())
    recall = float(cum_pos[last] / n_pos_total) if n_pos_total > 0 else float("nan")
    return recall, tau


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-features", nargs="+", required=True,
                    help="space-separated name=path entries")
    ap.add_argument("--output", required=True)
    ap.add_argument("--plots-dir", default=None)
    ap.add_argument("--p99-target", type=float, default=0.99,
                    help="precision target for recall_at_pXX")
    ap.add_argument("--report-positions", type=float, nargs="+", default=None,
                    help="If set, also report per-position AUROC/Brier for each "
                         "listed cut position in per_split[name][per_position] and "
                         "per-source auroc_at_pos_X keys.")
    args = ap.parse_args()

    eval_paths = parse_kv_list(args.eval_features)
    print(f"[load] checkpoint {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cargs = ckpt.get("args", {})
    cut_positions = np.asarray(ckpt.get("cut_positions", [0.10, 0.25, 0.50, 0.75, 1.00]),
                               dtype=np.float32)
    temperature = float(ckpt.get("temperature", 1.0))
    in_dim = int(cargs.get("hidden_dim", 3584)) + int(cargs.get("pos_dim", 1))
    domain_vocab = ckpt.get("domain_vocab", ["chat", "aug-math", "aug-code", "aug-mc"])
    n_domains = len(domain_vocab)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, fwd_args = build_model_from_ckpt(ckpt, in_dim, n_domains)
    model = model.to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[load] model arch={fwd_args.architecture} on {device}, temperature={temperature:.4f}")

    per_split: dict[str, dict] = {}
    per_split_per_cut: dict[str, dict] = {}
    per_source: dict[str, dict] = {}

    for name, path in eval_paths.items():
        print(f"\n[eval] {name} ← {path}")
        features, position_fractions, labels, sources, cuts_in_file = load_npz(path)
        if not np.allclose(cuts_in_file, cut_positions):
            print(f"  NOTE: cut positions in file {cuts_in_file.tolist()} differ "
                  f"from ckpt {cut_positions.tolist()}; using file cuts for indexing.")
        # File cuts drive indexing — the model is applied to every (sample, cut)
        # in the file regardless of what the ckpt was trained on.
        cuts_eval = np.asarray(cuts_in_file, dtype=np.float32)
        N, C, H = features.shape
        labels_per_pair = np.repeat(labels, C).astype(np.float32)
        cut_index = np.tile(np.arange(C), N)
        sources_per_pair = np.repeat(sources, C)
        target_ci = int(np.argmin(np.abs(cuts_eval - 1.0)))

        logits = score_split(model, fwd_args, device, features, position_fractions)
        logits_T = logits / temperature
        probs = 1.0 / (1.0 + np.exp(-logits_T))

        # Headline metrics use the 100% cut.
        m_top = cut_index == target_ci
        labels_top = labels_per_pair[m_top]
        probs_top = probs[m_top]
        if len(set(labels_top.tolist())) > 1:
            auroc = float(roc_auc_score(labels_top, probs_top))
        else:
            auroc = float("nan")
        brier = float(brier_score_loss(labels_top, probs_top))
        ece = expected_calibration_error(probs_top, labels_top)
        accept_rate = float(labels.mean())

        recall_p99, tau_p99 = recall_at_precision(probs_top, labels_top, args.p99_target)

        per_split[name] = {
            "n": int(N), "auroc": auroc, "brier": brier, "ece": ece,
            "accept_rate": accept_rate,
            "recall_at_p99": recall_p99, "tau_at_p99": tau_p99,
        }
        per_split_per_cut[name] = per_cut_auroc(probs, labels_per_pair, cut_index, cuts_eval)
        per_source[name] = per_source_auroc_at_cut(
            probs, labels_per_pair, cut_index, sources_per_pair, cuts_eval, target_cut=1.0
        )

        if args.report_positions:
            pp_split: dict[str, dict] = {}
            for p in args.report_positions:
                ci = int(np.argmin(np.abs(cuts_eval - p)))
                m = cut_index == ci
                ys = labels_per_pair[m]
                ps_ = probs[m]
                key = f"{float(p):.2f}"
                if len(ys) and len(set(ys.tolist())) > 1:
                    pp_split[key] = {
                        "auroc": float(roc_auc_score(ys, ps_)),
                        "brier": float(brier_score_loss(ys, ps_)),
                        "n": int(len(ys)),
                    }
                else:
                    pp_split[key] = {
                        "auroc": float("nan"),
                        "brier": float("nan"),
                        "n": int(len(ys)),
                    }
            per_split[name]["per_position"] = pp_split

            # Per-source auroc_at_pos_X for each requested position.
            for src in per_source[name]:
                m_src = sources_per_pair == src
                for p in args.report_positions:
                    ci = int(np.argmin(np.abs(cuts_eval - p)))
                    m = m_src & (cut_index == ci)
                    ys = labels_per_pair[m]
                    ss = probs[m]
                    key = f"auroc_at_pos_{float(p):.2f}"
                    if len(ys) >= 10 and len(set(ys.tolist())) > 1:
                        per_source[name][src][key] = float(roc_auc_score(ys, ss))
                    else:
                        per_source[name][src][key] = float("nan")
        print(f"  n={N} auroc={auroc:.4f} brier={brier:.4f} ece={ece:.4f} "
              f"accept_rate={accept_rate:.3f} recall@p{int(args.p99_target*100)}={recall_p99:.3f}")
        print(f"  per_cut={ {k: round(v, 3) for k, v in per_split_per_cut[name].items()} }")
        for src, m in per_source[name].items():
            print(f"    {src:<14} n={m['n']:>6} auroc={m['auroc']:.3f}")

        if args.plots_dir:
            os.makedirs(args.plots_dir, exist_ok=True)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                if len(set(labels_top.tolist())) > 1:
                    fpr, tpr, _ = roc_curve(labels_top, probs_top)
                    plt.figure()
                    plt.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
                    plt.plot([0, 1], [0, 1], "--", color="gray")
                    plt.xlabel("FPR"); plt.ylabel("TPR")
                    plt.title(f"{name} ROC (100% cut)")
                    plt.legend()
                    plt.savefig(f"{args.plots_dir}/{name}_roc.png", bbox_inches="tight")
                    plt.close()
            except ImportError:
                pass

    out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "architecture": fwd_args.architecture,
        "temperature": temperature,
        "cut_positions": cut_positions.tolist(),
        "p99_target": args.p99_target,
        "per_split": per_split,
        "per_split_per_cut": per_split_per_cut,
        "per_source": per_source,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] → {args.output}")


if __name__ == "__main__":
    main()
