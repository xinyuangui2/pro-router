"""External cascade head actor.

Replaces the inline head firing inside Qwen2Model.forward with a
separate Ray actor that consumes last-position hidden states from
finished requests and emits SHIP/REGEN decisions. Cut-1-only (the
post-generation hidden state); no cut-0 / OR-skip.

Why externalize: keeping the head inline forces the draft to run
in eager+sync mode ( τ-calibration mismatch under cudagraphs
drops ship rate 64% → 19% on long-CoT). With the head outside the
fork-modified inline path, the draft becomes pure vLLM under
graph_async, unlocking the ~25% throughput lift measured on
the base decoder.

Trade-off accepted: lose cut-0 early-SHIP latency, gain cudagraph
compatibility and a ~100-500× drop in head invocations (one per
request, not once per decode step per token position).

Usage:
    head_actor = CascadeHeadActor.options(num_gpus=0.1).remote(
        head_ckpt_path="weights/head.pt",
        tau_table_path="weights/tau.json",
        temperature=1.0,
    )
    decision = await head_actor.decide.remote(hidden_state, source)
    # decision ∈ {"SHIP", "REGEN"}

The decision uses τ_L2 only (post-generation threshold). τ_L1
(cut-0) is ignored. For the head + multitask_wide_tau_p_0.98
table, per-source τ_L2 is:
    y3_mmbench    0.768
    y3_mmmu       (look up)
    y3_mathvista  0.736
    y3_chartqa    0.901
    y3_docvqa     0.578

Ship_rate_per_cascade under cut-1-only will differ from γ.1
baselines (OR-skip rule) — some requests that would have early-
shipped via s00 ≥ τ_L1 are now subject to s10 ≥ τ_L2 alone. The
expected loss is small on workloads where cut-1 dominates SHIP
decisions, but every workload must be characterized empirically.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import ray
import torch


@ray.remote(num_cpus=1, num_gpus=0)
class CascadeHeadActor:
    """Loads the head + τ_L2 table once, serves decisions per request.

    Designed to colocate on rank-0's worker node — Ray placement is
    handled by the caller via NodeAffinitySchedulingStrategy or by
    pinning to a custom resource. The hidden_state argument is a
    fp16 tensor of shape [hidden_dim]; it gets `.float()` and a
    pos_fraction=1.0 column appended before the head's forward.
    """

    def __init__(
        self,
        head_ckpt_path: str,
        tau_table_path: str,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> None:
        from prorouter.eval_classifier_head import build_model_from_ckpt
        from prorouter.train_classifier_head import forward_logits

        self._device = torch.device(device)
        self._T = float(temperature)

        # τ table.
        with open(tau_table_path) as f:
            tau_data = json.load(f)
        # Per-source thresholds, plus a global fallback.
        global_pair = tau_data.get("global") or {}
        self._tau_global_l2 = float(global_pair.get("tau_l2", 0.5))
        self._tau_l2_per_source: dict[str, float] = {}
        for src, blob in (tau_data.get("per_source") or {}).items():
            self._tau_l2_per_source[src] = float(
                blob.get("tau_l2", self._tau_global_l2)
            )

        # Head model.
        ckpt = torch.load(
            head_ckpt_path, map_location="cpu", weights_only=False,
        )
        cargs = ckpt["args"]
        self._domain_vocab = ckpt.get("domain_vocab", ["chat"])
        n_domains = len(self._domain_vocab)
        hidden_dim = int(cargs["hidden_dim"])
        pos_dim = int(cargs["pos_dim"])
        in_dim = hidden_dim + pos_dim
        model, fwd_args = build_model_from_ckpt(ckpt, in_dim, n_domains)
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(self._device)
        self._model = model
        self._fwd_args = fwd_args
        self._forward_logits = forward_logits
        self._hidden_dim = hidden_dim
        self._pos_dim = pos_dim
        self._n_calls = 0

    async def ping(self) -> dict:
        return {
            "status": "ok",
            "hidden_dim": self._hidden_dim,
            "n_domains": len(self._domain_vocab),
            "tau_l2_global": self._tau_global_l2,
            "tau_l2_per_source": dict(self._tau_l2_per_source),
            "temperature": self._T,
        }

    async def decide(
        self,
        hidden_state: torch.Tensor,
        source: str | None = None,
    ) -> dict:
        """Apply head + τ_L2 to one last-position hidden state.

        Args:
            hidden_state: fp16 tensor of shape [hidden_dim] (CPU).
            source: per-record source label (e.g., "y3_mmbench").
                If None or not in the τ table, falls back to
                τ_global_l2.

        Returns:
            {"verdict": "SHIP"|"REGEN", "score": float, "src": int, "tau_l2": float}
        """
        self._n_calls += 1
        if hidden_state.dim() == 1:
            x = hidden_state.unsqueeze(0).float()
        else:
            x = hidden_state.float()
        # Append pos_fraction=1.0 column (cut-1).
        pos = torch.full(
            (x.shape[0], self._pos_dim), 1.0, dtype=x.dtype, device=x.device,
        )
        x = torch.cat([x, pos], dim=-1).to(self._device)
        with torch.inference_mode():
            logits, aux = self._forward_logits(self._model, self._fwd_args, x)
            score = torch.sigmoid(logits.float() / self._T).item()
            src_idx = 0
            if "domain_logits" in aux:
                src_idx = int(aux["domain_logits"][0].argmax().item())
            elif "gate" in aux:
                src_idx = int(aux["gate"][0].argmax().item())
        src_name = (
            self._domain_vocab[src_idx]
            if 0 <= src_idx < len(self._domain_vocab)
            else "global"
        )
        # Per-source τ takes precedence over global. The source argument
        # supplied by the caller (e.g. record's `source` field) wins over
        # the head's predicted src — caller's label is ground truth in our
        # benches, head's domain logits are an approximation we don't need
        # at cut-1 since the request's source is known.
        tau_lookup_key = source if source in self._tau_l2_per_source else src_name
        tau_l2 = self._tau_l2_per_source.get(
            tau_lookup_key, self._tau_global_l2,
        )
        verdict = "SHIP" if score >= tau_l2 else "REGEN"
        return {
            "verdict": verdict,
            "score": float(score),
            "src": src_idx,
            "src_name": src_name,
            "tau_l2": tau_l2,
        }

    async def stats(self) -> dict:
        return {"n_calls": self._n_calls}
