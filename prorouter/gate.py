"""Scheduler-side ship/escalate GATES for the baselines + ablations.

The cascade's ship-vs-escalate decision normally comes from the trained
hidden-state head fired inside the draft's vLLM fork (it emits
`head_decision ∈ {SHIP, REGEN}` on the final RequestOutput). These gates
replace that head with a *different signal*, computed entirely in the
scheduler — so the cascade structure (draft answers, then ship-or-escalate)
is byte-identical and only the DECISION SIGNAL varies. This seam hosts both
the faithful real-method baselines and the signal ablations:

  - hidden-state head        (the method; fork-side, NOT here)
  - answer_scorer_gate       (FrugalGPT baseline — trained scorer on
                              (query, answer); the real "judge the output"
                              cascade)
  - output_confidence_gate   (Gatekeeper-rule baseline via max_softmax /
                              neg_entropy stats; AND the raw-logprob signal
                              ablation A)
  - query_gate               (signal ablation B — prompt-only LR. The real
                              query-router baseline is RouteLLM, run in its
                              native routing topology by a separate harness,
                              NOT here.)

A gate is `gate(pending, item) -> "SHIP" | "REGEN"`:
  * `pending` is the scheduler's per-request dict (prompt, source, …).
  * `item` is the draft actor's finished payload (text, n_output_tokens,
    and — when the draft ran with logprobs — `mean_logprob`/`min_logprob`/
    `mean_max_prob`/`neg_mean_entropy`).

SHIP means "the draft answer is good enough, return it"; REGEN means
"escalate to the target." Every confidence/score stat is oriented so that
HIGHER = more confident / more likely correct → SHIP iff stat >= tau.

All gates are pure CPU and sub-millisecond — they run inline in
`V0Scheduler._on_draft_finished` with no extra RPC.
"""
from __future__ import annotations

import inspect

from typing import Callable

GateFn = Callable[[dict, dict], str]

# Confidence stats the output_confidence_gate can threshold. All are
# oriented "higher = more confident" so the gate rule is a single
# `stat >= tau`. The draft actor attaches whichever it can compute from
# the logprobs it was asked for (see DraftEngineAsync._drive):
#   mean_logprob / min_logprob   — need logprobs>=1 (chosen-token)
#   mean_max_prob / neg_mean_entropy — need logprobs>=2 (top-k dist)
CONFIDENCE_STATS = (
    "mean_logprob",      # mean chosen-token log-prob (fluency/confidence)
    "min_logprob",       # worst single-token log-prob (shakiest step)
    "mean_max_prob",     # Gatekeeper max-softmax: mean of exp(top1 logprob)
    "neg_mean_entropy",  # Gatekeeper neg predictive entropy (top-k approx)
)


def output_confidence_gate(
    tau: float,
    stat: str = "mean_logprob",
) -> GateFn:
    """Threshold the DRAFT's own output confidence. Serves two roles:

      * the **Gatekeeper-rule baseline** when `stat` is "mean_max_prob"
        (max-softmax) or "neg_mean_entropy" (negative predictive entropy) —
        Gatekeeper's *inference-time deferral rule*. NOTE: we reproduce only
        the rule, on the UN-FINE-TUNED 7B; the paper's calibration
        fine-tuning step is intentionally omitted (no public code; full
        fine-tuning would retrain the base model). State this in the writeup.
      * the **raw-logprob signal ablation A** when `stat` is "mean_logprob"
        or "min_logprob" — a thin attribution control, not a paper baseline.

    All stats are oriented higher = more confident, so the rule is
    `SHIP iff stat >= tau`. A missing stat (draft ran without the needed
    logprobs) is treated as REGEN — the conservative default that escalates
    rather than shipping blind.

    The τ sweep traces the confidence ship-rate / quality curve, compared
    against the hidden-state head at matched operating point.
    """
    if stat not in CONFIDENCE_STATS:
        raise ValueError(
            f"output_confidence_gate stat must be one of {CONFIDENCE_STATS}, "
            f"got {stat!r}"
        )

    def gate(pending: dict, item: dict) -> str:
        conf = item.get(stat)
        if conf is None:
            return "REGEN"
        return "SHIP" if float(conf) >= tau else "REGEN"

    return gate


def answer_scorer_gate(
    scorer: Callable[[dict, dict], float],
    tau: float,
) -> GateFn:
    """FrugalGPT baseline — a TRAINED scorer reads (query, answer) and
    predicts the draft answer's reliability; SHIP iff score >= tau.

    This is the faithful real-method "judge the output" cascade: FrugalGPT's
    DistilBERT regression scorer takes the question + the cheap model's
    generated answer and outputs a correctness/reliability score, escalating
    when it falls below a learned threshold. Here `scorer(pending, item) ->
    float` gets the prompt via `pending["prompt"]` and the draft's answer via
    `item["text"]` — it reads the actual generated answer (unlike the
    query-only router, which never sees the draft output), which is what
    makes it a strong, honest output-cascade baseline rather than a strawman.

    The trained scorer is supplied via the bench's `--scorer-callable`
    hook (e.g. a fine-tuned DistilBERT wrapped as `scorer(pending, item)`),
    fitted on (question, draft-answer) -> correct? labels. No baseline
    implementation ships with this repository; supply your own factory.

    The τ sweep traces FrugalGPT's cost-quality frontier on our cascade.
    """

    def gate(pending: dict, item: dict) -> str:
        try:
            score = float(scorer(pending, item))
        except Exception:
            return "REGEN"
        return "SHIP" if score >= tau else "REGEN"

    return gate


def query_gate(
    scorer: Callable[[dict], float],
    tau: float,
) -> GateFn:
    """Ablation B — threshold a QUERY-ONLY router score on the prompt.

    `scorer(pending) -> float` is any prompt-only predictor; the intended
    one is the source-free TF-IDF+LR classifier
    (`prorouter.pre_router.model_scorer`), reused here as the *gate* rather than
    just the buffer-ordering ranker. The score predicts P(SHIP) from the
    prompt text alone — it never reads the draft's output or hidden state.

    SHIP iff score >= tau. The τ sweep traces the *query-signal* ship-rate /
    quality curve, compared against the hidden-state head at matched
    operating point. This is the signal class of RouteLLM / Hybrid LLM /
    ECVL-ROUTER.

    NOTE: the score depends only on `pending` (the prompt), so it could be
    computed at submit time; we evaluate it post-draft so the cascade
    structure and the draft's wall-clock are identical to the other arms —
    the only thing that differs across A/B/head is the SIGNAL, not the
    pipeline.
    """

    def gate(pending: dict, item: dict) -> str:
        return "SHIP" if float(scorer(pending)) >= tau else "REGEN"

    return gate


def transformer_seq_gate(
    ckpt_path: str,
    tau_table: dict,
    use_global_tau: bool = False,
) -> GateFn:
    """T P18 — chosen production decider. Loads a trained
    TransformerSeq (or any SeqDataset-compatible model from
    sys22t_p16b_train_seq_models.py) and runs it on the draft's
    per-token feature sequence to make the ship/escalate decision.

    Input contract: the draft attaches
      item["per_token_features"] = list[list[float]]
                                  # rows = [chosen_lp, max_p,
                                  #         neg_entropy, pos_frac]
    when SamplingParams.logprobs >= 2 (see DraftEngineAsync._drive).

    Decision: per-source τ from the table. tau_table format matches
    verifier/build_tau_table.py output:
      {"per_source": {<source>: {"best": {"tau": float}}},
       "global":      {"tau": float}}

    Missing per-source entry → falls back to "global" τ.
    `use_global_tau=True` forces global for all sources (debugging / A/B).
    Missing per_token_features → REGEN (conservative).
    """
    import torch
    from prorouter import head_arch as _p16b
    from prorouter.head_model import TransformerSeq

    # TransformerSeq comes from head_model, not head_arch: the two are the same
    # network (identical state_dict, 67,329 params) but only head_model's takes
    # max_len, and the positional-encoding buffer is *in* the state_dict, so a
    # checkpoint trained at one max_len will not load into a class fixed at
    # another. The remaining three are ablation architectures.
    _ARCH_MAP = {
        "TransformerSeq": TransformerSeq,
        "BiLSTMSeq": _p16b.BiLSTMSeq,
        "CNN1D": _p16b.CNN1D,
        "AttnPool": _p16b.AttnPool,
    }

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    hp = ckpt["hparams"]
    arch = hp["arch"]
    if arch not in _ARCH_MAP:
        raise ValueError(
            f"{ckpt_path}: unknown head arch {arch!r}; "
            f"known: {sorted(_ARCH_MAP)}")
    cls = _ARCH_MAP[arch]
    accepted = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in hp.items() if k != "arch" and k in accepted}
    state = ckpt["state_dict"]
    # Older checkpoints predate the max_len hparam; recover it from the saved
    # positional-encoding buffer rather than falling back to a default that
    # would mismatch its shape.
    if "max_len" in accepted and "max_len" not in kwargs and "pe.pe" in state:
        kwargs["max_len"] = int(state["pe.pe"].shape[0])
    model = cls(**kwargs)
    model.load_state_dict(state)
    model.eval()
    # warmup so first inference doesn't pay PyTorch dispatch JIT cost
    with torch.inference_mode():
        _ = model(torch.zeros(1, 8, 4), torch.tensor([8]))

    per_source = tau_table.get("per_source", {})
    global_tau = (tau_table.get("global", {}) or {}).get("tau", 0.5)

    def gate(pending: dict, item: dict) -> str:
        feats = item.get("per_token_features")
        if not feats:
            return "REGEN"
        src = pending.get("source", "global")
        if use_global_tau:
            tau = global_tau
        else:
            entry = per_source.get(src)
            if entry is None:
                tau = global_tau
            else:
                tau = entry.get("best", {}).get("tau", global_tau)
        x = torch.tensor([feats], dtype=torch.float32)  # [1, T, 4]
        lens = torch.tensor([x.shape[1]])
        with torch.inference_mode():
            logit = model(x, lens)
        score = float(torch.sigmoid(logit).item())
        return "SHIP" if score >= float(tau) else "REGEN"

    return gate
