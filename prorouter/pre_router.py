"""Scorer for V0Scheduler's sorted-buffer routing.

The scorer maps a `pending` dict to a float: higher = more likely
to SHIP if cascaded. submit() inserts (score, seq, req_id) into the
shared sorted buffer; the draft dispatcher pulls from the high end
and the target dispatcher pulls from the low end. DIRECT vs cascade
split is emergent — no quantile cutoff, no `direct_fraction` to tune.

Per-source rate tables are the typical scorer (see
`per_source_scorer`); other implementations are free to use a
token-length heuristic, a regression model, or any function that
yields a useful ranking.
"""
from __future__ import annotations

from typing import Callable


# Per-source head ship rates anchored on measurements taken against
# the head + τ table on `bench_vlm_test_*`.
# These are population means observed AFTER head firing on cascade-
# routed requests — exactly the per-source `s` we want to use as a
# predicted SHIP probability at submit time.
DEFAULT_C18_SHIP_RATES: dict[str, float] = {
    "y3_docvqa":    1.00,
    "y3_mmbench":   0.88,
    "y3_mathvista": 0.68,
    "y3_chartqa":   0.37,
    "y3_mmmu":      0.27,
}


# Per-source SHIP rates for the MileBench head, anchored on the
# `tau_milebench_p_0.98_engine_fp16.json` per_source.skip_rate column.
# Spans the full [0, 1] range — 10 sources at 0 (always REGEN),
# 5 sources at 1.0 (always SHIP), 14 sources in the middle. Use with
# the head + tau bundle (`head_milebench_multitask_wide.pt`).
DEFAULT_MILEBENCH_SHIP_RATES: dict[str, float] = {
    "mile_ALFRED":                   0.0000,
    "mile_ActionLocalization":       0.0000,
    "mile_ActionPrediction":         1.0000,
    "mile_ActionSequence":           0.0000,
    "mile_CLEVR_Change":             0.0000,
    "mile_CharacterOrder":           0.0000,
    "mile_CounterfactualInference":  0.4815,
    "mile_DocVQA":                   0.6667,
    "mile_EgocentricNavigation":     0.0000,
    "mile_GPR1200":                  0.0000,
    "mile_IEdit":                    0.0000,
    "mile_ImageNeedleInAHaystack":   1.0000,
    "mile_MMCoQA":                   0.0714,
    "mile_MovingAttribute":          0.1481,
    "mile_MovingDirection":          0.2692,
    "mile_MultiModalQA":             0.7857,
    "mile_OCR_VQA":                  0.4167,
    "mile_ObjectExistence":          0.3462,
    "mile_ObjectInteraction":        1.0000,
    "mile_ObjectShuffle":            0.0000,
    "mile_SceneTransition":          0.9286,
    "mile_SlideVQA":                 0.1154,
    "mile_Spot_the_Diff":            0.0000,
    "mile_StateChange":              0.6458,
    "mile_TQA":                      0.8000,
    "mile_TextNeedleInAHaystack":    0.5500,
    "mile_WebQA":                    0.6000,
    "mile_WikiVQA":                  0.8182,
    "mile_nuscenes":                 1.0000,
}


# Registry: `--pre-router-scorer` CLI choice → builtin per-source table.
# Lets the bench select a scorer without exposing every default as a
# CLI flag and lets external scripts share the same lookup.
DEFAULT_C18_AOKVQA_SHIP_RATES: dict[str, float] = {
    # Part-C retest: MEASURED per-source ship_rate_per_cascade on this
    # fork (c18 from the S-probe; aokvqa from the Part-B cascade). Heterogeneous
    # moderate-s pool so the prescorer's buffer-ranking has real headroom.
    "y3_docvqa":    0.99,
    "y3_mmbench":   0.94,
    "y3_mmmu":      0.92,
    "y3_mathvista": 0.89,
    "y3_chartqa":   0.76,
    "aokvqa":       0.41,
}


BUILTIN_SOURCE_RATE_TABLES: dict[str, dict[str, float]] = {
    "c18_aokvqa": DEFAULT_C18_AOKVQA_SHIP_RATES,
    "c18":       DEFAULT_C18_SHIP_RATES,
    # Back-compat with — the CLI flag was originally just
    # `per_source` against the table.
    "per_source": DEFAULT_C18_SHIP_RATES,
    "milebench": DEFAULT_MILEBENCH_SHIP_RATES,
}


def per_source_scorer(
    per_source_ship_rate: dict[str, float],
    default: float = 0.5,
) -> Callable[[dict], float]:
    """Build a scorer(pending) -> float that looks up `pending["source"]`
    in the per-source ship-rate table. Unknown sources fall back to
    `default` (caller-controlled).
    """
    rates = dict(per_source_ship_rate)

    def scorer(pending: dict) -> float:
        return rates.get(pending.get("source"), default)

    return scorer


def model_scorer(model_path: str) -> Callable[[dict], float]:
    """Load a pickled `{vec, clf}` and build a source-free
    scorer(pending) -> float that featurizes `pending["prompt"]` +
    image count + prompt length, then returns clf.predict_proba()[:,1].

    Pickle schema:
      {"vec": TfidfVectorizer, "clf": LogisticRegression}

    Pure CPU; <1 ms/request budget. Does NOT read pending["source"] —
    that's the entire point of the source-free scorer.
    """
    import pickle
    import numpy as np
    from scipy.sparse import csr_matrix, hstack

    with open(model_path, "rb") as f:
        blob = pickle.load(f)
    vec = blob["vec"]
    clf = blob["clf"]
    # Honor persisted train-time meta normalization. Without it a
    # z-scored-trained LR sees raw n_chars/n_tokens (100-1000x the train
    # scale), the meta logit saturates the sigmoid (−20..−45), and every
    # realistic prompt scores ~0.0 → the sorted buffer degenerates to FIFO.
    # Schemas: sys65b/nomv persists meta_mu/meta_sd; a meta_norm dict is the
    # sys18 slot (None there = stats lost; such pickles stay raw and are NOT
    # fixed by this — check score spread before trusting one).
    _norm = blob.get("meta_norm")
    if _norm is not None:
        _meta_mu = np.asarray(_norm["mu"], dtype=np.float32)
        _meta_sd = np.asarray(_norm["sd"], dtype=np.float32)
    elif blob.get("meta_mu") is not None:
        _meta_mu = np.asarray(blob["meta_mu"], dtype=np.float32)
        _meta_sd = np.asarray(blob["meta_sd"], dtype=np.float32)
    else:
        _meta_mu = _meta_sd = None

    def scorer(pending: dict) -> float:
        prompt = pending.get("prompt") or ""
        n_imgs = len(pending.get("image_paths") or [])
        n_chars = len(prompt)
        n_tokens = len(prompt.split())
        X_text = vec.transform([prompt])
        meta = np.array([[n_imgs, n_chars, n_tokens]], dtype=np.float32)
        if _meta_mu is not None:
            meta = (meta - _meta_mu) / np.maximum(_meta_sd, 1e-6)
        X = hstack([X_text, csr_matrix(meta)]).tocsr()
        return float(clf.predict_proba(X)[0, 1])

    return scorer


