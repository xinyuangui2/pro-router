"""Inline draft-engine self-evaluation baselines.

Two published post-hoc confidence baselines, run INLINE on the draft
engine after a draft generation completes (blocking the request's
completion on them). This is the faithful "method overhead on the
serving path" variant of the offline verifier pass — the ship/
escalate decision needs the self-eval, so it must run before the
request finishes.

  * P(True)  (Kadavath et al. 2022, arXiv 2207.05221): zero-shot. Ask the
    model whether its proposed answer is True/False; score = softmax prob
    of the True token at the decision position. k=1, temperature 0,
    logprobs=20.

  * AutoMix  (Madaan et al. NeurIPS 2023, arXiv 2310.12963): few-shot
    self-verification. Sample k=8 verifications @ T=1.0; confidence =
    fraction that judge the answer "Correct".

Templates are VERBATIM from the published methods — they ARE the method,
do not paraphrase. The score-extraction helpers here mirror the offline
verifier pass exactly, so an inline score matches the offline score for
the same (question, answer, images)
to within sampler/graph nondeterminism (≈0 for P(True), which is greedy).
"""
from __future__ import annotations

import math
import re

# --- P(True) — Kadavath et al. 2022 §"Asking the model if its answer is True".
PTRUE_TEMPLATE = (
    "Question: {question}\n"
    "Proposed Answer: {answer}\n"
    "Is the proposed answer correct?\n"
    "(A) True\n"
    "(B) False\n"
    "The proposed answer is:"
)

# --- AutoMix few-shot self-verification — Madaan et al. 2023, Appendix D.1.
AUTOMIX_FEWSHOT = (
    "Context: The celestial event, known as the Pink Moon, is unique to the "
    "month of April and is named for the wild ground phlox, a flower that "
    "blooms in early spring.\n"
    "Question: In which month does the celestial event, the Pink Moon, occur?\n"
    "AI Generated Answer: July\n"
    "Instruction: Your task is to evaluate if the AI Generated Answer is "
    "correct, based on the provided context and question. Provide the "
    "judgement and reasoning for each case. Choose between Correct or "
    "Incorrect.\n"
    "Evaluation: The context clearly states that the Pink Moon is unique to "
    "the month of April. The AI Generated Answer of July is therefore "
    "Incorrect.\n\n"
)
AUTOMIX_TEMPLATE = (
    AUTOMIX_FEWSHOT
    + "Context: {context}\n"
    + "Question: {question}\n"
    + "AI Generated Answer: {answer}\n"
    + "Instruction: Your task is to evaluate if the AI Generated Answer is "
    "correct, based on the provided context and question. Provide the "
    "judgement and reasoning for each case. Choose between Correct or "
    "Incorrect.\n"
    + "Evaluation:"
)

# P(True) sampler: k=1, greedy, top-20 logprobs at the decision position.
PTRUE_MAX_TOKENS = 4
PTRUE_LOGPROBS = 20
# AutoMix sampler: k=8 samples @ T=1.0, short.
AUTOMIX_K = 8
AUTOMIX_MAX_TOKENS = 24
AUTOMIX_TEMPERATURE = 1.0

_AUTOMIX_RX = re.compile(r"\b(in)?correct\b")


def ptrue_text(question: str, answer: str) -> str:
    return PTRUE_TEMPLATE.format(question=question, answer=answer)


def automix_text(question: str, answer: str) -> str:
    # uses the question text for BOTH the context and question slots
    # of the multimodal adaptation (the image carries the real context).
    return AUTOMIX_TEMPLATE.format(context=question, question=question,
                                   answer=answer)


def prob_true(logprobs_at_pos) -> float | None:
    """Softmax prob mass on 'True'/'A'/'yes' vs 'False'/'B'/'no' at the
    decision position, from a top-logprobs dict {token_id: Logprob}.
    Verbatim from sys44_verifier_pass._prob_true."""
    if not logprobs_at_pos:
        return None
    true_lp, false_lp = [], []
    for lp in logprobs_at_pos.values():
        tok = (lp.decoded_token or "").strip().lower()
        if tok in ("true", "a", "yes", "(a", "(a)"):
            true_lp.append(lp.logprob)
        elif tok in ("false", "b", "no", "(b", "(b)"):
            false_lp.append(lp.logprob)
    if not true_lp and not false_lp:
        return None
    pt = sum(math.exp(x) for x in true_lp)
    pf = sum(math.exp(x) for x in false_lp)
    return float(pt / (pt + pf)) if (pt + pf) > 0 else None


def score_ptrue(completion) -> float:
    """P(True) score from a single CompletionOutput (max_tokens>=1,
    logprobs=20). Scans generated positions for the first verdict-token
    distribution; default 0.5 when none is found.."""
    if getattr(completion, "logprobs", None):
        for pos in completion.logprobs:
            s = prob_true(pos)
            if s is not None:
                return s
    return 0.5


def _automix_verdict(text: str):
    """Last decisive verdict word: 'incorrect' -> 0, 'correct' -> 1.
    Word-boundary match so 'correct' inside 'incorrect' doesn't false-fire.
    Verbatim from sys44_verifier_pass._verdict."""
    last = None
    for m in _AUTOMIX_RX.finditer(text.lower()):
        last = 0.0 if m.group(1) else 1.0
    return last


def score_automix(completion_outputs) -> float:
    """AutoMix confidence = fraction of the k=8 samples judged 'Correct'.
    Takes the list of CompletionOutput from one RequestOutput (n=8).
    Default 0.5 when no sample emits a decisive verdict.."""
    verdicts = [v for c in completion_outputs
                if (v := _automix_verdict(c.text)) is not None]
    return float(sum(verdicts) / len(verdicts)) if verdicts else 0.5
