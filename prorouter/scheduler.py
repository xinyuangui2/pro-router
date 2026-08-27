"""V0 head scheduler — pump loops on top of async actors.

Pipeline:

  client.submit(prompt, max_tokens)
        ↓
  [pick_draft] → drafts[i].submit(req_id, prompt, max_tokens)
                                       ↓ (vLLM continuous batching +
                                          cascade head fires inline)
                                  drafts[i].pop_finished
                                       ↓
   _draft_pump: dispatch based on head_decision
        head_decision == SHIP   → respond ACCEPT with draft text
                                  (no target call)
        head_decision == REGEN  → submit to target via submit_regen
                                       ↓
              [pick_target] → targets[j].submit_regen(req_id, prompt,
                                              max_tokens)
                                       ↓
                              targets[j].pop_finished
                                       ↓
   _target_pump: pop_finished → set client event → return Response

REGEN and DIRECT_TARGET (spillover) share a single router
— `_dispatch_direct_target`. Target's submit_regen takes one signature
for both, so KV-admission cost is uniform and target cannot
distinguish them at the queue level (removes the Gap-2 KV-
estimate mismatch that previously penalized cascade vs target_only
at saturation).

Picker policy:

  - Default (round-robin): preserves the measured fairness
    behavior for homogeneous workloads.
  - Load-aware (default ON since): `_pick_draft` returns
    argmin(_draft_load) with round-robin tiebreak. `_draft_load[i]`
    is incremented when scheduler submits to draft i and decremented
    when the corresponding draft-finished item is processed. Picker
    is recalibrated periodically against drafts[i].qsize().in_flight
    to correct local-counter drift.
  - Direct-target spillover (opt-in): when target has
    headroom past the expected REGEN inflow AND drafts are saturated,
    route incoming requests directly to target.submit_regen (no draft
    step). Saves draft latency on requests that would have REGEN'd
    anyway, and fills target idle capacity without burning compute
    on speculative dual-dispatch. Verdict = "DIRECT_TARGET".

Verdict semantics from the actor pipeline:
  ACCEPT         → cascade SHIP (draft ran, head said SHIP, no target call)
  REGEN          → target ran submit_regen after cascade head said REGEN
  DIRECT_TARGET  → request bypassed draft entirely (spillover) —
                   also calls submit_regen; identical to REGEN at target
  ERROR          → either draft or target reported an error (stage in payload)
"""
from __future__ import annotations

import asyncio
import bisect
import collections
import math
import random
import time
import uuid
from dataclasses import dataclass
from typing import Callable

# Positional-arg names of DraftEngineAsync._submit_impl, in order.
# Used to turn the dispatch loop's _submit_args tuple into the kwargs dict
# submit_batch carries. Must track the engine signature.
_DRAFT_SUBMIT_POS = (
    "req_id", "prompt", "max_tokens", "temperature",
    "ignore_eos", "image_path", "head_cascade", "image_paths",
)


@dataclass
class Response:
    request_id: str
    text: str
    verdict: str  # "ACCEPT" | "REGEN" | "DIRECT_TARGET" | "ERROR"
    arrival_t: float
    draft_completed_t: float
    completed_t: float
    error: str | None = None
    n_output_tokens: int | None = None
    routing_path: str = "cascade"  # "cascade" | "direct_target"
    # engine-side instrumentation — actor's perf_counter at
    # the moment _drive_decode/_drive_regen started and at the moment
    # the request finished vLLM. These are on the actor's process
    # clock, not the bench's, so they're only meaningful for
    # rate-on-actor-time analysis (sort, differentiate). None for
    # ACCEPT (never visited target).
    target_admit_actor_t: float | None = None
    target_finish_actor_t: float | None = None
    # Draft actor's perf_counter at _drive start and at vLLM
    # out.finished. Different process clock from target's so cross-
    # actor comparisons aren't valid; rates ON the draft actor clock
    # are valid (sort, differentiate).
    draft_admit_actor_t: float | None = None
    draft_finish_actor_t: float | None = None
    # Raw in-engine head score + tau for the request (probe diagnostics).
    head_score: float | None = None
    head_tau: float | None = None
    # Inline self-eval score (P(True)/AutoMix) + method, when
    # --inline-self-eval is on. None otherwise.
    self_eval_score: float | None = None
    self_eval_method: str | None = None
    # Wall time of the inline self-eval pass alone (ms), when exposed.
    self_eval_ms: float | None = None

    @property
    def end_to_end_ms(self) -> float:
        return (self.completed_t - self.arrival_t) * 1000.0

    @property
    def draft_ms(self) -> float:
        return (self.draft_completed_t - self.arrival_t) * 1000.0

    @property
    def target_ms(self) -> float:
        return (self.completed_t - self.draft_completed_t) * 1000.0


def pick_round_robin(handles: list, counter: int) -> int:
    if not handles:
        raise ValueError("no handles")
    return counter % len(handles)


class _TokenBucket:
    """Async rate limiter (token-bucket).

    `acquire(n)` waits until `n` tokens are available, then consumes
    them. Bucket refills at `rate` tokens/sec; max burst capped at
    one second of tokens (so a long-idle bucket can't release a
    massive burst that defeats the limit).

    `rate is None` → no-op acquire (returns immediately). This lets
    the same code path serve both rate-limited and unlimited
    configurations.
    """

    __slots__ = ("_rate", "_tokens", "_last_refill", "_lock", "_burst_s")

    def __init__(self, rate: float | None, burst_s: float = 1.0) -> None:
        self._rate = float(rate) if rate is not None and rate > 0 else None
        # rate-match: max burst = rate * burst_s tokens (default 1 s).
        # The credit scheme grants (tick + buffer) s of tokens per retune, so
        # the engine never starves between retunes — set burst_s = tick+buffer.
        self._burst_s = max(1e-3, float(burst_s))
        self._tokens = (self._rate * self._burst_s) if self._rate is not None else 0.0
        self._last_refill = time.perf_counter()
        self._lock = asyncio.Lock()

    def set_rate(self, rate: float | None) -> None:
        """Retune the limit at runtime. `acquire` reads `_rate` on
        every loop, so the new rate takes effect on the next acquire. The
        float write is atomic in CPython (no lock needed); tokens are
        clamped down to the new burst cap. Cannot enable a bucket that was
        constructed with `rate=None` — see `V0Scheduler.set_engine_send_rps`.
        """
        self._rate = float(rate) if rate is not None and rate > 0 else None
        if self._rate is not None:
            self._tokens = min(self._tokens, self._rate * self._burst_s)

    async def acquire(self, n: int = 1) -> None:
        if self._rate is None:
            return
        async with self._lock:
            while True:
                now = time.perf_counter()
                self._tokens = min(
                    self._rate * self._burst_s,
                    self._tokens + (now - self._last_refill) * self._rate,
                )
                self._last_refill = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                deficit = n - self._tokens
                # Release the lock while sleeping so other acquirers
                # can update the refill timestamp meaningfully — but
                # we want strict FIFO over the bucket. Easiest: hold
                # the lock and sleep. Acquirers serialize naturally.
                await asyncio.sleep(deficit / self._rate)


class V0Scheduler:
    """Head scheduler — routes requests across draft and target async actors.

    Public API:
      await scheduler.start()                      # spawn pump tasks
      response = await scheduler.submit(prompt)    # submit + await result
      await scheduler.stop()                       # graceful shutdown

    Load-aware routing:
      - `_draft_load[i]` / `_target_load[i]`: scheduler-local in-flight
        counter, ++ on dispatch and -- on finish. Maintained without
        per-pick RPC.
      - Periodic calibration (every `calibrate_every` submits) issues
        `actor.qsize()` for each draft/target and reconciles drift —
        catches any decrement leaks and stays robust to bugs.

    Routing model (sorted-buffer pull-from-both-ends):

      submit() scores the request (if a scorer is configured),
      inserts (score, seq, req_id) into a single shared sorted
      buffer, and waits on the client event. The routing decision
      itself is emergent — no `direct_ratio`, no `direct_fraction`,
      no d_opt knob.

      Two dispatcher tasks consume from opposite ends of the buffer:
        - `_draft_dispatcher` pulls the highest-scoring item
          (predicted-most-likely-to-SHIP) → enqueues to the
          min-load draft actor's send queue. The draft then runs
          its cascade head; SHIPs return to client, REGENs spill
          to the target via `_target_send_qs[i]` (NOT back into
          the sorted buffer).
        - `_target_dispatcher` pulls the lowest-scoring item
          (predicted-most-likely-to-REGEN-if-cascaded) → enqueues
          to the min-load target actor's send queue. The request
          bypasses draft entirely.

      Steady-state at saturation: draft eats top-of-buffer at rate
      D_solo, target eats bottom-of-buffer at rate T_solo minus the
      REGEN inflow from draft. The buffer's middle is the implicit
      "cutoff" — and it's exactly the d_opt fraction the closed-form
      formula picks, with the scorer's ranking providing the
      pre-router lift (s_top > population_s) automatically. No
      pre-measurement or knob tuning needed.

      Per-request overrides bypass the buffer entirely:
        - force_cascade=True   → straight to draft_send_q
        - force_direct=True    → straight to target_send_q
        - force_direct_target  → scheduler-wide variant of force_direct
                                  (every request bypasses draft)

      Scorer contract: a callable mapping a `pending` dict to a
      float. Higher score = more likely to SHIP if cascaded
      (predict-ACCEPT proxy). Static per-source tables in
      `prorouter.pre_router` are the typical scorer. When scorer is None,
      every request gets the same score (FIFO within ties) — the
      buffer still works but the draft/target split is whatever
      falls out of the consumption rates.
    """

    def __init__(
        self,
        drafts: list,
        targets: list,
        *,
        # ----- draft encoder pool (post BUILD) -----
        # When non-None, the scheduler routes the draft path through this
        # encoder pool: for each request going to draft, first call
        # `encoder.encode_return.remote(req_id, image_paths)` (Ray RPC)
        # → returns CPU image_embeds + image_grid_thw → then dispatch
        # `draft.submit(image_embeds=..., image_grid_thw=...)`. The draft
        # skips its own ViT. measured +28-31 % steady r/s at
        # λ=5,8 with 1× A10G encoder + g5.12 draft, Ray RPC transport.
        # Target side does NOT use encoder (showed −26 %); direct-
        # target and REGEN paths use raw image_paths as before.
        encoders: list | None = None,
        pop_max_n: int = 0,   # 0 = drain all finished per poll (see engine.pop_finished)
        # Polling cadence for actor.pop_finished — only matters when
        # the actor's finished_q is EMPTY (the call returns
        # immediately with up to pop_max_n items otherwise). Higher
        # values reduce idle polling RPCs linearly with no effect at
        # saturation; the cost is a slight rise in tail latency under
        # light load. Split per side: tail-latency-sensitive
        # configurations can tune draft and target independently.
        draft_pop_timeout_s: float = 0.1,
        target_pop_timeout_s: float = 0.1,
        force_accept_rate: float | None = None,
        # ----- load-aware draft picker -----
        load_aware: bool = True,
        calibrate_every: int = 64,
        # ----- scorer for sorted-buffer routing -----
        # Maps a `pending` dict to a float. Higher score = more
        # likely to SHIP if cascaded. submit() inserts (score, seq,
        # req_id) into the shared sorted buffer; dispatchers pull
        # top to draft, bottom to target. When None, every request
        # gets score 0.0 (FIFO within ties) — split is consumption-
        # rate dependent rather than scorer-guided.
        scorer: Callable[[dict], float] | None = None,
        # ----- scheduler-side ship/escalate gate (baseline ablations) -----
        # When set, this callable `gate(pending, item) -> "SHIP"|"REGEN"`
        # makes the ship/escalate decision IN PLACE OF the fork's
        # hidden-state head, for requests where the fork did not emit a
        # head_decision. It's the seam for the output-confidence (A) and
        # query-only (B) signal ablations — the cascade structure is
        # identical to the head cell; only the decision SIGNAL differs.
        # See prorouter/gate.py. None → no gate (head or auto-accept path).
        gate: Callable[[dict, dict], str] | None = None,
        # Number of top logprobs to request on draft generation so the
        # draft actor attaches confidence stats to its finished item.
        # 0/None → off (no plumbing cost on non-gate cells). 1 → chosen-
        # token stats (mean/min logprob). >=2 → also the Gatekeeper-rule
        # stats (max-softmax + neg predictive entropy over the top-k).
        # Required by output_confidence_gate; harmless otherwise.
        draft_logprobs: int = 0,
        # opt into the inline per-token feature seq
        # Path on lp-classifier-inline fork. When True, the draft
        # actor's submit() sets SamplingParams.emit_per_token_feature_seq
        # so the fork builds CompletionOutput.per_token_features inline
        # on the GPU side and the driver can consume it directly,
        # skipping the per-step Logprob dict construction +
        # detokenization. Must be combined with draft_logprobs >= 2.
        draft_emit_per_token_feature_seq: bool = False,
        # when True, every cascade-routed draft request
        # Opts into the in-engine attn_pool head; the engine returns
        # CompletionOutput.head_decision and _drive forwards it as
        # item["head_decision"]. Requires the draft actor to have been
        # booted with in_engine_cascade_head_{ckpt,tau} set.
        draft_in_engine_cascade_head: bool = False,
        # ----- target-only-through-scheduler mode -----
        # When True, every request bypasses the draft tier and routes
        # directly to a target. drafts=[] is allowed in this mode.
        # Used by the target_only baseline cell so it pays the same
        # scheduler + Ray-RPC overhead as cascade cells.
        force_direct_target: bool = False,
        # ----- per-engine send rate limiters -----
        # Token-bucket cap on the dispatch loop's buffer pop, in
        # request-rate units. None = unlimited (default).
        # POSITION MATTERS: the cap is at the dispatch loop BEFORE
        # the buffer pop, so backpressure propagates all the way to
        # the sorted buffer. (Earlier placed it after the
        # buffer in a send-pump layer; the buffer never filled and
        # score ordering never expressed — see fix doc.)
        # Per-actor bucket: with N actors at rate R, side throughput
        # caps at N·R r/s. Use case: match target's input rate
        # between target_only and cascade for fair contribution audit.
        draft_engine_send_rps: float | None = None,
        target_engine_send_rps: float | None = None,
        # target-input verification: when set, the scheduler
        # appends one CSV line per target dispatch (req_id, source,
        # prompt md5, n_imgs, max_tokens, perf_counter timestamp).
        # Used by cells to prove target receives a
        # bit-identical input set across target_only and cascade
        # configurations. None = disabled (no log).
        target_input_log_path: str | None = None,
        # ----- closed-loop DOPT + send-rate controller (opt-in) -----
        # ALL default to current static behavior. `direct_ratio` (also
        # settable at runtime via set_direct_ratio) makes submit() pin the
        # DIRECT-vs-CASCADE split by per-request sampling — like the bench's
        # static cascade_gate split, but mutable. `closed_loop=True` spawns a
        # periodic controller that reads piggybacked engine stats off each
        # finished item (no extra RPC) and re-tunes the split live: feedforward
        # DOPT from the draft's live ship-rate MA + the configured solo
        # ceilings T0/D0, plus a feedback trim from the tier GPU-util
        # imbalance, plus an optional KV-proximity throttle on target dispatch.
        direct_ratio: float | None = None,
        closed_loop: bool = False,
        control_tick_s: float = 1.0,
        control_T0: float | None = None,
        control_D0: float | None = None,
        control_trim_gain: float = 0.15,
        control_max_trim: float = 0.10,
        control_deadband: float = 0.01,
        control_ratio_bounds: tuple[float, float] = (0.0, 1.0),
        control_kv_guard: float | None = 0.92,
        # rate-match credit dispatch (alternative to DOPT). Per-engine
        # push rate is capped at the engine's MEASURED throughput (completions
        # observed for free on the pump path — no extra RPC). Every
        # rate_match_tick_s the bucket rate is retuned to throughput*(1+headroom)
        # with a (tick + buffer) s burst, so the engine's queue stays ~buffer s
        # deep: no overshoot, and the DIRECT/CASCADE split + Λ self-balance as
        # the workload (and s) drift. Default off → no buckets, old behavior.
        rate_match: bool = False,
        rate_match_tick_s: float = 2.0,
        rate_match_buffer_s: float = 1.0,
        rate_match_headroom: float = 0.15,
        # initial per-engine push-rate guess (r/s) before the first
        # Measurement; the headroom then ramps it up to the true capacity over
        # a few ticks. Replaces the old "seed unlimited" cold start.
        rate_match_init_rps: float = 16.0,
        # explicit two-queue dispatch (cleaner than encoding REGEN at
        # −∞ in the single sorted buffer): fresh requests in one deque (draft
        # pulls them = cascade; target pulls leftovers = direct), escalated
        # REGENs in a second deque the target drains FIRST (priority). Enabled
        # for rate-match / emergent; legacy force/scorer routing keeps the
        # sorted buffer.
        two_buffer: bool = False,
        # occupancy-gated admission: a dispatch loop HOLDS (stops
        # Popping for its engine) when that engine is "full" — in-flight ≥
        # hwm·max_inflight OR target KV proximity ≥ kv_hwm — and resumes once
        # it drains below lwm (hysteresis). Direct backpressure: add iff there's
        # literal room, off the KV-preemption cliff. Uses the scheduler-local
        # in-flight counter (fresh, no RPC) + the piggybacked target KV.
        occupancy_gate: bool = False,
        occupancy_max_inflight: int = 256,
        occupancy_hwm: float = 0.90,
        occupancy_lwm: float = 0.70,
        occupancy_kv_hwm: float = 0.90,
        # bound the pre-dispatch sorted buffer. The buffer is a
        # Bisect-sorted list — O(N) per insert (and O(N) for the target's
        # front pop / a −∞ REGEN's front insert). That is "fine for N ≤ a few
        # hundred" (the design invariant: N ≈ in-flight). But under OPEN-LOOP
        # overload (λ ≫ served rate) only *dispatch* is gated, not arrivals, so
        # the buffer grows unbounded (measured 31k deep) and the O(N) ops
        # saturate the single event loop → dispatch/pop starve → throughput
        # COLLAPSES (worst for the heaviest family, whose buffer is deepest —
        # llava). When set, a *fresh* arrival blocks (backpressure) until
        # the buffer drains below the cap; REGEN re-entries (−∞) bypass it (they
        # are already in-system, bounded by in-flight). 0 = unbounded (legacy).
        max_buffer_depth: int = 0,
        # opt-in: throughput-adaptive target batch dispatch. When on, the
        # target loop pops up to `clamp(MA_target_throughput × window, [min,max])`
        # eligible items and fires them in ONE submit_decode_batch RPC, so the
        # batch self-sizes to the actor's measured rate (vs the fixed single-item
        # dispatch). Off by default → byte-identical one-item-per-RPC behavior.
        adaptive_batch: bool = False,
        adaptive_batch_window_s: float = 0.05,
        adaptive_batch_min: int = 1,
        adaptive_batch_max: int = 16,
        # wire-latency fixes (both default OFF → behavior unchanged):
        # rtt_aware sizes the target batch as tput×(RTT_ema+window)+buffer
        # (a batch must cover the pump's round trip to keep the actor busy);
        # draft_submit_pipeline allows N draft submit-ACKs in flight instead
        # of awaiting each inline (the unbatched path's equivalent fix).
        adaptive_batch_rtt_aware: bool = False,
        adaptive_batch_buffer: int = 2,
        draft_submit_pipeline: int = 0,
        # wire-latency fix (default OFF → behavior unchanged):
        # coalesce up to N draft submits into one submit_batch RPC. The
        # pipeline caps dispatch at pipeline/RTT (and deepening it
        # measurably HURTS — 200 ms: pipeline 64 → 47.6 vs 16 → 62.6 r/s);
        # a batch instead makes one round trip carry whatever is eligible
        # in the buffer right now, so dispatch rate stops scaling ∝ 1/RTT.
        # Self-sizing: deep buffer → big batches, idle buffer → batch of 1.
        draft_submit_batch: int = 0,
        # wire-latency fix (the 1s residual; default OFF → unchanged):
        # allow N target dispatch RPCs in flight per target instead of the
        # serial one-at-a-time await. The serial path caps target dispatch at
        # batch/RTT — invisible at ≤200ms but the binding wall at 1s. Mirrors
        # the draft-side submit pipeline; fire + reap in a background task,
        # bounded by a per-target semaphore.
        target_submit_pipeline: int = 0,
        # direct per-request RPC dispatch (ablation arm b).
        # When True, submit() bypasses the sorted buffer AND all batched /
        # backpressure dispatch entirely: on arrival it immediately fires the
        # per-request engine RPC ("the scheduler just keeps calling the model
        # actor's RPC"). Plain/force_cascade → straight to a draft (cascade,
        # head fires); force_direct/force_direct_target and post-head REGENs →
        # straight to a target (one item per RPC). Force-disables
        # adaptive_batch / two_buffer / occupancy_gate / rate_match loudly.
        # Default off → behavior unchanged.
        dispatch_direct_rpc: bool = False,
    ) -> None:
        if not drafts and not force_direct_target:
            raise ValueError("at least one draft actor required "
                             "(or set force_direct_target=True)")
        if not targets:
            raise ValueError("at least one target actor required")
        if force_accept_rate is not None and not 0.0 <= force_accept_rate <= 1.0:
            raise ValueError("force_accept_rate must be in [0, 1]")
        # --dispatch-direct-rpc bypasses the buffer + batched
        # Dispatch. Force-disable everything that assumes the buffer/dispatch-
        # loop path so the ablation is a clean "just fire RPCs" baseline.
        self._dispatch_direct_rpc = bool(dispatch_direct_rpc)
        if self._dispatch_direct_rpc:
            _off = [n for n, v in (("adaptive_batch", adaptive_batch),
                                   ("two_buffer", two_buffer),
                                   ("occupancy_gate", occupancy_gate),
                                   ("rate_match", rate_match)) if v]
            if _off:
                print("[V0Scheduler] --dispatch-direct-rpc: force-disabling "
                      f"{', '.join(_off)} (direct per-request RPC bypasses the "
                      "sorted buffer + batched/backpressure dispatch)",
                      flush=True)
            adaptive_batch = False
            two_buffer = False
            occupancy_gate = False
            rate_match = False
            if encoders:
                raise ValueError(
                    "--dispatch-direct-rpc does not support the encoder pool "
                    "(no cell that uses it needs both)"
                )
        self._drafts = drafts
        self._targets = targets
        self._pop_max_n = pop_max_n
        self._draft_pop_timeout_s = float(draft_pop_timeout_s)
        self._target_pop_timeout_s = float(target_pop_timeout_s)
        # per-actor dispatch loops + rate-limit buckets.
        # One task + one bucket per actor; the loop acquires a
        # token, pops its end of the buffer, and fires the actor RPC.
        # No intermediate send queue — backpressure propagates from
        # the bucket → buffer → submit().
        self._draft_engine_send_rps = draft_engine_send_rps
        self._target_engine_send_rps = target_engine_send_rps
        self._draft_buckets: list[_TokenBucket | None] = []
        self._target_buckets: list[_TokenBucket | None] = []
        self._draft_dispatch_tasks: list[asyncio.Task] = []
        self._target_dispatch_tasks: list[asyncio.Task] = []
        self._target_input_log_path = target_input_log_path
        self._target_input_log_fh = None
        # When set, bypass the target's verify call entirely: each
        # request's verdict is sampled (ACCEPT with prob p, REGEN with
        # prob 1-p). Lets the bench measure the V0 dispatch mechanism
        # at a controlled ACCEPT rate without contamination from
        # judge calibration on synthetic prompts.
        self._force_accept_rate = force_accept_rate

        # Per-request tracking: prompt + arrival_t live here from
        # submit() until target finishes. Wiped on respond.
        self._pending: dict[str, dict] = {}
        self._client_events: dict[str, asyncio.Event] = {}
        self._client_results: dict[str, Response] = {}

        # ----- Load-aware picker state -----
        self._load_aware = bool(load_aware)
        # Per-actor outstanding-request count, maintained by the
        # scheduler. Decremented when the corresponding finished item
        # is observed. Periodic RPC calibration corrects drift.
        self._draft_load: list[int] = [0] * len(drafts)
        self._target_load: list[int] = [0] * len(targets)
        # Round-robin tiebreaker counters — used when multiple actors
        # share the minimum load to spread perfectly fairly (matches
        #'s measured fairness on homogeneous workloads).
        self._draft_rr_tiebreak = 0
        self._target_rr_tiebreak = 0

        # ----- encoder pool state -----
        # `encoders` is None or empty → fallback to draft's own ViT
        # (legacy path); the scheduler doesn't pre-encode. When non-
        # empty, min-load picker selects an encoder per request and
        # awaits its `encode_return.remote()` before dispatching to
        # the draft. Per-actor load counter mirrors the draft/target
        # pattern; calibration is currently in-process only (no
        # `qsize()` RPC) since the encoder has no internal queue.
        self._encoders: list = list(encoders) if encoders else []
        self._encoder_load: list[int] = [0] * len(self._encoders)
        self._encoder_rr_tiebreak = 0

        # Periodic calibration — every N submits, fetch qsize() for
        # all drafts and targets and overwrite the local counters.
        # N=64 keeps overhead well under 1% even at saturation
        # (one set of RPCs every ~64 requests).
        self._calibrate_every = max(1, int(calibrate_every))
        self._submits_since_calibrate = 0
        self._calibration_in_flight = False

        # ----- sorted-buffer routing state -----
        # Shared scored buffer; (score, seq, req_id) tuples ordered
        # by score ascending. Use bisect.insort for O(log N) lookup
        # + O(N) shift (fine for N ≤ a few hundred in-flight). seq
        # is a monotonic tiebreak so equal scores resolve FIFO.
        self._scorer: Callable[[dict], float] | None = scorer
        self._gate: Callable[[dict, dict], str] | None = gate
        self._draft_logprobs = int(draft_logprobs or 0)
        self._draft_emit_ptf_seq = bool(draft_emit_per_token_feature_seq)
        self._draft_in_engine_cascade_head = bool(
            draft_in_engine_cascade_head,
        )
        self._buffer: list[tuple[float, int, str]] = []
        self._buffer_cv = asyncio.Condition()
        self._buffer_seq = 0
        # force-direct-target mode. When True, every request
        # bypasses the sorted buffer and routes straight to a target
        # Via _dispatch_direct_target. Lets the target_only baseline
        # pay the same scheduler + Ray-RPC overhead as cascade cells.
        self._force_direct_target = bool(force_direct_target)

        # Pump task lifecycle.
        self._stop = asyncio.Event()
        self._draft_pump_task: asyncio.Task | None = None
        self._target_pump_task: asyncio.Task | None = None

        # ----- closed-loop controller state (opt-in; see __init__ doc) --
        self._direct_ratio = direct_ratio
        self._closed_loop = bool(closed_loop)
        self._control_tick_s = float(control_tick_s)
        self._control_T0 = control_T0
        self._control_D0 = control_D0
        self._control_trim_gain = float(control_trim_gain)
        self._control_max_trim = float(control_max_trim)
        self._control_deadband = float(control_deadband)
        self._control_ratio_lo, self._control_ratio_hi = control_ratio_bounds
        self._control_kv_guard = control_kv_guard
        self._control_task: asyncio.Task | None = None
        # Latest piggybacked engine-stats snapshots, updated in the pumps
        # from item["stats"] (no extra RPC). One slot per actor.
        self._draft_stats: list[dict] = [{} for _ in drafts]
        self._target_stats: list[dict] = [{} for _ in targets]

        # ----- throughput-adaptive target batch dispatch -----
        self._adaptive_batch = bool(adaptive_batch)
        self._adaptive_batch_window_s = float(adaptive_batch_window_s)
        self._adaptive_batch_min = int(adaptive_batch_min)
        self._adaptive_batch_max = int(adaptive_batch_max)
        # wire-latency fixes.
        self._adaptive_batch_rtt_aware = bool(adaptive_batch_rtt_aware)
        self._adaptive_batch_buffer = int(adaptive_batch_buffer)
        self._target_rpc_rtt_ema: list[float] = [0.0] * len(targets)
        self._draft_submit_pipeline = int(draft_submit_pipeline)
        self._draft_submit_batch = int(draft_submit_batch)
        self._draft_rpc_window = [
            asyncio.Semaphore(max(1, self._draft_submit_pipeline))
            for _ in drafts
        ]
        self._target_submit_pipeline = int(target_submit_pipeline)
        self._target_rpc_window = [
            asyncio.Semaphore(max(1, self._target_submit_pipeline))
            for _ in targets
        ]
        # Per-target completion-rate EMA (r/s), updated from completion
        # timestamps in _on_target_finished — no extra RPC.
        self._target_tput_ema: list[float] = [0.0] * len(targets)
        self._target_last_finish_t: list[float] = [0.0] * len(targets)
        self._tput_ema_alpha = 0.1

        # ----- rate-match credit dispatch state (opt-in) --------------
        self._rate_match = bool(rate_match)
        self._rate_match_tick_s = float(rate_match_tick_s)
        self._rate_match_buffer_s = float(rate_match_buffer_s)
        self._rate_match_headroom = float(rate_match_headroom)
        self._rate_match_init_rps = float(rate_match_init_rps)
        self._rate_match_task: asyncio.Task | None = None
        # explicit two-queue dispatch (opt-in). _fresh_q: fresh requests
        # (draft pulls = cascade; target pulls leftover = direct). _regen_q:
        # escalated REGENs (target drains FIRST). Guarded by _buffer_cv.
        #
        # with a scorer configured, _fresh_q holds (score, seq, req_id)
        # tuples kept ascending — draft pops the HIGH end (likely-SHIP), target
        # Pops the LOW end (likely-escalate), realizing the documented
        # "sorted buffer" semantics under emergent dispatch. The deque
        # implementation was FIFO regardless of scorer, so --scorer-model was
        # architecturally inert in every two-buffer run (ON==OFF
        # to 0.6% at 2x overload). Scorer-less runs keep the plain FIFO deque —
        # behavior unchanged.
        self._two_buffer = bool(two_buffer)
        self._fresh_sorted = self._two_buffer and scorer is not None
        self._fresh_q = [] if self._fresh_sorted else collections.deque()
        self._regen_q: "collections.deque[str]" = collections.deque()

        # ----- occupancy-gated admission state (opt-in) ---------------
        self._occupancy_gate = bool(occupancy_gate)
        self._occupancy_max_inflight = int(occupancy_max_inflight)
        self._occupancy_hwm = float(occupancy_hwm)
        self._occupancy_lwm = float(occupancy_lwm)
        self._occupancy_kv_hwm = float(occupancy_kv_hwm)
        self._max_buffer_depth = int(max_buffer_depth)
        # Per-engine "holding" flags for hysteresis.
        self._draft_holding = [False for _ in drafts]
        self._target_holding = [False for _ in targets]
        # Cumulative completions observed on the pump path (free piggyback).
        # draft = items the draft produced (SHIP + REGEN, all real draft work);
        # target = items the target produced. The retune loop diffs these to
        # get per-engine measured throughput.
        self._draft_completions = [0 for _ in drafts]
        self._target_completions = [0 for _ in targets]

        # Stats — verdicts + path counts + load-tracker diagnostics.
        self.stats = {
            "n_submit": 0,
            "n_accept": 0,
            "n_regen": 0,
            "n_direct_target": 0,
            "n_error": 0,
            "n_submit_per_draft": [0] * len(drafts),
            "n_dispatch_per_target": [0] * len(targets),
            # encoder pool: per-encoder dispatch counter.
            # Stays at zero when encoders=None (legacy draft-ViT path).
            "n_dispatch_per_encoder": [0] * len(self._encoders),
            # Load-tracker diagnostics — accumulated calibration drift
            # is a leak detector: ideal value is 0 if local counters
            # match actor truth perfectly.
            "calibration_total_abs_drift": 0,
            "calibration_calls": 0,
            # counts of dispatch path. n_buffer_via_top is the
            # number of requests the draft dispatcher pulled from the
            # Buffer's high-score end; n_buffer_via_bottom is what the
            # target dispatcher pulled from the low-score end. Together
            # with n_force_* they sum to n_submit (modulo errors).
            "n_buffer_inserts": 0,
            "n_buffer_via_top": 0,
            "n_buffer_via_bottom": 0,
            "n_force_cascade": 0,
            "n_force_direct": 0,
            "buffer_max_depth": 0,
            # running sums of the prescorer score at the two fresh-q
            # pop sites (draft = high end, target-direct = low end). The
            # Drafted-vs-direct mean gap is the "ordering is acting" receipt
            # (≈equal means = inert scorer, the bug signature). Zero
            # when no scorer is configured or dispatch bypasses the buffer.
            "scorer_drafted_score_sum": 0.0,
            "scorer_drafted_score_n": 0,
            "scorer_direct_score_sum": 0.0,
            "scorer_direct_score_n": 0,
            # number of ship/escalate decisions made by the
            # scheduler-side gate (output-confidence A or query B) instead
            # Of the fork's hidden-state head. Zero when no gate is set.
            "n_gate_decisions": 0,
            # controller diagnostics (inert when closed_loop=False).
            "control": {
                "enabled": self._closed_loop,
                "ticks": 0,
                "direct_ratio": self._direct_ratio,
                "ship_ma": None,
                "dopt_ff": None,
                "trim": 0.0,
                "target_throttled_rps": None,
            },
        }

    # ---------------- public API ----------------

    async def start(self) -> None:
        """Spawn the pump + dispatcher tasks. Idempotent.

        Tasks:
          - `_draft_pump` — drains draft finished_q.
          - `_target_pump` — drains target finished_q.
          - `_draft_dispatch_loop[i]` per draft — token-gated pop of
            buffer's high-score end + draft.submit.remote().
          - `_target_dispatch_loop[i]` per target — token-gated pop
            of buffer's low-score end + target.submit_decode_batch.remote().

        force_direct_target=True (scheduler-wide): production cell
        passes drafts=[] so draft dispatch loops aren't spawned and
        all (−∞-scored) submits flow to target. Tests using
        drafts=[d] + force_direct_target=True + per-request
        force_cascade=True are expected to disambiguate via the
        +∞/−∞ sentinel scores (force_cascade goes top → draft;
        default goes bottom → target).
        """
        if (
            self._target_input_log_path is not None
            and self._target_input_log_fh is None
        ):
            self._target_input_log_fh = open(self._target_input_log_path, "w")
            self._target_input_log_fh.write(
                "dispatched_t,req_id,source,prompt_md5,n_imgs,max_tokens\n"
            )
        if self._draft_pump_task is None:
            self._draft_pump_task = asyncio.create_task(self._draft_pump())
        if self._target_pump_task is None:
            self._target_pump_task = asyncio.create_task(self._target_pump())
        # rate-match: create buckets even with no static send-rps, seeded
        # high (effectively unlimited) so the first tick flows; the retune loop
        # then clamps each to its measured throughput. burst = (tick+buffer) s.
        _rm_burst = self._rate_match_tick_s + self._rate_match_buffer_s
        # Initial guess (not unlimited): the headroom ramps this up to the true
        # capacity over a few ticks; avoids the cold-start flood.
        _rm_seed = self._rate_match_init_rps
        if self._dispatch_direct_rpc:
            # Direct-RPC: no dispatch loops (submit() fires RPCs inline). The
            # pumps above still drain finished_q. Skip all bucket/loop setup.
            return
        if not self._draft_dispatch_tasks and self._drafts:
            self._draft_buckets = [
                _TokenBucket(_rm_seed, burst_s=_rm_burst) if self._rate_match
                else (_TokenBucket(self._draft_engine_send_rps)
                      if self._draft_engine_send_rps else None)
                for _ in self._drafts
            ]
            self._draft_dispatch_tasks = [
                asyncio.create_task(self._draft_dispatch_loop(i))
                for i in range(len(self._drafts))
            ]
        if not self._target_dispatch_tasks:
            self._target_buckets = [
                _TokenBucket(_rm_seed, burst_s=_rm_burst) if self._rate_match
                else (_TokenBucket(self._target_engine_send_rps)
                      if self._target_engine_send_rps else None)
                for _ in self._targets
            ]
            self._target_dispatch_tasks = [
                asyncio.create_task(self._target_dispatch_loop(i))
                for i in range(len(self._targets))
            ]
        if self._rate_match and self._rate_match_task is None:
            self._rate_match_task = asyncio.create_task(self._rate_match_loop())
        # opt-in closed-loop controller. Enable the engines'
        # piggybacked-stats path (off by default → zero cost otherwise),
        # then spawn the controller task.
        if self._closed_loop and self._control_task is None:
            try:
                await asyncio.gather(*[
                    a.set_emit_stats.remote(True)
                    for a in (*self._drafts, *self._targets)
                ])
            except Exception:
                pass
            self._control_task = asyncio.create_task(self._control_loop())

        # the occupancy gate's KV-proximity arm reads piggybacked target
        # KV stats (kv_in_flight/kv_threshold) from _target_stats — but those are
        # Only populated when emit_stats is on. Pre-it was enabled ONLY for
        # the closed-loop controller, so a gate-only run (no --closed-loop) had an
        # INERT KV arm (kvf≡0) and gated on in-flight alone — this is exactly why
        # the overload couldn't show "which arm binds". Enable it for the
        # gate too so the KV arm is live (and loggable via occupancy_snapshot()).
        if self._occupancy_gate and not self._closed_loop:
            try:
                await asyncio.gather(*[
                    a.set_emit_stats.remote(True)
                    for a in (*self._drafts, *self._targets)
                ])
            except Exception:
                pass

    def set_engine_send_rps(
        self,
        draft_rps: float | None = None,
        target_rps: float | None = None,
    ) -> dict:
        """retune the per-engine dispatch rate limits at runtime.

        Used by the saturation finder's rate-ramp: pre-fill the buffer
        (so per-arrival submit work — e.g. image decode — is already done)
        then step this up while watching the engine's completion rate, so
        the cap measured is the ENGINE's (not the bench driver's).

        Only retunes buckets that already exist. A bucket constructed with
        rate=None (the engine started unlimited) stays unlimited — boot the
        scheduler with a non-None `*_engine_send_rps` (any high value) if
        you want it ramp-controllable. Returns the buckets' new rates.
        """
        if draft_rps is not None:
            self._draft_engine_send_rps = draft_rps
            for b in self._draft_buckets:
                if b is not None:
                    b.set_rate(draft_rps)
        if target_rps is not None:
            self._target_engine_send_rps = target_rps
            for b in self._target_buckets:
                if b is not None:
                    b.set_rate(target_rps)
        return {
            "draft_send_rps": self._draft_engine_send_rps,
            "target_send_rps": self._target_engine_send_rps,
            "draft_buckets": sum(b is not None for b in self._draft_buckets),
            "target_buckets": sum(b is not None for b in self._target_buckets),
        }

    def set_direct_ratio(self, ratio: float | None) -> None:
        """set the DIRECT-vs-CASCADE split applied per request in
        submit(). `ratio` in [0,1] = fraction routed straight to target;
        None disables scheduler-side splitting (revert to per-request force
        flags / emergent buffer routing). Mutated live by the controller."""
        if ratio is not None:
            ratio = max(0.0, min(1.0, float(ratio)))
        self._direct_ratio = ratio
        self.stats["control"]["direct_ratio"] = ratio

    async def stop(self) -> None:
        """Signal stop and await pump + dispatcher tasks."""
        self._stop.set()
        # Wake any dispatcher waiting on the buffer cv.
        async with self._buffer_cv:
            self._buffer_cv.notify_all()
        for task in (
            self._draft_pump_task,
            self._target_pump_task,
            self._control_task,
            self._rate_match_task,
            *self._draft_dispatch_tasks,
            *self._target_dispatch_tasks,
        ):
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._draft_pump_task = None
        self._target_pump_task = None
        self._control_task = None
        self._rate_match_task = None
        self._draft_dispatch_tasks = []
        self._target_dispatch_tasks = []
        if self._target_input_log_fh is not None:
            try:
                self._target_input_log_fh.close()
            except Exception:
                pass
            self._target_input_log_fh = None

    async def submit(
        self, prompt: str, max_tokens: int = 256,
        ignore_eos: bool = False,
        image_path: str | None = None,
        image_paths: list[str] | None = None,
        force_direct: bool = False,
        force_cascade: bool = False,
        head_cascade: bool | None = None,
        force_draft_response: bool = False,
        source: str | None = None,
    ) -> Response:
        """Submit a request and await its response.

        Auto-starts pump tasks on first call (so simple test scripts
        can just call submit() without a separate start()).

        `image_path` (single, legacy) or `image_paths` (multi) attach
        images to the draft request. DraftEngineAsync.submit forwards
        them as multi_modal_data={"image": [...]}. When both are set,
        image_paths wins.

        per-request routing overrides for two-stream benches.
        `force_direct=True` pins this request to DIRECT_TARGET regardless
        of gate state — used by the bench's "target stream" so its
        responses come back through V0Scheduler's _target_pump (avoids
        a second pop_finished consumer racing the scheduler).
        `force_cascade=True` pins it to the draft path even when
        force_direct_target=True is set scheduler-wide — used by the
        bench's "cascade stream." Mutually exclusive (force_cascade
        wins on tie since the explicit no-DIRECT intent is louder).

 `force_draft_response=True` overrides the head's REGEN
        verdict to SHIP so the cascade head fires inside the draft
        actor (and pays its cost) but the request never dispatches to
        target. Isolates pure head-eval cost from REGEN-coordination
        cost for the Gap 3b decomposition.

 `source` is stashed on `pending` and consulted by the
        `scorer` (when configured) to map the request to a score. The
        score determines the request's position in the sorted buffer;
        the draft dispatcher pulls from the high-score end (predicted
        SHIPs), the target dispatcher pulls from the low-score end
        (predicted REGENs). None source is fine; the scorer should
        cope with it (typically returning its `default`).
        """
        if self._draft_pump_task is None:
            await self.start()

        req_id = uuid.uuid4().hex[:12]
        arrival_t = time.perf_counter()
        ev = asyncio.Event()

        if image_paths is None and image_path is not None:
            image_paths = [image_path]

        self._pending[req_id] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "ignore_eos": ignore_eos,
            "arrival_t": arrival_t,
            "image_paths": image_paths,
            # stashed for the scorer. Carries the upstream
            # record's `source` (e.g. y3_docvqa) so the scorer can
            # look up the predicted SHIP probability.
            "source": source,
            # Cascade_retesting: when explicitly False, the draft engine
            # skips head firing entirely → no SHIP/REGEN verdict; result
            # returns as plain draft generation. Used by the bench's
            # `--cell draft_only_via_scheduler` for the apples-to-apples
            # baseline against `target_only_via_scheduler` (both pay
            # V0Scheduler RPC overhead, neither involves cascade head).
            "head_cascade": head_cascade,
            # when True, the head_decision REGEN verdict is
            # overridden to SHIP — the head still fires inside the draft
            # actor (cost is paid), but the resulting REGEN never
            # Dispatches to target. Isolates pure head-eval cost from
            # REGEN-coordination cost for Gap 3b decomposition.
            "force_draft_response": force_draft_response,
        }
        self._client_events[req_id] = ev
        self.stats["n_submit"] += 1

        # direct per-request RPC dispatch — no buffer, no
        # backpressure, no batching. Fire the engine RPC on arrival.
        if self._dispatch_direct_rpc:
            if force_direct or self._force_direct_target:
                self.stats["n_force_direct"] += 1
                await self._direct_dispatch_target(req_id)
            else:
                if force_cascade:
                    self.stats["n_force_cascade"] += 1
                await self._direct_dispatch_draft(req_id)
            await ev.wait()
            return self._client_results.pop(req_id)

        # two-buffer dispatch: a plain (un-forced) submit goes into the
        # fresh deque — the draft pulls it (cascade), the target pulls leftovers
        # (direct). REGENs land in _regen_q (target-priority) from
        # _on_draft_finished. Force-routed requests still use the sorted buffer.
        if self._two_buffer and not (
            force_cascade or force_direct or self._force_direct_target
        ):
            async with self._buffer_cv:
                # backpressure: hold this fresh arrival while the queues
                # are at the cap. Bounds the pre-dispatch backlog (else λ ≫
                # served-rate lets _fresh_q grow to 10s of thousands, p50 → 100s
                # of seconds, and the system oscillates instead of settling at
                # its saturated rate). REGENs (_regen_q) are in-system, never
                # gated. Re-check after each wake (wait() drops the lock).
                if self._max_buffer_depth > 0:
                    while (len(self._fresh_q) + len(self._regen_q)
                           >= self._max_buffer_depth
                           and not self._stop.is_set()):
                        self.stats["n_buffer_backpressure_waits"] = (
                            self.stats.get("n_buffer_backpressure_waits", 0) + 1)
                        try:
                            await asyncio.wait_for(
                                self._buffer_cv.wait(), timeout=0.25)
                        except asyncio.TimeoutError:
                            pass
                if self._fresh_sorted:
                    score = self._scorer(self._pending[req_id])
                    self._pending[req_id]["score"] = score
                    self._buffer_seq += 1
                    bisect.insort(self._fresh_q, (score, self._buffer_seq, req_id))
                else:
                    self._fresh_q.append(req_id)
                self.stats["n_buffer_inserts"] += 1
                depth = len(self._fresh_q) + len(self._regen_q)
                if depth > self.stats["buffer_max_depth"]:
                    self.stats["buffer_max_depth"] = depth
                self._buffer_cv.notify_all()
            self._submits_since_calibrate += 1
            if (
                self._submits_since_calibrate >= self._calibrate_every
                and not self._calibration_in_flight
            ):
                self._submits_since_calibrate = 0
                asyncio.create_task(self._calibrate_loads())
            await ev.wait()
            return self._client_results.pop(req_id)

        # every submit goes through the sorted buffer.
        # Force routes use sentinel scores so they fall at the
        # expected end and the corresponding dispatch loop picks
        # them up naturally:
        #   Force_cascade=True             → score +∞  (buffer top → draft)
        #   force_direct / force_direct_target → score −∞ (buffer bottom → target)
        #   default                         → score = scorer(pending) or 0.0
        if force_cascade:
            self.stats["n_force_cascade"] += 1
            score: float = float("inf")
        elif force_direct or self._force_direct_target:
            self.stats["n_force_direct"] += 1
            score = float("-inf")
        elif self._direct_ratio is not None:
            # scheduler-owned split — pin the DIRECT fraction to the
            # (possibly controller-tuned) ratio by sampling per request,
            # exactly like the bench's static cascade_gate split but mutable.
            if random.random() < self._direct_ratio:
                self.stats["n_force_direct"] += 1
                score = float("-inf")
            else:
                self.stats["n_force_cascade"] += 1
                score = float("inf")
        else:
            score = (
                self._scorer(self._pending[req_id])
                if self._scorer is not None else 0.0
            )
        await self._insert_into_buffer(req_id, score=score)

        # Calibrate after the synchronous dispatch — never await the
        # calibration future, so calibration RPC latency doesn't enter
        # the per-submit critical path. Skip if a calibration is
        # already in flight (single-flight).
        self._submits_since_calibrate += 1
        if (
            self._submits_since_calibrate >= self._calibrate_every
            and not self._calibration_in_flight
        ):
            self._submits_since_calibrate = 0
            asyncio.create_task(self._calibrate_loads())

        await ev.wait()
        return self._client_results.pop(req_id)

    # ---------------- buffer + dispatch loops ----------------

    async def _insert_into_buffer(self, req_id: str, *, score: float) -> None:
        """Insert (score, seq, req_id) into the shared sorted buffer.

        Caller supplies the score. submit() picks it via scorer or
        sentinel; _on_draft_finished's REGEN branch uses −∞ so target
        picks the request up next.

        `seq` is a monotonic tiebreak so equal scores resolve FIFO.
        Wakes any dispatch loop blocked on `_buffer_cv`. O(log N)
        bisect + O(N) shift — fine for N ≤ a few hundred in-flight.
        """
        self._pending[req_id]["score"] = score
        async with self._buffer_cv:
            # backpressure: hold a FRESH arrival while the buffer is at
            # the cap so O(N) bisect ops stay bounded (see __init__). REGEN
            # re-entries (−∞) bypass — they are already in-system. Re-check
            # after each wake (wait() drops the lock) so the cap is not raced.
            if self._max_buffer_depth > 0 and math.isfinite(score):
                while (len(self._buffer) >= self._max_buffer_depth
                       and not self._stop.is_set()):
                    self.stats["n_buffer_backpressure_waits"] = (
                        self.stats.get("n_buffer_backpressure_waits", 0) + 1)
                    try:
                        await asyncio.wait_for(
                            self._buffer_cv.wait(), timeout=0.25)
                    except asyncio.TimeoutError:
                        pass
            self._buffer_seq += 1
            bisect.insort(self._buffer, (score, self._buffer_seq, req_id))
            self.stats["n_buffer_inserts"] += 1
            if len(self._buffer) > self.stats["buffer_max_depth"]:
                self.stats["buffer_max_depth"] = len(self._buffer)
            self._buffer_cv.notify_all()

    def _occupancy_room(self, load: int, kv_stats: dict,
                        holding: bool) -> tuple[bool, bool]:
        """occupancy gate. Returns (has_room, new_holding).

        'Full' = in-flight ≥ hwm·max_inflight OR target KV proximity ≥ kv_hwm.
        Hysteresis: once holding, resume only when BOTH drain below lwm — avoids
        flapping add/hold every step at the boundary."""
        maxn = self._occupancy_max_inflight
        lf = (load / maxn) if maxn else 0.0
        kf, kt = kv_stats.get("kv_in_flight"), kv_stats.get("kv_threshold")
        kvf = (kf / kt) if (kt and kf is not None) else 0.0
        if holding:
            still = (lf >= self._occupancy_lwm) or (kvf >= self._occupancy_lwm)
            return (not still, still)
        full = (lf >= self._occupancy_hwm) or (kvf >= self._occupancy_kv_hwm)
        return (not full, full)

    async def _direct_dispatch_draft(self, req_id: str) -> None:
        """fire the draft RPC immediately (no buffer, no gate,
        no rate limit). Mirrors the draft dispatch loop's RPC body (minus the
        encoder path, which direct-RPC does not support)."""
        pending = self._pending.get(req_id)
        if pending is None:
            return
        draft_idx = self._pick_draft()
        pending["draft_idx"] = draft_idx
        pending["routing_path"] = "cascade"
        self._draft_load[draft_idx] += 1
        self.stats["n_submit_per_draft"][draft_idx] += 1
        self.stats["n_buffer_via_top"] += 1
        image_paths = pending.get("image_paths")
        try:
            await self._drafts[draft_idx].submit.remote(
                req_id, pending["prompt"], pending["max_tokens"], 0.0,
                pending["ignore_eos"],
                image_paths[0] if image_paths and len(image_paths) == 1 else None,
                pending.get("head_cascade"),
                image_paths if image_paths and len(image_paths) > 1 else None,
                logprobs=(
                    None if self._draft_emit_ptf_seq
                    else (self._draft_logprobs or None)
                ),
                emit_per_token_feature_seq=(
                    self._draft_emit_ptf_seq
                    or self._draft_in_engine_cascade_head
                ),
                in_engine_cascade_head=self._draft_in_engine_cascade_head,
                cascade_source=pending.get("source"),
            )
        except Exception as e:
            self._draft_load[draft_idx] = max(
                0, self._draft_load[draft_idx] - 1,
            )
            self._respond_error(
                req_id, "draft_submit", f"{type(e).__name__}: {e}",
            )

    async def _direct_dispatch_target(self, req_id: str) -> None:
        """fire the target RPC immediately for a DIRECT or a
        post-head REGEN request (one item per RPC — no batching)."""
        pending = self._pending.get(req_id)
        if pending is None:
            return
        target_idx = self._pick_target()
        pending["target_idx"] = target_idx
        pending.setdefault("routing_path", "direct_target")
        pending.setdefault("draft_completed_t", pending["arrival_t"])
        self._target_load[target_idx] += 1
        self.stats["n_dispatch_per_target"][target_idx] += 1
        self.stats["n_buffer_via_bottom"] += 1
        image_paths = pending.get("image_paths")
        single = (
            image_paths[0] if image_paths and len(image_paths) == 1 else None
        )
        multi = image_paths if image_paths and len(image_paths) > 1 else None
        item = {
            "req_id":     req_id,
            "prompt":     pending["prompt"],
            "max_tokens": pending["max_tokens"],
            "ignore_eos": pending.get("ignore_eos", False),
            "image_path": single,
            "image_paths": multi,
            "source":     pending.get("source"),
        }
        try:
            await self._targets[target_idx].submit_decode_batch.remote([item])
        except Exception as e:
            self._target_load[target_idx] = max(
                0, self._target_load[target_idx] - 1,
            )
            self._respond_error(
                req_id, "submit_decode_batch", f"{type(e).__name__}: {e}",
            )

    async def _route_regen(self, req_id: str) -> None:
        """Route an escalated REGEN to the target. Direct-RPC mode: fire the
        target RPC immediately. Two-buffer mode: into the priority _regen_q
        (target drains it FIRST). Else: re-enter the sorted buffer at −∞ so
        the target dispatch picks it up next."""
        if self._dispatch_direct_rpc:
            await self._direct_dispatch_target(req_id)
            return
        if self._two_buffer:
            async with self._buffer_cv:
                self._regen_q.append(req_id)
                self._buffer_cv.notify_all()
        else:
            await self._insert_into_buffer(req_id, score=float("-inf"))

    async def _draft_dispatch_loop(self, draft_idx: int) -> None:
        """Token-gated draft consumer. One task per draft actor.

        Loop: acquire a token (the rate limit is BEFORE the buffer
        pop — that's what propagates backpressure to the buffer);
        wait for an item; pop the buffer's high-score end; stamp
        pending; (optionally) run encoder; fire `draft.submit.remote`.

        encoder pool: when encoders are configured and the
        request has images, the encoder RPC runs first
        (`encoder.encode_return.remote` → CPU image_embeds); draft
        consumes pre-encoded tensors and skips its own ViT.
        """
        bucket = self._draft_buckets[draft_idx]
        draft = self._drafts[draft_idx]
        while True:
            if bucket is not None:
                await bucket.acquire(1)
            # occupancy gate: hold if the draft is full (in-flight ≥ hwm).
            if self._occupancy_gate:
                while not self._stop.is_set():
                    room, self._draft_holding[draft_idx] = self._occupancy_room(
                        self._draft_load[draft_idx],
                        self._draft_stats[draft_idx],
                        self._draft_holding[draft_idx],
                    )
                    if room:
                        break
                    await asyncio.sleep(0.005)
            async with self._buffer_cv:
                if self._two_buffer:
                    # Two-buffer: draft pulls fresh requests only (cascade).
                    while not self._stop.is_set():
                        if self._fresh_q:
                            break
                        try:
                            await asyncio.wait_for(
                                self._buffer_cv.wait(),
                                timeout=self._draft_pop_timeout_s,
                            )
                        except asyncio.TimeoutError:
                            pass
                    if self._stop.is_set() and not self._fresh_q:
                        return
                    req_id = (self._fresh_q.pop()[2] if self._fresh_sorted
                              else self._fresh_q.popleft())
                    self.stats["n_buffer_via_top"] += 1
                    _sc = (self._pending.get(req_id) or {}).get("score")
                    if _sc is not None and _sc == _sc:
                        self.stats["scorer_drafted_score_sum"] += float(_sc)
                        self.stats["scorer_drafted_score_n"] += 1
                else:
                    while not self._stop.is_set():
                        # Eligibility: draft only pops items whose score
                        # is > −∞. A −∞ item is a `force_direct` or
                        # scheduler-wide `force_direct_target` request
                        # destined for target; draft must not steal it
                        # (single-item race otherwise — see T8/T9).
                        if self._buffer and self._buffer[-1][0] != float("-inf"):
                            break
                        try:
                            await asyncio.wait_for(
                                self._buffer_cv.wait(),
                                timeout=self._draft_pop_timeout_s,
                            )
                        except asyncio.TimeoutError:
                            pass
                    if self._stop.is_set() and (
                        not self._buffer
                        or self._buffer[-1][0] == float("-inf")
                    ):
                        return  # nothing eligible left
                    _score, _seq, req_id = self._buffer.pop()  # top = highest
                    self.stats["n_buffer_via_top"] += 1

            pending = self._pending.get(req_id)
            if pending is None:
                continue  # responded-to before we picked it
            pending["draft_idx"] = draft_idx
            pending["routing_path"] = "cascade"
            self._draft_load[draft_idx] += 1
            self.stats["n_submit_per_draft"][draft_idx] += 1

            image_paths = pending.get("image_paths")
            image_embeds = None
            image_grid_thw = None
            if self._encoders and image_paths:
                encoder_idx = self._pick_encoder()
                self._encoder_load[encoder_idx] += 1
                self.stats["n_dispatch_per_encoder"][encoder_idx] += 1
                pending["encoder_idx"] = encoder_idx
                try:
                    result = await self._encoders[encoder_idx].encode_return.remote(
                        req_id, image_paths,
                    )
                except Exception as e:
                    self._encoder_load[encoder_idx] = max(
                        0, self._encoder_load[encoder_idx] - 1,
                    )
                    self._draft_load[draft_idx] = max(
                        0, self._draft_load[draft_idx] - 1,
                    )
                    self._respond_error(
                        req_id, "encoder_encode", f"{type(e).__name__}: {e}",
                    )
                    continue
                self._encoder_load[encoder_idx] = max(
                    0, self._encoder_load[encoder_idx] - 1,
                )
                image_embeds = result.get("image_embeds")
                image_grid_thw = result.get("image_grid_thw")
                image_paths = None
            _submit_args = (
                req_id, pending["prompt"], pending["max_tokens"], 0.0,
                pending["ignore_eos"],
                image_paths[0] if image_paths and len(image_paths) == 1 else None,
                pending.get("head_cascade"),
                image_paths if image_paths and len(image_paths) > 1 else None,
            )
            _submit_kwargs = dict(
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
                # ask the draft for top-k per-token logprobs so
                # the gate can threshold its confidence stats. 0 → no
                # logprob overhead (non-gate cells).
                #
                # when emit_per_token_feature_seq is on,
                # We DO NOT request driver-visible logprobs. The
                # lp-classifier-inline fork internally forces the
                # sampler to emit logprobs_tensors (so the inline
                # FeatureSeqAccumulator hook still gets fed), but
                # the engine's LogprobsProcessor stays inactive ⇒
                # no per-step Logprob-dict construction + no
                # detokenization. The bulk of the driver-side cost.
                logprobs=(
                    None if self._draft_emit_ptf_seq
                    else (self._draft_logprobs or None)
                ),
                emit_per_token_feature_seq=(
                    self._draft_emit_ptf_seq
                    or self._draft_in_engine_cascade_head
                ),
                in_engine_cascade_head=(
                    self._draft_in_engine_cascade_head
                ),
                cascade_source=pending.get("source"),
            )
            try:
                if (self._draft_submit_batch > 1 and not self._encoders
                        and bucket is None):
                    # wire-latency fix: one submit_batch RPC carries
                    # this request PLUS everything draft-eligible in the
                    # buffer right now (cap: draft_submit_batch). A round
                    # trip then amortizes over the batch, so dispatch rate
                    # stops degrading ∝ 1/RTT (pipelining bounds it
                    # at pipeline/RTT; deepening the pipeline measurably
                    # hurts). Guards: encoder cells keep the per-request
                    # path (extras would need encode RPCs), and token-
                    # bucket (rate-match) cells keep it because extras
                    # would bypass the rate limit.
                    items = [dict(zip(_DRAFT_SUBMIT_POS, _submit_args),
                                  **_submit_kwargs)]
                    for rid2 in await self._pop_more_fresh_nowait(
                            self._draft_submit_batch - 1):
                        p2 = self._pending.get(rid2)
                        if p2 is None:
                            continue  # responded-to before we picked it
                        p2["draft_idx"] = draft_idx
                        p2["routing_path"] = "cascade"
                        self._draft_load[draft_idx] += 1
                        self.stats["n_submit_per_draft"][draft_idx] += 1
                        items.append(self._draft_submit_payload(rid2, p2))
                    rids = [it["req_id"] for it in items]
                    if self._draft_submit_pipeline > 0:
                        await self._draft_rpc_window[draft_idx].acquire()
                        _ref = draft.submit_batch.remote(items)
                        asyncio.create_task(
                            self._reap_draft_submit_batch(
                                draft_idx, rids, _ref)
                        )
                    else:
                        await draft.submit_batch.remote(items)
                elif self._draft_submit_pipeline > 0:
                    # wire-latency fix: don't serialize dispatch on the
                    # submit-ACK round trip. Fire the RPC and reap its ACK in
                    # a background task; a per-draft semaphore bounds in-flight
                    # ACKs so the pump can't outrun a dead/backlogged actor.
                    await self._draft_rpc_window[draft_idx].acquire()
                    _ref = draft.submit.remote(*_submit_args, **_submit_kwargs)
                    asyncio.create_task(
                        self._reap_draft_submit(draft_idx, req_id, _ref)
                    )
                else:
                    await draft.submit.remote(*_submit_args, **_submit_kwargs)
            except Exception as e:
                self._draft_load[draft_idx] = max(
                    0, self._draft_load[draft_idx] - 1,
                )
                self._respond_error(
                    req_id, "draft_submit", f"{type(e).__name__}: {e}",
                )

    def _draft_submit_payload(self, req_id: str, pending: dict) -> dict:
        """kwargs dict for DraftEngineAsync._submit_impl — the
        non-encoder path (raw image_paths through, embeds None). Must stay
        field-for-field identical to the inline _submit_args/_submit_kwargs
        build in _draft_dispatch_loop."""
        image_paths = pending.get("image_paths")
        return dict(
            req_id=req_id,
            prompt=pending["prompt"],
            max_tokens=pending["max_tokens"],
            temperature=0.0,
            ignore_eos=pending["ignore_eos"],
            image_path=(image_paths[0]
                        if image_paths and len(image_paths) == 1 else None),
            head_cascade=pending.get("head_cascade"),
            image_paths=(image_paths
                         if image_paths and len(image_paths) > 1 else None),
            image_embeds=None,
            image_grid_thw=None,
            logprobs=(None if self._draft_emit_ptf_seq
                      else (self._draft_logprobs or None)),
            emit_per_token_feature_seq=(
                self._draft_emit_ptf_seq
                or self._draft_in_engine_cascade_head
            ),
            in_engine_cascade_head=self._draft_in_engine_cascade_head,
            cascade_source=pending.get("source"),
        )

    async def _pop_more_fresh_nowait(self, n: int) -> list[str]:
        """opportunistically pop up to n MORE draft-eligible req_ids
        without waiting — the coalescing companion of the batched draft
        submit. Mirrors the blocking pop's eligibility rules (two-buffer:
        fresh_q only; single buffer: top must not be a −∞ direct-target
        item) and its stats bookkeeping."""
        out: list[str] = []
        if n <= 0:
            return out
        async with self._buffer_cv:
            while len(out) < n:
                if self._two_buffer:
                    if not self._fresh_q:
                        break
                    rid = (self._fresh_q.pop()[2] if self._fresh_sorted
                           else self._fresh_q.popleft())
                    _sc = (self._pending.get(rid) or {}).get("score")
                    if _sc is not None and _sc == _sc:
                        self.stats["scorer_drafted_score_sum"] += float(_sc)
                        self.stats["scorer_drafted_score_n"] += 1
                else:
                    if (not self._buffer
                            or self._buffer[-1][0] == float("-inf")):
                        break
                    _score, _seq, rid = self._buffer.pop()
                self.stats["n_buffer_via_top"] += 1
                out.append(rid)
        return out

    async def _reap_draft_submit(self, draft_idx: int, req_id: str,
                                 ref) -> None:
        """await a pipelined draft-submit ACK off the dispatch path;
        on failure route the request through the same error path the inline
        await used. if the ack carries stats (actor self-admit), store
        them as a fresh piggybacked snapshot for the controller/gate."""
        try:
            st = await ref
            if isinstance(st, dict):
                self._draft_stats[draft_idx] = st
        except Exception as e:
            self._draft_load[draft_idx] = max(
                0, self._draft_load[draft_idx] - 1,
            )
            self._respond_error(
                req_id, "draft_submit", f"{type(e).__name__}: {e}",
            )
        finally:
            self._draft_rpc_window[draft_idx].release()

    async def _reap_draft_submit_batch(self, draft_idx: int,
                                       req_ids: list[str], ref) -> None:
        """batch companion of _reap_draft_submit — a failed batch
        RPC errors every request it carried. (Per-item failures inside a
        delivered batch come back via finished_q instead.) A stats-carrying
        ack (actor self-admit) is stored as a fresh piggybacked snapshot."""
        try:
            st = await ref
            if isinstance(st, dict):
                self._draft_stats[draft_idx] = st
        except Exception as e:
            for rid in req_ids:
                self._draft_load[draft_idx] = max(
                    0, self._draft_load[draft_idx] - 1,
                )
                self._respond_error(
                    rid, "draft_submit", f"{type(e).__name__}: {e}",
                )
        finally:
            self._draft_rpc_window[draft_idx].release()

    def _target_batch_n(self, target_idx: int) -> int:
        """throughput-adaptive batch size for target dispatch:
        `clamp(round(MA_throughput × window), [min, max])`. Returns 1 when
        adaptive batching is off (→ one-item-per-RPC, pre-behavior) or
        before the rate EMA has warmed."""
        if not self._adaptive_batch:
            return 1
        window = self._adaptive_batch_window_s
        buffer_n = 0
        if self._adaptive_batch_rtt_aware:
            # wire-latency fix (operator): a batch must carry enough
            # work to cover the full pump round trip, not just the sizing
            # window — `batch ≈ tput × (RTT + window) + buffer` keeps the
            # actor busy while the next RPC is on the wire. With the legacy
            # formula a 50 ms RTT collapses batch→1 (tput falls → batch
            # falls → tput falls further).
            window = window + self._target_rpc_rtt_ema[target_idx]
            buffer_n = self._adaptive_batch_buffer
        est = self._target_tput_ema[target_idx] * window + buffer_n
        return max(
            self._adaptive_batch_min,
            min(self._adaptive_batch_max, int(round(est)) or 1),
        )

    async def _target_dispatch_loop(self, target_idx: int) -> None:
        """Token-gated target consumer. One task per target actor.

        Loop: acquire a token; pop the buffer's low-score end; stamp
        pending (preserving routing_path='cascade' if already set —
        REGEN re-entries from `_on_draft_finished` carry that); fire
        a single-item `submit_decode_batch` RPC.

        No batching — each pop fires one RPC. simplification:
        the era coalescing window is gone; if it's needed back
        later, add a `try_acquire` + `refund` dance on the bucket.
        """
        bucket = self._target_buckets[target_idx]
        target = self._targets[target_idx]
        while True:
            if bucket is not None:
                await bucket.acquire(1)
            # occupancy gate: hold if the target is full (in-flight ≥ hwm
            # OR KV proximity ≥ kv_hwm — keeps it off the KV-preemption cliff).
            if self._occupancy_gate:
                while not self._stop.is_set():
                    room, self._target_holding[target_idx] = self._occupancy_room(
                        self._target_load[target_idx],
                        self._target_stats[target_idx],
                        self._target_holding[target_idx],
                    )
                    if room:
                        break
                    await asyncio.sleep(0.005)
            # target batch self-sizes to the actor's measured rate.
            # Default off → batch_n=1 ⇒ byte-identical one-item-per-RPC.
            batch_n = self._target_batch_n(target_idx)
            req_ids: list[str] = []
            async with self._buffer_cv:
                # Block-wait for the FIRST eligible item only (never block to
                # FILL a batch — that would add latency); drain extras non-block.
                if self._two_buffer:
                    while not self._stop.is_set():
                        if self._regen_q or self._fresh_q:
                            break
                        try:
                            await asyncio.wait_for(
                                self._buffer_cv.wait(),
                                timeout=self._target_pop_timeout_s,
                            )
                        except asyncio.TimeoutError:
                            pass
                    if self._stop.is_set() and not (
                        self._regen_q or self._fresh_q
                    ):
                        return
                    while len(req_ids) < batch_n and (
                        self._regen_q or self._fresh_q
                    ):
                        # REGENs (priority) before fresh, matching single-pop.
                        if self._regen_q:
                            req_ids.append(self._regen_q.popleft())
                        else:
                            req_ids.append(
                                self._fresh_q.pop(0)[2] if self._fresh_sorted
                                else self._fresh_q.popleft())
                            _sc = (self._pending.get(req_ids[-1]) or {}).get(
                                "score")
                            if _sc is not None and _sc == _sc:
                                self.stats["scorer_direct_score_sum"] += \
                                    float(_sc)
                                self.stats["scorer_direct_score_n"] += 1
                        self.stats["n_buffer_via_bottom"] += 1
                else:
                    while not self._stop.is_set():
                        # Eligibility: target only pops items whose score
                        # is < +∞. A +∞ item is a `force_cascade` request
                        # destined for draft; target must not steal it.
                        if self._buffer and self._buffer[0][0] != float("inf"):
                            break
                        try:
                            await asyncio.wait_for(
                                self._buffer_cv.wait(),
                                timeout=self._target_pop_timeout_s,
                            )
                        except asyncio.TimeoutError:
                            pass
                    if self._stop.is_set() and (
                        not self._buffer
                        or self._buffer[0][0] == float("inf")
                    ):
                        return
                    while (
                        len(req_ids) < batch_n and self._buffer
                        and self._buffer[0][0] != float("inf")
                    ):
                        _s, _q, rid = self._buffer.pop(0)  # bottom=lowest
                        req_ids.append(rid)
                        self.stats["n_buffer_via_bottom"] += 1

            # Charge the rate limiter for the extra items popped this round
            # (the loop-top acquire(1) already covered the first).
            if bucket is not None and len(req_ids) > 1:
                await bucket.acquire(len(req_ids) - 1)

            items: list[dict] = []
            for req_id in req_ids:
                pending = self._pending.get(req_id)
                if pending is None:
                    continue
                pending["target_idx"] = target_idx
                # REGEN re-entries already have routing_path='cascade' +
                # draft_completed_t set; pure DIRECTs reach here without them.
                pending.setdefault("routing_path", "direct_target")
                pending.setdefault("draft_completed_t", pending["arrival_t"])
                self._target_load[target_idx] += 1
                self.stats["n_dispatch_per_target"][target_idx] += 1
                image_paths = pending.get("image_paths")
                single = (
                    image_paths[0]
                    if image_paths and len(image_paths) == 1 else None
                )
                multi = (
                    image_paths if image_paths and len(image_paths) > 1 else None
                )
                items.append({
                    "req_id":     req_id,
                    "prompt":     pending["prompt"],
                    "max_tokens": pending["max_tokens"],
                    "ignore_eos": pending.get("ignore_eos", False),
                    "image_path": single,
                    "image_paths": multi,
                    # carry source so the target's MA-length KV gate can
                    # bucket observed lengths per source.
                    "source":     pending.get("source"),
                })
                if self._target_input_log_fh is not None:
                    import hashlib
                    pmd5 = hashlib.md5(
                        (pending["prompt"] or "").encode()
                    ).hexdigest()[:12]
                    n_imgs = len(pending.get("image_paths") or [])
                    self._target_input_log_fh.write(
                        f"{time.perf_counter():.6f},{req_id},"
                        f"{pending.get('source','')},{pmd5},{n_imgs},"
                        f"{pending['max_tokens']}\n"
                    )
            if not items:
                continue
            self.stats.setdefault("target_batch_flushes", 0)
            self.stats.setdefault("target_batch_items_flushed", 0)
            self.stats["target_batch_flushes"] += 1
            self.stats["target_batch_items_flushed"] += len(items)
            _rpc_t0 = time.perf_counter()
            if self._target_submit_pipeline > 0:
                # wire-latency fix (the 1 s residual): the serial await
                # below holds exactly ONE dispatch RPC in flight per target —
                # during a 1 s round trip that target receives no new work
                # (cap = batch/RTT, and the pop loop stalls with it). Same
                # cure as the draft path: fire + reap in a background task,
                # bounded by a per-target semaphore. Default 0 = serial.
                await self._target_rpc_window[target_idx].acquire()
                _ref = target.submit_decode_batch.remote(items)
                asyncio.create_task(
                    self._reap_target_submit_batch(
                        target_idx, items, _ref, _rpc_t0)
                )
                continue
            try:
                await target.submit_decode_batch.remote(items)
                # EWMA of the dispatch-RPC round trip feeds the
                # rtt-aware batch sizing (α=0.2; first sample seeds).
                _dt = time.perf_counter() - _rpc_t0
                _old = self._target_rpc_rtt_ema[target_idx]
                self._target_rpc_rtt_ema[target_idx] = (
                    _dt if _old == 0.0 else 0.8 * _old + 0.2 * _dt
                )
            except Exception as e:
                for it in items:
                    self._target_load[target_idx] = max(
                        0, self._target_load[target_idx] - 1,
                    )
                    self._respond_error(
                        it["req_id"], "submit_decode_batch",
                        f"{type(e).__name__}: {e}",
                    )

    async def _reap_target_submit_batch(self, target_idx: int,
                                        items: list[dict], ref,
                                        rpc_t0: float) -> None:
        """await a pipelined target-dispatch ACK off the dispatch
        path; mirrors the serial path's RTT-EWMA update + all-items error
        routing. A stats-carrying ack (actor self-admit) is stored as a
        fresh piggybacked snapshot."""
        try:
            st = await ref
            if isinstance(st, dict):
                self._target_stats[target_idx] = st
            _dt = time.perf_counter() - rpc_t0
            _old = self._target_rpc_rtt_ema[target_idx]
            self._target_rpc_rtt_ema[target_idx] = (
                _dt if _old == 0.0 else 0.8 * _old + 0.2 * _dt
            )
        except Exception as e:
            for it in items:
                self._target_load[target_idx] = max(
                    0, self._target_load[target_idx] - 1,
                )
                self._respond_error(
                    it["req_id"], "submit_decode_batch",
                    f"{type(e).__name__}: {e}",
                )
        finally:
            self._target_rpc_window[target_idx].release()

    # ---------------- pickers ----------------

    def _pick_draft(self) -> int:
        """Pick the draft actor with minimum outstanding load.

        Tiebreaker: round-robin among the min-load drafts. Falls back
        to pure round-robin when load_aware=False (matches the
        pre-behavior).
        """
        if not self._load_aware:
            idx = pick_round_robin(self._drafts, self._draft_rr_tiebreak)
            self._draft_rr_tiebreak += 1
            return idx
        min_load = min(self._draft_load)
        # Build the list of indices at minimum load (almost always
        # short — at most len(drafts) entries).
        ties = [i for i, l in enumerate(self._draft_load) if l == min_load]
        if len(ties) == 1:
            return ties[0]
        idx = ties[self._draft_rr_tiebreak % len(ties)]
        self._draft_rr_tiebreak += 1
        return idx

    def _pick_target(self) -> int:
        """Pick the target with minimum outstanding load.

        Same min-load + RR-tiebreak policy as drafts. Falls back to
        pure round-robin when load_aware=False.
        """
        if not self._load_aware:
            idx = pick_round_robin(self._targets, self._target_rr_tiebreak)
            self._target_rr_tiebreak += 1
            return idx
        min_load = min(self._target_load)
        ties = [i for i, l in enumerate(self._target_load) if l == min_load]
        if len(ties) == 1:
            return ties[0]
        idx = ties[self._target_rr_tiebreak % len(ties)]
        self._target_rr_tiebreak += 1
        return idx

    def _pick_encoder(self) -> int:
        """pick the encoder actor with minimum outstanding
        encode count. Same min-load + RR-tiebreak policy as drafts /
        targets. Returns -1 if no encoders are configured."""
        if not self._encoders:
            return -1
        if not self._load_aware:
            idx = pick_round_robin(self._encoders, self._encoder_rr_tiebreak)
            self._encoder_rr_tiebreak += 1
            return idx
        min_load = min(self._encoder_load)
        ties = [i for i, l in enumerate(self._encoder_load) if l == min_load]
        if len(ties) == 1:
            return ties[0]
        idx = ties[self._encoder_rr_tiebreak % len(ties)]
        self._encoder_rr_tiebreak += 1
        return idx

    # ---------------- pump loops ----------------

    async def _draft_pump(self) -> None:
        """Drain each draft's finished_q; route each finished item to
        a target via submit_regen (REGEN verdict) or respond ACCEPT
        directly without a target call (SHIP verdict).

        Each item is tagged with its source draft index before being
        passed to _on_draft_finished — the handler uses it to decrement
        the per-draft load counter.

        when force_direct_target=True, drafts list may be
        empty. Empty gather returns immediately, so we must hand-pace
        the loop or it spins.
        """
        while not self._stop.is_set():
            if not self._drafts:
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self._draft_pop_timeout_s,
                    )
                except asyncio.TimeoutError:
                    pass
                continue
            polls = [
                d.pop_finished.remote(
                    max_n=self._pop_max_n, timeout_s=self._draft_pop_timeout_s,
                ) for d in self._drafts
            ]
            items_lists = await asyncio.gather(*polls)

            handlers: list = []
            for draft_idx, items in enumerate(items_lists):
                # capture the piggybacked engine-stats snapshot (all
                # items from one actor carry the same stats dict). Free —
                # rides the response the pump already drains.
                if items:
                    st = items[0].get("stats")
                    if st:
                        self._draft_stats[draft_idx] = st
                # Rate-match: count this draft's produced items (all real
                # draft work — SHIP + REGEN) for the measured-throughput retune.
                self._draft_completions[draft_idx] += len(items)
                for item in items:
                    # Tag source — useful for sanity if pending was
                    # somehow not tagged at submit time.
                    item.setdefault("_draft_idx", draft_idx)
                    handlers.append(self._on_draft_finished(item))
            if handlers:
                # Note: each handler's REGEN branch enqueues to the
                # target send queue (cheap) rather than awaiting a
                # target RPC — so the gather no longer produces an
                # N-concurrent-RPC stampede on target.
                await asyncio.gather(*handlers)
            # else: pop_finished already blocked for draft_pop_timeout_s,
            # natural pacing.

    async def _target_pump(self) -> None:
        """Drain each target's finished_q; resolve client futures."""
        while not self._stop.is_set():
            polls = [
                t.pop_finished.remote(
                    max_n=self._pop_max_n, timeout_s=self._target_pop_timeout_s,
                ) for t in self._targets
            ]
            items_lists = await asyncio.gather(*polls)
            for target_idx, items in enumerate(items_lists):
                # capture the piggybacked target stats (incl. KV
                # proximity) for the controller. Synthetic SHIP items never
                # reach this pump, so _target_stats reflects real target load.
                if items:
                    st = items[0].get("stats")
                    if st:
                        self._target_stats[target_idx] = st
                # Rate-match: count real target completions (synthetic
                # SHIPs never reach this pump) for the throughput retune.
                self._target_completions[target_idx] += len(items)
                # maintain a per-target completion-rate EMA for the
                # adaptive batch sizer (no extra RPC — rides the pump).
                if items and self._adaptive_batch:
                    now = time.perf_counter()
                    last = self._target_last_finish_t[target_idx]
                    if last > 0 and now > last:
                        inst = len(items) / (now - last)
                        a = self._tput_ema_alpha
                        self._target_tput_ema[target_idx] = (
                            a * inst
                            + (1 - a) * self._target_tput_ema[target_idx]
                        )
                    self._target_last_finish_t[target_idx] = now
                for item in items:
                    item.setdefault("_target_idx", target_idx)
                    self._on_target_finished(item)

    # ---------------- calibration ----------------

    async def _calibrate_loads(self) -> None:
        """Reconcile local _draft_load / _target_load against truthful
        actor counts via parallel qsize() RPCs.

        Single-flight — only one calibration in flight at a time. The
        scheduler's submit/finish loop continues during the RPC; when
        the call returns we overwrite locals with the actor's
        in_flight value. The unavoidable race window (submits/finishes
        between the RPC and the write) is small relative to the
        calibration cadence (every N=64 submits).
        """
        if self._calibration_in_flight:
            return
        self._calibration_in_flight = True
        try:
            d_futs = [d.qsize.remote() for d in self._drafts]
            t_futs = [t.qsize.remote() for t in self._targets]
            d_sizes = await asyncio.gather(*d_futs, return_exceptions=True)
            t_sizes = await asyncio.gather(*t_futs, return_exceptions=True)
            total_drift = 0
            for i, s in enumerate(d_sizes):
                if isinstance(s, Exception):
                    continue
                truth = int(s.get("in_flight", 0))
                drift = self._draft_load[i] - truth
                total_drift += abs(drift)
                self._draft_load[i] = truth
            for i, s in enumerate(t_sizes):
                if isinstance(s, Exception):
                    continue
                # Target qsize() includes verify_q + regen_q + in_flight
                # tasks. The local _target_load tracks what we've
                # submitted but haven't observed a finished item for.
                # Use in_flight (running) + queue depth as truth.
                truth = (
                    int(s.get("in_flight", 0))
                    + int(s.get("verify", 0))
                    + int(s.get("regen", 0))
                )
                drift = self._target_load[i] - truth
                total_drift += abs(drift)
                self._target_load[i] = truth
            self.stats["calibration_total_abs_drift"] += total_drift
            self.stats["calibration_calls"] += 1
        finally:
            self._calibration_in_flight = False

    # ---------------- closed-loop controller ----------------

    async def _control_loop(self) -> None:
        """Periodic closed-loop controller (opt-in via closed_loop=True).

        Every `control_tick_s`, reads the piggybacked engine-stats snapshots
        (in-memory, ZERO RPC) and re-tunes the DIRECT-vs-CASCADE split toward
        DOPT, plus an optional KV-proximity throttle on target dispatch.
        Best-effort: a step that raises is swallowed so the loop never dies.
        """
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._control_tick_s,
                )
                break  # stop signalled
            except asyncio.TimeoutError:
                pass
            try:
                self._control_step()
            except Exception:
                pass

    def _tier_util(self, stats_list: list[dict]) -> float | None:
        """Mean of available per-actor cached gpu_util% (None if no data)."""
        vals = [
            s.get("gpu_util") for s in stats_list
            if s.get("gpu_util") is not None
        ]
        return sum(vals) / len(vals) if vals else None

    def _control_step(self) -> None:
        """One controller tick: feedforward DOPT (from live `s` + T0/D0) +
        feedback trim (tier util imbalance) → set_direct_ratio; KV guard."""
        c = self.stats["control"]
        c["ticks"] += 1

        # Live ship rate — per-cascade by construction (the draft only sees
        # cascade requests). Aggregate the per-draft EMAs.
        ships = [
            s.get("ship_rate_ma") for s in self._draft_stats
            if s.get("ship_rate_ma") is not None
        ]
        if not ships:
            return  # not enough data yet — leave the split untouched
        s = sum(ships) / len(ships)
        c["ship_ma"] = s

        cur = self._direct_ratio if self._direct_ratio is not None else 0.5

        # Feedforward: the analytical optimum given the measured ship rate
        # and the configured solo ceilings. Without T0/D0 we fall back to
        # feedback-only around the current split.
        if self._control_T0 and self._control_D0:
            T0, D0 = self._control_T0, self._control_D0
            denom = T0 + D0 * s
            dopt = (T0 - D0 * (1.0 - s)) / denom if denom > 0 else cur
        else:
            dopt = cur
        c["dopt_ff"] = dopt

        # Feedback trim: draft busier than target → raise the direct
        # fraction (offload more straight to target), and vice versa.
        trim = 0.0
        du = self._tier_util(self._draft_stats)
        tu = self._tier_util(self._target_stats)
        if du is not None and tu is not None:
            trim = self._control_trim_gain * (du - tu) / 100.0
            trim = max(-self._control_max_trim,
                       min(self._control_max_trim, trim))
        c["trim"] = trim

        target = max(self._control_ratio_lo,
                     min(self._control_ratio_hi, dopt + trim))
        if abs(target - cur) >= self._control_deadband:
            self.set_direct_ratio(target)

        if self._control_kv_guard is not None:
            self._kv_throttle()

    def _kv_throttle(self) -> None:
        """Keep the target off the KV-preemption cliff (lever 1): if any
        target's kv_in_flight/kv_threshold exceeds the guard, ease target
        dispatch; relax when well below. No-op unless the target bucket was
        booted with a rate (target_engine_send_rps set)."""
        guard = self._control_kv_guard
        props = []
        for s in self._target_stats:
            kf, kt = s.get("kv_in_flight"), s.get("kv_threshold")
            if kf is not None and kt:
                props.append(kf / kt)
        if not props:
            return
        peak = max(props)
        cur = self._target_engine_send_rps
        if cur is None:
            return  # unlimited bucket — nothing to throttle
        if peak > guard:
            new = max(1.0, cur * 0.95)
        elif peak < guard * 0.8:
            new = cur * 1.05
        else:
            return
        self.set_engine_send_rps(target_rps=new)
        self.stats["control"]["target_throttled_rps"] = new

    async def _rate_match_loop(self) -> None:
        """rate-match credit dispatch (opt-in via rate_match=True).

        Every `rate_match_tick_s`, measure each engine's throughput from the
        completions the pumps already drained (zero extra RPC) and cap that
        engine's push-rate bucket at `throughput*(1+headroom)`, with a
        (tick+buffer)-second burst. Push tracks drain → the engine's queue
        stays ~buffer s deep (no KV overshoot), and because the target's budget
        is shared by the priority REGEN queue + fresh-direct, the DIRECT/CASCADE
        split and Λ self-balance as the workload (and s) drift — DOPT becomes an
        emergent property, no formula. We cap at the PEAK observed throughput so
        a cold-start tick can't lock the rate low; peak decays slowly so a
        genuinely harder workload still lowers the cap. Best-effort.
        """
        last_d = list(self._draft_completions)
        last_t = list(self._target_completions)
        peak_d = [0.0] * len(self._draft_buckets)
        peak_t = [0.0] * len(self._target_buckets)
        last_time = time.perf_counter()
        hr = 1.0 + self._rate_match_headroom
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._rate_match_tick_s)
            except asyncio.TimeoutError:
                pass
            if self._stop.is_set():
                break
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            if dt <= 0:
                continue
            diag = {"draft_rps": [], "target_rps": []}
            for i, b in enumerate(self._draft_buckets):
                if b is None:
                    continue
                tput = (self._draft_completions[i] - last_d[i]) / dt
                last_d[i] = self._draft_completions[i]
                peak_d[i] = max(tput, peak_d[i] * 0.9)  # slow decay
                if peak_d[i] > 0:
                    b.set_rate(peak_d[i] * hr)
                diag["draft_rps"].append(round(peak_d[i] * hr, 1))
            for i, b in enumerate(self._target_buckets):
                if b is None:
                    continue
                tput = (self._target_completions[i] - last_t[i]) / dt
                last_t[i] = self._target_completions[i]
                peak_t[i] = max(tput, peak_t[i] * 0.9)
                if peak_t[i] > 0:
                    b.set_rate(peak_t[i] * hr)
                diag["target_rps"].append(round(peak_t[i] * hr, 1))
            self.stats["rate_match"] = diag

    # ---------------- handlers ----------------

    async def _on_draft_finished(self, item: dict) -> None:
        rid = item.get("req_id")
        pending = self._pending.get(rid)
        if pending is None:
            return  # unknown / already responded

        # Decrement draft load — the draft is done with this request
        # regardless of where the verdict routes it next. Prefer the
        # draft_idx we stashed on submit; fall back to the pump's
        # source-tag if missing (shouldn't happen post-).
        draft_idx = pending.get("draft_idx", item.get("_draft_idx"))
        if draft_idx is not None and 0 <= draft_idx < len(self._draft_load):
            self._draft_load[draft_idx] = max(
                0, self._draft_load[draft_idx] - 1,
            )

        if "error" in item:
            self._respond_error(rid, "draft", item["error"])
            return

        pending["draft_response"] = item.get("text", "")
        # Stamp on the head clock — actor's completed_t is from a
        # different process/node so subtracting from arrival_t
        # (head-side) yields cross-clock garbage.
        pending["draft_completed_t"] = time.perf_counter()
        # stash draft actor-clock timestamps (separate clock
        # from target's) for engine-rate analysis.
        if "draft_admit_actor_t" in item:
            pending["draft_admit_actor_t"] = item["draft_admit_actor_t"]
        if "draft_finish_actor_t" in item:
            pending["draft_finish_actor_t"] = item["draft_finish_actor_t"]
        # Carry the inline self-eval score to the Response.
        # Stashed BEFORE the head_cascade=False early-respond branch so the
        # draft-solo self-eval arms (head off, force_draft_response) surface it.
        if "self_eval_score" in item:
            pending["self_eval_score"] = item.get("self_eval_score")
            pending["self_eval_method"] = item.get("self_eval_method")
            pending["self_eval_ms"] = item.get("self_eval_ms")

        # cascade_retesting: if the submitter explicitly opted out of
        # the cascade head (head_cascade=False), there's no verdict to
        # dispatch on — respond directly with the draft text. Used by
        # --cell draft_only_via_scheduler for the apples-to-apples
        # baseline against target_only_via_scheduler.
        # a scheduler-side gate takes precedence — when one is
        # configured we fall through to the gate logic below so the
        # head-less draft still gets a ship/escalate decision (from the
        # output-confidence or query signal instead of the fork head).
        # when the in-engine attn_pool head is in use,
        # item["head_decision"] already carries the verdict — also fall
        # Through to the head_decision routing path below.
        if (
            pending.get("head_cascade") is False
            and self._gate is None
            and not self._draft_in_engine_cascade_head
        ):
            self._on_target_finished({
                "req_id": rid,
                "verdict": "ACCEPT",
                "text": pending["draft_response"],
                "n_output_tokens": item.get("n_output_tokens"),
            })
            return

        # Cascade path: when the draft actor ran with head_cascade=True,
        # the fork emits head_decision ∈ {SHIP, REGEN} on the final
        # output. SHIP → respond directly with the draft text; REGEN →
        # forward to target.submit_regen with pre-encoded ViT payload
        # (the draft's target-ViT already ran on draft GPUs).
        head_decision = item.get("head_decision")
        # stash the raw head score/tau so it can ride on the Response
        # (probe diagnostics for picking a low-s tau).
        pending["head_score"] = item.get("head_decision_score")
        pending["head_tau"] = item.get("head_decision_tau")
        # force_draft_response overrides REGEN→SHIP so the head
        # still fires (cost is paid by the draft actor) but the verdict
        # never causes a target dispatch. Used by
        # --cell draft_only_with_head_via_scheduler to isolate pure
        # head-eval cost from REGEN-coordination cost.
        if pending.get("force_draft_response") and head_decision == "REGEN":
            head_decision = "SHIP"
        # Baseline ablations: when the fork head emitted no verdict
        # (head-less draft) and a scheduler-side gate is configured, derive
        # SHIP/REGEN from the gate's alternative signal (output-confidence
        # gate A, or query-only gate B). The cascade structure is identical
        # to the head cell — only the decision signal differs.
        if head_decision is None and self._gate is not None:
            try:
                head_decision = self._gate(pending, item)
            except Exception:
                # Conservative on gate error: escalate rather than ship
                # a possibly-wrong draft answer.
                head_decision = "REGEN"
            self.stats["n_gate_decisions"] = (
                self.stats.get("n_gate_decisions", 0) + 1
            )
        if head_decision == "SHIP":
            # Synthetic target-finished item: did NOT actually go
            # through a target, so don't tag a target_idx (the
            # decrement in _on_target_finished will then be a no-op).
            self._on_target_finished({
                "req_id": rid,
                "verdict": "ACCEPT",
                "text": pending["draft_response"],
                "n_output_tokens": item.get("n_output_tokens"),
            })
            return
        if head_decision == "REGEN":
            # REGEN re-enters the sorted buffer at −∞ so the
            # target dispatch loop picks it up next. routing_path is
            # already 'cascade' (stamped by the draft dispatch loop);
            # the target loop's setdefault preserves it so the verdict
            # surfaces as REGEN (not DIRECT_TARGET).
            await self._route_regen(rid)
            return

        # Forced-verdict path: skip the head entirely; sample a verdict
        # At the configured ACCEPT rate. ACCEPT ships the draft
        # directly; REGEN re-enters the buffer at −∞.
        if self._force_accept_rate is not None:
            if random.random() < self._force_accept_rate:
                self._on_target_finished({
                    "req_id": rid,
                    "verdict": "ACCEPT",
                    "text": pending["draft_response"],
                })
            else:
                await self._route_regen(rid)
            return

        # Fallthrough: head_decision is missing on a cascade request
        # (shouldn't happen with the current fork — head always emits
        # SHIP or REGEN). Treat as REGEN.
        await self._route_regen(rid)

    def _on_target_finished(self, item: dict) -> None:
        rid = item.get("req_id")
        pending = self._pending.pop(rid, None)
        if pending is None:
            return
        ev = self._client_events.pop(rid, None)
        if ev is None:
            return

        # Decrement target load only if we actually dispatched to a
        # target (cascade SHIP synthesizes a finished item without
        # ever touching a target).
        target_idx = pending.get("target_idx", item.get("_target_idx"))
        if (
            target_idx is not None
            and 0 <= target_idx < len(self._target_load)
            and "target_idx" in pending
        ):
            self._target_load[target_idx] = max(
                0, self._target_load[target_idx] - 1,
            )

        verdict = item.get("verdict", "ERROR")
        routing_path = pending.get("routing_path", "cascade")
        # Re-map verdict for direct-target requests: target reports
        # "REGEN" because it ran submit_regen, but routing_path tells
        # us the request never visited a draft. Keep "DIRECT_TARGET"
        # for downstream attribution.
        if routing_path == "direct_target" and verdict in ("REGEN", "ACCEPT"):
            verdict = "DIRECT_TARGET"

        # Head-clock stamp; ignore item["completed_t"] (actor clock).
        completed_t = time.perf_counter()
        draft_completed_t = pending.get("draft_completed_t",
                                        pending["arrival_t"])

        if verdict == "ACCEPT":
            self.stats["n_accept"] += 1
        elif verdict == "REGEN":
            self.stats["n_regen"] += 1
        elif verdict == "DIRECT_TARGET":
            self.stats["n_direct_target"] += 1
        else:
            self.stats["n_error"] += 1

        self._client_results[rid] = Response(
            request_id=rid,
            text=item.get("text", ""),
            verdict=verdict,
            arrival_t=pending["arrival_t"],
            draft_completed_t=draft_completed_t,
            completed_t=completed_t,
            n_output_tokens=item.get("n_output_tokens"),
            error=item.get("error") if verdict == "ERROR" else None,
            routing_path=routing_path,
            target_admit_actor_t=item.get("target_admit_actor_t"),
            target_finish_actor_t=item.get("target_finish_actor_t"),
            draft_admit_actor_t=pending.get("draft_admit_actor_t"),
            draft_finish_actor_t=pending.get("draft_finish_actor_t"),
            head_score=pending.get("head_score"),
            head_tau=pending.get("head_tau"),
            self_eval_score=pending.get("self_eval_score"),
            self_eval_method=pending.get("self_eval_method"),
            self_eval_ms=pending.get("self_eval_ms"),
        )
        ev.set()

    def _respond_error(self, rid: str, stage: str, error: str) -> None:
        pending = self._pending.pop(rid, None)
        if pending is None:
            return
        ev = self._client_events.pop(rid, None)
        if ev is None:
            return
        self.stats["n_error"] += 1
        now = time.perf_counter()
        self._client_results[rid] = Response(
            request_id=rid,
            text="",
            verdict="ERROR",
            arrival_t=pending["arrival_t"],
            draft_completed_t=pending.get("draft_completed_t", now),
            completed_t=now,
            error=f"[{stage}] {error}",
            routing_path=pending.get("routing_path", "cascade"),
        )
        ev.set()

    # ---------------- introspection ----------------

    def load_snapshot(self) -> dict:
        """Return a debug snapshot of scheduler-local load tracking."""
        return {
            "draft_load": list(self._draft_load),
            "target_load": list(self._target_load),
        }

    def occupancy_snapshot(self) -> dict:
        """A3: instantaneous occupancy-gate signals for time-series
        logging. Lets a sampler see WHICH gate arm binds under overload —
        in-flight (vs hwm·max) or target KV proximity (kv_in_flight/threshold) —
        the diagnostic the overload run lacked. Read-only; safe to call
        whether or not the gate is enabled."""
        maxn = self._occupancy_max_inflight or 1

        def _kvf(stats: dict):
            kf, kt = stats.get("kv_in_flight"), stats.get("kv_threshold")
            return round(kf / kt, 3) if (kt and kf is not None) else None

        return {
            # buffer depths so the arrival-loop sampler yields a
            # depth time series (buffer_max_depth alone hides oscillation).
            "fresh_q_depth": len(self._fresh_q),
            "regen_q_depth": len(self._regen_q),
            "legacy_buffer_depth": len(self._buffer),
            "draft_load": list(self._draft_load),
            "target_load": list(self._target_load),
            "draft_inflight_frac": [round(l / maxn, 3) for l in self._draft_load],
            "target_inflight_frac": [round(l / maxn, 3) for l in self._target_load],
            "draft_kv_frac": [_kvf(s) for s in self._draft_stats],
            "target_kv_frac": [_kvf(s) for s in self._target_stats],
            "draft_holding": list(self._draft_holding),
            "target_holding": list(self._target_holding),
            "occupancy_hwm": self._occupancy_hwm,
            "occupancy_kv_hwm": self._occupancy_kv_hwm,
            "occupancy_gate": self._occupancy_gate,
            # dispatch-RPC RTT EWMA + live batch size (rtt-aware fix
            # receipts).
            "target_rpc_rtt_ema_s": [round(x, 4)
                                     for x in self._target_rpc_rtt_ema],
            "target_batch_n_now": [self._target_batch_n(i)
                                   for i in range(len(self._targets))],
        }
