"""Single-run saturation finder.

Finds an engine's saturated output rate in ONE boot instead of
relaunching at many fixed λ. The bench drives an open-loop arrival
staircase (λ rising step by step) and feeds each step's measured
(offered λ, achieved completion rate, in-flight backlog trend) here.

The decision logic (`analyze_steps`) and the async staircase driver
(`run_staircase`) live here with NO Ray / vLLM imports, so the exact
code the bench ships is unit-tested against a synthetic server. The
bench injects a real
`submit_one`; the test injects a fake server with a known cap.

Saturation model. While λ ≤ capacity the engine keeps up: achieved ≈
offered and the in-flight set stays bounded. Past capacity, offered
arrivals outrun service: achieved flattens at the cap and the
in-flight backlog grows roughly linearly (open-loop has no back-
pressure). The knee is where achieved *decouples* from offered. The
reported cap is the best sustained achieved rate at or before the knee.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, asdict
from typing import Awaitable, Callable


@dataclass
class Step:
    """One staircase step's measurements (after its warmup sub-window)."""
    lam: float              # offered arrival rate (req/s)
    achieved_rps: float     # completion rate over the step's measure window
    mean_in_flight: float   # mean bench-side in-flight (submitted, not done)
    inflight_slope: float   # linear growth of in-flight (tasks/s) in window
    p50_ms: float           # median latency in the measure window
    n_completed: int = 0
    p90_ms: float = 0.0     # tail latency at the same operating point
    p99_ms: float = 0.0


def _slope(ts: list[float], ys: list[float]) -> float:
    """Least-squares slope of ys vs ts (0.0 if degenerate)."""
    n = len(ts)
    if n < 2:
        return 0.0
    mt = sum(ts) / n
    my = sum(ys) / n
    den = sum((t - mt) ** 2 for t in ts)
    if den == 0:
        return 0.0
    return sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / den


class StepAccumulator:
    """Collects raw samples for one staircase step and computes its Step.

    Shared by the bench's async run-loop and the offline simulation test
    so the metric math (achieved rate, backlog slope, p50) is identical
    in both. Only completions/samples inside the post-warmup measure
    window [measure_start, measure_end] count toward the metrics.
    """

    def __init__(self, lam: float):
        self.lam = lam
        self._completions: list[tuple[float, float]] = []   # (t, latency_ms)
        self._inflight: list[tuple[float, int]] = []        # (t, n_in_flight)

    def add_completion(self, t: float, latency_ms: float) -> None:
        self._completions.append((t, latency_ms))

    def add_inflight_sample(self, t: float, n: int) -> None:
        self._inflight.append((t, n))

    def finalize(self, measure_start: float, measure_end: float) -> Step:
        dur = max(measure_end - measure_start, 1e-9)
        comps = [(t, l) for (t, l) in self._completions
                 if measure_start <= t <= measure_end]
        lats = sorted(l for _, l in comps)
        p50 = lats[len(lats) // 2] if lats else 0.0
        p90 = lats[min(int(len(lats) * 0.90), len(lats) - 1)] if lats else 0.0
        p99 = lats[min(int(len(lats) * 0.99), len(lats) - 1)] if lats else 0.0
        infl = [(t, n) for (t, n) in self._inflight
                if measure_start <= t <= measure_end]
        mean_if = sum(n for _, n in infl) / len(infl) if infl else 0.0
        slope = _slope([t for t, _ in infl], [float(n) for _, n in infl])
        return Step(
            lam=self.lam, achieved_rps=len(comps) / dur,
            mean_in_flight=mean_if, inflight_slope=slope, p50_ms=p50,
            n_completed=len(comps), p90_ms=p90, p99_ms=p99,
        )


@dataclass
class SatVerdict:
    saturated: bool
    cap_rps: float | None        # best sustained achieved rate (the answer)
    knee_lam: float | None       # offered λ at which keep-up broke
    knee_step_idx: int | None
    reason: str
    steps: list[dict]            # per-step records + per-step classification


def _classify(
    step: Step,
    prev_achieved: float | None,
    keepup: float,
    gain_eps: float,
    slope_thresh: float,
    min_lam_for_slope: float = 0.0,
) -> tuple[bool, str]:
    """Is THIS step saturated? Returns (saturated, reason).

    Two independent saturation tells, OR'd:
      - keep-up broke: achieved/offered < `keepup` (engine can't drain
        what's offered), AND backlog is growing (`inflight_slope` >
        `slope_thresh`) — the open-loop divergence signature.
      - throughput plateau: achieved barely rose vs the previous step
        despite a higher offered λ (relative gain < `gain_eps`).
    """
    ratio = step.achieved_rps / step.lam if step.lam > 0 else 1.0
    backlog_growing = step.inflight_slope > slope_thresh
    if ratio < keepup and backlog_growing:
        return True, (
            f"keep-up broke: achieved/offered={ratio:.2f}<{keepup} "
            f"and backlog slope={step.inflight_slope:.2f}>{slope_thresh} tasks/s"
        )
    if prev_achieved is not None and prev_achieved > 0:
        gain = (step.achieved_rps - prev_achieved) / prev_achieved
        if gain < gain_eps and backlog_growing:
            return True, (
                f"throughput plateau: gain={gain*100:.1f}%<{gain_eps*100:.0f}% "
                f"vs prev with backlog growing (slope={step.inflight_slope:.2f})"
            )
    return False, "sustainable"


def analyze_steps(
    steps: list[Step],
    *,
    keepup: float = 0.92,
    gain_eps: float = 0.04,
    slope_thresh: float = 0.5,
    min_steps_before_call: int = 2,
) -> SatVerdict:
    """Classify a completed staircase and estimate the cap.

    `min_steps_before_call` guards against calling saturation on the
    very first ramp step (cold-start transients can look like a stall).
    The cap is the max achieved rate among steps classified sustainable
    up to and including the knee — i.e. the best rate the engine
    actually delivered while keeping up.
    """
    out: list[dict] = []
    prev_achieved: float | None = None
    knee_idx: int | None = None
    reason = "no saturation observed in the swept range"
    for i, s in enumerate(steps):
        sat, why = _classify(
            s, prev_achieved, keepup, gain_eps, slope_thresh
        )
        if i < min_steps_before_call:
            sat, why = False, "warmup step (below min_steps_before_call)"
        rec = asdict(s)
        rec["classification"] = "saturated" if sat else "sustainable"
        rec["why"] = why
        out.append(rec)
        if sat and knee_idx is None:
            knee_idx = i
            reason = f"saturation at step {i} (λ={s.lam:.1f}): {why}"
        prev_achieved = s.achieved_rps

    if knee_idx is None:
        # Never saturated in range — cap is at least the best achieved,
        # but it's a lower bound (engine had headroom at the top λ).
        cap = max((s.achieved_rps for s in steps), default=None)
        return SatVerdict(
            saturated=False, cap_rps=cap, knee_lam=None, knee_step_idx=None,
            reason=reason + " — cap is a LOWER BOUND (raise λ_max)", steps=out,
        )

    sustained = [
        s.achieved_rps
        for j, s in enumerate(steps[: knee_idx + 1])
        if out[j]["classification"] == "sustainable"
    ]
    cap = max(sustained) if sustained else steps[knee_idx].achieved_rps
    return SatVerdict(
        saturated=True, cap_rps=cap, knee_lam=steps[knee_idx].lam,
        knee_step_idx=knee_idx, reason=reason, steps=out,
    )


async def run_staircase(
    submit_one: Callable[[dict], Awaitable],
    next_record: Callable[[], dict],
    *,
    lam_start: float,
    lam_max: float,
    lam_step: float = 0.0,         # additive step; 0 → use geometric
    lam_mult: float = 1.3,         # geometric ratio when lam_step == 0
    step_dur_s: float = 30.0,
    warmup_frac: float = 0.4,      # discard this fraction of each step
    sample_hz: float = 5.0,        # in-flight sampling rate
    distribution: str = "poisson",
    keepup: float = 0.92,
    gain_eps: float = 0.04,
    slope_thresh: float = 0.5,
    min_steps_before_call: int = 2,
    confirm_saturated_steps: int = 1,
    on_step: Callable[[Step, SatVerdict], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> SatVerdict:
    """Drive an open-loop λ staircase until saturation, in ONE run.

    Submits records via `submit_one` (an async callable; exceptions are
    swallowed per-request so one failure doesn't abort the ramp) at a
    rising arrival rate. After each step's post-warmup window it builds a
    `Step` and re-runs `analyze_steps`; once the latest step reads
    saturated for `confirm_saturated_steps` consecutive steps (past
    `min_steps_before_call`), it stops and returns the verdict. The cap
    is `verdict.cap_rps`.

    Caller-injectable `clock` + `next_record` keep this testable against
    a synthetic server. In-flight = bench-side submitted-but-not-done.
    """
    in_flight: set[asyncio.Task] = set()
    steps: list[Step] = []
    cur: StepAccumulator | None = None

    def _on_done(task: asyncio.Task, submit_t: float) -> None:
        in_flight.discard(task)
        if cur is not None and not task.cancelled():
            cur.add_completion(clock(), (clock() - submit_t) * 1000.0)

    # geometric or additive ramp schedule
    lams: list[float] = []
    lam = lam_start
    while lam <= lam_max + 1e-9:
        lams.append(lam)
        lam = lam + lam_step if lam_step > 0 else lam * lam_mult

    verdict = SatVerdict(False, None, None, None, "not run", [])
    consecutive_sat = 0
    sample_dt = 1.0 / sample_hz

    for lam in lams:
        cur = StepAccumulator(lam)
        step_start = clock()
        measure_start = step_start + warmup_frac * step_dur_s
        next_sample = step_start
        next_arrival = step_start
        while True:
            now = clock()
            if now >= step_start + step_dur_s:
                break
            if now >= next_sample:
                cur.add_inflight_sample(now, len(in_flight))
                next_sample += sample_dt
            if now >= next_arrival:
                rec = next_record()
                submit_t = now
                t = asyncio.create_task(submit_one(rec))
                t.add_done_callback(lambda tk, st=submit_t: _on_done(tk, st))
                in_flight.add(t)
                gap = (random.expovariate(lam) if distribution == "poisson"
                       else 1.0 / lam)
                next_arrival += gap
            await asyncio.sleep(min(sample_dt, 0.01))

        step = cur.finalize(measure_start, step_start + step_dur_s)
        steps.append(step)
        verdict = analyze_steps(
            steps, keepup=keepup, gain_eps=gain_eps,
            slope_thresh=slope_thresh,
            min_steps_before_call=min_steps_before_call,
        )
        if on_step is not None:
            on_step(step, verdict)
        latest_sat = steps and verdict.steps[-1]["classification"] == "saturated"
        consecutive_sat = consecutive_sat + 1 if latest_sat else 0
        if consecutive_sat >= confirm_saturated_steps and verdict.saturated:
            break

    cur = None
    for t in in_flight:
        t.cancel()
    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)
    return verdict


# --------------------- closed-loop concurrency ramp ---------------------
# The open-loop λ ramp above measures the *bench-pipeline* cap: each
# arrival's submit path (e.g. per-request image decode) runs on the
# driver, so on image-heavy workloads the finder saturates on the driver
# long before the GPU does (2.3 r/s open-loop vs ~28 r/s GPU).
# The concurrency ramp keeps a fixed pool of `c` requests in flight and
# refills on completion — decode overlaps across slots, so the GPU
# becomes the bottleneck and the reported cap is the engine's real
# throughput ceiling. Saturation here is a throughput *plateau* as `c`
# rises (Little's law: past the knee, more concurrency only adds latency,
# not throughput) — there is no backlog-divergence signal (in-flight is
# pinned at `c` by construction).


def analyze_concurrency_steps(
    steps: list[Step],
    *,
    gain_eps: float = 0.05,
    min_steps_before_call: int = 1,
    latency_rise: float = 1.5,
) -> SatVerdict:
    """Classify a concurrency staircase. `Step.lam` holds the concurrency
    level `c` (not an arrival rate). Saturated at the first step whose
    throughput gain over the previous step falls below `gain_eps` (a
    plateau) **and** whose p50 has risen to ≥ `latency_rise` × the lowest
    p50 seen so far. Cap = max achieved throughput across the sweep.

    The `latency_rise` guard is the closed-loop knee signal (Little's law:
    once throughput is flat, more concurrency only inflates latency). It
    prevents the failure mode where a slow-warming low-c step shows
    a small step-to-step gain and gets misread as the plateau even though
    the engine is still in the cold, under-utilized part of the curve —
    there, throughput is flat-ish AND latency is still low. Set
    `latency_rise <= 0` to disable (pure throughput-plateau detection).
    """
    out: list[dict] = []
    prev: float | None = None
    knee: int | None = None
    min_p50: float | None = None
    reason = "throughput still rising across the swept concurrency range"
    for i, s in enumerate(steps):
        if s.p50_ms > 0:
            min_p50 = s.p50_ms if min_p50 is None else min(min_p50, s.p50_ms)
        sat, why = False, "throughput still rising"
        if i >= min_steps_before_call and prev is not None and prev > 0:
            gain = (s.achieved_rps - prev) / prev
            if gain < gain_eps:
                lat_ok = (
                    latency_rise <= 0
                    or not min_p50
                    or s.p50_ms >= latency_rise * min_p50
                )
                if lat_ok:
                    sat = True
                    why = (f"plateau: gain={gain*100:.1f}%<{gain_eps*100:.0f}% "
                           f"AND p50={s.p50_ms:.0f}ms≥{latency_rise:.1f}×"
                           f"{(min_p50 or 0):.0f} at c={s.lam:.0f}")
                else:
                    why = (f"flat throughput (gain={gain*100:.1f}%) but p50="
                           f"{s.p50_ms:.0f}ms<{latency_rise:.1f}×"
                           f"{(min_p50 or 0):.0f} — still warming, not the knee")
        rec = asdict(s)
        rec["classification"] = "saturated" if sat else "sustainable"
        rec["why"] = why
        out.append(rec)
        if sat and knee is None:
            knee = i
            reason = f"plateau at step {i} (c={s.lam:.0f}): {why}"
        prev = s.achieved_rps

    cap = max((s.achieved_rps for s in steps), default=None)
    if knee is None:
        return SatVerdict(
            saturated=False, cap_rps=cap, knee_lam=None, knee_step_idx=None,
            reason=reason + " — cap is a LOWER BOUND (raise c_max)", steps=out,
        )
    return SatVerdict(
        saturated=True, cap_rps=cap, knee_lam=steps[knee].lam,
        knee_step_idx=knee, reason=reason, steps=out,
    )


async def run_concurrency_staircase(
    submit_one: Callable[[dict], Awaitable],
    next_record: Callable[[], dict],
    *,
    c_start: int = 8,
    c_max: int = 512,
    c_step: int = 0,            # additive; 0 → geometric (c_mult)
    c_mult: float = 2.0,
    step_dur_s: float = 30.0,    # legacy/fallback: fixed-window step length
    warmup_frac: float = 0.4,    # legacy/fallback: discard-this-fraction
    gain_eps: float = 0.05,
    confirm_saturated_steps: int = 1,
    min_steps_before_call: int = 1,
    latency_rise: float = 1.5,
    # --- "wait for stable" knobs ---
    # When `stable_windows > 0` and `stable_tol > 0`, replace the fixed
    # warmup_frac measurement with a rolling-window stability detector:
    # the step finalizes once the last `stable_windows` non-overlapping
    # `stable_window_s` windows agree to within ±`stable_tol` of their
    # mean throughput. Bounded by [step_min_dur_s, step_max_dur_s].
    # `step_dur_s` (the legacy knob) is used as `step_max_dur_s` when
    # `step_max_dur_s is None`.
    stable_window_s: float = 0.0,        # 0 → disable wait-for-stable
    stable_windows: int = 0,             # # windows that must agree
    stable_tol: float = 0.10,            # relative spread allowed
    step_min_dur_s: float | None = None,
    step_max_dur_s: float | None = None,
    on_step: Callable[[Step, SatVerdict], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> SatVerdict:
    """Find the engine's GPU throughput cap in ONE boot via a closed-loop
    concurrency ramp. Holds exactly `c` requests in flight (refilling on
    each completion) for a step, measures achieved throughput over the
    post-warmup window, then raises `c` until throughput plateaus.

    Reuses the same `submit_one` as the bench's closed-loop sweep, so the
    cap matches a manual concurrency sweep — but in one boot.

    **Wait-for-stable mode** (enabled when `stable_window_s > 0` and
    `stable_windows >= 2`): instead of finalizing each step at a fixed
    `step_dur_s`, the driver waits inside the step until the last
    `stable_windows` non-overlapping `stable_window_s`-second windows
    agree (max−min divided by mean ≤ `stable_tol`). This fixes the
    failure where the autosat called saturation during the
    engine's slow per-c warmup on image-heavy workloads. The step is
    bounded by `[step_min_dur_s, step_max_dur_s]`. The Step.achieved_rps
    reported is the mean over the last `stable_windows` windows (the
    converged steady-state throughput), not a single fixed window.
    """
    levels: list[int] = []
    c = c_start
    while c <= c_max:
        levels.append(int(c))
        c = c + c_step if c_step > 0 else int(c * c_mult)

    steps: list[Step] = []
    verdict = SatVerdict(False, None, None, None, "not run", [])
    consecutive_sat = 0

    wait_for_stable = stable_window_s > 0 and stable_windows >= 2
    if step_max_dur_s is None:
        step_max_dur_s = step_dur_s
    if step_min_dur_s is None:
        step_min_dur_s = warmup_frac * step_dur_s if not wait_for_stable \
            else min(stable_window_s * stable_windows, step_max_dur_s)

    for c in levels:
        acc = StepAccumulator(float(c))
        step_start = clock()
        in_flight: set[asyncio.Task] = set()
        running = True

        def _launch() -> None:
            rec = next_record()
            submit_t = clock()
            task = asyncio.create_task(submit_one(rec))

            def _done(tk: asyncio.Task, st: float = submit_t) -> None:
                in_flight.discard(tk)
                if not tk.cancelled():
                    acc.add_completion(clock(), (clock() - st) * 1000.0)
                if running and clock() < step_start + step_max_dur_s:
                    _launch()      # refill to keep the pool at `c`

            task.add_done_callback(_done)
            in_flight.add(task)

        for _ in range(c):
            _launch()

        # Roll up completions into non-overlapping windows; the last
        # `stable_windows` of them must agree before we finalize the step.
        # We track windows by counting completions whose timestamp lands
        # in [step_start + k*W, step_start + (k+1)*W). 0 disables.
        window_throughputs: list[float] = []
        next_window_end = step_start + stable_window_s if wait_for_stable else 0.0
        stable_reached_at: float | None = None
        # Legacy mode (no wait_for_stable): measure from warmup_frac*step_dur_s
        # to step_max_dur_s, finalize at step_max_dur_s.
        legacy_measure_start = step_start + warmup_frac * step_max_dur_s

        while clock() < step_start + step_max_dur_s:
            now = clock()
            acc.add_inflight_sample(now, len(in_flight))
            if wait_for_stable and now >= next_window_end:
                w_start = next_window_end - stable_window_s
                w_end = next_window_end
                n_in_win = sum(
                    1 for (t, _) in acc._completions
                    if w_start <= t < w_end
                )
                window_throughputs.append(n_in_win / stable_window_s)
                next_window_end += stable_window_s
                # Check stability across the last `stable_windows` windows,
                # but only after `step_min_dur_s` has elapsed.
                if (
                    (now - step_start) >= step_min_dur_s
                    and len(window_throughputs) >= stable_windows
                ):
                    recent = window_throughputs[-stable_windows:]
                    mean = sum(recent) / len(recent)
                    if mean > 0:
                        spread = (max(recent) - min(recent)) / mean
                        if spread <= stable_tol:
                            stable_reached_at = now
                            break
            await asyncio.sleep(0.2)
        running = False

        # Choose measurement window:
        #   wait-for-stable: last `stable_windows` × `stable_window_s`
        #   legacy: warmup_frac to step_max_dur_s
        if wait_for_stable and stable_reached_at is not None:
            measure_end = stable_reached_at
            measure_start = stable_reached_at - stable_windows * stable_window_s
            why_extra = (
                f" (stable after {measure_end - step_start:.0f}s; "
                f"last {stable_windows}×{stable_window_s:.0f}s windows="
                f"{[f'{w:.2f}' for w in window_throughputs[-stable_windows:]]})"
            )
        elif wait_for_stable:
            # Timed out before stable — fall back to last `stable_windows` worth.
            measure_end = clock()
            measure_start = max(
                step_start + step_min_dur_s,
                measure_end - stable_windows * stable_window_s,
            )
            why_extra = (
                f" (max_dur={step_max_dur_s:.0f}s reached without stable; "
                f"window history={[f'{w:.2f}' for w in window_throughputs]})"
            )
        else:
            measure_start = legacy_measure_start
            measure_end = step_start + step_max_dur_s
            why_extra = ""

        step = acc.finalize(measure_start, measure_end)
        steps.append(step)
        verdict = analyze_concurrency_steps(
            steps, gain_eps=gain_eps,
            min_steps_before_call=min_steps_before_call,
            latency_rise=latency_rise,
        )
        # Append diagnostic to the latest step record.
        if verdict.steps and why_extra:
            verdict.steps[-1]["why"] = verdict.steps[-1]["why"] + why_extra
        if on_step is not None:
            on_step(step, verdict)

        for t in list(in_flight):
            t.cancel()
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

        latest_sat = verdict.steps[-1]["classification"] == "saturated"
        consecutive_sat = consecutive_sat + 1 if latest_sat else 0
        if consecutive_sat >= confirm_saturated_steps and verdict.saturated:
            break

    return verdict
