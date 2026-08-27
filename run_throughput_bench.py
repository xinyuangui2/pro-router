"""Cascade vs target_only throughput bench.

One unified harness for the three cells compares:

  --cell target_only       Boot target only (no draft, no V0Scheduler).
                           Every request → TargetEngineAsync.submit_regen
                           with raw image_path(s) → target runs its own ViT.
                           Equivalent to sys9_target_burst_bench --mode vit
                           but with the same record/load/output schema as
                           the cascade cells, so phase-B summary tables line
                           up across cells.

  --cell cascade_no_vit    Boot draft + target + V0Scheduler. On REGEN
                           the V0Scheduler forwards the raw image path(s)
                           to the target — target runs its own ViT.
                           (The cascade_with_vit variant — pre-encoded
                           image_embeds offloaded from draft to target —
                           was removed; the offload didn't pay off and
                           the dual code paths just added confusion.)

Load axes mirror sys9_target_burst_bench.py:

  --concurrencies 32,64,128       (closed-loop: c bounded in-flight)
  --arrival-rates 60,120,200      (open-loop: Poisson λ for --duration s)

Workload axes specified via record file + per-request flags:

  --records  path/to/records.json   (each record has prompt + images list)
  --images-per-record 1|4           (use 1st image dup'd N times)
  --max-tokens 256|2048|4096
  --ignore-eos                      (forces 7B/72B to keep generating to max_tokens)

Output: one JSON per (cell, c|λ) cell, plus a console summary table.
The schema includes requests_per_s, latency p50/p95/p99, SHIP rate
(cascade cells only), output-token stats, draft+target queue depth
samples, and the verdict breakdown.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
import uuid
from pathlib import Path

import ray


def _load_scorer_callable(spec: str, arg: str | None = None):
    """Load a `scorer(pending, item) -> float` from a dotted spec.

    `spec` is "module.path:factory". The factory is imported and called
    with `arg` (e.g. a checkpoint dir) when `--scorer-callable-arg` is set,
    else with no args; it must return the scorer callable. Used by the
    FrugalGPT gate to plug in a trained (query, answer) reliability scorer
    without baking model-loading into the bench. No baseline implementation
    ships with this repository; supply your own factory.
    """
    import importlib
    if ":" not in spec:
        raise ValueError(
            f"--scorer-callable must be 'module:factory', got {spec!r}"
        )
    mod_name, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    factory = getattr(mod, attr)
    scorer = factory(arg) if arg is not None else factory()
    if not callable(scorer):
        raise TypeError(
            f"{spec} did not return a callable scorer (got {type(scorer)})"
        )
    return scorer


def _load_records(
    path: str, limit: int | None,
    source_filter: str | None = None,
    allow_text: bool = False,
) -> list[dict]:
    # Auto-detect .jsonl (one JSON per line) vs .json (single JSON array).
    # joined records are JSONL / MileBench / outputs are JSON.
    if path.endswith(".jsonl"):
        with open(path) as _f:
            rows = [json.loads(_l) for _l in _f if _l.strip()]
    else:
        rows = json.load(open(path))
    if limit is not None:
        rows = rows[:limit]
    # Filter to records that have at least one image and a prompt.
    # --allow-text-records keeps image-less records (text benches —
    # MMLU/GSM8K/CoQA/TriviaQA ship "images": []).
    if allow_text:
        rows = [r for r in rows if r.get("prompt")]
    else:
        rows = [r for r in rows if r.get("images") and r.get("prompt")]
    # Source_filter narrows the records pool to the
    # listed sources (comma-separated). Useful for bit-identical
    # target-input cells.
    if source_filter:
        keep = set(s.strip() for s in source_filter.split(",") if s.strip())
        rows = [r for r in rows if r.get("source") in keep]
    if not rows:
        raise SystemExit(f"no usable records in {path}")
    return rows


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


# ----------------------------- target_only path -----------------------------


class _TargetOnlyRouter:
    """Pops finished items off TargetEngineAsync directly (no V0Scheduler)."""

    def __init__(self, target):
        self._target = target
        self._waiters: dict[str, asyncio.Future] = {}
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def register(self, rid: str) -> asyncio.Future:
        fut = asyncio.get_event_loop().create_future()
        self._waiters[rid] = fut
        return fut

    async def _loop(self):
        while not self._stop.is_set():
            try:
                items = await self._target.pop_finished.remote(
                    max_n=64, timeout_s=0.05,
                )
            except Exception:
                await asyncio.sleep(0.05)
                continue
            for it in items:
                rid = it.get("req_id")
                fut = self._waiters.pop(rid, None)
                if fut is not None and not fut.done():
                    fut.set_result(it)


async def _target_only_submit(
    target, router: _TargetOnlyRouter, sem: asyncio.Semaphore,
    rec: dict, max_tokens: int, ignore_eos: bool,
    images_per_record: int,
    embeds_cache: dict | None = None,
):
    """Submit one request to target. If embeds_cache is None, target runs
    its own ViT on raw image paths (the legacy "vit" mode). If embeds_cache
    is given, target consumes pre-encoded image_embeds + image_grid_thw
    via vLLM's public multi_modal_data path (no_vit mode)."""
    async with sem:
        rid = f"{rec['id']}__{uuid.uuid4().hex[:6]}"
        fut = router.register(rid)
        t0 = time.perf_counter()
        if embeds_cache is not None:
            entry = embeds_cache.get(rec["id"])
            if entry is None:
                return rec, RuntimeError(
                    f"no_vit: rid={rec['id']} missing in embeds cache"
                ), time.perf_counter() - t0
            try:
                await target.submit_decode.remote(
                    rid, rec["prompt"], max_tokens, None, ignore_eos,
                    None, None,
                    image_embeds=entry["image_embeds"],
                    image_grid_thw=entry["image_grid_thw"],
                )
            except Exception as e:
                return rec, e, time.perf_counter() - t0
        else:
            if images_per_record <= 0:
                image_paths = list(rec["images"])
            else:
                image_paths = [rec["images"][0]] * images_per_record
            try:
                await target.submit_decode.remote(
                    rid, rec["prompt"], max_tokens, None, ignore_eos,
                    image_paths[0] if len(image_paths) == 1 else None,
                    image_paths if len(image_paths) > 1 else None,
                )
            except Exception as e:
                return rec, e, time.perf_counter() - t0
        try:
            item = await asyncio.wait_for(fut, timeout=900.0)
        except asyncio.TimeoutError:
            return rec, None, time.perf_counter() - t0
        wall = time.perf_counter() - t0
        return rec, item, wall


# ----------------------------- draft_only path -----------------------------


class _DraftOnlyRouter:
    """Pops finished items off DraftEngineAsync directly (no V0Scheduler,
    no head_cascade — just 7B-VL generating answers)."""

    def __init__(self, draft):
        self._draft = draft
        self._waiters: dict[str, asyncio.Future] = {}
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def register(self, rid: str) -> asyncio.Future:
        fut = asyncio.get_event_loop().create_future()
        self._waiters[rid] = fut
        return fut

    async def _loop(self):
        while not self._stop.is_set():
            try:
                items = await self._draft.pop_finished.remote(
                    max_n=64, timeout_s=0.05,
                )
            except Exception:
                await asyncio.sleep(0.05)
                continue
            for it in items:
                rid = it.get("req_id")
                fut = self._waiters.pop(rid, None)
                if fut is not None and not fut.done():
                    fut.set_result(it)


async def _draft_only_submit(
    draft, router: _DraftOnlyRouter, sem: asyncio.Semaphore,
    rec: dict, max_tokens: int, ignore_eos: bool,
    images_per_record: int,
    head_cascade: bool = False,
):
    async with sem:
        rid = f"{rec['id']}__{uuid.uuid4().hex[:6]}"
        fut = router.register(rid)
        t0 = time.perf_counter()
        if images_per_record <= 0:
            image_paths = list(rec["images"])
        else:
            image_paths = [rec["images"][0]] * images_per_record
        try:
            await draft.submit.remote(
                rid, rec["prompt"], max_tokens, 0.0, ignore_eos,
                image_paths[0] if len(image_paths) == 1 else None,
                head_cascade,  # opt-in head firing on draft_only
                image_paths if len(image_paths) > 1 else None,
            )
        except Exception as e:
            return rec, e, time.perf_counter() - t0
        try:
            item = await asyncio.wait_for(fut, timeout=900.0)
        except asyncio.TimeoutError:
            return rec, None, time.perf_counter() - t0
        wall = time.perf_counter() - t0
        return rec, item, wall


# Draft_only with dedicated encoder pool (route image_paths
# through one encoder actor, get back image_embeds via Ray RPC, then
# submit (prompt, image_embeds) to the draft). The draft's ViT is
# skipped (it consumes the precomputed image_embeds directly).
class _RoundRobin:
    def __init__(self, n: int):
        self._n = n
        self._i = 0

    def next(self) -> int:
        idx = self._i
        self._i = (self._i + 1) % self._n
        return idx


async def _draft_only_pool_submit(
    draft, encoders: list, rr: "_RoundRobin",
    router: _DraftOnlyRouter, sem: asyncio.Semaphore,
    rec: dict, max_tokens: int, ignore_eos: bool,
    images_per_record: int,
):
    async with sem:
        rid = f"{rec['id']}__{uuid.uuid4().hex[:6]}"
        fut = router.register(rid)
        t0 = time.perf_counter()
        if images_per_record <= 0:
            image_paths = list(rec["images"])
        else:
            image_paths = [rec["images"][0]] * images_per_record
        # 1) route image_paths to one encoder; encoder returns CPU embeds
        enc = encoders[rr.next()]
        try:
            payload = await enc.encode_return.remote(rid, image_paths)
        except Exception as e:
            return rec, e, time.perf_counter() - t0
        # 2) submit (prompt, image_embeds) to draft, ViT skipped
        try:
            await draft.submit.remote(
                rid, rec["prompt"], max_tokens, 0.0, ignore_eos,
                None,  # image_path (unused; embeds path)
                False,  # head_cascade disabled
                None,  # image_paths (unused)
                image_embeds=payload["image_embeds"],
                image_grid_thw=payload["image_grid_thw"],
            )
        except Exception as e:
            return rec, e, time.perf_counter() - t0
        try:
            item = await asyncio.wait_for(fut, timeout=900.0)
        except asyncio.TimeoutError:
            return rec, None, time.perf_counter() - t0
        wall = time.perf_counter() - t0
        return rec, item, wall


# ----------------------------- cascade path -----------------------------


async def _cascade_submit(
    sched, sem: asyncio.Semaphore,
    rec: dict, max_tokens: int, ignore_eos: bool,
    images_per_record: int,
    force_direct: bool = False,
    force_cascade: bool = False,
    head_cascade: bool | None = None,
    force_draft_response: bool = False,
):
    async with sem:
        t0 = time.perf_counter()
        if images_per_record <= 0:
            image_paths = list(rec["images"])
        else:
            image_paths = [rec["images"][0]] * images_per_record
        try:
            r = await asyncio.wait_for(
                sched.submit(
                    prompt=rec["prompt"], max_tokens=max_tokens,
                    ignore_eos=ignore_eos,
                    image_paths=image_paths,
                    force_direct=force_direct,
                    force_cascade=force_cascade,
                    head_cascade=head_cascade,
                    force_draft_response=force_draft_response,
                    # Pre-router reads pending["source"] for
                    # per-source scoring. None when records lack a source.
                    source=rec.get("source"),
                ),
                timeout=900.0,
            )
        except asyncio.TimeoutError:
            return rec, None, time.perf_counter() - t0
        except Exception as e:
            return rec, e, time.perf_counter() - t0
        wall = time.perf_counter() - t0
        return rec, r, wall


# ----------------------------- closed-loop -----------------------------


async def _run_closed_loop(
    submit_one, records: list[dict], concurrency: int,
    out_path: Path, label: str, cell: str, max_tokens: int,
    ignore_eos: bool, images_per_record: int,
    capture_text: bool = False,
) -> dict:
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    tasks = [
        asyncio.create_task(
            submit_one(sem, r, max_tokens, ignore_eos, images_per_record)
        )
        for r in records
    ]
    results: list[dict] = []
    n_done = 0
    next_progress = max(50, len(records) // 4)
    for fut in asyncio.as_completed(tasks):
        rec, item, wall = await fut
        n_done += 1
        rec_id = rec.get("id")
        source = rec.get("source")
        if isinstance(item, Exception):
            results.append({"rid": rec_id, "source": source,
                            "error": f"{type(item).__name__}: {item}",
                            "wall_ms": wall * 1000.0})
        elif item is None:
            results.append({"rid": rec_id, "source": source,
                            "error": "timeout", "wall_ms": wall * 1000.0})
        elif isinstance(item, dict):
            # Legacy direct-router paths (target_only_no_vit, draft_only,
            # and target_only with --target-only-legacy-router).
            if item.get("error") or item.get("verdict") == "ERROR":
                results.append({"rid": rec_id, "source": source,
                                "wall_ms": wall * 1000.0,
                                "error": item.get("error") or "verdict=ERROR"})
            else:
                # when --draft-head-cascade is on, the draft
                # Actor returns item["head_decision"] ∈ {"SHIP","REGEN"};
                # map SHIP→ACCEPT and REGEN→REGEN so per-source ship
                # rates surface in the existing aggregation path.
                hd = item.get("head_decision")
                if hd == "SHIP":
                    _verdict = "ACCEPT"
                elif hd == "REGEN":
                    _verdict = "REGEN"
                else:
                    _verdict = ("REGEN" if cell.startswith("target_only")
                                else "DRAFT")
                rec_out = {
                    "rid": rec_id, "source": source, "wall_ms": wall * 1000.0,
                    "n_output_tokens": item.get("n_output_tokens"),
                    "finish_reason": item.get("finish_reason"),
                    "verdict": _verdict,
                }
                if hd is not None:
                    rec_out["head_decision"] = hd
                # capture the raw head score/tau so a probe yields the
                # Per-source score distribution (lets us set tau for a target
                # ship rate offline instead of re-booting to guess).
                if item.get("head_decision_score") is not None:
                    rec_out["head_score"] = item.get("head_decision_score")
                    rec_out["head_tau"] = item.get("head_decision_tau")
                if capture_text:
                    rec_out["text"] = item.get("text")
                results.append(rec_out)
        else:
            # Response dataclass — cascade path OR
            # target_only-through-scheduler.
            if item.verdict == "ERROR":
                results.append({"rid": rec_id, "source": source,
                                "wall_ms": wall * 1000.0,
                                "error": item.error or "verdict=ERROR"})
            else:
                rec_out = {
                    "rid": rec_id, "source": source,
                    "wall_ms": wall * 1000.0,
                    "verdict": item.verdict,
                    "draft_ms": item.draft_ms,
                    "target_ms": item.target_ms,
                    "n_output_tokens": len((item.text or "").split()),
                }
                if capture_text:
                    rec_out["text"] = item.text
                results.append(rec_out)
        if n_done >= next_progress:
            elapsed = time.perf_counter() - t0
            rate = n_done / elapsed if elapsed else 0
            eta = (len(records) - n_done) / rate if rate else 0
            print(f"  [{n_done:>4}/{len(records)}] c={concurrency} "
                  f"elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s",
                  flush=True)
            next_progress += max(50, len(records) // 4)
    wall = time.perf_counter() - t0
    return _summarize(results, label, cell, wall, max_tokens, ignore_eos,
                      images_per_record, out_path, concurrency=concurrency)


# ----------------------------- open-loop -----------------------------


async def _run_open_loop(
    submit_one, records: list[dict], arrival_rate: float,
    duration_s: float, distribution: str,
    out_path: Path, label: str, cell: str, max_tokens: int,
    ignore_eos: bool, images_per_record: int,
    capture_text: bool = False,
    burst_window_s: float = 0.0,
    drain_cap_s: float | None = None,
) -> dict:
    sem = asyncio.Semaphore(100000)  # uncapped
    completions: list[dict] = []
    in_flight: set[asyncio.Task] = set()

    async def _wrap(rec, arrival_t):
        try:
            out_rec, item, _ = await submit_one(
                sem, rec, max_tokens, ignore_eos, images_per_record,
            )
            completed_t = time.perf_counter()
            wall_ms = (completed_t - arrival_t) * 1000.0
            if isinstance(item, Exception):
                completions.append({
                    "rid": rec.get("id"), "source": rec.get("source"),
                    "wall_ms": wall_ms, "arrival_t": arrival_t,
                    "completed_t": completed_t,
                    "error": f"{type(item).__name__}: {item}",
                })
            elif item is None:
                completions.append({
                    "rid": rec.get("id"), "source": rec.get("source"),
                    "wall_ms": wall_ms, "arrival_t": arrival_t,
                    "completed_t": completed_t, "error": "timeout",
                })
            elif isinstance(item, dict):
                # Legacy direct-router paths.
                if item.get("error") or item.get("verdict") == "ERROR":
                    completions.append({
                        "rid": rec.get("id"), "source": rec.get("source"),
                        "wall_ms": wall_ms, "arrival_t": arrival_t,
                        "completed_t": completed_t,
                        "error": item.get("error") or "verdict=ERROR",
                    })
                else:
                    # when --draft-head-cascade is on, the draft
                    # Actor returns item["head_decision"] ∈ {"SHIP","REGEN"};
                    # map SHIP→ACCEPT and REGEN→REGEN so per-source ship
                    # rates surface in the open-loop aggregation too.
                    hd = item.get("head_decision")
                    if hd == "SHIP":
                        _verdict = "ACCEPT"
                    elif hd == "REGEN":
                        _verdict = "REGEN"
                    else:
                        _verdict = ("REGEN" if cell.startswith("target_only")
                                    else "DRAFT")
                    rec_out = {
                        "rid": rec.get("id"), "source": rec.get("source"),
                        "wall_ms": wall_ms, "arrival_t": arrival_t,
                        "completed_t": completed_t,
                        "n_output_tokens": item.get("n_output_tokens"),
                        "finish_reason": item.get("finish_reason"),
                        "verdict": _verdict,
                    }
                    if hd is not None:
                        rec_out["head_decision"] = hd
                    # capture raw head score/tau (open-loop path) so a
                    # Probe yields the per-source score distribution.
                    if item.get("head_decision_score") is not None:
                        rec_out["head_score"] = item.get("head_decision_score")
                        rec_out["head_tau"] = item.get("head_decision_tau")
                    if capture_text:
                        rec_out["text"] = item.get("text")
                    completions.append(rec_out)
            else:
                # Response dataclass — cascade or target_only-through-scheduler.
                if item.verdict == "ERROR":
                    completions.append({
                        "rid": rec.get("id"), "source": rec.get("source"),
                        "wall_ms": wall_ms, "arrival_t": arrival_t,
                        "completed_t": completed_t,
                        "error": item.error or "verdict=ERROR",
                    })
                else:
                    rec_out = {
                        "rid": rec.get("id"), "source": rec.get("source"),
                        "wall_ms": wall_ms, "arrival_t": arrival_t,
                        "completed_t": completed_t,
                        "verdict": item.verdict,
                        "draft_ms": item.draft_ms,
                        "target_ms": item.target_ms,
                        "n_output_tokens": len((item.text or "").split()),
                    }
                    # engine-side instrumentation: actor's
                    # perf_counter at _drive_regen start (admit) and at
                    # vLLM out.finished. Lets us compute admit-rate and
                    # finish-rate ON THE ACTOR'S CLOCK, free of any
                    # bench-side _target_pump polling artefacts.
                    if item.target_admit_actor_t is not None:
                        rec_out["target_admit_actor_t"] = item.target_admit_actor_t
                    if item.target_finish_actor_t is not None:
                        rec_out["target_finish_actor_t"] = item.target_finish_actor_t
                    if item.draft_admit_actor_t is not None:
                        rec_out["draft_admit_actor_t"] = item.draft_admit_actor_t
                    if item.draft_finish_actor_t is not None:
                        rec_out["draft_finish_actor_t"] = item.draft_finish_actor_t
                    # raw head score/tau (Response path) for probe diag.
                    if getattr(item, "head_score", None) is not None:
                        rec_out["head_score"] = item.head_score
                        rec_out["head_tau"] = item.head_tau
                    if capture_text:
                        rec_out["text"] = item.text
                    completions.append(rec_out)
        except Exception as e:
            completions.append({
                "rid": rec.get("id"), "source": rec.get("source"),
                "wall_ms": (time.perf_counter() - arrival_t) * 1000.0,
                "arrival_t": arrival_t,
                "completed_t": time.perf_counter(),
                "error": f"{type(e).__name__}: {e}",
            })

    burst_mode = burst_window_s > 0.0
    extra = f"  burst_window={burst_window_s*1000:.0f}ms" if burst_mode else ""
    print(f"[open-loop] start: λ={arrival_rate:.1f} req/s  dist={distribution}  "
          f"duration={duration_s:.0f}s{extra}", flush=True)
    t0 = time.perf_counter()
    n_records = len(records)
    i = 0
    deadline = t0 + duration_s
    next_progress = 5.0
    burst_buf: list[tuple[dict, float]] = []
    last_flush = t0
    n_bursts_fired = 0
    burst_sizes: list[int] = []

    async def _run_burst(coros):
        await asyncio.gather(*coros)

    def _flush_burst():
        nonlocal burst_buf, n_bursts_fired
        if not burst_buf:
            return
        # asyncio.gather of all accumulated wraps — they get scheduled
        # together so the underlying scheduler.submit_request calls
        # land on the engine in a tight cluster, mirroring the shape
        # of cascade's _draft_pump REGEN dispatch.
        burst_sizes.append(len(burst_buf))
        n_bursts_fired += 1
        coros = [_wrap(rec, at) for rec, at in burst_buf]
        burst_buf = []
        task = asyncio.create_task(_run_burst(coros))
        in_flight.add(task)
        task.add_done_callback(in_flight.discard)

    while True:
        now = time.perf_counter()
        if now >= deadline:
            break
        if distribution == "poisson":
            gap = random.expovariate(arrival_rate)
        else:
            gap = 1.0 / arrival_rate
        await asyncio.sleep(gap)
        rec = records[i % n_records]
        i += 1
        arrival_t = time.perf_counter()
        if burst_mode:
            burst_buf.append((rec, arrival_t))
            if arrival_t - last_flush >= burst_window_s:
                _flush_burst()
                last_flush = arrival_t
        else:
            task = asyncio.create_task(_wrap(rec, arrival_t))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)
        elapsed = arrival_t - t0
        if elapsed >= next_progress:
            print(f"  [t={elapsed:.0f}s] submitted={i} completed={len(completions)} "
                  f"in_flight={len(in_flight)}", flush=True)
            next_progress += 5.0
    if burst_mode and burst_buf:
        _flush_burst()
    if burst_mode:
        mean_b = sum(burst_sizes) / len(burst_sizes) if burst_sizes else 0
        print(f"[open-loop] burst stats: n_bursts={n_bursts_fired} "
              f"mean_size={mean_b:.1f} "
              f"max_size={max(burst_sizes) if burst_sizes else 0}", flush=True)
    # deep-overload cells build ~λ·duration backlogs whose drain adds
    # Up to 2×duration of wall time but contributes nothing to the in-window
    # metrics (post_ramp / rolling-30s). --ol-drain-cap-s bounds it; cancelled
    # stragglers vanish from completions (they are neither n_ok nor n_err) —
    # quote in-window percentiles for cross-arm latency comparisons.
    _drain_cap = 2 * duration_s if drain_cap_s is None else float(drain_cap_s)
    print(f"[open-loop] arrivals stopped at t={time.perf_counter() - t0:.1f}s "
          f"({i} submitted). Draining {len(in_flight)} in-flight "
          f"(cap={_drain_cap:.0f}s)…", flush=True)
    if in_flight:
        try:
            await asyncio.wait_for(
                asyncio.gather(*in_flight, return_exceptions=True),
                timeout=_drain_cap,
            )
        except asyncio.TimeoutError:
            print(f"[open-loop] drain capped; {len(in_flight)} still in flight, "
                  "cancelling and continuing", flush=True)
            for t in in_flight:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*in_flight, return_exceptions=True)
    drain_wall = time.perf_counter() - t0

    stats = _summarize(completions, label, cell, drain_wall, max_tokens,
                       ignore_eos, images_per_record, out_path,
                       arrival_rate=arrival_rate, distribution=distribution,
                       duration_s=duration_s, n_submitted=i)
    return stats


# ----------------------------- two-stream open-loop -----------------------------


async def _run_two_stream(
    submit_target, submit_cascade,
    records: list[dict],
    lambda_target: float, lambda_cascade: float,
    duration_s: float, distribution: str,
    out_dir: Path, label: str,
    max_tokens: int, ignore_eos: bool, images_per_record: int,
    capture_text: bool = False,
) -> dict:
    """Two independent Poisson streams. Stream T calls submit_target (which
    pins force_direct=True so V0Scheduler routes the req to target),
    Stream D calls submit_cascade (force_cascade=True). Both streams share
    the same V0Scheduler, so target's responses route through the
    scheduler's single _target_pump (no race against a separate router).

    Returns a combined stats dict containing per-stream sub-dicts plus
    overall totals. Per-stream cell JSONs are also written to out_dir
    so the existing analysis scripts can pick each up unchanged.
    """
    sem = asyncio.Semaphore(100000)
    completions_target: list[dict] = []
    completions_cascade: list[dict] = []
    in_flight: set[asyncio.Task] = set()

    def _build_completion(rec, item, arrival_t, completed_t, stream: str) -> dict:
        wall_ms = (completed_t - arrival_t) * 1000.0
        base = {
            "rid": rec.get("id"), "source": rec.get("source"),
            "wall_ms": wall_ms, "arrival_t": arrival_t,
            "completed_t": completed_t, "_stream": stream,
        }
        if isinstance(item, Exception):
            base["error"] = f"{type(item).__name__}: {item}"
            return base
        if item is None:
            base["error"] = "timeout"
            return base
        if isinstance(item, dict):
            if item.get("error") or item.get("verdict") == "ERROR":
                base["error"] = item.get("error") or "verdict=ERROR"
                return base
            base["n_output_tokens"] = item.get("n_output_tokens")
            base["finish_reason"] = item.get("finish_reason")
            base["verdict"] = "REGEN" if stream == "target" else "DRAFT"
            if capture_text:
                base["text"] = item.get("text")
            return base
        # Response dataclass — cascade verdict or DIRECT_TARGET.
        if item.verdict == "ERROR":
            base["error"] = item.error or "verdict=ERROR"
            return base
        base["verdict"] = item.verdict
        base["draft_ms"] = item.draft_ms
        base["target_ms"] = item.target_ms
        base["n_output_tokens"] = len((item.text or "").split())
        if capture_text:
            base["text"] = item.text
        return base

    async def _wrap(rec, arrival_t, stream: str):
        submit = submit_target if stream == "target" else submit_cascade
        store = completions_target if stream == "target" else completions_cascade
        try:
            out_rec, item, _ = await submit(
                sem, rec, max_tokens, ignore_eos, images_per_record,
            )
            store.append(_build_completion(
                rec, item, arrival_t, time.perf_counter(), stream,
            ))
        except Exception as e:
            store.append({
                "rid": rec.get("id"), "source": rec.get("source"),
                "wall_ms": (time.perf_counter() - arrival_t) * 1000.0,
                "arrival_t": arrival_t,
                "completed_t": time.perf_counter(),
                "_stream": stream,
                "error": f"{type(e).__name__}: {e}",
            })

    t0 = time.perf_counter()
    deadline = t0 + duration_s
    n_records = len(records)
    submitted = {"target": 0, "cascade": 0}
    next_progress = 5.0

    async def _stream_loop(rate: float, stream_name: str, offset: int):
        """Independent Poisson generator. `offset` staggers the two
        streams' record cursors so cascade and target don't fight for
        the same record at the same instant."""
        if rate <= 0:
            return
        i = offset
        while True:
            now = time.perf_counter()
            if now >= deadline:
                return
            if distribution == "poisson":
                gap = random.expovariate(rate)
            else:
                gap = 1.0 / rate
            await asyncio.sleep(gap)
            rec = records[i % n_records]
            i += 1
            arrival_t = time.perf_counter()
            submitted[stream_name] += 1
            task = asyncio.create_task(_wrap(rec, arrival_t, stream_name))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)

    async def _progress_pump():
        nonlocal next_progress
        while True:
            now = time.perf_counter()
            if now >= deadline:
                return
            elapsed = now - t0
            if elapsed >= next_progress:
                print(
                    f"  [t={elapsed:.0f}s] target_submitted="
                    f"{submitted['target']} cascade_submitted="
                    f"{submitted['cascade']} target_completed="
                    f"{len(completions_target)} cascade_completed="
                    f"{len(completions_cascade)} in_flight={len(in_flight)}",
                    flush=True,
                )
                next_progress += 5.0
            await asyncio.sleep(0.5)

    print(f"[two-stream] start: λ_target={lambda_target:.1f} λ_cascade="
          f"{lambda_cascade:.1f} dist={distribution} duration={duration_s:.0f}s",
          flush=True)
    target_loop = asyncio.create_task(_stream_loop(lambda_target, "target", 0))
    cascade_loop = asyncio.create_task(_stream_loop(lambda_cascade, "cascade", n_records // 2))
    progress = asyncio.create_task(_progress_pump())
    await asyncio.gather(target_loop, cascade_loop, progress, return_exceptions=True)

    print(
        f"[two-stream] arrivals stopped at t={time.perf_counter() - t0:.1f}s "
        f"(target={submitted['target']}, cascade={submitted['cascade']}). "
        f"Draining {len(in_flight)} in-flight (cap=2×duration)…",
        flush=True,
    )
    if in_flight:
        try:
            await asyncio.wait_for(
                asyncio.gather(*in_flight, return_exceptions=True),
                timeout=2 * duration_s,
            )
        except asyncio.TimeoutError:
            print(f"[two-stream] drain capped; {len(in_flight)} still in-flight",
                  flush=True)
            for t in in_flight:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*in_flight, return_exceptions=True)
    drain_wall = time.perf_counter() - t0

    # Per-stream cell JSONs.
    target_label = f"{label}_target_lam{lambda_target:g}"
    cascade_label = f"{label}_cascade_lam{lambda_cascade:g}"
    target_path = out_dir / f"{target_label}.json"
    cascade_path = out_dir / f"{cascade_label}.json"
    target_stats = _summarize(
        completions_target, target_label, "two_stream_target", drain_wall,
        max_tokens, ignore_eos, images_per_record, target_path,
        arrival_rate=lambda_target, distribution=distribution,
        duration_s=duration_s, n_submitted=submitted["target"],
    )
    cascade_stats = _summarize(
        completions_cascade, cascade_label, "two_stream_cascade", drain_wall,
        max_tokens, ignore_eos, images_per_record, cascade_path,
        arrival_rate=lambda_cascade, distribution=distribution,
        duration_s=duration_s, n_submitted=submitted["cascade"],
    )

    combined = {
        "label": label,
        "cell": "two_stream",
        "lambda_target": lambda_target,
        "lambda_cascade": lambda_cascade,
        "duration_s": duration_s,
        "distribution": distribution,
        "wall_s": drain_wall,
        "n_submitted_target": submitted["target"],
        "n_submitted_cascade": submitted["cascade"],
        "n_completed_target": len(completions_target),
        "n_completed_cascade": len(completions_cascade),
        "target_stats": target_stats,
        "cascade_stats": cascade_stats,
        "total_steady_rps": (
            target_stats.get("steady_rps", 0.0)
            + cascade_stats.get("steady_rps", 0.0)
        ),
    }
    return combined


# ----------------------------- summarize -----------------------------


def _compute_post_ramp_metric(
    results: list[dict], t0: float, duration_s: float, wall_s: float,
) -> dict:
    """Adaptive post-cache-warmup throughput estimator.

    Discovered in under graph_async + prefix caching + a workload
    with substantial image/prefix reuse, the per-second finish rate during
    the arrival window has a long ramp (the engine is admit-saturated until
    enough records have been seen for the prefix cache to start cushioning
    new admits). The bench's `lambda_steady_rps` over [60s, duration_s]
    averages the bad ramp with the good post-ramp regime, producing a number
    that is neither the engine's saturated capacity nor a meaningful steady
    rate.

    Algorithm:
      1. Bin completions into 1-s bins indexed against `t0` (the first
         arrival's perf_counter).
      2. Build a 30-s rolling-mean finish-rate series over the arrival window.
      3. Report the finish rate over the last third of the arrival window
         as `post_ramp_rps` (a saturation proxy that captures the post-
         cache-warmup regime when the system stabilized; degenerates to
         "best portion of the run" otherwise).
      4. Tag `post_ramp_stabilized` = True if the last 30-s rolling rate
         is within 90% of the max 30-s rolling rate AND the max occurred
         before the final 30 s — i.e. the system DID reach an asymptote
         during arrivals.

    Returns a dict that is merged into the stats JSON.
    """
    if not results or duration_s is None or duration_s <= 0:
        return {}

    # also bin by verdict so the post-ramp rate decomposes
    # Into target_contribution (REGEN + DIRECT_TARGET) vs
    # draft_contribution (ACCEPT) — those are the two halves of
    # Λ_cascade = target_contribution + draft_contribution. Both
    # sum to post_ramp_rps by construction.
    completed_pairs = [
        (r["completed_t"] - t0, (r.get("verdict") or "UNKNOWN"))
        for r in results
        if r.get("completed_t") is not None and not r.get("error")
    ]
    if not completed_pairs:
        return {}

    n_bins = int(duration_s)
    if n_bins < 30:
        return {}  # arrival window too short to characterize ramp

    bins = [0] * n_bins
    bins_target = [0] * n_bins
    bins_draft = [0] * n_bins
    for ct_rel, v in completed_pairs:
        if 0.0 <= ct_rel < duration_s:
            idx = int(ct_rel)
            bins[idx] += 1
            if v == "ACCEPT":
                bins_draft[idx] += 1
            elif v in ("REGEN", "DIRECT_TARGET"):
                bins_target[idx] += 1

    # Last-third-of-arrivals window as the post-ramp proxy.
    third = n_bins // 3
    post_ramp_start_s = float(n_bins - third)
    post_ramp_end_s = float(n_bins)
    post_ramp_n = sum(bins[n_bins - third:])
    post_ramp_rps = post_ramp_n / third
    post_ramp_target_rps = sum(bins_target[n_bins - third:]) / third
    post_ramp_draft_rps = sum(bins_draft[n_bins - third:]) / third

    # 30-s rolling-mean rate series for stabilization detection.
    win = 30
    rolling = []
    for i in range(n_bins - win + 1):
        rolling.append(sum(bins[i:i + win]) / win)
    max_rate = max(rolling) if rolling else 0.0
    max_idx = rolling.index(max_rate) if rolling else 0
    final_rate = rolling[-1] if rolling else 0.0
    stabilized = bool(
        max_rate > 0
        and final_rate >= 0.9 * max_rate
        and max_idx < len(rolling) - 1
    )

    # Heuristic ramp-end: first index where rolling rate reaches 90% of
    # the max and stays within 80% of max through end of arrivals.
    ramp_end_idx: int | None = None
    if max_rate > 0:
        thresh_hi = 0.9 * max_rate
        thresh_lo = 0.8 * max_rate
        for i in range(len(rolling)):
            if rolling[i] >= thresh_hi and all(
                r >= thresh_lo for r in rolling[i:]
            ):
                ramp_end_idx = i
                break

    return {
        # canonical throughput name — completed requests per second in
        # The steady window (last third of the ARRIVAL window; excludes cold
        # ramp and post-arrival drain). Alias of post_ramp_rps; only a capacity
        # reading when the cell is saturated (else it echoes the offered λ),
        # and can overshoot capacity when a cold-start backlog drains inside
        # the window — check post_ramp_stabilized + the rolling rates.
        "finished_items_per_second": post_ramp_rps,
        "post_ramp_rps": post_ramp_rps,
        "post_ramp_target_rps": post_ramp_target_rps,
        "post_ramp_draft_rps": post_ramp_draft_rps,
        "post_ramp_window_start_s": post_ramp_start_s,
        "post_ramp_window_end_s": post_ramp_end_s,
        "post_ramp_window_s": float(third),
        "rolling30s_max_rps": max_rate,
        "rolling30s_max_t_s": float(max_idx) + win / 2.0 if rolling else float("nan"),
        "rolling30s_final_rps": final_rate,
        "post_ramp_stabilized": stabilized,
        "ramp_end_t_s": (
            float(ramp_end_idx) + win / 2.0 if ramp_end_idx is not None
            else float("nan")
        ),
    }


def _summarize(
    results: list[dict], label: str, cell: str, wall_s: float,
    max_tokens: int, ignore_eos: bool, images_per_record: int,
    out_path: Path, *, concurrency: int | None = None,
    arrival_rate: float | None = None, distribution: str | None = None,
    duration_s: float | None = None, n_submitted: int | None = None,
) -> dict:
    ok = [r for r in results if not r.get("error")]
    err = [r for r in results if r.get("error")]
    lat_ms = [r["wall_ms"] for r in ok]
    tokens = sum((r.get("n_output_tokens") or 0) for r in ok)

    # Verdict breakdown
    verdicts = {}
    for r in ok:
        v = r.get("verdict") or "UNKNOWN"
        verdicts[v] = verdicts.get(v, 0) + 1
    n_accept = verdicts.get("ACCEPT", 0)
    n_regen  = verdicts.get("REGEN", 0)
    n_direct = verdicts.get("DIRECT_TARGET", 0)
    # contribution decomposition: Λ_cascade = target + draft.
    # target_contribution gathers everything target finished (REGEN
    # path + DIRECT bypass); draft_contribution is the ACCEPT
    # ship-rate. Together they sum to requests_per_s exactly.
    target_contribution_rps = (
        (n_regen + n_direct) / wall_s if wall_s > 0 else float("nan")
    )
    draft_contribution_rps  = (
        n_accept / wall_s if wall_s > 0 else float("nan")
    )
    # Two ship-rate metrics — they answer different questions and the
    # cluster-session confusion in came from conflating them:
    #
    #   ship_rate_per_cascade = ACCEPT / (ACCEPT + REGEN)
    #     The head's positive rate on cascade-routed requests. This is
    #     the `s` in the closed-form  d_opt = (T - D(1-s)) / (T + D·s).
    #     Independent of `direct_ratio`; depends only on head + workload.
    #
    #   ship_rate_per_total = ACCEPT / total_ok
    #     ACCEPT count diluted by DIRECT_TARGET. Drops as `direct_ratio`
    #     rises (more requests bypass the head entirely). Useful for the
    #     "what fraction of all requests SHIPPED" question, but NOT a
    #     substitute for `s` in d_opt.
    #
    # See CLAUDE.md "Bench metric definitions" for the disambiguation.
    cascade_count = n_accept + n_regen
    ship_rate_per_cascade = (n_accept / cascade_count) if cascade_count else float("nan")
    ship_rate_per_total   = (n_accept / len(ok)) if ok else float("nan")
    ship_rate = ship_rate_per_total  # legacy alias — use *_per_total or *_per_cascade in new code

    # Per-source verdict breakdown
    per_source = {}
    for r in ok:
        s = r.get("source") or "unknown"
        d = per_source.setdefault(s, {"n": 0, "accept": 0, "regen": 0})
        d["n"] += 1
        if r.get("verdict") == "ACCEPT":
            d["accept"] += 1
        elif r.get("verdict") == "REGEN":
            d["regen"] += 1

    stats: dict = {
        "label": label,
        "cell": cell,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
        "images_per_record": images_per_record,
        "wall_s": wall_s,
        "n_total": len(results),
        "n_ok": len(ok),
        "n_err": len(err),
        "total_output_tokens": tokens,
        "requests_per_s": len(ok) / wall_s if wall_s > 0 else float("nan"),
        "tokens_per_s": tokens / wall_s if wall_s > 0 else float("nan"),
        "p50_ms": _percentile(lat_ms, 0.50),
        "p95_ms": _percentile(lat_ms, 0.95),
        "p99_ms": _percentile(lat_ms, 0.99),
        "mean_ms": statistics.mean(lat_ms) if lat_ms else float("nan"),
        "verdicts": verdicts,
        "ship_rate": ship_rate,                       # legacy alias for ship_rate_per_total
        "ship_rate_per_total": ship_rate_per_total,   # ACCEPT / total_ok
        "ship_rate_per_cascade": ship_rate_per_cascade,  # ACCEPT / (ACCEPT + REGEN); the `s` in d_opt
        # cascade Λ decomposition. The two _rps fields sum to
        # Requests_per_s by construction (post-ramp versions live
        # under post_ramp_target_rps / post_ramp_draft_rps from the
        # post-ramp metric block). Compare these to the matched
        # `target_only` + `draft_only_with_head_via_scheduler` cell's
        # `post_ramp_rps` to compute the cross-engine interference gap.
        "target_contribution_rps": target_contribution_rps,  # (REGEN + DIRECT_TARGET) / wall
        "draft_contribution_rps":  draft_contribution_rps,   # ACCEPT / wall
        "per_source": per_source,
    }
    if concurrency is not None:
        stats["concurrency"] = concurrency
    if arrival_rate is not None:
        stats["arrival_rate"] = arrival_rate
        stats["distribution"] = distribution
        stats["duration_s"] = duration_s
        stats["n_submitted"] = n_submitted
        # Achieved rps within the arrival window (exclude drain)
        window_completions = [
            r for r in ok
            if r.get("arrival_t") is not None and duration_s is not None
            and r["arrival_t"] - results[0].get("arrival_t", r["arrival_t"]) <= duration_s
        ] if results else []
        # safer: count completions per duration window
        n_in_window = sum(
            1 for r in ok
            if r.get("completed_t") is not None and r.get("arrival_t") is not None
        )
        stats["achieved_rps"] = n_in_window / wall_s if wall_s > 0 else float("nan")
        # wall-averaged rps blends a transient (boot/warmup) head and
        # A draining tail with the actual steady-state — the cascade lam90
        # cell's 52.5 r/s headline came 50% from the 181 s drain. Λ_steady
        # excludes the first STEADY_WARMUP_S of arrivals and the post-arrival
        # drain, giving the rate the system actually sustains under load.
        STEADY_WARMUP_S = 60.0
        if (
            results and duration_s is not None
            and duration_s > STEADY_WARMUP_S
        ):
            t0 = min(
                (r["arrival_t"] for r in results if r.get("arrival_t") is not None),
                default=None,
            )
            if t0 is not None:
                n_steady = sum(
                    1 for r in ok
                    if r.get("completed_t") is not None
                    and STEADY_WARMUP_S
                    <= (r["completed_t"] - t0) <= duration_s
                )
                n_arrival = sum(
                    1 for r in ok
                    if r.get("completed_t") is not None
                    and (r["completed_t"] - t0) <= duration_s
                )
                n_drain = len(ok) - n_arrival
                drain_s = max(0.0, wall_s - duration_s)
                stats["lambda_steady_rps"] = (
                    n_steady / (duration_s - STEADY_WARMUP_S)
                )
                stats["lambda_arrival_window_rps"] = n_arrival / duration_s
                stats["lambda_drain_rps"] = (
                    n_drain / drain_s if drain_s > 0 else float("nan")
                )
                stats["steady_warmup_s"] = STEADY_WARMUP_S

                # adaptive post-cache-warmup throughput.
                # The bench-side completion-rate signal has a long
                # Cache-warming transient: under graph_async with prefix
                # caching ON, sustained finish-rate is initially close
                # to zero (the engine is admit-saturated, every step
                # spends most of its 2048-token budget on prefill chunks
                # of new arrivals — each unique image+prompt costs ~1000
                # prefill tokens cold). As the prefix cache populates
                # against the workload's unique-record pool, per-admit
                # prefill cost drops and finish rate ramps up an order
                # of magnitude. A λ=31.5 cell takes ~200 s of the 300 s
                # arrival window to stabilize; at λ=70 it never stabilizes
                # at all within 300 s.
                #
                # The static STEADY_WARMUP_S = 60 s heuristic averages
                # the bad ramp with the good steady-state and reports
                # ~16 r/s when the engine actually sustains ~36 r/s
                # post-ramp. The post_ramp_rps metric below uses the
                # last third of the arrival window as a saturation
                # proxy and tags whether the system stabilized.
                stats.update(_compute_post_ramp_metric(
                    results=results, t0=t0, duration_s=duration_s,
                    wall_s=wall_s,
                ))
    if err:
        stats["err_samples"] = err[:5]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    # Sidecar: per-request records for steady-state windowing + Phase E
    # quality grading. Only written for open-loop runs (closed-loop
    # records don't have arrival_t/completed_t).
    if arrival_rate is not None and results:
        records_path = out_path.with_name(out_path.stem + "_records.jsonl")
        with open(records_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        stats["records_path"] = str(records_path)
    steady_tag = (
        f" Λ_steady={stats['lambda_steady_rps']:.2f}"
        if "lambda_steady_rps" in stats else ""
    )
    # surface the cascade Λ decomposition (target_contribution
    # vs draft_contribution) in the console summary. Both fields are
    # Always emitted; for non-cascade cells (target_only with no draft
    # routing, or draft_only_with_head where target idle) one half is
    # ~0 and the other half ≈ requests_per_s.
    print(f"[{cell} {label}] n_ok={len(ok)} n_err={len(err)} "
          f"req/s={stats['requests_per_s']:.2f}{steady_tag} "
          f"tgt_contrib={target_contribution_rps:.2f} "
          f"drft_contrib={draft_contribution_rps:.2f} "
          f"tok/s={stats['tokens_per_s']:.1f} "
          f"p50={stats['p50_ms']:.0f}ms p99={stats['p99_ms']:.0f}ms "
          f"ship_per_cascade={ship_rate_per_cascade*100:.1f}% "
          f"ship_per_total={ship_rate_per_total*100:.1f}% "
          f"→ {out_path}", flush=True)
    if err:
        print(f"  [err] showing up to 3: {[e.get('error') for e in err[:3]]}", flush=True)
    return stats


# ----------------------------- amain -----------------------------


async def amain(args) -> int:
    actor_env = {
        "HF_HOME": args.hf_home,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # NFS-shared HF cache races (OSError 116 stale file
        # handle) when N drafts boot concurrently and all call
        # Snapshot_download() to validate the same model dir. Offline
        # mode short-circuits the metadata fetch — workers go straight
        # to the cached snapshot dir. Cache must be pre-populated.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_PAPER_EXPLORE_PATH": os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ),
    }
    if not ray.is_initialized():
        # A worker image without scikit-learn breaks the head loader: vLLM's
        # `_load_cascade_head` transitively imports
        # `prorouter.eval_classifier_head`, whose top-level
        # `from sklearn.metrics import ...` fails there. Inject the dep through
        # Ray's runtime_env so worker actors get it at boot. The first boot is
        # slow (~30 s pip install in a venv); later boots reuse the cache.
        #
        # working_dir is pinned to this repo. Left to a managed cluster's
        # default it can pick a directory that also slurps in a sibling vLLM
        # checkout (multi-GB) and trip Ray's 512 MB cap regardless of the
        # excludes, since the excludes are repo-relative. Pinning it makes the
        # excludes meaningful and the upload a few MB.
        repo_root = os.path.dirname(os.path.abspath(__file__))
        ray.init(address="auto", log_to_driver=True,
                 runtime_env={
                     "working_dir": repo_root,
                     "env_vars": actor_env,
                     "pip": ["scikit-learn"],
                     # Keep the working_dir upload under Ray's 512 MB cap:
                     # workers need the code, never the run outputs.
                     "excludes": [
                         ".git/**",
                         "runs/**",
                         "results/**",
                         "**/*.jsonl",
                         "**/*.pkl",
                         "**/*.npz",
                         "**/*.pt",
                     ],
                 })

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from prorouter.engine import DraftEngineAsync, TargetEngineAsync  # noqa: E402
    from prorouter.launcher import DRAFT_RESOURCE, TARGET_RESOURCE  # noqa: E402
    from prorouter.scheduler import V0Scheduler  # noqa: E402

    # optional draft encoder pool. Same module as bench;
    # the encoder lives on a separate g5.xlarge / 1-A10G node tagged
    # With ENCODER_RESOURCE. Defaults to DRAFT_RESOURCE so single-node
    # tests still place the encoder (will share a g5.12 with the
    # draft if there's no separate node — fine for smoke / local runs).
    ENCODER_RESOURCE = os.environ.get("ENCODER_RESOURCE", DRAFT_RESOURCE)

    cell = args.cell
    print(f"[sys11-bench] cell={cell}  records={args.records} "
          f"images_per_record={args.images_per_record} "
          f"max_tokens={args.max_tokens} ignore_eos={args.ignore_eos}")

    # When --images-per-record 0 (sentinel = honor rec['images'] verbatim),
    # derive limit_mm_per_prompt from the workload's max image count so
    # vLLM accepts the longest record.
    mm_image_limit = max(args.images_per_record, 1)
    if args.images_per_record <= 0:
        try:
            _rows = _load_records(args.records, args.limit, source_filter=args.source_filter, allow_text=getattr(args, "allow_text_records", False))
            mm_image_limit = max(
                1, max(len(r.get("images") or []) for r in _rows)
            )
        except SystemExit:
            mm_image_limit = 1
        print(f"[sys11-bench] images-per-record=auto, "
              f"limit_mm_per_prompt={{image: {mm_image_limit}}}")

    # ---- Optionally load embeds cache for no_vit / cached cells ----
    embeds_cache = None
    if cell == "target_only_no_vit":
        if not args.embeds_cache:
            print("ERROR: --cell target_only_no_vit requires --embeds-cache PATH")
            return 2
        import torch as _torch
        print(f"[sys11-bench] loading embeds cache {args.embeds_cache}…")
        embeds_cache = _torch.load(args.embeds_cache, map_location="cpu",
                                   weights_only=False)
        print(f"[sys11-bench] {len(embeds_cache)} pre-encoded rids in cache "
              f"(target_only_no_vit ships tensors via Ray pickle per request)")
    elif cell == "target_only_cached":
        # isolation bench. Cache is loaded INSIDE target actor
        # (post-boot, via target.init_embeds_cache.remote) so per-request
        # dispatch only passes cache_key — no 11 MB tensor pickle.
        if not args.embeds_cache:
            print("ERROR: --cell target_only_cached requires --embeds-cache PATH")
            return 2
    elif cell == "draft_only_cached":
        # isolation bench (mirror of on draft side). Cache
        # is loaded INSIDE draft actor; per-request dispatch passes cache_key.
        if not args.embeds_cache:
            print("ERROR: --cell draft_only_cached requires --embeds-cache PATH "
                  "(produced by sys9_preencode_embeds.py with the DRAFT model)")
            return 2
    elif cell == "draft_only_no_vit":
        # same cache required, but bench-side loaded
        # (per-request Ray-pickle ship of 11 MB tensor — the biased
        # Lower-bound mirror of target_only_no_vit).
        if not args.embeds_cache:
            print("ERROR: --cell draft_only_no_vit requires --embeds-cache PATH "
                  "(produced by sys9_preencode_embeds.py with the DRAFT model)")
            return 2

    # ---- Boot engines (only what the cell needs) ----
    t = time.perf_counter()
    target = None
    draft = None
    sched = None
    # --mock-target spawns a zero-GPU MockTargetActor that
    # satisfies V0Scheduler's "at least one target" requirement +
    # Periodic poll RPCs without booting a real 8xA100 target. Only
    # valid for force_cascade cells (draft_only_*_via_scheduler);
    # any real target call raises loudly. Enables style
    # draft-side benches on 2× g5.12 clusters with no p4d/8xA100 node.
    use_mock_target = bool(getattr(args, "mock_target", False))
    _mock_serve_cells = (
        "draft_only_via_scheduler",
        "draft_only_with_head_via_scheduler",
    )
    # cascade cells with mock-target need behavioural serving
    # so REGEN dispatches + DIRECT bypasses don't crash. The mock then
    # Synthesises REGEN-verdict items; the head's real ACCEPT/REGEN on
    # the cascade arm is untouched (head fires inside the draft actor).
    _mock_serve_decode = cell.startswith("cascade") or cell == "two_stream"
    if use_mock_target and cell not in _mock_serve_cells and not _mock_serve_decode:
        print(f"ERROR: --mock-target only valid for draft_only_via_scheduler / "
              f"draft_only_with_head_via_scheduler / cascade_* / two_stream "
              f"(cell={cell})")
        return 2
    if use_mock_target:
        from prorouter.mock_target import MockTargetActor  # noqa: E402
        if _mock_serve_decode:
            print("[sys11-bench] booting MockTargetActor (0 GPU, serve_decode=True "
                  "— synthesises REGEN-verdict items so cascade routing runs "
                  "end-to-end without a real target)")
            target = MockTargetActor.remote(serve_decode=True)
        else:
            print("[sys11-bench] booting MockTargetActor (0 GPU, satisfies "
                  "V0Scheduler init only)")
            target = MockTargetActor.remote()
    elif (
        cell in ("target_only", "target_only_no_vit", "target_only_cached")
        or cell.startswith("cascade")
        or cell == "two_stream"
        or cell == "draft_only_via_scheduler"
        or cell == "draft_only_with_head_via_scheduler"
    ):
        target_kwargs = dict(
            model_id=args.target_model,
            tensor_parallel_size=args.target_tp,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.target_mem_util,
            distributed_executor_backend="mp",
            limit_mm_per_prompt={"image": mm_image_limit},
        )
        if args.max_pixels is not None:
            target_kwargs["mm_processor_kwargs"] = {"max_pixels": args.max_pixels}
        # c18x9 — target-side twin of --draft-no-prefix-caching: in a
        # cascade the target serves the same cycled/duplicated prompts
        # (direct + REGEN), so its prefix cache inflates throughput the same way.
        if getattr(args, "target_no_prefix_caching", False):
            target_kwargs["enable_prefix_caching"] = False
        # expose _kv_pool_threshold so we can test whether
        # the default 200K is a binding constraint in cascade workloads.
        if args.target_kv_pool_threshold is not None:
            target_kwargs["kv_pool_threshold"] = args.target_kv_pool_threshold
        # MA-length KV gating (per-source p90 of observed output len).
        if getattr(args, "ma_length_gating", False):
            target_kwargs["ma_length_gating"] = True
        if getattr(args, "rpc_fake_latency_ms", 0.0):
            target_kwargs["rpc_fake_latency_ms"] = args.rpc_fake_latency_ms
        if getattr(args, "actor_self_admit", False):
            target_kwargs["actor_self_admit"] = True
            target_kwargs["actor_admit_interval_ms"] = args.actor_admit_interval_ms
            target_kwargs["actor_admit_max_inflight"] = args.actor_admit_max_inflight
        target = TargetEngineAsync.options(
            num_gpus=args.target_tp, resources={TARGET_RESOURCE: 1},
        ).remote(**target_kwargs)

    drafts: list = []
    if (
        cell in ("draft_only", "draft_only_cached", "draft_only_no_vit",
                 "draft_only_via_scheduler",
                 "draft_only_with_head_via_scheduler")
        or cell.startswith("cascade")
        or cell == "two_stream"
    ):
        draft_kwargs = dict(
            model_id=args.draft_model,
            tensor_parallel_size=args.draft_tp,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.draft_mem_util,
            distributed_executor_backend="mp",
            limit_mm_per_prompt={"image": mm_image_limit},
        )
        if args.max_pixels is not None:
            draft_kwargs["mm_processor_kwargs"] = {"max_pixels": args.max_pixels}
        # disable draft prefix caching at engine boot (records
        # cycle under stop-on-stable; identical repeated prompts must not be
        # served from the KV prefix cache, which would inflate throughput).
        if getattr(args, "draft_no_prefix_caching", False):
            draft_kwargs["enable_prefix_caching"] = False
        # inline self-eval baseline (P(True) / AutoMix), run
        # blocking on the draft engine after each generation. Default off.
        if getattr(args, "inline_self_eval", None):
            draft_kwargs["inline_self_eval"] = args.inline_self_eval
        # the draft_only_no_vit / draft_only_cached cells feed
        # vLLM pre-computed image_embeds. vLLM rejects those unless
        # enable_mm_embeds was set at AsyncEngineArgs time. added
        # This on the target side; this is the matching draft-side fix.
        if cell in ("draft_only_no_vit", "draft_only_cached"):
            draft_kwargs["enable_mm_embeds"] = True
        # draft_only with --n-encoders > 0 routes image_paths
        # through a dedicated encoder pool and submits image_embeds back
        # to the draft (Ray-RPC). Draft must accept embeds.
        if cell == "draft_only" and getattr(args, "n_encoders", 0) > 0:
            draft_kwargs["enable_mm_embeds"] = True
        # cascade cells with an encoder pool (--n-encoders > 0)
        # also feed pre-encoded image_embeds into the draft via the
        # V0Scheduler dispatch path. The draft needs enable_mm_embeds=True
        # for vLLM to accept those embeds.
        if cell.startswith("cascade") and getattr(args, "n_encoders", 0) > 0:
            draft_kwargs["enable_mm_embeds"] = True
        # expose vLLM's per-step token budget + per-step seq cap
        # to the bench so we can probe the prefill-vs-decode-in-one-step
        # trade-off. C3b confirmed throughput saturates at c=128 on
        # g5.12 TP=4 with default max_num_batched_tokens=2048; bumping
        # The budget should let one step pack more prefill alongside
        # decode and close the 35→46 r/s gap to the pure-decode ceiling.
        if getattr(args, "draft_max_num_batched_tokens", None) is not None:
            draft_kwargs["max_num_batched_tokens"] = args.draft_max_num_batched_tokens
        if getattr(args, "draft_max_num_seqs", None) is not None:
            draft_kwargs["max_num_seqs"] = args.draft_max_num_seqs
        if getattr(args, "draft_logprobs_mode", None):
            draft_kwargs["logprobs_mode"] = args.draft_logprobs_mode
        if getattr(args, "rpc_fake_latency_ms", 0.0):
            draft_kwargs["rpc_fake_latency_ms"] = args.rpc_fake_latency_ms
        if getattr(args, "actor_self_admit", False):
            draft_kwargs["actor_self_admit"] = True
            draft_kwargs["actor_admit_interval_ms"] = args.actor_admit_interval_ms
            draft_kwargs["actor_admit_max_inflight"] = args.actor_admit_max_inflight
            draft_kwargs["actor_admit_kv_threshold"] = args.actor_admit_kv_threshold
        if getattr(args, "in_engine_head_ckpt", None) and \
           getattr(args, "in_engine_head_tau", None):
            draft_kwargs["in_engine_cascade_head_ckpt"] = (
                args.in_engine_head_ckpt
            )
            draft_kwargs["in_engine_cascade_head_tau"] = (
                args.in_engine_head_tau
            )
        if (
            (cell.startswith("cascade") and cell != "cascade_gate")
            or cell == "two_stream"
            or cell == "draft_only_with_head_via_scheduler"
            or (cell == "draft_only"
                and getattr(args, "draft_head_cascade", False))
        ):
            draft_kwargs.update(
                head_cascade=True,
                head_checkpoint_path=args.head_ckpt,
                head_tau_table_path=args.tau,
                extract_hidden_states_layer=args.layer_from_end,
            )
            # explicitly choose the cascade engine mode.
            # DraftEngineAsync defaults to (eager=True, async=False) for
            # legacy compatibility. Flip via --draft-engine-mode.
            _enforce_eager, _async_sched = {
                "eager_baseline": (True, False),
                "graph_only":     (False, False),
                "async_only":     (True, True),
                "graph_async":    (False, True),
            }[args.draft_engine_mode]
            draft_kwargs["cascade_enforce_eager"] = _enforce_eager
            draft_kwargs["cascade_async_scheduling"] = _async_sched
            print(f"[sys11-bench] draft-engine-mode={args.draft_engine_mode} "
                  f"(eager={_enforce_eager}, async={_async_sched})")
        elif cell in ("draft_only_via_scheduler", "cascade_gate"):
            # the head-off via_scheduler + gate cells
            # need engine mode override for apples-to-apples with the
            # head-on cascade cell (eager+sync). Neither loads the fork
            # head (head_cascade=False at actor level), so the
            # Head_cascade-gated kwarg path above doesn't fire.
            #
            # cascade_gate: the draft runs head-less; the ship/
            # escalate decision comes from the scheduler-side gate
            # (output-confidence A or query B), NOT the fork head.
            # extract_hidden_states_layer is set so the engine installs
            # the same per-step extract buffer as the head cell (fairness:
            # both arms pay the extract install; only the head arm
            # additionally fires the classifier). No request opts into
            # hidden_states emission here, so it's a small per-step cost.
            # Set extract_hidden_states_layer to trigger engine.py's
            # mode-kwarg conditional, AND match step0/A's extract_buf
            # install overhead. The extract_buf write fires every
            # step but no request opts into hidden_states emission,
            # so it's a small per-step cost matching step0/A.
            _enforce_eager, _async_sched = {
                "eager_baseline": (True, False),
                "graph_only":     (False, False),
                "async_only":     (True, True),
                "graph_async":    (False, True),
            }[args.draft_engine_mode]
            draft_kwargs["extract_hidden_states_layer"] = args.layer_from_end
            draft_kwargs["cascade_enforce_eager"] = _enforce_eager
            draft_kwargs["cascade_async_scheduling"] = _async_sched
            print(f"[sys11-bench] draft-engine-mode={args.draft_engine_mode} "
                  f"(eager={_enforce_eager}, async={_async_sched}) — "
                  f"head-off via_scheduler cell")
        # two_stream: the cascade arm uses the encoder pool when
        # --n-encoders > 0, same as cascade_no_vit.
        if cell == "two_stream" and getattr(args, "n_encoders", 0) > 0:
            draft_kwargs["enable_mm_embeds"] = True
        # --n-drafts > 1 boots multiple draft actors, one per
        # A10G node. Ray's resource scheduler picks distinct nodes since
        # each g5.12xlarge node carries exactly 1 unit of DRAFT_RESOURCE.
        # draft_only cell keeps the legacy single-handle path.
        # Draft_only_via_scheduler cells also accept n_drafts>1
        # (V0Scheduler takes a list of drafts).
        n_drafts = (
            args.n_drafts
            if (cell.startswith("cascade") or cell == "two_stream"
                or cell in ("draft_only_via_scheduler",
                            "draft_only_with_head_via_scheduler"))
            else 1
        )
        # heterogeneous draft fleet (mixture cell). --draft-mixture
        # "llava:2,pixtral:2" boots a mixed pool sharing ONE Qwen-72B target.
        # The V0Scheduler is model-agnostic (round-robin / argmin-load over
        # generic .submit() handles), and each draft loads its own in-engine
        # Head, so per-family actors co-serve a single request stream cleanly.
        # ⚠ All families MUST consume the SAME on-disk image preprocessing
        # (one shared --records); the shared sorted buffer hands any request to
        # any draft, so the mixture is measured at one shared resolution
        # (LE-256, the Pixtral-compatible config) — NOT per-family native res.
        if getattr(args, "draft_mixture", None) and (
            cell.startswith("cascade")
            or cell == "draft_only_with_head_via_scheduler"
        ):
            import copy as _copy
            _MIX_MODEL = {
                "llava": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
                "pixtral": "mistral-community/pixtral-12b",
            }
            _MIX_MNBT = {"llava": None, "pixtral": 8192}
            hdir = args.mixture_heads_dir
            mix_manifest = []
            for tok in args.draft_mixture.split(","):
                fam, _, cnt = tok.strip().partition(":")
                fam = fam.strip(); cnt = int(cnt)
                if fam not in _MIX_MODEL:
                    raise ValueError(f"--draft-mixture: unknown family {fam!r}")
                ckpt = os.path.join(hdir, f"{fam}_GEN-ALL.pt")
                tau = os.path.join(hdir, f"{fam}_c18_tau.json")
                if not (os.path.exists(ckpt) and os.path.exists(tau)):
                    raise FileNotFoundError(
                        f"--draft-mixture {fam}: missing head/tau under {hdir}"
                    )
                fk = _copy.deepcopy(draft_kwargs)
                fk["model_id"] = _MIX_MODEL[fam]
                fk["in_engine_cascade_head_ckpt"] = ckpt
                fk["in_engine_cascade_head_tau"] = tau
                if _MIX_MNBT[fam] is not None:
                    fk["max_num_batched_tokens"] = _MIX_MNBT[fam]
                else:
                    fk.pop("max_num_batched_tokens", None)
                for _ in range(cnt):
                    drafts.append(DraftEngineAsync.options(
                        num_gpus=args.draft_tp, resources={DRAFT_RESOURCE: 1},
                    ).remote(**fk))
                mix_manifest.append(f"{fam}×{cnt}→{_MIX_MODEL[fam]}")
            print(f"[sys11-bench] draft-mixture: {'; '.join(mix_manifest)} "
                  f"(heads={hdir}); {len(drafts)} draft actor(s) total")
        else:
            for _ in range(n_drafts):
                drafts.append(DraftEngineAsync.options(
                    num_gpus=args.draft_tp, resources={DRAFT_RESOURCE: 1},
                ).remote(**draft_kwargs))
        draft = drafts[0]  # back-compat for draft_only path

    pings = []
    for d in drafts:
        pings.append(d.ping.remote())
    if target is not None:
        pings.append(target.ping.remote())
    print(f"[sys11-bench] waiting for {len(drafts)} draft(s) + "
          f"{1 if target is not None else 0} target to load "
          f"(this can take 3-6 min)…")
    ray.get(pings)
    print(f"[sys11-bench] engines up in {time.perf_counter() - t:.0f}s")

    # optional draft encoder pool. When --n-encoders > 0,
    # boot that many DraftVisionEncoder actors on ENCODER_RESOURCE
    # nodes and hand them to V0Scheduler. Per-request the scheduler
    # routes draft work through the pool (Ray-RPC encode → image_embeds
    # → draft.submit). measured +28-31 % draft steady r/s,
    # 0.4-0.6× p50 latency; this brings the same win to the cascade
    # path. Target / REGEN paths stay raw (target-side
    # offload is −26 %). N=0 → legacy behaviour (draft runs its own ViT).
    encoders: list = []
    if (
        getattr(args, "n_encoders", 0) > 0
        and (
            cell.startswith("cascade")
            or cell == "draft_only"
            or cell == "two_stream"
        )
    ):
        from prorouter.draft_vision import DraftVisionEncoder  # noqa: E402
        # ENCODER_RESOURCE=NONE colocates encoders on whatever
        # node has free GPUs (relies on num_gpus=1 alone). Needed when
        # Pattern A only has one A10G node free for encoders and we want
        # n_encoders > 1 — the per-node Ray tokens (provider:aws,
        # accelerator_shape:4xA10G, etc.) are all 1-per-node, so the
        # Resources={…:1} constraint deadlocks the 2nd encoder.
        use_resource_token = ENCODER_RESOURCE and ENCODER_RESOURCE.upper() != "NONE"
        scheduling_label = ENCODER_RESOURCE if use_resource_token else "NONE (colocate by num_gpus)"
        print(f"[sys11-bench] booting {args.n_encoders} encoder(s) on "
              f"{scheduling_label} (1 GPU each, model={args.draft_model})")
        for _ in range(args.n_encoders):
            opts = {"num_gpus": 1}
            if use_resource_token:
                opts["resources"] = {ENCODER_RESOURCE: 1}
            encoders.append(DraftVisionEncoder.options(**opts).remote(
                model_id=args.draft_model,
                dtype=args.dtype,
            ))
        ray.get([e.ping.remote() for e in encoders])
        print(f"[sys11-bench] {len(encoders)} encoder(s) ready")

    # target_only now defaults to going through V0Scheduler
    # (force_direct_target=True, drafts=[]) so it pays the same
    # scheduler + Ray-RPC overhead as the cascade cells. Opt out via
    # --target-only-legacy-router to use the old direct _TargetOnlyRouter.
    use_glue_for_target_only = (
        cell == "target_only" and not args.target_only_legacy_router
    )
    if (
        cell.startswith("cascade")
        or use_glue_for_target_only
        or cell == "two_stream"
        or cell == "draft_only_via_scheduler"
        or cell == "draft_only_with_head_via_scheduler"
    ):
        # build the scorer BEFORE the scheduler so the same
        # callable is used across all submits in the run. The scorer
        # maps a `pending` dict to a float; submit() inserts into the
        # sorted buffer and dispatchers pull from opposite ends. No
        # quantile threshold needed — the routing emerges from
        # consumption rates.
        scorer = None
        if args.scorer_model:
            # source-free content-based scorer. Mutually
            # exclusive with --scorer (which selects a per-source
            # ship-rate table requiring `pending["source"]`).
            if args.scorer:
                raise ValueError(
                    "--scorer-model and --scorer are mutually exclusive"
                )
            from prorouter.pre_router import model_scorer  # noqa: E402
            scorer = model_scorer(args.scorer_model)
            print(f"[sys11-bench] scorer_model={args.scorer_model} "
                  f"(source-free TF-IDF+LR)")
        elif args.scorer:
            from prorouter.pre_router import (  # noqa: E402
                BUILTIN_SOURCE_RATE_TABLES,
                per_source_scorer,
            )
            table = BUILTIN_SOURCE_RATE_TABLES.get(args.scorer)
            if table is None:
                raise ValueError(
                    f"unknown --scorer={args.scorer!r}; "
                    f"known: {sorted(BUILTIN_SOURCE_RATE_TABLES)}"
                )
            scorer = per_source_scorer(table)
            print(f"[sys11-bench] scorer={args.scorer} "
                  f"(per-source ship-rate table, n={len(table)})")
        # scheduler-side ship/escalate gate for the baseline
        # ablations (--cell cascade_gate). Replaces the fork's hidden-
        # state head with an alternative signal so we can A/B the signal
        # class on identical cascade/models/hardware.
        gate = None
        draft_logprobs = 0
        if getattr(args, "gate", None):
            from prorouter.gate import (  # noqa: E402
                answer_scorer_gate,
                output_confidence_gate,
                query_gate,
                transformer_seq_gate,
            )
            if args.gate == "output_conf":
                if args.gate_tau is None:
                    raise ValueError("--gate output_conf requires --gate-tau")
                gate = output_confidence_gate(args.gate_tau, stat=args.gate_stat)
                # Chosen-token stats need logprobs=1; the Gatekeeper-rule
                # stats (max-softmax / neg-entropy) need a top-k distribution.
                draft_logprobs = (
                    1 if args.gate_stat in ("mean_logprob", "min_logprob")
                    else args.gate_logprobs_k
                )
                _role = ("Gatekeeper-rule baseline (inference rule only, "
                         "no base fine-tune)"
                         if args.gate_stat in ("mean_max_prob", "neg_mean_entropy")
                         else "raw-logprob ablation A")
                print(f"[sys11-bench] gate=output_conf stat={args.gate_stat} "
                      f"tau={args.gate_tau} logprobs_k={draft_logprobs} "
                      f"({_role}; SHIP iff {args.gate_stat} >= tau)")
            elif args.gate == "frugalgpt":
                if args.gate_tau is None:
                    raise ValueError("--gate frugalgpt requires --gate-tau")
                if not args.scorer_callable:
                    raise ValueError(
                        "--gate frugalgpt requires --scorer-callable "
                        "module:fn (the trained (query,answer) scorer)"
                    )
                ascorer = _load_scorer_callable(
                    args.scorer_callable, args.scorer_callable_arg,
                )
                gate = answer_scorer_gate(ascorer, args.gate_tau)
                print(f"[sys11-bench] gate=frugalgpt tau={args.gate_tau} "
                      f"scorer_callable={args.scorer_callable} (FrugalGPT "
                      f"baseline — trained scorer on (query, draft answer); "
                      f"SHIP iff score >= tau)")
            elif args.gate == "query":
                if args.gate_tau is None:
                    raise ValueError("--gate query requires --gate-tau")
                if not args.scorer_model:
                    raise ValueError(
                        "--gate query requires --scorer-model (the "
                        "prompt classifier used as the gate)"
                    )
                if scorer is None:
                    raise ValueError("--gate query: scorer failed to build")
                gate = query_gate(scorer, args.gate_tau)
                # The query gate reuses the prompt scorer as the GATE; it
                # should not also drive buffer ordering (that would conflate
                # ranking with gating). Force-cascade routing in the
                # cascade_gate submit path already neutralizes ordering, but
                # we drop the scorer here so the buffer stays FIFO.
                scorer = None
                print(f"[sys11-bench] gate=query tau={args.gate_tau} "
                      f"(Ablation B — query-only router as gate; "
                      f"SHIP iff prompt-classifier P(SHIP) >= tau)")
            elif args.gate == "transformer_seq":
                # T P18 — chosen production decider.
                if not args.transformer_ckpt:
                    raise ValueError(
                        "--gate transformer_seq requires --transformer-ckpt"
                    )
                if not args.transformer_tau_table:
                    raise ValueError(
                        "--gate transformer_seq requires "
                        "--transformer-tau-table"
                    )
                import json as _json
                with open(args.transformer_tau_table) as _f:
                    _tau_table = _json.load(_f)
                gate = transformer_seq_gate(
                    ckpt_path=args.transformer_ckpt,
                    tau_table=_tau_table,
                    use_global_tau=args.transformer_use_global_tau,
                )
                # Transformer consumes per-token features which need
                # logprobs >= 2 from the draft (top-K distribution).
                draft_logprobs = max(args.gate_logprobs_k, 2)
                print(f"[sys11-bench] gate=transformer_seq "
                      f"ckpt={args.transformer_ckpt} "
                      f"tau_table={args.transformer_tau_table} "
                      f"use_global={args.transformer_use_global_tau} "
                      f"draft_logprobs={draft_logprobs} "
                      f"(P18 chosen production decider — per-token "
                      f"sequence model with per-source τ)")
            else:
                raise ValueError(f"unknown --gate={args.gate!r}")
        if getattr(args, "draft_logprobs_only", 0):
            if gate is not None:
                raise ValueError(
                    "--draft-logprobs-only is mutually exclusive with --gate"
                )
            if getattr(args, "in_engine_cascade_head", False):
                raise ValueError(
                    "--draft-logprobs-only is mutually exclusive with "
                    "--in-engine-cascade-head"
                )
            draft_logprobs = int(args.draft_logprobs_only)
            print(f"[sys11-bench] draft_logprobs_only={draft_logprobs} "
                  f"(logprobs-only config, no head, no gate)")
        # the scheduler-owned split's initial value. Set iff the
        # controller is on or --direct-ratio was given explicitly; otherwise
        # None → the scheduler keeps its old static behavior (default-off
        # invariant). The static --cascade-direct-ratio path is independent
        # of this and stays bench-sampled.
        _sys60_dr = getattr(args, "direct_ratio", None)
        _sys60_closed = bool(getattr(args, "closed_loop", False))
        _sys60_initial_direct_ratio = (
            _sys60_dr if _sys60_dr is not None
            else (0.5 if _sys60_closed else None)
        )
        sched = V0Scheduler(
            drafts=(drafts if not use_glue_for_target_only else []),
            targets=[target],
            encoders=(encoders or None),
            load_aware=not args.disable_load_aware,
            calibrate_every=args.calibrate_every,
            force_direct_target=use_glue_for_target_only,
            # sorted-buffer routing. scorer is the ranking
            # function; dispatchers pull top-of-buffer → draft and
            # bottom-of-buffer → target. No d_opt knob.
            scorer=scorer,
            # scheduler-side ship/escalate gate + logprob plumbing
            # for the output-confidence ablation. None/False on non-gate
            # cells (no behavior change).
            gate=gate,
            draft_logprobs=draft_logprobs,
            # opt the draft into the
            # lp-classifier-inline fork's per-token feature seq path.
            draft_emit_per_token_feature_seq=bool(
                getattr(args, "draft_emit_per_token_feature_seq", False)
            ),
            # opt the draft into the in-engine attn_pool
            # head. Each cascade-routed draft request returns
            # CompletionOutput.head_decision with the SHIP/REGEN verdict.
            draft_in_engine_cascade_head=bool(
                getattr(args, "in_engine_cascade_head", False)
            ),
            # Polling cadence for actor.pop_finished. Both sides
            # Independently configurable; raising reduces idle RPC
            # load at a small tail-latency cost under light load.
            draft_pop_timeout_s=args.draft_pop_timeout_ms / 1000.0,
            target_pop_timeout_s=args.target_pop_timeout_ms / 1000.0,
            # per-engine send rate limiters. Token-bucket
            # cap on the send pump in request-rate units; None
            # disables. Use case: match target's input rate between
            # target_only and cascade for fair contribution audit.
            draft_engine_send_rps=args.draft_engine_send_rps,
            target_engine_send_rps=args.target_engine_send_rps,
            # when set, scheduler appends one CSV line
            # per target dispatch for bit-identical-input sanity check.
            target_input_log_path=args.target_input_log,
            # closed-loop controller (all default to current static
            # behavior; pass through verbatim). When --closed-loop or
            # --direct-ratio is set, the scheduler owns the DIRECT/CASCADE
            # Split via its mutable _direct_ratio and the cascade cell submits
            # WITHOUT force_direct/force_cascade (see below).
            direct_ratio=_sys60_initial_direct_ratio,
            closed_loop=bool(getattr(args, "closed_loop", False)),
            control_tick_s=float(getattr(args, "control_tick_s", 1.0) or 1.0),
            control_T0=getattr(args, "control_t0", None),
            control_D0=getattr(args, "control_d0", None),
            control_trim_gain=float(
                getattr(args, "control_trim_gain", 0.15) or 0.15),
            control_kv_guard=getattr(args, "control_kv_guard", 0.92),
            # rate-match credit dispatch (per-engine push rate capped at
            # measured throughput; the split + Λ self-balance, no DOPT formula).
            rate_match=bool(getattr(args, "rate_match", False)),
            rate_match_tick_s=float(getattr(args, "rate_match_tick_s", 2.0) or 2.0),
            rate_match_buffer_s=float(getattr(args, "rate_match_buffer_s", 1.0) or 1.0),
            rate_match_headroom=float(getattr(args, "rate_match_headroom", 0.15) or 0.15),
            rate_match_init_rps=float(getattr(args, "rate_match_init_rps", 16.0) or 16.0),
            # occupancy-gated admission (direct "is the engine full?" gate).
            occupancy_gate=bool(getattr(args, "occupancy_gate", False)),
            occupancy_max_inflight=int(getattr(args, "occupancy_max_inflight", 256) or 256),
            occupancy_hwm=float(getattr(args, "occupancy_hwm", 0.90) or 0.90),
            occupancy_lwm=float(getattr(args, "occupancy_lwm", 0.70) or 0.70),
            occupancy_kv_hwm=float(getattr(args, "occupancy_kv_hwm", 0.90) or 0.90),
            # throughput-adaptive target batch dispatch.
            adaptive_batch=bool(getattr(args, "adaptive_batch", False)),
            adaptive_batch_window_s=float(
                getattr(args, "adaptive_batch_window_s", 0.05) or 0.05),
            adaptive_batch_min=int(getattr(args, "adaptive_batch_min", 1) or 1),
            adaptive_batch_max=int(getattr(args, "adaptive_batch_max", 16) or 16),
            # wire-latency fixes (default off).
            adaptive_batch_rtt_aware=bool(
                getattr(args, "adaptive_batch_rtt_aware", False)),
            adaptive_batch_buffer=int(
                getattr(args, "adaptive_batch_buffer", 2) or 2),
            draft_submit_pipeline=int(
                getattr(args, "draft_submit_pipeline", 0) or 0),
            # wire-latency fixes (default off / default 32).
            draft_submit_batch=int(
                getattr(args, "draft_submit_batch", 0) or 0),
            target_submit_pipeline=int(
                getattr(args, "target_submit_pipeline", 0) or 0),
            pop_max_n=int(getattr(args, "pop_max_n", 0) or 0),
            max_buffer_depth=int(getattr(args, "max_buffer_depth", 0) or 0),
            # Explicit two-queue dispatch for the rate-match / emergent /
            # occupancy pipelines (the clean fresh + priority-regen split).
            two_buffer=bool(getattr(args, "rate_match", False)
                            or getattr(args, "emergent_dispatch", False)
                            or getattr(args, "occupancy_gate", False)),
            # direct per-request RPC dispatch (ablation arm b).
            dispatch_direct_rpc=bool(getattr(args, "dispatch_direct_rpc", False)),
        )
        await sched.start()

    records = _load_records(args.records, args.limit, source_filter=args.source_filter, allow_text=getattr(args, "allow_text_records", False))
    print(f"[sys11-bench] {len(records)} records per load-point")

    # ---- Submit helper ----
    router = None
    if cell == "target_only" and use_glue_for_target_only:
        # route through V0Scheduler (force_direct_target=True).
        # Same submit path as cascade cells → same RPC + scheduler
        # overhead, so comparing r/s and $/M-req is apples-to-apples.
        async def submit_one(sem, rec, mt, ieos, ipr):
            return await _cascade_submit(sched, sem, rec, mt, ieos, ipr)
    elif cell == "target_only":
        # Legacy direct path (opt-in via --target-only-legacy-router).
        router = _TargetOnlyRouter(target)
        router.start()

        async def submit_one(sem, rec, mt, ieos, ipr):
            return await _target_only_submit(target, router, sem, rec, mt, ieos, ipr)
    elif cell == "target_only_no_vit":
        router = _TargetOnlyRouter(target)
        router.start()

        async def submit_one(sem, rec, mt, ieos, ipr):
            return await _target_only_submit(target, router, sem, rec, mt, ieos, ipr,
                                              embeds_cache=embeds_cache)
    elif cell == "target_only_cached":
        # isolation bench. Target actor already has the embeds
        # cache loaded; bench dispatches by rec_id only.
        #
        # --embeds-transfer-mode controls how the actor hands tensors
        # to vLLM:
        #   gpu_side_stream — actor side-streams CPU→GPU on cuda:0
        #     (requires `--target-mem-util` ≤ ~0.78 so the actor's
        #     per-request GPU copies don't OOM against vLLM's KV pool).
        #     This is the original pattern.
        #   cpu_pinned — actor passes pinned CPU tensors directly;
        #     vLLM does H2D inside its own worker process. No actor-side
        #     GPU footprint, so KV cap can stay at the default 0.85.
        #     Closer to production (NCCL-into-worker pattern).
        print(f"[sys11-bench] loading embeds cache into target actor "
              f"({args.embeds_cache}) — mode={args.embeds_transfer_mode}")
        cache_info = ray.get(
            target.init_embeds_cache.remote(
                args.embeds_cache,
                transfer_mode=args.embeds_transfer_mode,
            )
        )
        print(f"[sys11-bench] target actor cache: {cache_info}")
        router = _TargetOnlyRouter(target)
        router.start()

        async def submit_one(sem, rec, mt, ieos, ipr):
            async with sem:
                rid = f"{rec['id']}__{uuid.uuid4().hex[:6]}"
                fut = router.register(rid)
                t0 = time.perf_counter()
                try:
                    await target.submit_regen_by_cache_key.remote(
                        rid, rec["id"], rec["prompt"], mt, ieos,
                    )
                except Exception as e:
                    return rec, e, time.perf_counter() - t0
                try:
                    item = await asyncio.wait_for(fut, timeout=900.0)
                except asyncio.TimeoutError:
                    return rec, None, time.perf_counter() - t0
                return rec, item, time.perf_counter() - t0
    elif cell == "draft_only":
        router = _DraftOnlyRouter(draft)
        router.start()

        if encoders:
            # route image_paths through dedicated encoder pool,
            # then submit (prompt, image_embeds) to the draft. The draft's
            # in-engine ViT is skipped. Real encoder offload, not cached.
            print(f"[sys11-bench] draft_only with {len(encoders)} "
                  f"encoder pool — real offload mode")
            rr = _RoundRobin(len(encoders))

            async def submit_one(sem, rec, mt, ieos, ipr):
                return await _draft_only_pool_submit(
                    draft, encoders, rr, router, sem, rec, mt, ieos, ipr,
                )
        else:
            _hc = bool(getattr(args, "draft_head_cascade", False))
            async def submit_one(sem, rec, mt, ieos, ipr):
                return await _draft_only_submit(
                    draft, router, sem, rec, mt, ieos, ipr,
                    head_cascade=_hc,
                )
    elif cell == "draft_only_no_vit":
        # mirror of target_only_no_vit on the draft.
        # Bench loads cache and ships per-request 11 MB embed tensor
        # via Ray pickle to draft.submit() (biased lower bound — the
        # Ray serialization tax that diagnosed).
        if not args.embeds_cache:
            print("ERROR: --cell draft_only_no_vit requires --embeds-cache PATH")
            return 2
        if embeds_cache is None:
            import torch as _torch
            print(f"[sys11-bench] loading embeds cache {args.embeds_cache}…")
            embeds_cache = _torch.load(
                args.embeds_cache, map_location="cpu", weights_only=False,
            )
        router = _DraftOnlyRouter(draft)
        router.start()

        async def submit_one(sem, rec, mt, ieos, ipr):
            async with sem:
                rid = f"{rec['id']}__{uuid.uuid4().hex[:6]}"
                fut = router.register(rid)
                t0 = time.perf_counter()
                entry = embeds_cache.get(rec["id"])
                if entry is None:
                    return rec, RuntimeError(
                        f"no_vit: rid={rec['id']} missing in embeds cache"
                    ), time.perf_counter() - t0
                try:
                    await draft.submit.remote(
                        rid, rec["prompt"], mt, 0.0, ieos,
                        None,                  # image_path
                        False,                 # head_cascade
                        None,                  # image_paths
                        image_embeds=entry["image_embeds"],
                        image_grid_thw=entry["image_grid_thw"],
                    )
                except Exception as e:
                    return rec, e, time.perf_counter() - t0
                try:
                    item = await asyncio.wait_for(fut, timeout=900.0)
                except asyncio.TimeoutError:
                    return rec, None, time.perf_counter() - t0
                return rec, item, time.perf_counter() - t0
    elif cell == "draft_only_cached":
        # isolation bench. Draft actor has the embed cache; per-
        # request dispatch via cache_key only (no Ray pickle of the
        # 11 MB embed tensor per request). Mirrors target_only_cached
        # in shape but uses draft model's ViT embeds (different
        # hidden_dim — must be encoded with the draft model).
        transfer_mode = getattr(args, "embeds_transfer_mode", "cpu_pinned")
        print(f"[sys11-bench] loading embeds cache into draft actor "
              f"(mode={transfer_mode}, path={args.embeds_cache})")
        cache_info = ray.get(
            draft.init_embeds_cache.remote(args.embeds_cache, transfer_mode)
        )
        print(f"[sys11-bench] draft actor cache: {cache_info}")
        router = _DraftOnlyRouter(draft)
        router.start()

        async def submit_one(sem, rec, mt, ieos, ipr):
            async with sem:
                rid = f"{rec['id']}__{uuid.uuid4().hex[:6]}"
                fut = router.register(rid)
                t0 = time.perf_counter()
                try:
                    await draft.submit_by_cache_key.remote(
                        rid, rec["id"], rec["prompt"], mt, 0.0, ieos,
                        False,  # head_cascade disabled — plain generation
                    )
                except Exception as e:
                    return rec, e, time.perf_counter() - t0
                try:
                    item = await asyncio.wait_for(fut, timeout=900.0)
                except asyncio.TimeoutError:
                    return rec, None, time.perf_counter() - t0
                return rec, item, time.perf_counter() - t0
    elif cell == "draft_only_via_scheduler":
        # cascade_retesting: pure draft generation routed THROUGH
        # V0Scheduler so the bench-side overhead matches what
        # cascade pays. Apples-to-apples baseline against
        # target_only_via_scheduler (the existing target_only cell
        # without --target-only-legacy-router). head_cascade=False
        # → draft skips head firing entirely; no SHIP/REGEN verdict;
        # result returns through V0Scheduler.submit() as a plain draft
        # response (no target involvement).
        async def submit_one(sem, rec, mt, ieos, ipr):
            return await _cascade_submit(
                sched, sem, rec, mt, ieos, ipr,
                force_cascade=True,         # pin to draft path
                head_cascade=False,         # disable head firing
            )
    elif cell == "draft_only_with_head_via_scheduler":
        # same path as draft_only_via_scheduler but with the
        # cascade head firing inside the draft actor. REGEN verdict is
        # short-circuited to SHIP via force_draft_response so the head
        # cost is paid (hidden-state extract + classifier) but no
        # request ever reaches target. Pure head-eval cost baseline:
        # comparing against draft_only_via_scheduler isolates the head
        # overhead from REGEN-coordination overhead.
        async def submit_one(sem, rec, mt, ieos, ipr):
            return await _cascade_submit(
                sched, sem, rec, mt, ieos, ipr,
                force_cascade=True,             # pin to draft path
                head_cascade=True,              # fire the head
                force_draft_response=True,      # override REGEN→SHIP
            )
    elif cell == "cascade_gate":
        # baseline ablation: every request goes through the draft
        # (force_cascade=True), the draft runs head-less (head_cascade=
        # False), and the scheduler-side gate decides SHIP (return draft
        # answer) vs REGEN (escalate to target).
        #
        # optional --cascade-direct-ratio R in [0,1)
        # implements the co-saturation protocol — each request is
        # randomly routed DIRECT (force_direct=True, bypass draft, hit
        # target only) with probability R, else CASCADE (force_cascade=True,
        # gate decides ship/escalate). At R = DOPT = (T − D(1−s))/(T + D·s)
        # both engines co-saturate and `Λ_cascade ≈ T + s·D` (otherwise the
        # cascade is single-tier-bound and the formula doesn't apply).
        _direct_ratio_cg = float(getattr(args, "cascade_direct_ratio", 0.0) or 0.0)
        _sys60_closed = bool(getattr(args, "closed_loop", False))
        _sys60_dr = getattr(args, "direct_ratio", None)
        _sys60_emergent = bool(getattr(args, "emergent_dispatch", False))
        _sys60_rm = bool(getattr(args, "rate_match", False))
        _sys60_og = bool(getattr(args, "occupancy_gate", False))
        # direct-RPC also submits plain (scheduler owns routing).
        _sys60_direct_rpc = bool(getattr(args, "dispatch_direct_rpc", False))
        if (_sys60_closed or _sys60_dr is not None or _sys60_emergent
                or _sys60_rm or _sys60_og or _sys60_direct_rpc):
            # scheduler-owned split — submit every request plain (no
            # force_direct/force_cascade); the SCHEDULER decides routing. Three
            # sub-modes, all submit identically (no force):
            #  * controller (--closed-loop): mutable _direct_ratio tuned each
            #    tick from live s + tier util.
            #  * pinned (--direct-ratio R): runtime-mutable static split.
            #  * EMERGENT (--emergent-dispatch): _direct_ratio=None → no pinned
            #    split at all. The draft pulls fresh from the buffer top at its
            #    capacity; the target drains REGENs (re-buffered at −∞) first,
            #    then fresh, at its capacity. The DIRECT/CASCADE split + DOPT
            #    throughput EMERGE from backpressure — no formula, no measured
            #    T0/D0/s (the 2-buffer credit idea; −∞ end = the priority REGEN
            #    queue). The in-engine head still fires on draft-pulled requests.
            # Mutually exclusive with the static --cascade-direct-ratio path.
            if _direct_ratio_cg > 0.0:
                raise ValueError(
                    "--closed-loop / --direct-ratio / --emergent-dispatch "
                    "is mutually exclusive with the static --cascade-direct-ratio"
                )
            print(f"[sys11-bench] scheduler-owned split: "
                  f"closed_loop={_sys60_closed}, emergent={_sys60_emergent}, "
                  f"init direct_ratio={_sys60_initial_direct_ratio}")
            async def submit_one(sem, rec, mt, ieos, ipr):
                return await _cascade_submit(
                    sched, sem, rec, mt, ieos, ipr,
                    head_cascade=False,
                )
        elif _direct_ratio_cg <= 0.0:
            async def submit_one(sem, rec, mt, ieos, ipr):
                return await _cascade_submit(
                    sched, sem, rec, mt, ieos, ipr,
                    force_cascade=True,
                    head_cascade=False,
                )
        else:
            import random as _rnd
            _rng_cg = _rnd.Random(0)  # deterministic split for reproducibility
            print(f"[sys11-bench] cascade_gate co-sat mode: "
                  f"--cascade-direct-ratio={_direct_ratio_cg:.3f}")
            async def submit_one(sem, rec, mt, ieos, ipr):
                if _rng_cg.random() < _direct_ratio_cg:
                    return await _cascade_submit(
                        sched, sem, rec, mt, ieos, ipr,
                        force_direct=True,
                    )
                return await _cascade_submit(
                    sched, sem, rec, mt, ieos, ipr,
                    force_cascade=True,
                    head_cascade=False,
                )
    elif cell == "two_stream":
        # two independent Poisson streams share the same
        # V0Scheduler so target responses route through one consumer
        # (avoids racing _target_pump with a separate _TargetOnlyRouter).
        # Stream T pins force_direct=True so requests skip draft+encoder;
        # Stream D pins force_cascade=True so requests go through
        # encoder→draft→ACCEPT-or-REGEN regardless of gate state.
        async def submit_target_stream(sem, rec, mt, ieos, ipr):
            return await _cascade_submit(
                sched, sem, rec, mt, ieos, ipr, force_direct=True,
            )

        async def submit_cascade_stream(sem, rec, mt, ieos, ipr):
            return await _cascade_submit(
                sched, sem, rec, mt, ieos, ipr, force_cascade=True,
            )

        # Warmup callable: hit both streams equally so the cascade head +
        # encoder pool + target all see traffic before measurement begins.
        async def submit_one(sem, rec, mt, ieos, ipr):
            # Alternate streams during warmup based on record id hash.
            if hash(rec.get("id", "")) & 1:
                return await submit_target_stream(sem, rec, mt, ieos, ipr)
            return await submit_cascade_stream(sem, rec, mt, ieos, ipr)
    else:
        # per-source routing via force_direct /
        # force_cascade flags. Records in --target-sources go through
        # force_direct (score=-inf → target dispatcher only); records
        # in --cascade-sources go through force_cascade + force_draft_response
        # (score=+inf → draft dispatcher; head's REGEN verdict is
        # absorbed so target never sees a REGEN re-injection). This
        # lets target_only and cascade cells deliver the SAME source
        # multiset to target for bit-identical-input verification.
        target_sources_set = (
            {s.strip() for s in args.target_sources.split(",") if s.strip()}
            if args.target_sources else set()
        )
        cascade_sources_set = (
            {s.strip() for s in args.cascade_sources.split(",") if s.strip()}
            if args.cascade_sources else set()
        )
        async def submit_one(sem, rec, mt, ieos, ipr):
            source = rec.get("source")
            fd = source in target_sources_set
            fc = source in cascade_sources_set
            return await _cascade_submit(
                sched, sem, rec, mt, ieos, ipr,
                force_direct=fd,
                force_cascade=fc,
                force_draft_response=fc,
            )

    # ---- encoder-capacity guard ----
    if cell == "two_stream":
        if args.lambda_target <= 0 and args.lambda_draft <= 0:
            print("ERROR: --cell two_stream requires --lambda-target > 0 "
                  "and/or --lambda-draft > 0")
            return 2
        # Per-A10G ceiling from (B=1): ~12 r/s single-image.
        # Multi-image divides this by imgs_per_rec because each request
        # is one encoder pass per image.
        est_imgs = args.images_per_record if args.images_per_record > 0 else (
            sum(len(r.get("images", [])) for r in records) / max(1, len(records))
        )
        n_enc = max(0, getattr(args, "n_encoders", 0))
        if args.lambda_draft > 0 and n_enc > 0:
            cap = n_enc * 12.0 / max(1.0, float(est_imgs))
            threshold = 0.8 * cap
            print(f"[sys17o] encoder capacity check: n_encoders={n_enc}, "
                  f"est_imgs_per_rec={est_imgs:.2f}, "
                  f"cap≈{cap:.1f} r/s, threshold(0.8×cap)≈{threshold:.1f} r/s, "
                  f"lambda_draft={args.lambda_draft:.1f}")
            if args.lambda_draft > threshold and not args.allow_encoder_bound:
                print(
                    f"ERROR: lambda_draft={args.lambda_draft:.1f} exceeds 80%% of "
                    f"encoder capacity ({threshold:.1f} r/s). The cascade "
                    f"ceiling you measure would be encoder-bound, not "
                    f"draft-bound. Either (a) drop lambda_draft below "
                    f"{threshold:.1f}, (b) raise --n-encoders, or "
                    f"(c) pass --allow-encoder-bound to measure encoder-"
                    f"bound regimes intentionally."
                )
                return 2
        elif args.lambda_draft > 0 and n_enc == 0:
            print(f"[sys17o] cascade stream WITHOUT encoder pool "
                  f"(--n-encoders 0). Draft runs its own in-engine ViT; "
                  f"cascade ceiling will reflect that pre-baseline.")

    # ---- Warmup ----
    if args.warmup > 0:
        # cap warmup generations — llava/pixtral never hit EOS on some
        # text records (triviaqa hang), so an uncapped warmup can wedge
        # the whole bench before the sweep starts.
        warm_mt = (min(args.max_tokens, args.warmup_max_tokens)
                   if args.warmup_max_tokens > 0 else args.max_tokens)
        print(f"[sys11-bench] warmup {args.warmup} records "
              f"(max_tokens={warm_mt}, results discarded)…")
        warm_sem = asyncio.Semaphore(min(args.warmup, 8))
        await asyncio.gather(*[
            submit_one(warm_sem, r, warm_mt, args.ignore_eos,
                       args.images_per_record)
            for r in records[: args.warmup]
        ], return_exceptions=True)

    # ---- Sweep ----
    out_dir = Path(args.out_dir)
    all_stats = []
    # Closed-loop first (concurrencies), then open-loop (arrival rates).
    # Closed-loop is bounded by the in-flight cap, so it always finishes
    # in known wall time. Open-loop overloads more aggressively and
    # depends on having the saturated rate measured first.
    def _snapshot_sched_stats():
        """per-draft submit counts for fairness analysis.
        per-target dispatch counts, direct-target verdict count,
        calibration drift totals, plus the live load_snapshot()."""
        if sched is None:
            return None
        snap = {
            "n_submit_per_draft": list(sched.stats.get("n_submit_per_draft", [])),
            "n_submit_total": sched.stats.get("n_submit", 0),
            "n_accept": sched.stats.get("n_accept", 0),
            "n_regen": sched.stats.get("n_regen", 0),
            "n_gate_decisions": sched.stats.get("n_gate_decisions", 0),
            "n_direct_target": sched.stats.get("n_direct_target", 0),
            "n_dispatch_per_target": list(
                sched.stats.get("n_dispatch_per_target", [])
            ),
            "n_dispatch_per_encoder": list(
                sched.stats.get("n_dispatch_per_encoder", [])
            ),
            "calibration_calls": sched.stats.get("calibration_calls", 0),
            "calibration_total_abs_drift": sched.stats.get(
                "calibration_total_abs_drift", 0,
            ),
            "target_batch_flushes": sched.stats.get("target_batch_flushes", 0),
            "target_batch_items_flushed": sched.stats.get(
                "target_batch_items_flushed", 0,
            ),
            "n_buffer_inserts": sched.stats.get("n_buffer_inserts", 0),
            "n_buffer_via_top": sched.stats.get("n_buffer_via_top", 0),
            "n_buffer_via_bottom": sched.stats.get("n_buffer_via_bottom", 0),
            # prescorer score sums at the two fresh-q pop sites.
            "scorer_drafted_score_sum": sched.stats.get(
                "scorer_drafted_score_sum", 0.0),
            "scorer_drafted_score_n": sched.stats.get(
                "scorer_drafted_score_n", 0),
            "scorer_direct_score_sum": sched.stats.get(
                "scorer_direct_score_sum", 0.0),
            "scorer_direct_score_n": sched.stats.get(
                "scorer_direct_score_n", 0),
            "buffer_max_depth": sched.stats.get("buffer_max_depth", 0),
            # Controller diagnostics snapshot (inert dict when the
            # controller is off: enabled=False, ticks=0).
            "control": dict(sched.stats.get("control", {}) or {}),
            # rate-match per-engine push-rate diagnostic (None if off).
            "rate_match": (dict(sched.stats.get("rate_match"))
                           if sched.stats.get("rate_match") else None),
        }
        try:
            ls = sched.load_snapshot()
            snap["load_snapshot"] = {
                "draft_load": ls["draft_load"],
                "target_load": ls["target_load"],
            }
        except Exception:
            pass
        return snap

    if getattr(args, "probe_oneby", False):
        # per-request routing latency: probes submitted strictly one
        # At a time (fixed file order) over a steady closed-loop background
        # decode load from a DISJOINT record slice, so each probe's wall time
        # includes production-like contention. The paired per-record delta vs
        # the same family's A0 cell isolates the routing signal's added
        # latency. Background requests use the SAME submit path as probes —
        # the signal runs on the whole stream (the mechanism), which is
        # what makes the background completion rate a throughput echo.
        label = f"{args.label}_probe1by1"
        out_path = out_dir / f"{label}.json"
        probes = records[: args.probe_n]
        bg_recs = records[args.bg_start: args.bg_end]
        if args.bg_concurrency > 0:
            if not bg_recs:
                print(f"ERROR: --probe-oneby background slice "
                      f"[{args.bg_start}:{args.bg_end}) is empty "
                      f"(records file has {len(records)})")
                return 2
            if args.bg_start < len(probes):
                print("ERROR: --bg-start overlaps the probe prefix "
                      f"({args.bg_start} < {len(probes)})")
                return 2

        def _mk_probe_result(rec, item, wall, idx):
            out = {"probe_idx": idx, "rid": rec.get("id"),
                   "source": rec.get("source"), "wall_ms": wall * 1000.0}
            if isinstance(item, Exception):
                out["error"] = f"{type(item).__name__}: {item}"
            elif item is None:
                out["error"] = "timeout"
            elif isinstance(item, dict):
                if item.get("error") or item.get("verdict") == "ERROR":
                    out["error"] = item.get("error") or "verdict=ERROR"
                else:
                    out["n_output_tokens"] = item.get("n_output_tokens")
                    out["finish_reason"] = item.get("finish_reason")
                    if item.get("head_decision") is not None:
                        out["head_decision"] = item.get("head_decision")
            else:
                if item.verdict == "ERROR":
                    out["error"] = item.error or "verdict=ERROR"
                else:
                    out["verdict"] = item.verdict
                    out["draft_ms"] = item.draft_ms
                    out["target_ms"] = item.target_ms
                    out["n_output_tokens"] = len((item.text or "").split())
                    if getattr(item, "head_score", None) is not None:
                        out["head_score"] = item.head_score
                    if getattr(item, "self_eval_score", None) is not None:
                        out["self_eval_score"] = item.self_eval_score
                        out["self_eval_method"] = item.self_eval_method
                    if getattr(item, "self_eval_ms", None) is not None:
                        out["self_eval_ms"] = item.self_eval_ms
            return out

        bg_stop = asyncio.Event()
        bg_counts = {"done": 0, "err": 0}
        bg_log: list[dict] = []
        bg_sem = asyncio.Semaphore(10 ** 7)
        bg_t0 = time.perf_counter()

        async def _bg_worker(wid: int):
            i = wid
            n_bg = len(bg_recs)
            while not bg_stop.is_set():
                rec = bg_recs[i % n_bg]
                i += args.bg_concurrency
                try:
                    _, item, wall = await submit_one(
                        bg_sem, rec, args.max_tokens, args.ignore_eos,
                        args.images_per_record)
                except Exception as e:
                    item, wall = e, 0.0
                err = (isinstance(item, Exception) or item is None
                       or (isinstance(item, dict) and item.get("error"))
                       or (not isinstance(item, (dict, type(None)))
                           and getattr(item, "verdict", None) == "ERROR"))
                bg_counts["err" if err else "done"] += 1
                bg_log.append({
                    "rid": rec.get("id"), "wall_ms": wall * 1000.0,
                    "t": round(time.perf_counter() - bg_t0, 3),
                    "err": bool(err)})

        bg_tasks = []
        if args.bg_concurrency > 0:
            print(f"[probe-1by1] background load: c={args.bg_concurrency} "
                  f"over records [{args.bg_start}:{args.bg_end}) cycled; "
                  f"settling {args.bg_settle_s:.0f}s…", flush=True)
            bg_tasks = [asyncio.create_task(_bg_worker(w))
                        for w in range(args.bg_concurrency)]
            await asyncio.sleep(args.bg_settle_s)
            print(f"[probe-1by1] settle done: bg completed="
                  f"{bg_counts['done']} err={bg_counts['err']}", flush=True)
        else:
            print("[probe-1by1] IDLE control: no background load", flush=True)

        probe_sem = asyncio.Semaphore(1)
        probe_results: list[dict] = []
        bg_done_at_start = bg_counts["done"]
        probes_t0 = time.perf_counter()
        for idx, rec in enumerate(probes):
            _, item, wall = await submit_one(
                probe_sem, rec, args.max_tokens, args.ignore_eos,
                args.images_per_record)
            r = _mk_probe_result(rec, item, wall, idx)
            r["t"] = round(time.perf_counter() - probes_t0, 3)
            probe_results.append(r)
            if (idx + 1) % 25 == 0:
                el = time.perf_counter() - probes_t0
                walls_so_far = [x["wall_ms"] for x in probe_results
                                if "error" not in x]
                p50 = _percentile(walls_so_far, 0.50) if walls_so_far else 0
                print(f"  [probe {idx + 1:>4}/{len(probes)}] "
                      f"elapsed={el:.0f}s p50={p50:.0f}ms "
                      f"bg_done={bg_counts['done']} bg_err={bg_counts['err']}",
                      flush=True)
        probes_wall_s = time.perf_counter() - probes_t0
        bg_done_during = bg_counts["done"] - bg_done_at_start

        if bg_tasks:
            bg_stop.set()
            done_set, pending_set = await asyncio.wait(bg_tasks, timeout=120.0)
            for t in pending_set:
                t.cancel()
            await asyncio.gather(*bg_tasks, return_exceptions=True)

        probes_jsonl = out_dir / f"{label}_probes.jsonl"
        with open(probes_jsonl, "w") as f:
            for r in probe_results:
                f.write(json.dumps(r) + "\n")
        bg_jsonl = out_dir / f"{label}_bg.jsonl"
        with open(bg_jsonl, "w") as f:
            for r in bg_log:
                f.write(json.dumps(r) + "\n")

        walls = [r["wall_ms"] for r in probe_results if "error" not in r]
        n_err = len(probe_results) - len(walls)
        se_walls = [r["self_eval_ms"] for r in probe_results
                    if r.get("self_eval_ms") is not None]
        stats = {
            "label": label, "cell": cell, "mode": "probe_oneby",
            "n_probes": len(probe_results), "n_ok": len(walls),
            "n_err": n_err,
            "probe_wall_s": probes_wall_s,
            "mean_ms": (sum(walls) / len(walls)) if walls else None,
            "p50_ms": _percentile(walls, 0.50) if walls else None,
            "p95_ms": _percentile(walls, 0.95) if walls else None,
            "p99_ms": _percentile(walls, 0.99) if walls else None,
            "min_ms": min(walls) if walls else None,
            "max_ms": max(walls) if walls else None,
            "self_eval_ms_mean": (sum(se_walls) / len(se_walls))
                                 if se_walls else None,
            "probe_n_requested": args.probe_n,
            "max_tokens": args.max_tokens,
            "bg": {
                "concurrency": args.bg_concurrency,
                "records_slice": [args.bg_start, args.bg_end],
                "settle_s": args.bg_settle_s,
                "completed_total": bg_counts["done"],
                "errors_total": bg_counts["err"],
                "completed_during_probes": bg_done_during,
                "rate_during_probes_rps":
                    (bg_done_during / probes_wall_s) if probes_wall_s else None,
            },
            "probes_jsonl": str(probes_jsonl),
            "bg_jsonl": str(bg_jsonl),
        }
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        all_stats.append(stats)
        _mean = stats["mean_ms"] or 0
        _p50 = stats["p50_ms"] or 0
        _p99 = stats["p99_ms"] or 0
        print(f"\n[probe-1by1] ===> n={len(walls)}/{len(probe_results)} "
              f"mean={_mean:.1f}ms p50={_p50:.1f}ms p99={_p99:.1f}ms "
              f"n_err={n_err} bg_rate={stats['bg']['rate_during_probes_rps'] or 0:.2f} r/s\n",
              flush=True)

    if args.auto_saturate:
        # single-boot saturation finder. Ramp until the knee, then
        # report the cap — no relaunching at fixed λ / fixed c.
        import itertools
        from prorouter.autosat import run_staircase, run_concurrency_staircase

        label = f"{args.label}_autosat"
        out_path = out_dir / f"{label}.json"
        sat_sem = asyncio.Semaphore(10 ** 7)  # uncapped (driver-side)
        pool = list(records)
        random.shuffle(pool)
        rec_cycle = itertools.cycle(pool)

        # tally head_decision verdicts both aggregate
        # AND per-c-step. The per-step list lets us A/B the scorer's
        # Contribution to ship_rate at the same c (i.e., same engine
        # utilization) — the "is draft throughput invariant and ship
        # rate the thing that moves" sanity check.
        sat_verdicts = {"ship": 0, "regen": 0, "error": 0, "other": 0}
        # Per-step counters; index advances in `_on_step`.
        sat_per_step = [{"ship": 0, "regen": 0, "error": 0, "other": 0}]
        sat_step_idx = [0]

        def _tally(item):
            cur = sat_per_step[sat_step_idx[0]]
            try:
                if isinstance(item, Exception) or item is None:
                    sat_verdicts["error"] += 1
                    cur["error"] += 1
                elif isinstance(item, dict):
                    hd = item.get("head_decision")
                    bucket = "ship" if hd == "SHIP" else (
                        "regen" if hd == "REGEN" else "other"
                    )
                    sat_verdicts[bucket] += 1
                    cur[bucket] += 1
                else:
                    v = getattr(item, "verdict", None)
                    bucket = "ship" if v == "ACCEPT" else (
                        "regen" if v == "REGEN" else "other"
                    )
                    sat_verdicts[bucket] += 1
                    cur[bucket] += 1
            except Exception:
                sat_verdicts["other"] += 1
                cur["other"] += 1

        async def _sat_submit(rec):
            r = await _unpack_submit_one(rec)
            return r

        async def _unpack_submit_one(rec):
            rec_out, item, _wall = await submit_one(
                sat_sem, rec, args.max_tokens, args.ignore_eos,
                args.images_per_record,
            )
            _tally(item)
            return rec_out

        # sample the controller diagnostics throughout the ramp so the
        # cell JSON carries the direct_ratio / ship_ma trajectory (cold-start →
        # settled). Inert (empty) on non-controller runs. Best-effort: never
        # interferes with the staircase.
        _ctl_series: list[dict] = []
        async def _ctl_sampler():
            t_start = time.perf_counter()
            while True:
                try:
                    c = sched.stats.get("control", {}) if sched else {}
                    _ctl_series.append({
                        "t": round(time.perf_counter() - t_start, 1),
                        "ticks": c.get("ticks"),
                        "direct_ratio": c.get("direct_ratio"),
                        "ship_ma": c.get("ship_ma"),
                        "dopt_ff": c.get("dopt_ff"),
                        "trim": c.get("trim"),
                        "target_throttled_rps": c.get("target_throttled_rps"),
                    })
                except Exception:
                    pass
                await asyncio.sleep(2.0)
        _ctl_task = (
            asyncio.create_task(_ctl_sampler())
            if (sched is not None and getattr(args, "closed_loop", False))
            else None
        )

        if args.sat_mode == "concurrency":
            # analyze_concurrency_steps rebuilds verdict.steps from scratch
            # at every iteration of the staircase, so stashing onto
            # verdict.steps[-1] in _on_step only survives for the final
            # iteration. Keep a parallel list keyed by step index and merge
            # back into the final verdict.steps after the staircase exits.
            per_step_meta: list[dict] = []

            def _on_step(step, verdict):
                cur = sat_per_step[sat_step_idx[0]]
                n_cas = cur["ship"] + cur["regen"]
                step_ship_rate = (cur["ship"] / n_cas) if n_cas else None
                print(
                    f"[autosat] c={step.lam:6.0f}  "
                    f"throughput={step.achieved_rps:6.1f} r/s  "
                    f"p50={step.p50_ms:7.0f}ms  "
                    f"ship={cur['ship']}/{n_cas} "
                    f"({(step_ship_rate or 0)*100:.1f}%) "
                    f"-> {verdict.steps[-1]['classification']}",
                    flush=True,
                )
                per_step_meta.append({
                    "ship_n": cur["ship"],
                    "regen_n": cur["regen"],
                    "other_n": cur["other"],
                    "error_n": cur["error"],
                    "ship_rate_per_cascade": step_ship_rate,
                })
                sat_per_step.append(
                    {"ship": 0, "regen": 0, "error": 0, "other": 0}
                )
                sat_step_idx[0] += 1
            # robustness: a plain `--sat-mode concurrency` (stability
            # knobs left at 0) auto-engages the wait-for-stable + skip-low-c
            # defaults the round found necessary on image-heavy VLM
            # workloads — naive fixed-window ramps from c=8 under-call the
            # cap ~10× (they read the cold per-c warmup as the plateau).
            _c_start = args.sat_c_start
            _stable_window_s = args.sat_stable_window_s
            _stable_windows = args.sat_stable_windows
            _stable_tol = args.sat_stable_tol
            _step_min = args.sat_step_min_dur
            _step_max = args.sat_step_max_dur
            if args.sat_stable_windows == 0:
                _stable_window_s = 15.0
                _stable_windows = 3
                _stable_tol = 0.12
                _step_min = 30.0 if _step_min is None else _step_min
                _step_max = 180.0 if _step_max is None else _step_max
                if _c_start < 32:
                    _c_start = 32
                print(
                    f"[autosat] concurrency mode: auto-engaged robust "
                    f"defaults (wait-for-stable 3×15s tol 0.12, c_start="
                    f"{_c_start}, step 30-180s, latency_rise "
                    f"{args.sat_latency_rise}). Override with --sat-stable-* "
                    f"/ --sat-c-start.",
                    flush=True,
                )
            snap0 = _snapshot_sched_stats()
            verdict = await run_concurrency_staircase(
                _sat_submit, lambda: next(rec_cycle),
                c_start=_c_start, c_max=args.sat_c_max,
                c_step=args.sat_c_step, c_mult=args.sat_c_mult,
                step_dur_s=args.sat_step_dur, warmup_frac=args.sat_warmup_frac,
                gain_eps=args.sat_gain_eps,
                confirm_saturated_steps=args.sat_confirm_steps,
                latency_rise=args.sat_latency_rise,
                # wait-for-stable knobs (auto-engaged above).
                stable_window_s=_stable_window_s,
                stable_windows=_stable_windows,
                stable_tol=_stable_tol,
                step_min_dur_s=_step_min,
                step_max_dur_s=_step_max,
                on_step=_on_step,
            )
            for i, meta in enumerate(per_step_meta):
                if i < len(verdict.steps):
                    verdict.steps[i].update(meta)
        else:
            def _on_step(step, verdict):
                print(
                    f"[autosat] λ={step.lam:6.1f}  "
                    f"achieved={step.achieved_rps:6.1f} r/s  "
                    f"in_flight={step.mean_in_flight:6.1f}  "
                    f"slope={step.inflight_slope:+6.2f}/s  p50={step.p50_ms:7.0f}ms "
                    f"-> {verdict.steps[-1]['classification']}",
                    flush=True,
                )
            snap0 = _snapshot_sched_stats()
            verdict = await run_staircase(
                _sat_submit, lambda: next(rec_cycle),
                lam_start=args.sat_lam_start, lam_max=args.sat_lam_max,
                lam_step=args.sat_lam_step, lam_mult=args.sat_lam_mult,
                step_dur_s=args.sat_step_dur, warmup_frac=args.sat_warmup_frac,
                distribution=args.arrival_distribution,
                keepup=args.sat_keepup, gain_eps=args.sat_gain_eps,
                slope_thresh=args.sat_slope_thresh,
                confirm_saturated_steps=args.sat_confirm_steps,
                on_step=_on_step,
            )
        snap1 = _snapshot_sched_stats()
        if _ctl_task is not None:
            _ctl_task.cancel()
            try:
                await _ctl_task
            except (asyncio.CancelledError, Exception):
                pass
        sat_stats = {
            "label": label, "cell": cell, "mode": "auto_saturate",
            "sat_mode": args.sat_mode,
            "bench_cli_args": sys.argv,
            "max_pixels": args.max_pixels,
            "draft_engine_mode": args.draft_engine_mode,
            "draft_model": args.draft_model, "target_model": args.target_model,
            "draft_mixture": getattr(args, "draft_mixture", None),
            "records_path": args.records,
            "cap_rps": verdict.cap_rps, "saturated": verdict.saturated,
            # canonical name (2026-07-07 convention): completed items/s in the
            # steady window of the winning ramp step — alias of cap_rps here
            "finished_items_per_second": verdict.cap_rps,
            "knee_lam": verdict.knee_lam, "knee_step_idx": verdict.knee_step_idx,
            "reason": verdict.reason, "steps": verdict.steps,
            "sat_config": {
                "sat_mode": args.sat_mode,
                "step_dur_s": args.sat_step_dur,
                "warmup_frac": args.sat_warmup_frac,
                "gain_eps": args.sat_gain_eps,
                "confirm_steps": args.sat_confirm_steps,
                # arrival-mode knobs
                "lam_start": args.sat_lam_start, "lam_max": args.sat_lam_max,
                "lam_step": args.sat_lam_step, "lam_mult": args.sat_lam_mult,
                "keepup": args.sat_keepup, "slope_thresh": args.sat_slope_thresh,
                "distribution": args.arrival_distribution,
                # concurrency-mode knobs (requested)
                "c_start": args.sat_c_start, "c_max": args.sat_c_max,
                "c_step": args.sat_c_step, "c_mult": args.sat_c_mult,
                "latency_rise": args.sat_latency_rise,
                # concurrency-mode knobs (effective, after auto-engage)
                "effective_c_start":
                    _c_start if args.sat_mode == "concurrency" else None,
                "effective_stable_window_s":
                    _stable_window_s if args.sat_mode == "concurrency" else None,
                "effective_stable_windows":
                    _stable_windows if args.sat_mode == "concurrency" else None,
                "effective_stable_tol":
                    _stable_tol if args.sat_mode == "concurrency" else None,
            },
        }
        if snap0 is not None and snap1 is not None:
            sat_stats["load_snapshot_end"] = snap1.get("load_snapshot")
        # controller final snapshot + the ramp-long direct_ratio/
        # ship_ma trajectory (empty list on non-controller runs).
        if snap1 is not None:
            sat_stats["control"] = snap1.get("control")
        sat_stats["control_series"] = _ctl_series
        # aggregate ship_rate from per-request
        # head_decision counts collected by _sat_submit.
        n_cascade = sat_verdicts["ship"] + sat_verdicts["regen"]
        sat_stats["sat_verdict_counts"] = dict(sat_verdicts)
        sat_stats["ship_rate_per_cascade"] = (
            sat_verdicts["ship"] / n_cascade if n_cascade else None
        )
        sat_stats["ship_rate_per_total"] = (
            sat_verdicts["ship"] / sum(sat_verdicts.values())
            if sum(sat_verdicts.values()) else None
        )
        with open(out_path, "w") as f:
            json.dump(sat_stats, f, indent=2)
        all_stats.append(sat_stats)
        _knee_unit = "c" if args.sat_mode == "concurrency" else "λ"
        print(
            f"\n[autosat] ===> estimated cap = "
            f"{verdict.cap_rps:.1f} r/s  (saturated={verdict.saturated}, "
            f"knee {_knee_unit}={verdict.knee_lam})\n          {verdict.reason}\n",
            flush=True,
        )

    for c in args.concurrencies:
        label = f"{args.label}_c{c}"
        out_path = out_dir / f"{label}.json"
        snap0 = _snapshot_sched_stats()
        # A3: per-c occupancy time-series — sample in-flight + target KV
        # proximity + which gate arm holds, so the overload sweep shows WHICH
        # arm binds (the gap). Read-only; ~2 Hz; no-op if no scheduler.
        _occ_series: list[dict] = []
        async def _occ_sampler(_c=c):
            t0 = time.perf_counter()
            while True:
                try:
                    s = sched.occupancy_snapshot()
                    s["t"] = round(time.perf_counter() - t0, 1)
                    s["c"] = _c
                    _occ_series.append(s)
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        _occ_task = (
            asyncio.create_task(_occ_sampler()) if sched is not None else None
        )
        stats = await _run_closed_loop(
            submit_one, records, c, out_path, label, cell,
            args.max_tokens, args.ignore_eos, args.images_per_record,
            capture_text=args.capture_text,
        )
        if _occ_task is not None:
            _occ_task.cancel()
            stats["occupancy_series"] = _occ_series
        snap1 = _snapshot_sched_stats()
        if snap0 is not None and snap1 is not None:
            stats["per_draft_submit"] = [
                a - b for a, b in zip(
                    snap1["n_submit_per_draft"], snap0["n_submit_per_draft"]
                )
            ]
            stats["per_target_dispatch"] = [
                a - b for a, b in zip(
                    snap1.get("n_dispatch_per_target", []),
                    snap0.get("n_dispatch_per_target", []),
                )
            ]
            stats["per_encoder_dispatch"] = [
                a - b for a, b in zip(
                    snap1.get("n_dispatch_per_encoder", []),
                    snap0.get("n_dispatch_per_encoder", []),
                )
            ]
            stats["n_direct_target"] = (
                snap1.get("n_direct_target", 0)
                - snap0.get("n_direct_target", 0)
            )
            stats["target_batch_flushes"] = (
                snap1.get("target_batch_flushes", 0)
                - snap0.get("target_batch_flushes", 0)
            )
            stats["target_batch_items_flushed"] = (
                snap1.get("target_batch_items_flushed", 0)
                - snap0.get("target_batch_items_flushed", 0)
            )
            stats["calibration_calls"] = (
                snap1.get("calibration_calls", 0)
                - snap0.get("calibration_calls", 0)
            )
            stats["calibration_total_abs_drift"] = (
                snap1.get("calibration_total_abs_drift", 0)
                - snap0.get("calibration_total_abs_drift", 0)
            )
            stats["load_snapshot_end"] = snap1.get("load_snapshot")
            with open(out_path, "w") as f:
                json.dump(stats, f, indent=2)
        all_stats.append(stats)
    # two_stream short-circuits the (util, λ) sweep entirely.
    # Single run with fixed λ_target + λ_cascade; emits per-stream cell
    # JSONs plus a combined summary. Scheduler stop happens in the
    # post-sweep cleanup below, shared with all other cells.
    if cell == "two_stream":
        label = (
            f"{args.label}_lamT{args.lambda_target:g}_lamD{args.lambda_draft:g}"
        )
        snap0 = _snapshot_sched_stats()
        combined = await _run_two_stream(
            submit_target_stream, submit_cascade_stream, records,
            args.lambda_target, args.lambda_draft, args.duration,
            args.arrival_distribution, out_dir, label,
            args.max_tokens, args.ignore_eos, args.images_per_record,
            capture_text=args.capture_text,
        )
        snap1 = _snapshot_sched_stats()
        if snap0 is not None and snap1 is not None:
            combined["per_draft_submit"] = [
                a - b for a, b in zip(
                    snap1["n_submit_per_draft"], snap0["n_submit_per_draft"]
                )
            ]
            combined["per_target_dispatch"] = [
                a - b for a, b in zip(
                    snap1.get("n_dispatch_per_target", []),
                    snap0.get("n_dispatch_per_target", []),
                )
            ]
            combined["per_encoder_dispatch"] = [
                a - b for a, b in zip(
                    snap1.get("n_dispatch_per_encoder", []),
                    snap0.get("n_dispatch_per_encoder", []),
                )
            ]
            combined["n_direct_target"] = (
                snap1.get("n_direct_target", 0)
                - snap0.get("n_direct_target", 0)
            )
        # flag encoder-bound regimes so downstream analysis
        # doesn't mistake encoder ceiling for draft ceiling. Per-encoder
        # ceiling is ~12 r/s on a single A10G at single-image (
        # Cell 1, B=1). Threshold at 80% of that.
        if args.n_encoders > 0 and combined.get("per_encoder_dispatch"):
            steady_window_s = max(1.0, args.duration * 0.6)  # 60-240 of 300s
            for i, dispatches in enumerate(combined["per_encoder_dispatch"]):
                per_enc_rps = dispatches / steady_window_s
                if per_enc_rps > 0.8 * 12.0:
                    combined.setdefault("encoder_bound_flags", []).append(i)
        combined_path = out_dir / f"{label}_combined.json"
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"[sys17o] wrote combined summary → {combined_path}")
        all_stats.append(combined)

    # two_stream already ran in the block above; skip the
    # arrival-rate sweep entirely so we don't no-op-loop with empty
    # args.arrival_rates and so the post-cleanup runs as normal.
    if cell != "two_stream":
        for lam in args.arrival_rates:
            label = f"{args.label}_lam{lam:g}"
            out_path = out_dir / f"{label}.json"
            snap0 = _snapshot_sched_stats()
            # sample the controller diagnostics throughout this
            # open-loop run so Cell B's per-phase direct_ratio / ship_ma
            # trajectory is captured. Inert on non-controller runs.
            _ol_ctl_series: list[dict] = []
            async def _ol_ctl_sampler():
                t0c = time.perf_counter()
                while True:
                    try:
                        c = sched.stats.get("control", {}) if sched else {}
                        _ol_ctl_series.append({
                            "t": round(time.perf_counter() - t0c, 1),
                            "ticks": c.get("ticks"),
                            "direct_ratio": c.get("direct_ratio"),
                            "ship_ma": c.get("ship_ma"),
                            "dopt_ff": c.get("dopt_ff"),
                            "trim": c.get("trim"),
                        })
                    except Exception:
                        pass
                    await asyncio.sleep(2.0)
            _ol_ctl_task = (
                asyncio.create_task(_ol_ctl_sampler())
                if (sched is not None and getattr(args, "closed_loop", False))
                else None
            )
            # occupancy + buffer-depth time series for open-loop
            # cells (previously concurrency-cells-only). Shows which gate
            # arm binds AND how the backlog evolves; 1 Hz, read-only.
            _ol_occ_series: list[dict] = []
            async def _ol_occ_sampler():
                t0o = time.perf_counter()
                while True:
                    try:
                        s = sched.occupancy_snapshot()
                        s["t"] = round(time.perf_counter() - t0o, 1)
                        _ol_occ_series.append(s)
                    except Exception:
                        pass
                    await asyncio.sleep(1.0)
            _ol_occ_task = (
                asyncio.create_task(_ol_occ_sampler())
                if sched is not None else None
            )
            stats = await _run_open_loop(
                submit_one, records, lam, args.duration,
                args.arrival_distribution, out_path, label, cell,
                args.max_tokens, args.ignore_eos, args.images_per_record,
                capture_text=args.capture_text,
                burst_window_s=args.burst_window_s,
                drain_cap_s=getattr(args, "ol_drain_cap_s", None),
            )
            if _ol_ctl_task is not None:
                _ol_ctl_task.cancel()
                try:
                    await _ol_ctl_task
                except (asyncio.CancelledError, Exception):
                    pass
            if _ol_occ_task is not None:
                _ol_occ_task.cancel()
                try:
                    await _ol_occ_task
                except (asyncio.CancelledError, Exception):
                    pass
                stats["occupancy_series"] = _ol_occ_series
            stats["control"] = dict(sched.stats.get("control", {}) or {}) \
                if sched is not None else None
            stats["control_series"] = _ol_ctl_series
            snap1 = _snapshot_sched_stats()
            if snap0 is not None and snap1 is not None:
                stats["per_draft_submit"] = [
                    a - b for a, b in zip(
                        snap1["n_submit_per_draft"], snap0["n_submit_per_draft"]
                    )
                ]
                stats["per_target_dispatch"] = [
                    a - b for a, b in zip(
                        snap1.get("n_dispatch_per_target", []),
                        snap0.get("n_dispatch_per_target", []),
                    )
                ]
                stats["per_encoder_dispatch"] = [
                    a - b for a, b in zip(
                        snap1.get("n_dispatch_per_encoder", []),
                        snap0.get("n_dispatch_per_encoder", []),
                    )
                ]
                stats["n_direct_target"] = (
                    snap1.get("n_direct_target", 0)
                    - snap0.get("n_direct_target", 0)
                )
                stats["target_batch_flushes"] = (
                    snap1.get("target_batch_flushes", 0)
                    - snap0.get("target_batch_flushes", 0)
                )
                stats["target_batch_items_flushed"] = (
                    snap1.get("target_batch_items_flushed", 0)
                    - snap0.get("target_batch_items_flushed", 0)
                )
                stats["calibration_calls"] = (
                    snap1.get("calibration_calls", 0)
                    - snap0.get("calibration_calls", 0)
                )
                stats["calibration_total_abs_drift"] = (
                    snap1.get("calibration_total_abs_drift", 0)
                    - snap0.get("calibration_total_abs_drift", 0)
                )
                stats["n_buffer_inserts"] = (
                    snap1.get("n_buffer_inserts", 0)
                    - snap0.get("n_buffer_inserts", 0)
                )
                stats["n_buffer_via_top"] = (
                    snap1.get("n_buffer_via_top", 0)
                    - snap0.get("n_buffer_via_top", 0)
                )
                stats["n_buffer_via_bottom"] = (
                    snap1.get("n_buffer_via_bottom", 0)
                    - snap0.get("n_buffer_via_bottom", 0)
                )
                stats["buffer_max_depth"] = snap1.get("buffer_max_depth", 0)
                # expose the scheduler's gate-loop counters so the
                # plumbing check `n_gate_decisions == n_accept + n_regen`
                # is verifiable from the cell JSON.
                stats["n_accept_sched"] = (
                    snap1.get("n_accept", 0) - snap0.get("n_accept", 0)
                )
                stats["n_regen_sched"] = (
                    snap1.get("n_regen", 0) - snap0.get("n_regen", 0)
                )
                stats["n_gate_decisions"] = (
                    snap1.get("n_gate_decisions", 0)
                    - snap0.get("n_gate_decisions", 0)
                )
                # drafted-vs-direct mean prescorer score for THIS
                # Cell (delta over the invocation's earlier cells). The
                # ordering-acting receipt: a clear gap (e.g. ~0.82 vs
                # ~0.52) = scorer routing is live; ≈equal = inert scorer.
                _dn = (snap1.get("scorer_drafted_score_n", 0)
                       - snap0.get("scorer_drafted_score_n", 0))
                _ds = (snap1.get("scorer_drafted_score_sum", 0.0)
                       - snap0.get("scorer_drafted_score_sum", 0.0))
                _tn = (snap1.get("scorer_direct_score_n", 0)
                       - snap0.get("scorer_direct_score_n", 0))
                _ts = (snap1.get("scorer_direct_score_sum", 0.0)
                       - snap0.get("scorer_direct_score_sum", 0.0))
                stats["scorer_drafted_mean_score"] = (
                    _ds / _dn if _dn else None)
                stats["scorer_drafted_score_n"] = _dn
                stats["scorer_direct_mean_score"] = (
                    _ts / _tn if _tn else None)
                stats["scorer_direct_score_n"] = _tn
                stats["load_snapshot_end"] = snap1.get("load_snapshot")
                with open(out_path, "w") as f:
                    json.dump(stats, f, indent=2)
            all_stats.append(stats)

    if router is not None:
        await router.stop()
    if sched is not None:
        await sched.stop()

    # Explicitly kill Ray actors so GPU memory is released before the
    # process exits. Ray's default actor lifetime is the driver, but the
    # teardown lag can leave GPU memory occupied long enough that the
    # NEXT bench's engine boot trips "Free memory < gpu_memory_utilization".
    # Killing forces immediate release; the subsequent sleep in the sweep
    # driver covers any residual destructor delay.
    for actor in (*drafts, target):
        if actor is not None:
            try:
                ray.kill(actor, no_restart=True)
            except Exception as e:
                print(f"[sys11-bench] ray.kill warning: {e}", flush=True)

    print()
    # ship/casc = ACCEPT / (ACCEPT + REGEN) — the head's positive rate
    # (the `s` in d_opt). ship/tot column omitted for compactness; read
    # from JSON if needed. See CLAUDE.md "Bench metric definitions".
    print(f"{'label':<32} {'req/s':>7} {'tok/s':>8} {'p50':>7} {'p99':>7} "
          f"{'ship/casc':>9} {'n_err':>5}")
    for s in all_stats:
        # two_stream: outer dict carries per-stream sub-dicts plus
        # total_steady_rps. Flatten to one row per stream for the table.
        if s.get("cell") == "two_stream":
            for sub in (s.get("target_stats"), s.get("cascade_stats")):
                if not sub or sub.get("n_ok", 0) == 0:
                    continue
                ship_casc = sub.get("ship_rate_per_cascade", sub.get("ship_rate", 0))
                print(f"{sub['label']:<32} {sub['requests_per_s']:>7.2f} "
                      f"{sub['tokens_per_s']:>8.1f} {sub['p50_ms']:>6.0f}ms "
                      f"{sub['p99_ms']:>6.0f}ms {ship_casc*100:>8.1f}% "
                      f"{sub['n_err']:>5}")
            continue
        # auto_saturate rows don't carry per-request aggregate fields
        # (requests_per_s/tokens_per_s/p50_ms/...) — print a cap-rps line
        # instead so the summary doesn't crash on missing keys.
        if s.get("mode") == "probe_oneby":
            _m = s.get("mean_ms")
            _p50 = s.get("p50_ms")
            _p99 = s.get("p99_ms")
            _bgr = (s.get("bg") or {}).get("rate_during_probes_rps")
            print(f"{s['label']:<32} mean="
                  f"{_m:.1f}ms p50={_p50:.1f}ms p99={_p99:.1f}ms "
                  f"n={s.get('n_ok')}/{s.get('n_probes')} "
                  f"n_err={s.get('n_err')} "
                  f"bg_rps={_bgr if _bgr is not None else 'n/a'} "
                  f"(probe_oneby)" if _m is not None else
                  f"{s['label']:<32} probe_oneby: NO OK PROBES "
                  f"(n_err={s.get('n_err')})")
            continue
        if s.get("mode") == "auto_saturate":
            cap = s.get("cap_rps")
            knee = s.get("knee_lam")
            ship_casc = s.get("ship_rate_per_cascade") or 0
            cap_s = f"{cap:.2f}" if cap is not None else "n/a"
            knee_s = f"{knee}" if knee is not None else "n/a"
            print(f"{s['label']:<32} cap={cap_s:>7} r/s "
                  f"knee={knee_s:<5} ship/casc={ship_casc*100:.1f}% "
                  f"(auto_saturate)")
            continue
        ship_casc = s.get("ship_rate_per_cascade", s.get("ship_rate", 0))
        print(f"{s['label']:<32} {s['requests_per_s']:>7.2f} "
              f"{s['tokens_per_s']:>8.1f} {s['p50_ms']:>6.0f}ms "
              f"{s['p99_ms']:>6.0f}ms {ship_casc*100:>8.1f}% "
              f"{s['n_err']:>5}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True,
                    choices=["target_only", "target_only_no_vit",
                             "target_only_cached", "draft_only",
                             "draft_only_via_scheduler",
                             "draft_only_with_head_via_scheduler",
                             "draft_only_no_vit", "draft_only_cached",
                             "cascade_no_vit", "cascade_gate", "two_stream"])
    ap.add_argument("--embeds-cache", default=None,
                    help="Pre-encoded ViT embeds .pt path (output of "
                         "sys9_preencode_embeds.py). Required for "
                         "--cell target_only_no_vit (bench-side cache, "
                         "ships tensor per request via Ray pickle) and "
                         "--cell target_only_cached (target-actor-side "
                         "cache, ships rec_id per request).")
    ap.add_argument("--embeds-transfer-mode",
                    choices=["gpu_side_stream", "cpu_pinned"],
                    default="gpu_side_stream",
                    help="For --cell target_only_cached: how the target "
                         "actor hands embeds to vLLM. gpu_side_stream "
                         "(default original) side-streams CPU→GPU "
                         "in the actor process. cpu_pinned passes pinned "
                         "CPU tensors directly so vLLM does its own H2D "
                         "inside the worker — closer to the "
                         "production NCCL-into-worker pattern and avoids "
                         "competing with vLLM's KV pool on GPU 0.")
    ap.add_argument("--draft-engine-mode",
                    choices=["eager_baseline", "graph_only", "async_only",
                             "graph_async"],
                    default="graph_async",
                    help="Cascade draft engine mode. (2026-05-21) "
                         "tested graph_async and observed ship_rate collapse "
                         "from 64.4% → 18.7% on long-CoT with the head/τ "
                         "— head-cascade verdict logic mis-fired under "
                         "cudagraphs despite the Phase B / 2026-05-13 "
                         "fixes. Default reverted to eager_baseline (eager=True, "
                         "async=False) at the time. 2× g5.12 system "
                         "test (2026-05-28) re-validates on the mixed "
                         "workload (`bench_vlm_test_repath.json`): if ship "
                         "rate holds at ≈0.605 under graph_async, the "
                         "long-CoT regression is workload-specific and "
                         "graph_async is the right default; revert this "
                         "default if it collapses again. "
                         "Pass eager_baseline / graph_only / async_only to "
                         "isolate a specific axis.")
    ap.add_argument("--draft-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--draft-mixture", default=None,
                    help="heterogeneous draft fleet for cascade cells, "
                         "e.g. 'llava:2,pixtral:2'. Boots a mixed pool sharing "
                         "ONE target. Overrides --draft-model/--n-drafts/"
                         "--in-engine-head-* for the draft tier; each family's "
                         "model + GEN-ALL head + c18 tau are resolved from "
                         "--mixture-heads-dir. All families share ONE --records "
                         "(one image resolution).")
    ap.add_argument("--mixture-heads-dir",
                    default=os.getenv("PROROUTER_HEADS_DIR", "runs/heads"),
                    help="dir holding <family>_GEN-ALL.pt + "
                         "<family>_c18_tau.json for --draft-mixture.")
    ap.add_argument("--target-model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--draft-tp", type=int, default=4)
    ap.add_argument("--n-drafts", type=int, default=1,
                    help="boot N draft actors. Each requires a "
                         "distinct DRAFT_RESOURCE node (g5.12xlarge). "
                         "Only honored for --cell cascade_no_vit.")
    ap.add_argument("--n-encoders", type=int, default=0,
                    help="boot N DraftVisionEncoder actors on "
                         "ENCODER_RESOURCE nodes (1 GPU each). When >0, "
                         "the scheduler routes draft work through the "
                         "pool (encoder runs ViT, draft consumes embeds). "
                         "Honored for --cell cascade_no_vit and draft_only. "
                         "Set ENCODER_RESOURCE env var to point at a "
                         "separate g5.xlarge / 1×A10G node label; "
                         "defaults to DRAFT_RESOURCE.")
    ap.add_argument("--target-tp", type=int, default=8)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--draft-mem-util", type=float, default=0.85)
    ap.add_argument("--draft-no-prefix-caching", action="store_true",
                    help="Part 1 — boot the draft engine with prefix "
                         "caching OFF (records cycle under stop-on-stable; "
                         "identical repeated prompts must not be served from "
                         "the KV prefix cache, which would inflate throughput).")
    ap.add_argument("--target-no-prefix-caching", action="store_true",
                    help="c18x9 — boot the TARGET engine with prefix "
                         "caching OFF (twin of --draft-no-prefix-caching; "
                         "cascade targets see the same cycled prompts).")
    ap.add_argument("--draft-max-num-batched-tokens", type=int, default=None,
                    help="vLLM max_num_batched_tokens for the draft engine "
                         "(default: vLLM internal default = 2048 for cudagraph "
                         "ranges). Bumping raises the per-step prefill+decode "
                         "token budget — useful to keep new-admit prefills "
                         "from preempting decode. probe.")
    ap.add_argument("--draft-max-num-seqs", type=int, default=None,
                    help="vLLM max_num_seqs for the draft engine (default: "
                         "vLLM internal default). Going above 256 falls off "
                         "cudagraph capture range; useful only with eager mode.")
    ap.add_argument("--target-mem-util", type=float, default=0.85)
    ap.add_argument("--target-kv-pool-threshold", type=int, default=None,
                    help="probe — override TargetEngineAsync's "
                         "_kv_pool_threshold (default 200_000 tokens). "
                         "Raising it tests whether KV admission is the "
                         "binding constraint on cascade-side target rate.")
    ap.add_argument("--draft-pop-timeout-ms", type=float, default=100.0,
                    help="_draft_pump.pop_finished poll timeout in ms. "
                         "Only matters when the draft's finished_q is "
                         "empty (call returns immediately with items "
                         "otherwise). Higher = fewer idle polling "
                         "RPCs; cost is a small tail-latency rise "
                         "under light load. Default 100.")
    ap.add_argument("--target-pop-timeout-ms", type=float, default=100.0,
                    help="_target_pump.pop_finished poll timeout in "
                         "ms. Independent of the draft side. Default 100.")
    ap.add_argument("--draft-engine-send-rps", type=float, default=None,
                    help=" — token-bucket cap on the per-draft "
                         "send pump in request-rate units (r/s). None "
                         "(default) = unlimited. Use to match draft's "
                         "input rate between draft_only and cascade "
                         "cells for fair contribution audit. Each draft "
                         "actor has its own bucket; with N drafts the "
                         "total cap is N × this value.")
    ap.add_argument("--target-engine-send-rps", type=float, default=None,
                    help=" — token-bucket cap on the per-target "
                         "send pump in request-rate units (r/s). None "
                         "(default) = unlimited. Use to match target's "
                         "input rate between target_only and cascade "
                         "cells. Billed per request (not per batched "
                         "RPC) so the cap stays in r/s units regardless "
                         "of target_batch_enabled.")
    ap.add_argument("--source-filter", default=None,
                    help="verify — restrict the records pool to "
                         "the listed sources (comma-separated). Filters "
                         "at load time. Use to send a specific source "
                         "subset to target in target_only mode for the "
                         "bit-identical-input audit.")
    ap.add_argument("--allow-text-records", action="store_true",
                    help=" — keep records with no images (text "
                         "benches: MMLU/GSM8K/CoQA/TriviaQA ship "
                         "\"images\": []). Default keeps the historical "
                         "image-required filter.")
    ap.add_argument("--target-sources", default=None,
                    help="verify — comma-separated source names "
                         "to mark with force_direct=True at submit time. "
                         "These records get score=−∞ → bottom of buffer "
                         "→ target dispatcher only. Used with --cascade-"
                         "sources to deliver a bit-identical target "
                         "input multiset in cascade_no_vit cells.")
    ap.add_argument("--cascade-sources", default=None,
                    help="verify — comma-separated source names "
                         "to mark with force_cascade=True AND "
                         "force_draft_response=True at submit time. "
                         "These records get score=+∞ → top of buffer "
                         "→ draft dispatcher only; head's REGEN verdict "
                         "is absorbed (no re-route to target). Used "
                         "with --target-sources for the bit-identical "
                         "verify experiment.")
    ap.add_argument("--target-input-log", default=None,
                    help="verify — write one CSV line per target "
                         "dispatch (dispatched_t, req_id, source, "
                         "prompt_md5, n_imgs, max_tokens) to this path. "
                         "Diff sorted CSVs across two cells to verify "
                         "the multiset of target inputs is identical.")
    ap.add_argument("--mock-target", action="store_true",
                    help=" — boot a zero-GPU MockTargetActor instead "
                         "of TargetEngineAsync. Satisfies V0Scheduler's "
                         "'at least one target' requirement + its periodic "
                         "poll RPCs without booting a real target. Only "
                         "valid for force_cascade cells "
                         "(draft_only_via_scheduler / "
                         "draft_only_with_head_via_scheduler) where the "
                         "target is never actually called. Enables draft-"
                         "side benches on 2× g5.12 clusters with no p4d.")
    ap.add_argument("--draft-head-cascade", action="store_true",
                    help=" — fire the cascade head on the draft "
                         "even for plain --cell draft_only (no target, no "
                         "V0Scheduler). Enables head_cascade on the draft "
                         "actor + per-request, and maps "
                         "item['head_decision'] to ACCEPT/REGEN verdicts so "
                         "per-source ship_rate_per_cascade is emitted. "
                         "Used by to bench head firing cost on 2× "
                         "g5.12 clusters that have no target node.")
    ap.add_argument("--head-ckpt",
                    default=os.getenv("PROROUTER_HEAD_CKPT", "weights/head.pt"))
    ap.add_argument("--tau",
                    default=os.getenv("PROROUTER_TAU_TABLE", "weights/tau.json"))
    ap.add_argument("--layer-from-end", type=int, default=7)
    ap.add_argument("--records", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--images-per-record", type=int, default=1,
                    help="0 = honor each record's images list verbatim (mixed workloads); "
                         ">0 = duplicate rec['images'][0] N times")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="Bound Qwen2.5-VL ViT tokens via "
                         "mm_processor_kwargs={'max_pixels': N}. "
                         "1280*28*28=1003520 caps each image at ~1280 tokens. "
                         "None = unbounded (full-res, the Phase-A config).")
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--warmup-max-tokens", type=int, default=0,
                    help="cap max_tokens for warmup generations only "
                         "(0 = use --max-tokens). llava/pixtral hang "
                         "on some text records at warmup (triviaqa).")
    # ---- one-by-one probe latency mode ----
    ap.add_argument("--probe-oneby", action="store_true",
                    help="per-request routing latency. Submits "
                         "records[:probe-n] strictly one at a time (c=1, "
                         "fixed file order) and reports per-request wall "
                         "time, over an optional steady closed-loop "
                         "background load from a DISJOINT slice of the same "
                         "records file. Mutually exclusive with sweeps / "
                         "--auto-saturate.")
    ap.add_argument("--probe-n", type=int, default=300,
                    help="number of probe records (prefix of the records "
                         "file, in file order).")
    ap.add_argument("--bg-start", type=int, default=300,
                    help="background slice start index into the records "
                         "file (must not overlap [0, probe-n)).")
    ap.add_argument("--bg-end", type=int, default=600,
                    help="background slice end index (exclusive).")
    ap.add_argument("--bg-concurrency", type=int, default=32,
                    help="closed-loop background concurrency. 0 = idle "
                         "control (no background load).")
    ap.add_argument("--bg-settle-s", type=float, default=45.0,
                    help="seconds to let the background load reach steady "
                         "state before the first probe.")
    ap.add_argument("--concurrencies", type=lambda s: [int(x) for x in s.split(",")],
                    default=[])
    ap.add_argument("--arrival-rates",
                    type=lambda s: [float(x) for x in s.split(",")],
                    default=[])
    ap.add_argument("--rpc-fake-latency-ms", type=float, default=0.0,
                    help=" — network-latency resilience probe: each "
                         "serving RPC (submit/submit_decode[_batch]/"
                         "pop_finished) sleeps this long at entry on BOTH "
                         "engine actors (one-way wire latency; asyncio, "
                         "non-serializing). 0 = off.")
    ap.add_argument("--ol-drain-cap-s", type=float, default=None,
                    help=" — cap the open-loop post-arrival drain (s). "
                         "Default None = legacy 2×duration. In-window metrics "
                         "are unaffected; cancelled stragglers are dropped "
                         "from completions (neither ok nor err).")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--arrival-distribution", choices=["constant", "poisson"],
                    default="poisson")
    # ---- single-run saturation finder ----
    ap.add_argument("--auto-saturate", action="store_true",
                    help="find the engine's saturated output rate in "
                         "ONE boot by ramping open-loop λ as a staircase and "
                         "detecting the knee (achieved-rate plateau + in-flight "
                         "backlog divergence). Replaces relaunching at many "
                         "fixed λ. Writes <label>_autosat.json with the cap, "
                         "knee λ, and per-step table. Ignores "
                         "--concurrencies/--arrival-rates.")
    ap.add_argument("--sat-lam-start", type=float, default=8.0,
                    help="first staircase λ (req/s).")
    ap.add_argument("--sat-lam-max", type=float, default=200.0,
                    help="stop ramping past this λ (safety ceiling).")
    ap.add_argument("--sat-lam-step", type=float, default=0.0,
                    help="additive λ step (req/s). 0 → geometric "
                         "(--sat-lam-mult).")
    ap.add_argument("--sat-lam-mult", type=float, default=1.3,
                    help="geometric λ ratio when --sat-lam-step=0.")
    ap.add_argument("--sat-step-dur", type=float, default=30.0,
                    help="seconds per staircase step.")
    ap.add_argument("--sat-warmup-frac", type=float, default=0.4,
                    help="fraction of each step discarded as transient "
                         "before measuring achieved rate / backlog slope.")
    ap.add_argument("--sat-keepup", type=float, default=0.92,
                    help="achieved/offered below this (with backlog "
                         "growing) flags the step saturated.")
    ap.add_argument("--sat-gain-eps", type=float, default=0.04,
                    help="relative achieved-rate gain vs prev step "
                         "below this (with backlog growing) flags a plateau.")
    ap.add_argument("--sat-slope-thresh", type=float, default=0.5,
                    help="in-flight backlog growth (tasks/s) above "
                         "which the step counts as backlog-diverging.")
    ap.add_argument("--sat-confirm-steps", type=int, default=1,
                    help="stop after this many consecutive saturated "
                         "steps (≥2 is more robust against a noisy step).")
    ap.add_argument("--sat-mode", choices=["arrival", "concurrency"],
                    default="arrival",
                    help="'arrival' ramps open-loop λ (finds the "
                         "bench-PIPELINE cap — bottlenecked by per-arrival "
                         "submit work like image decode). 'concurrency' ramps "
                         "closed-loop in-flight count c and detects the "
                         "throughput plateau (finds the engine's GPU cap — "
                         "the number you want for image-heavy workloads).")
    ap.add_argument("--sat-c-start", type=int, default=8,
                    help="concurrency mode: first in-flight level c.")
    ap.add_argument("--sat-c-max", type=int, default=512,
                    help="concurrency mode: stop past this c.")
    ap.add_argument("--sat-c-step", type=int, default=0,
                    help="concurrency mode: additive c step. 0 → "
                         "geometric (--sat-c-mult).")
    ap.add_argument("--sat-c-mult", type=float, default=2.0,
                    help="concurrency mode: geometric c ratio when "
                         "--sat-c-step=0.")
    # --- wait-for-stable knobs (concurrency mode) ---
    ap.add_argument("--sat-stable-window-s", type=float, default=0.0,
                    help="concurrency mode: rolling-window size (s) "
                         "for in-step stability detection. 0 → disable "
                         "wait-for-stable; finalize each step at "
                         "--sat-step-dur (legacy). When set ≥ ~p50 of the "
                         "engine (e.g. 20-30s for image-heavy VLM), the "
                         "step finalizes only when the last "
                         "--sat-stable-windows windows agree.")
    ap.add_argument("--sat-stable-windows", type=int, default=0,
                    help="concurrency mode: # consecutive stable "
                         "windows required before finalizing the step. "
                         "Needs ≥ 2 to engage wait-for-stable.")
    ap.add_argument("--sat-stable-tol", type=float, default=0.10,
                    help="concurrency mode: relative throughput "
                         "spread (max-min)/mean over the recent windows "
                         "that counts as stable. 0.10 = ±10%.")
    ap.add_argument("--sat-latency-rise", type=float, default=1.5,
                    help="concurrency mode: a throughput plateau only "
                         "counts as saturation if p50 has risen to ≥ this × "
                         "the lowest p50 seen (the closed-loop queue-bound "
                         "knee signal — guards against reading a cold low-c "
                         "warmup stall as the plateau). 0 = disable.")
    ap.add_argument("--sat-step-min-dur", type=float, default=None,
                    help="concurrency mode: minimum seconds before "
                         "the step can finalize even if 'stable' (anti-"
                         "false-positive). Default: max(60, "
                         "stable_window*stable_windows).")
    ap.add_argument("--sat-step-max-dur", type=float, default=None,
                    help="concurrency mode: hard ceiling on step "
                         "duration (s) — if no stable read by then, "
                         "finalize on whatever last windows we have. "
                         "Default: --sat-step-dur.")
    ap.add_argument("--hf-home", default=os.getenv("HF_HOME", ""))
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--burst-window-s", type=float, default=0.0,
                    help="open-loop: if > 0, accumulate arrivals for this "
                         "window then dispatch as one asyncio.gather. "
                         "Mirrors cascade's _draft_pump REGEN burst shape. "
                         "0 = per-arrival fire-and-forget (default).")
    ap.add_argument("--capture-text", action="store_true",
                    help="store full per-request text in the per-record JSONL "
                         "sidecar (needed for Phase E quality grading)")
    # ---- scheduler knobs ----
    ap.add_argument("--disable-load-aware", action="store_true",
                    help="Fall back to pure round-robin picker (pre-)")
    ap.add_argument("--calibrate-every", type=int, default=64,
                    help="Recalibrate scheduler-local load counters every N submits")
    ap.add_argument("--scorer", default=None,
                    choices=[None, "per_source", "c18", "milebench", "c18_aokvqa"],
                    help=" — scorer for sorted-buffer routing. "
                         "Selects a per-source SHIP-rate table from "
                         "prorouter.pre_router.BUILTIN_SOURCE_RATE_TABLES. "
                         "'c18'/'per_source' → DEFAULT_C18_SHIP_RATES; "
                         "'milebench' → DEFAULT_MILEBENCH_SHIP_RATES. "
                         "When set, submit() inserts each request into "
                         "the scheduler's sorted buffer keyed by score; "
                         "the draft dispatcher pulls the highest scorer "
                         "and the target dispatcher pulls the lowest. "
                         "Routing split is emergent — no quantile cutoff. "
                         "When unset, all requests get score 0.0 and the "
                         "buffer behaves FIFO.")
    ap.add_argument("--scorer-model", default=None,
                    help=" — load a pickled source-free scorer "
                         "(TF-IDF + LR) from this path and use it for "
                         "sorted-buffer routing. Mutually exclusive with "
                         "--scorer. Unlike --scorer, this works on "
                         "untagged production traffic (does not read "
                         "pending['source']). End-to-end Pattern A "
                         "validation pending — needs a target node.")
    # ---- baseline gate knobs (--cell cascade_gate) ----
    ap.add_argument("--gate", default=None,
                    choices=[None, "frugalgpt", "output_conf", "query",
                             "transformer_seq"],
                    help=" — scheduler-side ship/escalate GATE (use with "
                         "--cell cascade_gate). Replaces the fork's hidden-"
                         "state head with another signal on identical cascade/"
                         "models/hardware. "
                         "'frugalgpt' → FrugalGPT baseline: a trained scorer "
                         "on (query, draft answer) via --scorer-callable "
                         "(score >= --gate-tau → SHIP). "
                         "'output_conf' → with --gate-stat mean_max_prob / "
                         "neg_mean_entropy this is the Gatekeeper-rule baseline "
                         "(inference rule on the un-fine-tuned 7B; with "
                         "mean/min_logprob it is the raw-logprob ablation A). "
                         "'query' → ablation B: prompt-only classifier "
                         "(--scorer-model) as the gate. (The real query-router "
                         "baseline is RouteLLM, run in its native routing "
                         "topology by a separate harness, NOT here.) "
                         "Sweep --gate-tau to trace the ship-rate/quality "
                         "curve vs the head.")
    ap.add_argument("--gate-tau", type=float, default=None,
                    help=" — SHIP threshold for --gate. output_conf: a "
                         "logprob/confidence threshold (higher = more "
                         "confident = ship). frugalgpt/query: a probability. "
                         "Sweep across cells to trace the curve.")
    ap.add_argument("--gate-stat", default="mean_logprob",
                    choices=["mean_logprob", "min_logprob",
                             "mean_max_prob", "neg_mean_entropy"],
                    help=" — output_conf confidence statistic. "
                         "mean/min_logprob = chosen-token (ablation A); "
                         "mean_max_prob (max-softmax) / neg_mean_entropy "
                         "(negative predictive entropy, top-k approx) = the "
                         "Gatekeeper deferral rule.")
    ap.add_argument("--gate-logprobs-k", type=int, default=20,
                    help=" — top-k logprobs requested from the draft for "
                         "the Gatekeeper-rule stats (max-softmax / entropy). "
                         "Ignored for the chosen-token logprob stats (k=1).")
    ap.add_argument("--scorer-callable", default=None,
                    help=" — 'module:factory' returning a "
                         "scorer(pending, item) -> float for --gate frugalgpt. "
                         "The factory is called with --scorer-callable-arg "
                         "(e.g. the trained DistilBERT ckpt dir). Supply "
                         "your own; none ships here.")
    ap.add_argument("--scorer-callable-arg", default=None,
                    help=" — argument passed to the --scorer-callable "
                         "factory (typically a checkpoint dir/path).")
    # ---- T P18 transformer_seq gate ----
    ap.add_argument("--transformer-ckpt", default=None,
                    help="P18 — path to the trained TransformerSeq L=2 (or "
                         "any sys22t_p16b architecture) checkpoint .pt. "
                         "Required when --gate transformer_seq.")
    ap.add_argument("--transformer-tau-table", default=None,
                    help="P18 — path to the per-source τ table JSON "
                         "(verifier/build_tau_table.py output). "
                         "Required when --gate transformer_seq.")
    ap.add_argument("--transformer-use-global-tau", action="store_true",
                    help="P18 — force global τ for all sources (debug/A/B). "
                         "Default uses per-source τ from the table.")
    ap.add_argument("--in-engine-cascade-head",
                    action="store_true",
                    help="Phase 4 — bring the cascade gate inside "
                         "the vLLM engine. Requires --in-engine-head-ckpt "
                         "and --in-engine-head-tau. Every cascade-routed "
                         "draft request returns CompletionOutput."
                         "head_decision with the SHIP/REGEN verdict; the "
                         "scheduler routes from that directly (no driver-"
                         "side gate forward). One engine call returns "
                         "text AND the routing decision.")
    ap.add_argument("--in-engine-head-ckpt", default=None,
                    help="Phase 4 — path to attn_pool.pt loaded inside "
                         "vLLM. Sets VLLM_CASCADE_ATTN_POOL_CKPT env var "
                         "before engine boot.")
    ap.add_argument("--in-engine-head-tau", default=None,
                    help="Phase 4 — path to the per-source τ table .json. "
                         "Sets VLLM_CASCADE_ATTN_POOL_TAU env var.")
    ap.add_argument("--draft-logprobs-mode",
                    choices=["raw_logprobs", "raw_logits"],
                    default=None,
                    help="logit-features retrofit — pass through to "
                         "the draft engine's AsyncEngineArgs.logprobs_mode. "
                         "'raw_logprobs' (default) is the normal "
                         "log_softmax-space top-K. 'raw_logits' makes vLLM "
                         "skip the log_softmax over the 152k vocab and "
                         "return raw logits in completion.logprobs; "
                         "DraftEngineAsync._drive then builds LOGIT-GAP "
                         "per-token features (t1-t2, t1-t5, t1-t20, "
                         "pos_frac) consumed by a gate trained on the "
                         "matching schema. No vLLM fork patch required.")
    ap.add_argument("--draft-logprobs-only", type=int, default=0,
                    help="Part A — force the draft to request "
                         "logprobs=K on every cascade request WITHOUT "
                         "enabling any gate or head. Isolates the "
                         "logprobs-IPC cost from head-compute. Use K=20 "
                         "(matches the in-engine head's default top-K). "
                         "Errors if combined with --gate or "
                         "--in-engine-cascade-head.")
    ap.add_argument("--draft-emit-per-token-feature-seq",
                    action="store_true",
                    help="P1-followup — set "
                         "SamplingParams.emit_per_token_feature_seq=True "
                         "on every cascade-routed draft request. Requires "
                         "the lp-classifier-inline fork to have the "
                         "patch (per-token feature buffer in "
                         "cascade_lp_classifier + plumbing). When set, "
                         "the fork builds CompletionOutput.per_token_features "
                         "inline; the bench's _drive consumes it directly "
                         "and skips the driver-side per-step Logprob-dict "
                         "construction + detokenization. Saves the ~3-5%% "
                         "of driver-side overhead py-spy measured on B1.")
    ap.add_argument("--cascade-direct-ratio", type=float, default=0.0,
                    help="follow-up — for --cell cascade_gate, "
                         "the fraction R in [0,1) of requests that bypass "
                         "the cascade and go straight to target (force_direct"
                         "=True). At R = DOPT = (T − D(1−s))/(T + D·s) both "
                         "engines co-saturate and `Λ_cascade ≈ T + s·D` "
                         "(recipe). Default 0.0 = all-through-cascade "
                         "(single-tier-bound regime; original behavior).")
    # closed-loop DOPT controller. All default to current static
    # behavior; when --closed-loop (or --direct-ratio) is set on a
    # cascade_gate cell, the SCHEDULER owns the DIRECT/CASCADE split (mutable
    # _direct_ratio) and the bench submits requests plain. Mutually exclusive
    # with --cascade-direct-ratio.
    ap.add_argument("--closed-loop", action="store_true",
                    help=" — enable the scheduler's closed-loop DOPT "
                         "controller (cold-starts the split at 0.5 unless "
                         "--direct-ratio is given, then tunes it each tick "
                         "from the live ship-rate + tier-util imbalance).")
    ap.add_argument("--direct-ratio", type=float, default=None,
                    help=" — initial scheduler-owned DIRECT fraction in "
                         "[0,1]. Sets V0Scheduler(direct_ratio=...). On its "
                         "own (no --closed-loop) this is a runtime-mutable "
                         "static split; with --closed-loop it's the cold-start "
                         "value the controller tunes from.")
    ap.add_argument("--control-t0", type=float, default=None,
                    help=" — target solo ceiling T₀ (r/s) for the "
                         "feedforward DOPT term.")
    ap.add_argument("--control-d0", type=float, default=None,
                    help=" — draft+head solo ceiling D₀ (r/s) for the "
                         "feedforward DOPT term.")
    ap.add_argument("--control-tick-s", type=float, default=1.0,
                    help=" — controller tick period in seconds "
                         "(default 1.0).")
    ap.add_argument("--control-trim-gain", type=float, default=0.15,
                    help=" — feedback trim gain on the tier GPU-util "
                         "imbalance (default 0.15).")
    ap.add_argument("--control-kv-guard", type=float, default=0.92,
                    help=" — KV-proximity throttle guard (target "
                         "kv_in_flight/kv_threshold). Only active if the "
                         "target bucket was booted with a send rate.")
    ap.add_argument("--rate-match", action="store_true",
                    help=" — rate-match credit dispatch: per-engine push "
                         "rate is capped at the engine's MEASURED throughput "
                         "(completions seen on the pump path, no extra RPC), "
                         "retuned every --rate-match-tick-s with a "
                         "(tick+buffer)s credit burst. Push tracks drain so the "
                         "queue stays shallow; the split + throughput "
                         "self-balance, no DOPT formula. Submit plain.")
    ap.add_argument("--rate-match-tick-s", type=float, default=2.0,
                    help=" — rate-match retune period (s).")
    ap.add_argument("--rate-match-buffer-s", type=float, default=1.0,
                    help=" — extra credit burst (s) so the engine never "
                         "starves between retunes.")
    ap.add_argument("--rate-match-headroom", type=float, default=0.15,
                    help=" — push-rate headroom over measured throughput "
                         "(keeps the engine fed).")
    ap.add_argument("--rate-match-init-rps", type=float, default=16.0,
                    help=" — initial per-engine push-rate guess (r/s) "
                         "before the first measurement; headroom ramps it to "
                         "the true capacity over a few ticks (not unlimited).")
    ap.add_argument("--occupancy-gate", action="store_true",
                    help=" — occupancy-gated admission: a dispatch loop "
                         "HOLDS when its engine is full (in-flight ≥ hwm·max OR "
                         "target KV proximity ≥ kv_hwm), resumes below lwm. "
                         "Direct backpressure off the KV cliff. Submit plain.")
    ap.add_argument("--occupancy-max-inflight", type=int, default=256,
                    help=" — per-engine in-flight cap for the occupancy "
                         "gate (≈ engine max_num_seqs).")
    ap.add_argument("--occupancy-hwm", type=float, default=0.90,
                    help=" — high watermark (fraction) to start holding.")
    ap.add_argument("--occupancy-lwm", type=float, default=0.70,
                    help=" — low watermark (fraction) to resume.")
    ap.add_argument("--occupancy-kv-hwm", type=float, default=0.90,
                    help=" — target KV-proximity high watermark.")
    # #1 MA-length KV gating + #2 throughput-adaptive batch dispatch.
    ap.add_argument("--ma-length-gating", action="store_true",
                    help=" #1 — reserve regen KV from the per-source p90 of "
                         "OBSERVED output lengths (capped at max_tokens) instead "
                         "of worst-case max_tokens. Higher KV utilization; vLLM "
                         "preemption is the backstop on under-prediction.")
    ap.add_argument("--adaptive-batch", action="store_true",
                    help=" #2 — target batch self-sizes to the actor's "
                         "measured throughput (clamp(MA_tput × window, [min,max])) "
                         "instead of one item per RPC.")
    ap.add_argument("--adaptive-batch-window-s", type=float, default=0.05,
                    help=" #2 — batch window (s): batch ≈ MA_tput × this.")
    ap.add_argument("--adaptive-batch-min", type=int, default=1,
                    help=" #2 — min batch size.")
    ap.add_argument("--adaptive-batch-max", type=int, default=16,
                    help=" #2 — max batch size.")
    ap.add_argument("--adaptive-batch-rtt-aware", action="store_true",
                    help="wire-latency fix — size the target batch as "
                         "tput×(RTT_ema+window)+buffer instead of "
                         "tput×window, so a batch covers the dispatch RPC "
                         "round trip (keeps the actor busy on slow links).")
    ap.add_argument("--adaptive-batch-buffer", type=int, default=2,
                    help=" — additive batch headroom when rtt-aware.")
    ap.add_argument("--draft-submit-pipeline", type=int, default=0,
                    help="wire-latency fix — allow N draft submit-ACK "
                         "RPCs in flight instead of awaiting each inline "
                         "(0 = legacy serial dispatch).")
    ap.add_argument("--draft-submit-batch", type=int, default=0,
                    help="wire-latency fix — coalesce up to N draft "
                         "submits into one submit_batch RPC so a round trip "
                         "amortizes over the batch (0 = per-request submit). "
                         "Skipped on encoder-pool / token-bucket cells.")
    ap.add_argument("--target-submit-pipeline", type=int, default=0,
                    help="wire-latency fix (1s residual) — allow N target "
                         "dispatch RPCs in flight per target instead of the "
                         "serial one-at-a-time await (0 = legacy serial). The "
                         "serial path caps target dispatch at batch/RTT; raise "
                         "on high-latency links (e.g. 8).")
    ap.add_argument("--actor-self-admit", action="store_true",
                    help=" — move admission control into the model actors "
                         "(draft AND target): the RPC just buffers the request "
                         "and returns stats; a background loop admits into vLLM "
                         "bounded by a local capacity cap. Dispatch stops being "
                         "gated by RTT-inflated in-flight. Default off.")
    ap.add_argument("--actor-admit-interval-ms", type=float, default=5.0,
                    help=" — background admission-loop cadence (ms).")
    ap.add_argument("--actor-admit-max-inflight", type=int, default=256,
                    help=" — per-actor concurrency cap for self-admit "
                         "(the KV-safe local admission limit).")
    ap.add_argument("--actor-admit-kv-threshold", type=int, default=0,
                    help=" — draft self-admit KV-token budget (reserve "
                         "learned prefill KV per request; hold at this "
                         "threshold). 0 = auto-derive 0.85x(actual draft KV "
                         "pool); <0 = disable (count cap only).")
    ap.add_argument("--pop-max-n", type=int, default=0,
                    help=" — cap on finished items drained per pop_finished "
                         "poll. DEFAULT 0 = drain everything finished (recommended: "
                         "a finished item is already done, so a finite cap only "
                         "defers it to the next poll = another whole RTT, silently "
                         "throttling the serial return pump to max_n/RTT). Set >0 "
                         "only to bound the RPC response size.")
    ap.add_argument("--max-buffer-depth", type=int, default=0,
                    help=" — bound the scheduler's pre-dispatch buffer "
                         "(fresh + regen). A fresh arrival backpressures while "
                         "the buffer is at the cap; REGEN re-entries bypass. "
                         "0 = unbounded (legacy). Set to ~2× the concurrency "
                         "cap under open-loop overload to keep the backlog (and "
                         "p50) bounded instead of growing to 10s of thousands.")
    ap.add_argument("--emergent-dispatch", action="store_true",
                    help=" — emergent 2-buffer backpressure dispatch: no "
                         "pinned split, no DOPT formula. Submit plain; the draft "
                         "pulls fresh at its capacity, the target drains REGENs "
                         "(re-buffered at −∞, the priority queue) then fresh at "
                         "its capacity. The split + DOPT throughput emerge from "
                         "backpressure. Mutually exclusive with --closed-loop / "
                         "--direct-ratio / --cascade-direct-ratio.")
    ap.add_argument("--dispatch-direct-rpc", action="store_true",
                    help="(ablation arm b) — bypass the sorted "
                         "buffer AND all batched/backpressure dispatch; fire "
                         "the per-request engine RPC on arrival ('the scheduler "
                         "just keeps calling the model actor's RPC'). "
                         "Force-disables --adaptive-batch / --emergent-dispatch "
                         "/ --occupancy-gate / --rate-match (with a loud log).")
    ap.add_argument("--inline-self-eval", default=None,
                    choices=["ptrue", "automix"],
                    help="Part 0b — after each draft generation, run a "
                         "blocking self-eval pass on the SAME draft engine "
                         "(P(True) k=1 greedy, or AutoMix k=8 @ T=1.0 "
                         "prompts). Score is logged on the Response; routing "
                         "is unaffected. Default off.")
    ap.add_argument("--target-only-legacy-router", action="store_true",
                    help="opt out of routing target_only through "
                         "V0Scheduler(force_direct_target=True). When set, "
                         "target_only uses the pre-direct _TargetOnlyRouter "
                         "path — faster but pays no scheduler overhead, so r/s "
                         "is not apples-to-apples with cascade cells.")
    # ---- two-stream knobs ----
    ap.add_argument("--lambda-target", type=float, default=0.0,
                    help="Poisson rate for the target-direct stream "
                         "in --cell two_stream. Bypasses V0Scheduler; calls "
                         "target.submit_regen with raw images (target's own "
                         "ViT). Set to 0 to disable this stream.")
    ap.add_argument("--lambda-draft", type=float, default=0.0,
                    help="Poisson rate for the cascade stream in "
                         "--cell two_stream. Goes through V0Scheduler + "
                         "encoder pool (if --n-encoders > 0) + draft + "
                         "ACCEPT/REGEN routing. Encoder capacity is checked "
                         "against this rate (see --allow-encoder-bound).")
    ap.add_argument("--allow-encoder-bound", action="store_true",
                    help="skip the encoder-capacity sanity check "
                         "before running. By default, --cell two_stream "
                         "aborts if lambda_draft × imgs_per_rec > 0.8 × "
                         "n_encoders × 12 r/s (per-A10G ceiling from). "
                         "Pass this flag to measure encoder-bound regimes "
                         "intentionally.")
    args = ap.parse_args()
    if args.cell == "two_stream":
        if args.lambda_target <= 0 and args.lambda_draft <= 0:
            print("ERROR: --cell two_stream needs --lambda-target > 0 "
                  "and/or --lambda-draft > 0")
            return 2
    elif args.probe_oneby:
        if args.concurrencies or args.arrival_rates or args.auto_saturate:
            print("ERROR: --probe-oneby is mutually exclusive with "
                  "--concurrencies / --arrival-rates / --auto-saturate")
            return 2
    elif not args.concurrencies and not args.arrival_rates \
            and not args.auto_saturate:
        print("ERROR: pass --concurrencies, --arrival-rates, --auto-saturate, "
              "or --probe-oneby")
        return 2
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
