"""Run the two-tier cascade on a Ray cluster.

A draft engine and a target engine are launched as Ray actors on separate
nodes. Each request is scored by the pre-scorer before generation (which orders
a shared buffer, so likely-shippable work reaches the draft first), answered by
the draft, then judged by the confidence head: above threshold the draft answer
ships, below it the request is re-generated on the target. Dispatch is
backpressure-driven -- the draft pulls from the confident end of the buffer and
the target drains escalations plus whatever the draft could not absorb -- and
admission is held while an engine is full, so neither tier idles and neither
overloads.

    python run_pipeline.py --records bench_test.jsonl \
        --head weights/head.pt --tau weights/tau.json \
        --scorer weights/prescorer.pkl --concurrency 256

Needs a running Ray cluster (`ray start`) with the draft and target resources
advertised; see README.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hardware  # noqa: E402


def _records(path: str, limit: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[:limit] if limit else rows


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


async def _drive(sched, records, concurrency, max_tokens, duration_s,
                 request_timeout_s=300.0, force_cascade=False):
    """Closed-loop load: `concurrency` workers pulling from a shared cursor.

    Every submit is bounded by `request_timeout_s`. Without that, `duration_s`
    is only consulted *between* requests, so a single submit that never resolves
    (an engine that failed to start, say) hangs the whole run with no output --
    the deadline can never fire because every worker is parked inside await.
    """
    results, cursor, stop_at = [], 0, time.perf_counter() + duration_s
    lock = asyncio.Lock()

    async def worker():
        nonlocal cursor
        while True:
            async with lock:
                if cursor >= len(records) or time.perf_counter() > stop_at:
                    return
                rec = records[cursor]
                cursor += 1
            t0 = time.perf_counter()
            try:
                resp = await asyncio.wait_for(
                    sched.submit(
                        prompt=rec["prompt"], max_tokens=max_tokens,
                        image_paths=rec.get("images") or None,
                        source=rec.get("source"),
                        force_cascade=force_cascade,
                    ),
                    timeout=request_timeout_s,
                )
            except asyncio.TimeoutError:
                results.append({
                    "id": rec["id"], "source": rec.get("source"),
                    "verdict": "ERROR", "routing_path": None,
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                    "n_output_tokens": None, "hit_token_cap": False,
                    "text": "", "gold": rec.get("gold"),
                    "error": f"no response within --request-timeout-s "
                             f"({request_timeout_s}s)",
                })
                continue
            results.append({
                "id": rec["id"], "source": rec.get("source"),
                "verdict": resp.verdict, "routing_path": resp.routing_path,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "n_output_tokens": resp.n_output_tokens,
                "hit_token_cap": (resp.n_output_tokens is not None
                                  and resp.n_output_tokens >= max_tokens),
                # kept so a ship decision can be checked against the answer it
                # shipped: without the text, ship_rate is unfalsifiable
                "text": resp.text,
                "gold": rec.get("gold"),
                "error": resp.error,
            })

    t_start = time.perf_counter()
    await asyncio.gather(*[asyncio.create_task(worker())
                           for _ in range(concurrency)])
    return results, time.perf_counter() - t_start


#: escalation rates outside this band mean the head is not discriminating.
DEGENERATE_BAND = (0.05, 0.95)
#: below this many cascaded requests a per-source rate is too noisy to judge.
MIN_SOURCE_N = 20


def _escalation_by_source(ok: list[dict]) -> dict:
    """Per-source {n_cascaded, escalation_rate}, for the degeneracy check."""
    out: dict[str, dict] = {}
    for r in ok:
        if r["verdict"] in ("ACCEPT", "REGEN"):
            e = out.setdefault(r.get("source") or "unknown",
                               {"n_cascaded": 0, "n_regen": 0})
            e["n_cascaded"] += 1
            e["n_regen"] += int(r["verdict"] == "REGEN")
    for e in out.values():
        e["escalation_rate"] = round(e["n_regen"] / e["n_cascaded"], 4)
    return dict(sorted(out.items()))


def degenerate_sources(by_source: dict) -> list[tuple[str, float, int]]:
    """Sources with enough traffic to judge and an escalation rate off the band."""
    lo, hi = DEGENERATE_BAND
    return [(src, e["escalation_rate"], e["n_cascaded"])
            for src, e in (by_source or {}).items()
            if e["n_cascaded"] >= MIN_SOURCE_N
            and not lo <= e["escalation_rate"] <= hi]


def summarize(results: list[dict], wall_s: float) -> dict:
    ok = [r for r in results if r["verdict"] != "ERROR"]
    lat = [r["latency_ms"] for r in ok]
    capped = [r for r in ok if r.get("hit_token_cap")]
    accept = sum(1 for r in ok if r["verdict"] == "ACCEPT")
    regen = sum(1 for r in ok if r["verdict"] == "REGEN")
    direct = sum(1 for r in ok if r["verdict"] == "DIRECT_TARGET")
    cascaded = accept + regen
    return {
        "n_total": len(results),
        "n_error": len(results) - len(ok),
        "wall_s": round(wall_s, 2),
        "throughput_rps": round(len(ok) / wall_s, 3) if wall_s > 0 else None,
        "routing": {"accept": accept, "regen": regen, "direct_target": direct},
        # the head's positive rate on cascade-routed requests; independent of
        # how much traffic bypassed the draft
        "ship_rate_per_cascade": round(accept / cascaded, 4) if cascaded else None,
        # a generation stopped by the cap ends while the model is still
        # locally confident, which lifts the head's score: read the ship rate
        # together with this fraction, never on its own
        "token_cap_hit_rate": round(len(capped) / len(ok), 4) if ok else None,
        # at a near-0% or near-100% escalation rate the head is not being
        # asked to discriminate, and the ship rate carries no information --
        # any head scores well there
        "escalation_rate_per_cascade": round(regen / cascaded, 4) if cascaded else None,
        # tau is per-source, so degeneracy has to be checked per-source: one
        # source escalating healthily hides another escalating nothing, and the
        # aggregate stays inside any sane band
        "escalation_by_source": _escalation_by_source(ok),
        "latency_ms": {
            "p50": round(_percentile(lat, 0.50), 1),
            "p90": round(_percentile(lat, 0.90), 1),
            "p99": round(_percentile(lat, 0.99), 1),
            "mean": round(statistics.fmean(lat), 1) if lat else None,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--head", required=True, help="Confidence head checkpoint.")
    ap.add_argument("--tau", required=True, help="Per-source threshold table.")
    ap.add_argument("--scorer", default=None,
                    help="Pre-scorer pickle. Omitted -> FIFO buffer order.")
    ap.add_argument("--small-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--large-model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--small-tp", type=int, default=0,
                    help="0 = derive from the GPUs on the small-model node.")
    ap.add_argument("--large-tp", type=int, default=0,
                    help="0 = derive from the GPUs on the large-model node.")
    ap.add_argument("--small-mem-util", type=float, default=0.85)
    ap.add_argument("--large-mem-util", type=float, default=0.93)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="Generation cap. Short caps truncate answers "
                         "mid-sequence, which raises the ship rate; the "
                         "summary reports the cap-hit fraction so the "
                         "effect is visible.")
    ap.add_argument("--max-pixels", type=int, default=1003520,
                    help="0 disables; ignored by non-Qwen-VL processors.")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-images", type=int, default=2)
    ap.add_argument("--small-resource", default=None,
                    help="Ray node resource for the small-model engine. "
                         "Ignored if the cluster does not advertise it.")
    ap.add_argument("--large-resource", default=None,
                    help="Ray node resource for the large-model engine.")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--duration-s", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--request-timeout-s", type=float, default=300.0,
                    help="Per-request ceiling. Keeps a stalled engine from "
                         "hanging the run forever.")
    ap.add_argument("--engine-timeout-s", type=float, default=1200.0,
                    help="How long to wait for both engines to answer ping "
                         "before giving up with a diagnostic.")
    ap.add_argument("--force-cascade", action="store_true",
                    help="Route every request through the small model and the "
                         "head, disabling direct-to-large spillover (and with "
                         "it two-buffer dispatch). Needed with --mock-large: "
                         "the stub reports unlimited headroom, so backpressure "
                         "otherwise sends everything straight past the head and "
                         "the gate never runs. Not a throughput configuration.")
    ap.add_argument("--mock-large", action="store_true",
                    help="Substitute a stub large-model actor (no large-model GPUs).")
    ap.add_argument("--ray-address", default="auto",
                    help="'auto' joins a running cluster; 'local' starts a single-node one.")
    ap.add_argument("--out", default="cascade_run.json")
    args = ap.parse_args()

    import ray

    from prorouter.engine import DraftEngineAsync, TargetEngineAsync
    from prorouter.gate import transformer_seq_gate
    from prorouter.launcher import DRAFT_RESOURCE, TARGET_RESOURCE
    from prorouter.scheduler import V0Scheduler

    # `prorouter` has to be importable in the actor processes. Ray only puts the
    # driver's cwd on the worker path, so unless the driver happens to run from
    # pipeline/ the actors die with ModuleNotFoundError: No module named 'prorouter'.
    # py_modules ships the package itself, which works from any cwd and needs
    # no install on the workers.
    pkg_dir = str(Path(__file__).resolve().parent / "prorouter")
    ray.init(address=args.ray_address,
             runtime_env={"py_modules": [pkg_dir]})

    records = _records(args.records, args.limit)
    print(f"[cascade] {len(records)} records", flush=True)

    print(f"[cascade] cluster: {hardware.describe_cluster()}", flush=True)
    small_tp = hardware.pick_tensor_parallel(args.small_tp, label="small model")
    # Only resolved when a real large-model engine is launched. --mock-large
    # substitutes a zero-GPU stub that takes no tensor-parallel size, and on a
    # CPU-only driver this call refuses to guess -- so asking for it here would
    # fail the smoke path over a model that is never started.
    large_tp = (0 if args.mock_large else
                hardware.pick_tensor_parallel(args.large_tp, label="large model"))
    common = dict(dtype=hardware.pick_dtype(args.dtype),
                  max_model_len=args.max_model_len,
                  distributed_executor_backend="mp")

    if args.mock_large:
        from prorouter.mock_target import MockTargetActor
        # serve_decode=True is required: the scheduler routes REGEN escalations
        # and DIRECT overflow to the target, and the stub's default mode raises
        # on those calls instead of serving them.
        target = MockTargetActor.remote(serve_decode=True)
    else:
        target = TargetEngineAsync.options(
            **hardware.placement_kwargs(
                args.large_resource or TARGET_RESOURCE, large_tp, "large model"),
        ).remote(model_id=args.large_model,
                 tensor_parallel_size=large_tp,
                 gpu_memory_utilization=args.large_mem_util,
                 **hardware.mm_kwargs(args.large_model, args.max_pixels,
                                      args.max_images), **common)

    draft = DraftEngineAsync.options(
        **hardware.placement_kwargs(
            args.small_resource or DRAFT_RESOURCE, small_tp, "small model"),
    ).remote(model_id=args.small_model,
             tensor_parallel_size=small_tp,
             gpu_memory_utilization=args.small_mem_util,
             **hardware.mm_kwargs(args.small_model, args.max_pixels,
                                  args.max_images), **common)

    # An actor that cannot be scheduled stays PENDING indefinitely and Ray says
    # nothing. Ping both engines up front so a placement or OOM problem surfaces
    # here, with a message, instead of as a silent hang at the first request.
    print(f"[cascade] waiting for engines (up to {args.engine_timeout_s:.0f}s)",
          flush=True)
    try:
        ray.get([draft.ping.remote(), target.ping.remote()],
                timeout=args.engine_timeout_s)
    except ray.exceptions.GetTimeoutError:
        raise SystemExit(
            f"[cascade] an engine did not become ready within "
            f"{args.engine_timeout_s:.0f}s. Usual causes:\n"
            f"  * the node cannot satisfy the GPU request (small-tp="
            f"{small_tp}, large-tp={large_tp}); check `ray status`\n"
            f"  * a requested node resource is advertised but already claimed\n"
            f"  * leftover vLLM workers are holding the GPUs -- see the README\n"
            f"  * the model is still downloading; raise --engine-timeout-s")
    print("[cascade] engines ready", flush=True)

    gate = transformer_seq_gate(ckpt_path=args.head,
                                tau_table=json.load(open(args.tau)))
    scorer = None
    if args.scorer:
        from prorouter.pre_router import model_scorer
        scorer = model_scorer(args.scorer)
        print(f"[cascade] pre-scorer loaded from {args.scorer}", flush=True)

    sched = V0Scheduler(
        drafts=[draft], targets=[target],
        scorer=scorer, gate=gate, draft_logprobs=20,
        # backpressure dispatch: one buffer, the draft pulls the confident
        # end and the target drains escalations -- the split self-balances.
        # force_cascade routes through the *sorted* buffer instead, which the
        # two_buffer dispatch loops do not drain, so the two are mutually
        # exclusive: asking for both leaves every request unclaimed until
        # --request-timeout-s fires.
        two_buffer=not args.force_cascade,
        direct_ratio=None,
        occupancy_gate=True,
        occupancy_max_inflight=args.concurrency,
    )

    async def run():
        await sched.start()
        try:
            return await _drive(sched, records, args.concurrency,
                                args.max_tokens, args.duration_s,
                                request_timeout_s=args.request_timeout_s,
                                force_cascade=args.force_cascade)
        finally:
            await sched.stop()

    results, wall = asyncio.run(run())
    summary = summarize(results, wall)

    Path(args.out).write_text(json.dumps(
        {"summary": summary, "config": vars(args), "requests": results},
        indent=1, default=str))

    print("\n=== cascade run ===")
    print(f"  served              {summary['n_total'] - summary['n_error']}"
          f" ({summary['n_error']} errors) in {summary['wall_s']}s")
    print(f"  throughput          {summary['throughput_rps']} req/s")
    print(f"  routing             {summary['routing']}")
    print(f"  ship rate (cascade) {summary['ship_rate_per_cascade']}"
          f"   (token-cap hit {summary['token_cap_hit_rate']})")
    print(f"  latency ms          {summary['latency_ms']}")
    esc = summary.get("escalation_rate_per_cascade")
    if esc is not None:
        print(f"  escalation (cascade) {esc}")
    by_src = summary.get("escalation_by_source") or {}
    for src, e in by_src.items():
        print(f"    {src:<12s} {e['escalation_rate']:.4f} "
              f"({e['n_regen']}/{e['n_cascaded']})")

    lo, hi = DEGENERATE_BAND
    bad = degenerate_sources(by_src)
    aggregate_bad = esc is not None and not lo <= esc <= hi
    if bad or aggregate_bad:
        print("\n  WARNING: the head is not discriminating at this operating "
              "point.")
        if aggregate_bad:
            print(f"    aggregate escalation {esc:.3f} is outside "
                  f"{lo:.0%}-{hi:.0%}")
        for src, rate, n in bad:
            print(f"    {src}: {rate:.3f} over {n} cascaded requests")
        # tau is per-source, so a healthy aggregate proves nothing: one source
        # escalating normally masks another escalating nothing at all.
        if bad and not aggregate_bad:
            print(f"    (the aggregate {esc:.3f} sits inside the band and "
                  f"hides this)")
        print("\n  The threshold table is calibrated for a particular model "
              "pair and\n  answer-length distribution; on a different setup it "
              "lands at the wrong\n  quantile and the head ships (or escalates)"
              " almost everything. A ship\n  rate measured here says nothing "
              "about routing quality -- recalibrate tau\n  on your own "
              "validation split first, per source.")
    if summary.get("token_cap_hit_rate", 0):
        print(f"\n  note: {summary['token_cap_hit_rate']:.1%} of generations hit "
              f"the token cap; quote this beside the ship rate.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
