"""Unit smoke test for V0Scheduler (load-aware + direct-target).

No Ray, no GPU. Stub actors implement the subset of the Ray actor
interface the scheduler uses:
    actor.submit.remote(...)
    actor.pop_finished.remote(max_n, timeout_s)
    actor.qsize.remote()
    actor.gpu_util.remote()
    actor.submit_verify.remote(...)   (target)
    actor.submit_regen.remote(...)    (target)

Tests:
  T1  load-aware picker distributes correctly when one draft is slow
  T2  calibration reconciles drift after manual desync
  T3  direct-target spillover triggers when drafts saturated +
      target idle
  T4  direct-target spillover suppressed when expected REGEN inflow
      reserves capacity
  T5  Response.routing_path / verdict labelling correct on each path
"""
from __future__ import annotations

import asyncio
import random
import time

from prorouter.scheduler import V0Scheduler


# ---------------- Stub actor scaffolding ----------------


class _RemoteStub:
    """Make an async method respond to .remote(...) like a Ray actor."""
    def __init__(self, fn):
        self._fn = fn
    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class StubDraftActor:
    """In-process draft actor stub.

    submit(req_id, prompt, max_tokens, temperature, ignore_eos,
           image_path, head_cascade, image_paths)
      → schedules a coroutine that completes after `decode_delay_s`
        seconds and pushes a finished item.

    Item shape mirrors what DraftEngineAsync.submit produces:
      {req_id, text, n_output_tokens, finish_reason, head_decision}
    """
    def __init__(
        self,
        name: str,
        *,
        decode_delay_s: float = 0.01,
        head_decision: str = "SHIP",  # "SHIP" | "REGEN" | None
        gpu_util_value: int = 10,
        submit_delay_s: float = 0.0,  # fake wire RTT on submit RPCs
    ) -> None:
        self.name = name
        self._decode_delay_s = decode_delay_s
        self._head_decision = head_decision
        self._gpu_util_value = gpu_util_value
        self._submit_delay_s = submit_delay_s
        self._finished_q: asyncio.Queue = asyncio.Queue()
        self._in_flight = 0
        self.submit = _RemoteStub(self._submit)
        self.submit_batch = _RemoteStub(self._submit_batch)
        self.pop_finished = _RemoteStub(self._pop_finished)
        self.qsize = _RemoteStub(self._qsize)
        self.gpu_util = _RemoteStub(self._gpu_util)
        # Counters for assertions.
        self.n_submitted = 0
        self.batch_sizes: list[int] = []

    async def _submit(
        self, req_id, prompt, max_tokens, temperature, ignore_eos,
        image_path, head_cascade, image_paths,
        image_embeds=None, image_grid_thw=None,
        # Current DraftEngineAsync.submit surface — the dispatch loop passes
        # these unconditionally. Accept + ignore so the
        # stub tracks the live RPC signature.
        **kwargs,
    ):
        if self._submit_delay_s:
            await asyncio.sleep(self._submit_delay_s)
        self._in_flight += 1
        self.n_submitted += 1
        asyncio.create_task(self._drive(req_id, head_cascade))

    async def _submit_batch(self, items):
        """Mirrors DraftEngineAsync.submit_batch — one wire delay
        for the whole batch, then per-item submit."""
        if self._submit_delay_s:
            await asyncio.sleep(self._submit_delay_s)
        self.batch_sizes.append(len(items))
        for it in items:
            self._in_flight += 1
            self.n_submitted += 1
            asyncio.create_task(
                self._drive(it["req_id"], it.get("head_cascade"))
            )

    async def _drive(self, req_id, head_cascade):
        await asyncio.sleep(self._decode_delay_s)
        item = {
            "req_id": req_id,
            "text": f"draft-resp-{req_id}",
            "n_output_tokens": 16,
            "finish_reason": "stop",
            "completed_t": time.perf_counter(),
        }
        # cascade_retesting: head_decision only emitted when head fires.
        # head_cascade=False (explicit opt-out) → no verdict in item.
        if head_cascade is not False:
            item["head_decision"] = self._head_decision
        await self._finished_q.put(item)
        self._in_flight -= 1

    async def _pop_finished(self, max_n=32, timeout_s=0.05):
        items = []
        try:
            first = await asyncio.wait_for(
                self._finished_q.get(), timeout=timeout_s,
            )
            items.append(first)
        except asyncio.TimeoutError:
            return items
        while len(items) < max_n:
            try:
                items.append(self._finished_q.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    async def _qsize(self):
        return {"finished": self._finished_q.qsize(), "in_flight": self._in_flight}

    async def _gpu_util(self):
        return [{"gpu": 0, "util_pct": self._gpu_util_value, "mem_pct": 50}]


class StubTargetActor:
    """In-process target actor stub.

    Implements submit_verify and submit_regen — both schedule a
    completion after `target_delay_s` and push a finished item.

    verify produces verdict="ACCEPT" (deterministic — drafts in this
    test stream a fixed "SHIP" cascade head_decision so the verify
    path isn't exercised by default; flip via the stub config if
    needed).
    """
    def __init__(
        self,
        name: str,
        *,
        target_delay_s: float = 0.02,
        gpu_util_value: int = 5,
        regen_verdict: str = "REGEN",
        submit_delay_s: float = 0.0,  # fake wire RTT on dispatch RPCs
    ) -> None:
        self.name = name
        self._target_delay_s = target_delay_s
        self._submit_delay_s = submit_delay_s
        self._max_concurrent_batches = 0
        self._inflight_batches = 0
        self._gpu_util_value = gpu_util_value
        self._regen_verdict = regen_verdict
        self._finished_q: asyncio.Queue = asyncio.Queue()
        # All in-flight requests + queue depth. Mirrors what
        # TargetEngineAsync.qsize returns.
        self._in_flight = 0
        self._verify_q_depth = 0
        self._regen_q_depth = 0
        self.submit_verify = _RemoteStub(self._submit_verify)
        self.submit_regen = _RemoteStub(self._submit_regen)
        # Scheduler now calls submit_decode (submit_regen kept
        # as a server-side alias on TargetEngineAsync). Mock both for
        # apples-to-apples behaviour in the stub.
        self.submit_decode = _RemoteStub(self._submit_regen)
        # Batched dispatch path.
        self.submit_decode_batch = _RemoteStub(self._submit_decode_batch)
        self.pop_finished = _RemoteStub(self._pop_finished)
        self.qsize = _RemoteStub(self._qsize)
        self.gpu_util = _RemoteStub(self._gpu_util)
        self.n_verify = 0
        self.n_regen = 0
        # Batched-RPC counters.
        self.n_batch_calls = 0
        self.batch_call_sizes: list[int] = []
        # Capture every submit_regen call's args so tests can
        # assert REGEN and DIRECT pass identical signatures (target
        # cannot distinguish them).
        self.regen_calls: list[dict] = []

    async def _submit_verify(
        self, req_id, prompt, draft_response, max_tokens, ignore_eos,
    ):
        self._in_flight += 1
        self.n_verify += 1
        asyncio.create_task(self._drive_verify(req_id, draft_response))

    async def _drive_verify(self, req_id, draft_response):
        await asyncio.sleep(self._target_delay_s)
        await self._finished_q.put({
            "req_id": req_id,
            "verdict": "ACCEPT",
            "text": draft_response,
            "completed_t": time.perf_counter(),
        })
        self._in_flight -= 1

    async def _submit_regen(
        self, req_id, prompt, max_tokens, ignore_eos,
        image_path, image_paths,
    ):
        self._in_flight += 1
        self.n_regen += 1
        self.regen_calls.append({
            "req_id": req_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "ignore_eos": ignore_eos,
            "image_path": image_path,
            "image_paths": image_paths,
        })
        asyncio.create_task(self._drive_regen(req_id))

    async def _submit_decode_batch(self, items):
        """Batched dispatch entry. Replays each item as if it
        had arrived via the single-call path so the rest of the stub
        (n_regen, regen_calls, _drive_regen) keeps working unchanged."""
        self.n_batch_calls += 1
        self.batch_call_sizes.append(len(items))
        # Fake the dispatch-RPC wire round trip so a serial caller is
        # capped at 1/submit_delay_s while a pipelined caller overlaps them.
        # Track peak concurrency to prove the pipeline is actually in flight.
        if self._submit_delay_s:
            self._inflight_batches += 1
            self._max_concurrent_batches = max(
                self._max_concurrent_batches, self._inflight_batches,
            )
            try:
                await asyncio.sleep(self._submit_delay_s)
            finally:
                self._inflight_batches -= 1
        for it in items:
            await self._submit_regen(
                it["req_id"], it["prompt"], it["max_tokens"],
                it.get("ignore_eos", False),
                it.get("image_path"), it.get("image_paths"),
            )

    async def _drive_regen(self, req_id):
        await asyncio.sleep(self._target_delay_s)
        await self._finished_q.put({
            "req_id": req_id,
            "verdict": self._regen_verdict,
            "text": f"target-resp-{req_id}",
            "n_output_tokens": 32,
            "finish_reason": "stop",
            "completed_t": time.perf_counter(),
        })
        self._in_flight -= 1

    async def _pop_finished(self, max_n=32, timeout_s=0.05):
        items = []
        try:
            first = await asyncio.wait_for(
                self._finished_q.get(), timeout=timeout_s,
            )
            items.append(first)
        except asyncio.TimeoutError:
            return items
        while len(items) < max_n:
            try:
                items.append(self._finished_q.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    async def _qsize(self):
        return {
            "verify": self._verify_q_depth,
            "regen": self._regen_q_depth,
            "finished": self._finished_q.qsize(),
            "in_flight": self._in_flight,
            "kv_in_flight": 0,
            "kv_threshold": 1_000_000,
        }

    async def _gpu_util(self):
        return [{"gpu": 0, "util_pct": self._gpu_util_value, "mem_pct": 50}]


# ---------------- Tests ----------------


async def _drain(sched: V0Scheduler, n: int, **kwargs) -> list:
    """Submit n requests concurrently and gather responses."""
    return await asyncio.gather(*[
        sched.submit(f"prompt-{i}", **kwargs) for i in range(n)
    ])


async def _stream(
    sched: V0Scheduler, n: int, inter_arrival_s: float, **kwargs,
) -> list:
    """Submit n requests with `inter_arrival_s` between submits, then
    gather responses. Models a continuous-arrival workload so the
    load-aware picker sees completions interleaved with new submits."""
    tasks: list = []
    for i in range(n):
        tasks.append(asyncio.create_task(
            sched.submit(f"prompt-{i}", **kwargs)
        ))
        if inter_arrival_s > 0:
            await asyncio.sleep(inter_arrival_s)
    return await asyncio.gather(*tasks)


async def t1_multi_draft_dispatch_serves_all() -> None:
    """with multiple draft actors, each runs its own
    rate-gated dispatch loop popping the buffer's top end. Both
    actors get work; all submits complete. (Pre-this
    test asserted a fast/slow split via a central load-aware
    picker; the per-actor loops produced by the collapse
    don't have a central picker, so load balancing now relies on
    the actor's submit RPC backpressure — invisible with stub
    actors that return submit() immediately.)
    """
    fast = StubDraftActor("fast", decode_delay_s=0.005)
    slow = StubDraftActor("slow", decode_delay_s=0.060)
    tgt = StubTargetActor("tgt", target_delay_s=0.001)
    sched = V0Scheduler(
        drafts=[fast, slow],
        targets=[tgt],
        draft_pop_timeout_s=0.002,
        calibrate_every=1000,
    )
    await sched.start()
    try:
        responses = await _stream(
            sched, 30, inter_arrival_s=0.010, force_cascade=True,
        )
    finally:
        await sched.stop()
    assert fast.n_submitted + slow.n_submitted == 30, (
        f"some submits lost: fast={fast.n_submitted} slow={slow.n_submitted}"
    )
    assert fast.n_submitted > 0 and slow.n_submitted > 0, (
        f"one draft got nothing: fast={fast.n_submitted} slow={slow.n_submitted}"
    )
    assert all(r.verdict == "ACCEPT" for r in responses)
    print(f"  T1 OK  fast={fast.n_submitted} slow={slow.n_submitted} (both got work)")


async def t1b_picker_unit() -> None:
    """Direct test of _pick_draft logic against manipulated loads."""
    d0 = StubDraftActor("d0")
    d1 = StubDraftActor("d1")
    d2 = StubDraftActor("d2")
    tgt = StubTargetActor("tgt")
    sched = V0Scheduler(drafts=[d0, d1, d2], targets=[tgt])
    # All equal → RR tiebreak should cycle 0, 1, 2, 0, 1, 2.
    picks = [sched._pick_draft() for _ in range(6)]
    assert picks == [0, 1, 2, 0, 1, 2], picks
    # Skew: d1 has lower load → picker should always pick d1.
    sched._draft_load = [3, 1, 3]
    picks = [sched._pick_draft() for _ in range(4)]
    assert all(p == 1 for p in picks), picks
    # Two-way tie: d0 + d2 are tied below d1 → RR between them.
    sched._draft_load = [2, 5, 2]
    sched._draft_rr_tiebreak = 0
    picks = [sched._pick_draft() for _ in range(4)]
    assert picks == [0, 2, 0, 2], picks
    print(f"  T1b OK  picker unit (equal, skewed, two-way tie)")


async def t2_calibration_reconciles_drift() -> None:
    """After we manually poison the local load counter, the next
    calibration round should restore it to the actor's truth."""
    d = StubDraftActor("d", decode_delay_s=0.005)
    tgt = StubTargetActor("tgt")
    sched = V0Scheduler(
        drafts=[d],
        targets=[tgt],
        draft_pop_timeout_s=0.005,
        calibrate_every=2,
    )
    await sched.start()
    try:
        await _drain(sched, 4)
        # Wait for any background calibration to flush.
        await asyncio.sleep(0.05)
        # Poison the local counter.
        sched._draft_load[0] = 999
        sched._target_load[0] = 999
        # Submit a couple more to trigger calibration.
        await _drain(sched, 4)
        await asyncio.sleep(0.05)
        # After calibration, locals should be near truthful (≤ a few
        # in flight).
        assert sched._draft_load[0] < 50, (
            f"calibration didn't fix draft drift: {sched._draft_load[0]}"
        )
        assert sched._target_load[0] < 50, (
            f"calibration didn't fix target drift: {sched._target_load[0]}"
        )
        assert sched.stats["calibration_calls"] > 0
        assert sched.stats["calibration_total_abs_drift"] > 100
    finally:
        await sched.stop()
    print(f"  T2 OK  calibration_calls={sched.stats['calibration_calls']} "
          f"abs_drift={sched.stats['calibration_total_abs_drift']}")


async def t5_response_labelling() -> None:
    """SHIP → verdict ACCEPT, routing_path cascade. REGEN cascade →
    REGEN. Direct-route → DIRECT_TARGET."""
    # SHIP path
    d_ship = StubDraftActor("ship", head_decision="SHIP")
    tgt = StubTargetActor("tgt")
    s = V0Scheduler(drafts=[d_ship], targets=[tgt], draft_pop_timeout_s=0.005)
    await s.start()
    try:
        r = await s.submit("hi")
    finally:
        await s.stop()
    assert r.verdict == "ACCEPT" and r.routing_path == "cascade", repr(r)

    # REGEN path
    d_regen = StubDraftActor("regen", head_decision="REGEN")
    tgt2 = StubTargetActor("tgt2")
    s2 = V0Scheduler(drafts=[d_regen], targets=[tgt2], draft_pop_timeout_s=0.005)
    await s2.start()
    try:
        r2 = await s2.submit("hi")
    finally:
        await s2.stop()
    assert r2.verdict == "REGEN" and r2.routing_path == "cascade", repr(r2)

    # DIRECT_TARGET path — drive via force_direct_target=True (every
    # request bypasses draft). Exercises the verdict-relabel logic in
    # _on_target_finished.
    d_slow = StubDraftActor("slow", decode_delay_s=0.200)
    tgt3 = StubTargetActor("tgt3")
    s3 = V0Scheduler(
        drafts=[d_slow], targets=[tgt3], draft_pop_timeout_s=0.005,
        force_direct_target=True,
    )
    await s3.start()
    try:
        r3 = await s3.submit("hi")
    finally:
        await s3.stop()
    assert r3.verdict == "DIRECT_TARGET" and r3.routing_path == "direct_target", repr(r3)
    # Direct-target path should have draft_ms == 0 (no draft step).
    assert r3.draft_ms == 0, r3.draft_ms
    print(f"  T5 OK  ACCEPT/REGEN/DIRECT_TARGET labels correct")


async def t6_force_direct_target_no_drafts() -> None:
    """target_only-through-scheduler uses force_direct_target=True
    with drafts=[]. Every request should route via direct-target path,
    no draft work, verdict='DIRECT_TARGET'."""
    tgt = StubTargetActor("tgt", target_delay_s=0.005)
    sched = V0Scheduler(
        drafts=[],                  # no draft tier
        targets=[tgt],
        draft_pop_timeout_s=0.002,
        force_direct_target=True,
        calibrate_every=1000,
    )
    await sched.start()
    try:
        responses = await _stream(sched, 20, inter_arrival_s=0.005)
    finally:
        await sched.stop()
    assert all(r.verdict == "DIRECT_TARGET" for r in responses), (
        f"verdicts={[r.verdict for r in responses]}"
    )
    assert all(r.routing_path == "direct_target" for r in responses)
    assert all(r.draft_ms == 0 for r in responses)
    assert tgt.n_regen == 20, f"target.n_regen={tgt.n_regen}"
    print(f"  T6 OK  20 reqs all DIRECT_TARGET via empty-drafts scheduler")


async def t6b_force_direct_target_rejects_empty_both() -> None:
    """force_direct_target=True still requires at least one target."""
    try:
        V0Scheduler(drafts=[], targets=[], force_direct_target=True)
    except ValueError as e:
        assert "target" in str(e).lower()
        print("  T6b OK  empty targets correctly rejected")
        return
    raise AssertionError("expected ValueError for empty targets")


async def t6c_drafts_required_without_force() -> None:
    """Without force_direct_target, empty drafts is still an error."""
    tgt = StubTargetActor("tgt")
    try:
        V0Scheduler(drafts=[], targets=[tgt])
    except ValueError as e:
        assert "draft" in str(e).lower()
        print("  T6c OK  empty drafts rejected (no force_direct_target)")
        return
    raise AssertionError("expected ValueError for empty drafts")


class StubEncoderActor:
    """stub for DraftVisionEncoder. Returns synthetic
    image_embeds + image_grid_thw via Ray-RPC-shaped interface."""

    def __init__(self, name: str, *, encode_delay_s: float = 0.001,
                 fail: bool = False) -> None:
        self.name = name
        self._encode_delay_s = encode_delay_s
        self._fail = fail
        self.encode_return = _RemoteStub(self._encode_return)
        self.ping = _RemoteStub(self._ping)
        self.n_encoded = 0

    async def _encode_return(self, req_id, image_paths):
        await asyncio.sleep(self._encode_delay_s)
        if self._fail:
            raise RuntimeError(f"stub-encode-fail rid={req_id}")
        self.n_encoded += 1
        # Synthetic embed shape — what matters for the scheduler is that
        # we return *something* under these keys; the draft stub just
        # records that it got them.
        return {
            "req_id": req_id,
            "image_embeds": f"embeds-{req_id}",  # placeholder; scheduler treats as opaque
            "image_grid_thw": f"thw-{req_id}",
            "n_image_tokens": 250,
            "hidden_dim": 3584,
            "n_images": len(image_paths),
        }

    async def _ping(self):
        return "ok"


async def t7_encoder_pool_intercepts_draft() -> None:
    """When an encoder is configured, _dispatch_to_draft should call
    encoder.encode_return.remote() before draft.submit.remote(), and
    pass image_embeds + image_grid_thw to the draft. The draft should
    receive None image_paths."""

    # Augment the draft stub to record what it was given for embeds.
    captured: list[dict] = []

    class _RecordingDraft(StubDraftActor):
        async def _submit(self, req_id, prompt, max_tokens, temperature,
                          ignore_eos, image_path, head_cascade, image_paths,
                          image_embeds=None, image_grid_thw=None, **kwargs):
            captured.append({
                "req_id": req_id,
                "image_path": image_path,
                "image_paths": image_paths,
                "image_embeds": image_embeds,
                "image_grid_thw": image_grid_thw,
            })
            self._in_flight += 1
            self.n_submitted += 1
            asyncio.create_task(self._drive(req_id, head_cascade))

    draft = _RecordingDraft("d", decode_delay_s=0.005)
    target = StubTargetActor("tgt")
    encoder = StubEncoderActor("enc")

    sched = V0Scheduler(
        drafts=[draft],
        targets=[target],
        encoders=[encoder],
        draft_pop_timeout_s=0.005,
        calibrate_every=1000,
    )
    await sched.start()
    try:
        # Submit with image_paths — scheduler should encode then dispatch.
        r = await sched.submit("describe image", image_paths=["/tmp/img.png"])
    finally:
        await sched.stop()

    assert r.verdict == "ACCEPT", repr(r)
    assert encoder.n_encoded == 1, f"encoder should have been called once, got {encoder.n_encoded}"
    assert len(captured) == 1
    cap = captured[0]
    assert cap["image_paths"] is None, f"draft should get None image_paths, got {cap['image_paths']!r}"
    assert cap["image_path"] is None, f"draft should get None image_path, got {cap['image_path']!r}"
    assert cap["image_embeds"] is not None, "draft should get image_embeds from encoder"
    assert cap["image_grid_thw"] is not None
    assert sched.stats["n_dispatch_per_encoder"] == [1]
    print(f"  T7 OK  encoder called, image_paths cleared, embeds attached")


async def t7b_encoder_error_propagates() -> None:
    """If encoder.encode_return.remote() raises, the request should
    surface as ERROR with stage='encoder_encode' (not reach draft)."""

    submitted_to_draft: list[str] = []

    class _CountingDraft(StubDraftActor):
        async def _submit(self, req_id, *a, **k):
            submitted_to_draft.append(req_id)
            await super()._submit(req_id, *a, **k)

    draft = _CountingDraft("d", decode_delay_s=0.005)
    target = StubTargetActor("tgt")
    encoder = StubEncoderActor("enc", fail=True)

    sched = V0Scheduler(
        drafts=[draft],
        targets=[target],
        encoders=[encoder],
        draft_pop_timeout_s=0.005,
        calibrate_every=1000,
    )
    await sched.start()
    try:
        r = await sched.submit("p", image_paths=["/tmp/img.png"])
    finally:
        await sched.stop()

    assert r.verdict == "ERROR", repr(r)
    assert r.error and "encoder_encode" in r.error, r.error
    assert submitted_to_draft == [], (
        f"draft should not be reached when encoder fails, got {submitted_to_draft}"
    )
    print(f"  T7b OK  encoder failure → ERROR, draft skipped")


async def t8_force_direct_per_request() -> None:
    """force_direct=True on a single request pins it to
    DIRECT_TARGET even when no scheduler-wide direct routing is
    configured (force_direct_target=False, direct_ratio=0)."""
    d = StubDraftActor("d")
    tgt = StubTargetActor("tgt")
    sched = V0Scheduler(
        drafts=[d], targets=[tgt],
        load_aware=True, calibrate_every=1000,
        force_direct_target=False,
    )
    # No force_direct → cascade (DRAFT path).
    r_cascade = await sched.submit(prompt="p1", max_tokens=4)
    # With force_direct → DIRECT_TARGET.
    r_direct = await sched.submit(prompt="p2", max_tokens=4, force_direct=True)
    assert r_cascade.verdict in ("ACCEPT", "REGEN"), r_cascade.verdict
    assert r_direct.verdict == "DIRECT_TARGET", r_direct.verdict
    print(f"  T8 OK  per-request force_direct overrides scheduler-wide config")
    await sched.stop()


async def t10_buffer_without_scorer_serves_all() -> None:
    """with no scorer (every request gets score 0.0), the
    sorted buffer still serves all submits — draft and target
    dispatchers race for FIFO-ordered items, both ends are the same
    when scores are tied. End behavior: every submit completes; the
    split between cascade and DIRECT is consumption-rate dependent
    rather than scorer-guided. This test only asserts liveness and
    the n_buffer_inserts stat — the split is not asserted."""
    d = StubDraftActor("d", head_decision="SHIP")
    tgt = StubTargetActor("tgt")
    sched = V0Scheduler(
        drafts=[d], targets=[tgt],
        load_aware=True, calibrate_every=10000,
        # scorer=None (default)
    )
    direct_count = 0
    cascade_count = 0
    for i in range(50):
        r = await sched.submit(prompt=f"p{i}", max_tokens=4)
        if r.verdict == "DIRECT_TARGET":
            direct_count += 1
        else:
            cascade_count += 1
    # All 50 completed and went through the buffer.
    assert direct_count + cascade_count == 50, (direct_count, cascade_count)
    assert sched.stats["n_buffer_inserts"] == 50, sched.stats
    assert (
        sched.stats["n_buffer_via_top"] + sched.stats["n_buffer_via_bottom"]
        == 50
    ), sched.stats
    print(f"  T10 OK  buffer served 50 reqs → "
          f"{cascade_count} cascade / {direct_count} direct "
          f"(top={sched.stats['n_buffer_via_top']}, "
          f"bot={sched.stats['n_buffer_via_bottom']})")
    await sched.stop()


async def t12_head_cascade_false_skips_head() -> None:
    """cascade_retesting: scheduler.submit(head_cascade=False) routes
    through draft path but disables head firing. Used by
    --cell draft_only_via_scheduler for apples-to-apples baseline
    against target_only_via_scheduler."""
    d = StubDraftActor("d", head_decision="SHIP")  # would SHIP if head fired
    tgt = StubTargetActor("tgt")
    sched = V0Scheduler(
        drafts=[d], targets=[tgt],
        load_aware=True, calibrate_every=1000,
    )
    # Normal request: head fires, returns SHIP → ACCEPT.
    r_normal = await sched.submit(prompt="p1", max_tokens=4)
    assert r_normal.verdict == "ACCEPT", r_normal.verdict
    assert d.n_submitted == 1

    # head_cascade=False: head doesn't fire; should still return ACCEPT
    # but via the no-head path (no target call).
    target_calls_before = tgt.n_verify
    r_noh = await sched.submit(prompt="p2", max_tokens=4, head_cascade=False)
    assert r_noh.verdict == "ACCEPT", r_noh.verdict
    assert "draft-resp-" in (r_noh.text or ""), r_noh.text
    # Critical: no target call should have happened.
    assert tgt.n_verify == target_calls_before, (
        f"target was called {tgt.n_verify - target_calls_before} times "
        "but head_cascade=False should skip target entirely"
    )
    print(f"  T12 OK  head_cascade=False → ACCEPT via no-head path, target untouched")
    await sched.stop()


async def t9_force_cascade_per_request() -> None:
    """force_cascade=True on a single request pins it to the
    draft path even when force_direct_target=True is set scheduler-wide
    (used by the cascade stream in two_stream bench)."""
    d = StubDraftActor("d")
    tgt = StubTargetActor("tgt")
    sched = V0Scheduler(
        drafts=[d], targets=[tgt],
        load_aware=True, calibrate_every=1000,
        force_direct_target=True,           # scheduler-wide forces DIRECT
    )
    # No override → DIRECT_TARGET.
    r_direct = await sched.submit(prompt="p1", max_tokens=4)
    # With force_cascade → goes to draft despite scheduler-wide force.
    r_cascade = await sched.submit(prompt="p2", max_tokens=4, force_cascade=True)
    assert r_direct.verdict == "DIRECT_TARGET", r_direct.verdict
    assert r_cascade.verdict in ("ACCEPT", "REGEN"), r_cascade.verdict
    print(f"  T9 OK  per-request force_cascade overrides force_direct_target")
    await sched.stop()


async def t13_regen_shares_direct_router() -> None:
    """REGEN dispatch routes through the same code path as
    DIRECT (V0Scheduler._dispatch_direct_target). The two submit_regen
    calls must have identical argument shapes (modulo req_id and
    prompt) so target's submit_regen + KV-admission cannot
    distinguish a cascade-REGEN from a DIRECT spillover.

    Side-effect invariants the merged router preserves:
      - REGEN verdicts still surface as 'REGEN' (not 'DIRECT_TARGET')
        because draft set routing_path='cascade' before head fired.
      - DIRECT verdicts surface as 'DIRECT_TARGET' (no draft step).
    """
    # 8 REGEN requests via cascade head. Pin force_cascade=True so they
    # all go to draft (otherwise the buffer's target dispatcher would
    # pop some and they'd be DIRECT_TARGET, which is correct for the
    # but isn't what this REGEN-shape test wants to
    # exercise).
    d_regen = StubDraftActor("regen", head_decision="REGEN")
    tgt_regen = StubTargetActor("tgt_regen")
    sched_regen = V0Scheduler(
        drafts=[d_regen], targets=[tgt_regen],
        load_aware=True, calibrate_every=1000,
    )
    regen_responses = await _stream(
        sched_regen, 8, inter_arrival_s=0.002, force_cascade=True,
    )
    await sched_regen.stop()

    # 4 DIRECT requests with no draft tier.
    tgt_direct = StubTargetActor("tgt_direct")
    sched_direct = V0Scheduler(
        drafts=[], targets=[tgt_direct],
        force_direct_target=True, calibrate_every=1000,
    )
    direct_responses = await _stream(sched_direct, 4, inter_arrival_s=0.002)
    await sched_direct.stop()

    # Verdict labels still distinguish provenance for analytics.
    assert all(r.verdict == "REGEN" for r in regen_responses), (
        f"REGEN verdicts={[r.verdict for r in regen_responses]}"
    )
    assert all(r.verdict == "DIRECT_TARGET" for r in direct_responses), (
        f"DIRECT verdicts={[r.verdict for r in direct_responses]}"
    )

    # Both target stubs saw exactly the expected number of submit_regen
    # calls — REGEN goes through the merged router, not a separate path.
    assert tgt_regen.n_regen == 8, tgt_regen.n_regen
    assert tgt_direct.n_regen == 4, tgt_direct.n_regen
    assert tgt_regen.n_verify == 0 and tgt_direct.n_verify == 0, (
        "submit_verify must never be called (dead code removed)"
    )

    # The submit_regen call signature is identical between REGEN and
    # DIRECT — same keys, same image_path/image_paths types, same
    # ignore_eos, same max_tokens. (Per-request prompt + req_id
    # differ, those are stripped before the comparison.)
    def shape(call: dict) -> tuple:
        return (
            call["max_tokens"],
            call["ignore_eos"],
            type(call["image_path"]).__name__,
            type(call["image_paths"]).__name__,
            tuple(sorted(call.keys())),
        )
    regen_shapes  = {shape(c) for c in tgt_regen.regen_calls}
    direct_shapes = {shape(c) for c in tgt_direct.regen_calls}
    assert regen_shapes == direct_shapes, (
        "REGEN vs DIRECT submit_regen call shapes diverge: "
        f"REGEN={regen_shapes}  DIRECT={direct_shapes}"
    )

    # And: no call carries a length_hint key (the field was removed
    # from submit_regen in this cleanup).
    for c in tgt_regen.regen_calls + tgt_direct.regen_calls:
        assert "length_hint" not in c, (
            f"submit_regen still receiving length_hint: {c}"
        )

    print(f"  T13 OK  REGEN ({tgt_regen.n_regen}) and DIRECT "
          f"({tgt_direct.n_regen}) share one router; "
          f"call shapes identical; no length_hint")


async def t14_scorer_buffer_partitions_by_score() -> None:
    """the scorer maps requests to floats; `_insert_into_buffer`
    keeps the buffer sorted by score ascending. Top end (popped by
    draft dispatcher) is highest-scoring (predicted SHIPs); bottom
    (popped by target dispatcher) is lowest-scoring (predicted
    REGENs).

    Tests the insertion path directly because the dispatch-side race
    is timing-sensitive: in a unit-test env, dispatchers drain as
    fast as items arrive, so an end-to-end "draft eats easies / target
    eats hards" assertion would be flaky. The buffer-ordering
    invariant is what makes the correct; if it holds,
    consumption-side behaviour at steady state follows mechanically.
    """
    from prorouter.pre_router import per_source_scorer

    rates = {"easy": 0.95, "medium": 0.60, "hard": 0.20}
    scorer = per_source_scorer(rates)
    sources = ["easy"] * 3 + ["medium"] * 3 + ["hard"] * 4
    d = StubDraftActor("d")
    tgt = StubTargetActor("tgt")
    sched = V0Scheduler(
        drafts=[d], targets=[tgt], load_aware=True, calibrate_every=1000,
        scorer=scorer,
    )
    # Insert directly without starting dispatchers so the buffer
    # accumulates and we can inspect its order.
    for i, s in enumerate(sources):
        rid = f"r{i}"
        sched._pending[rid] = {
            "source": s, "arrival_t": 0.0, "prompt": "",
            "max_tokens": 4, "ignore_eos": False, "image_paths": None,
            "head_cascade": None, "force_draft_response": False,
        }
        sched._client_events[rid] = asyncio.Event()
        score = scorer({"source": s})
        await sched._insert_into_buffer(rid, score=score)

    # Buffer is sorted ascending by score. Tail (popped by target
    # dispatcher) is lowest, head (popped by draft dispatcher) is
    # highest.
    scores = [tup[0] for tup in sched._buffer]
    assert scores == sorted(scores), f"buffer not sorted: {scores}"
    # Bottom 4 are all "hard" (0.20); top 3 are "easy" (0.95).
    bottom4_sources = [
        sched._pending[tup[2]]["source"] for tup in sched._buffer[:4]
    ]
    top3_sources = [
        sched._pending[tup[2]]["source"] for tup in sched._buffer[-3:]
    ]
    assert bottom4_sources == ["hard"] * 4, bottom4_sources
    assert top3_sources == ["easy"] * 3, top3_sources
    assert sched.stats["n_buffer_inserts"] == 10, sched.stats
    assert sched.stats["buffer_max_depth"] == 10, sched.stats
    print(f"  T14 OK  10 reqs scored + sorted; bottom 4 = hard, top 3 = easy")


async def t17_target_engine_send_rps_caps_rate() -> None:
    """target_engine_send_rps caps the target send pump's
    request-rate regardless of submit rate or batching. With a cap
    of 20 r/s and a 1.5s burst of submits, the target stub should
    see ~30 ± a few items, not the full burst."""
    d = StubDraftActor("d", head_decision="REGEN")  # all cascade REGEN
    tgt = StubTargetActor("tgt")
    sched = V0Scheduler(
        drafts=[d], targets=[tgt],
        load_aware=True, calibrate_every=10000,
        target_engine_send_rps=20.0,
    )
    # Submit 100 force_cascade requests as fast as possible — they
    # all REGEN, so they pile up in the target send queue.
    submit_tasks = [
        asyncio.create_task(
            sched.submit(prompt=f"p{i}", max_tokens=4, force_cascade=True)
        )
        for i in range(100)
    ]
    # Sleep 1.5s. With a 20-r/s cap, target should have seen ~30
    # items (1.0s of cap + ~0.5s extra-window, less startup). Cancel
    # pending tasks; we're done measuring.
    await asyncio.sleep(1.5)
    for t in submit_tasks:
        if not t.done():
            t.cancel()
    n_target_seen = tgt.n_regen
    # Allow [12, 50] — wide bounds for ramp + asyncio scheduling
    # jitter, narrow enough to detect a missing rate limit (would
    # see ~100).
    assert 12 <= n_target_seen <= 50, (
        f"target saw {n_target_seen} items; expected ~30 under "
        f"20 r/s cap (allowed 12-50). Rate limiter not working."
    )
    print(f"  T17 OK  target_engine_send_rps=20 capped n_regen to "
          f"{n_target_seen} in 1.5s (vs 100 submitted)")
    await sched.stop()


async def t18_direct_rpc_ship() -> None:
    """--dispatch-direct-rpc fires the draft RPC on arrival
    (no buffer). SHIP verdict responds ACCEPT; target is never touched."""
    d = StubDraftActor("d", decode_delay_s=0.005, head_decision="SHIP")
    tgt = StubTargetActor("tgt")
    sched = V0Scheduler(
        drafts=[d], targets=[tgt], dispatch_direct_rpc=True,
        calibrate_every=1000,
    )
    await sched.start()
    try:
        responses = await _drain(sched, 12)
    finally:
        await sched.stop()
    assert all(r.verdict == "ACCEPT" for r in responses), (
        [r.verdict for r in responses]
    )
    assert d.n_submitted == 12, d.n_submitted
    assert tgt.n_regen == 0, f"target touched on SHIP: {tgt.n_regen}"
    # No sorted buffer / dispatch loop used.
    assert not sched._buffer, sched._buffer
    assert not sched._draft_dispatch_tasks and not sched._target_dispatch_tasks
    assert sched.stats["n_buffer_via_top"] == 12
    print(f"  T18 OK  direct-RPC SHIP: 12 fired to draft, 0 to target, "
          f"no buffer/loops")


async def t19_direct_rpc_regen() -> None:
    """under --dispatch-direct-rpc a head REGEN fires the
    target RPC immediately (one item per RPC)."""
    d = StubDraftActor("d", decode_delay_s=0.005, head_decision="REGEN")
    tgt = StubTargetActor("tgt", target_delay_s=0.002, regen_verdict="REGEN")
    sched = V0Scheduler(
        drafts=[d], targets=[tgt], dispatch_direct_rpc=True,
        calibrate_every=1000,
    )
    await sched.start()
    try:
        responses = await _drain(sched, 8)
    finally:
        await sched.stop()
    assert all(r.verdict == "REGEN" for r in responses), (
        [r.verdict for r in responses]
    )
    assert d.n_submitted == 8 and tgt.n_regen == 8, (
        f"draft={d.n_submitted} target={tgt.n_regen}"
    )
    # Each target RPC carried exactly one item (no batching).
    assert tgt.batch_call_sizes == [1] * 8, tgt.batch_call_sizes
    print(f"  T19 OK  direct-RPC REGEN: 8 draft→8 target, one item/RPC")


async def t20_direct_rpc_force_disables_batching() -> None:
    """--dispatch-direct-rpc force-disables adaptive_batch /
    two_buffer / occupancy_gate / rate_match at construction."""
    d = StubDraftActor("d")
    tgt = StubTargetActor("tgt")
    sched = V0Scheduler(
        drafts=[d], targets=[tgt], dispatch_direct_rpc=True,
        adaptive_batch=True, occupancy_gate=True, two_buffer=True,
        rate_match=True,
    )
    assert sched._adaptive_batch is False
    assert sched._occupancy_gate is False
    assert sched._two_buffer is False
    assert sched._rate_match is False
    print(f"  T20 OK  direct-RPC force-disabled adaptive/occupancy/2buf/rate")


async def t21_self_eval_score_flows_to_response() -> None:
    """a self_eval_score on the draft finished item surfaces on
    the Response (the plumbing the inline self-eval arms rely on)."""
    class _SEDraft(StubDraftActor):
        async def _drive(self, req_id, head_cascade):
            await asyncio.sleep(self._decode_delay_s)
            item = {
                "req_id": req_id, "text": f"draft-resp-{req_id}",
                "n_output_tokens": 16, "finish_reason": "stop",
                "completed_t": time.perf_counter(),
                "self_eval_score": 0.73, "self_eval_method": "ptrue",
            }
            if head_cascade is not False:
                item["head_decision"] = self._head_decision
            await self._finished_q.put(item)
            self._in_flight -= 1

    d = _SEDraft("d", decode_delay_s=0.005, head_decision="SHIP")
    tgt = StubTargetActor("tgt")
    sched = V0Scheduler(drafts=[d], targets=[tgt], calibrate_every=1000)
    await sched.start()
    try:
        responses = await _drain(sched, 5, force_cascade=True)
    finally:
        await sched.stop()
    assert all(r.self_eval_score == 0.73 for r in responses), (
        [r.self_eval_score for r in responses]
    )
    assert all(r.self_eval_method == "ptrue" for r in responses)
    print(f"  T21 OK  self_eval_score/method flow to Response (5 reqs)")


async def t22_self_eval_scoring_unit() -> None:
    """self_eval.prob_true / score_ptrue / score_automix match
    the reference math on hand-built logprob/completion fakes."""
    import math

    from prorouter import self_eval as SE

    class _LP:
        def __init__(self, decoded_token, logprob):
            self.decoded_token = decoded_token
            self.logprob = logprob

    # P(True): equal logits for True/False → prob 0.5; skew → skew.
    pos_even = {1: _LP("True", math.log(0.6)), 2: _LP("False", math.log(0.6))}
    assert abs(SE.prob_true(pos_even) - 0.5) < 1e-9, SE.prob_true(pos_even)
    pos_skew = {1: _LP(" A", math.log(0.9)), 2: _LP(" B", math.log(0.1))}
    assert abs(SE.prob_true(pos_skew) - 0.9) < 1e-9, SE.prob_true(pos_skew)
    # No verdict token at this position → None (scanner should skip it).
    assert SE.prob_true({1: _LP("hello", -0.1)}) is None

    class _Comp:
        def __init__(self, logprobs=None, text=""):
            self.logprobs = logprobs
            self.text = text

    # score_ptrue scans to the first verdict position; default 0.5 if none.
    comp = _Comp(logprobs=[{9: _LP("the", -0.1)}, pos_skew])
    assert abs(SE.score_ptrue(comp) - 0.9) < 1e-9, SE.score_ptrue(comp)
    assert SE.score_ptrue(_Comp(logprobs=None)) == 0.5

    # AutoMix: fraction of the k samples judged "Correct" (word-boundary; the
    # 'correct' inside 'incorrect' must NOT count as correct).
    outs = [_Comp(text="The answer is Correct."),
            _Comp(text="This is Incorrect."),
            _Comp(text="reasoning... Correct"),
            _Comp(text="no verdict word here")]
    # 2 correct, 1 incorrect, 1 undecided → 2/3.
    assert abs(SE.score_automix(outs) - (2.0 / 3.0)) < 1e-9, SE.score_automix(outs)
    assert SE.score_automix([_Comp(text="nothing")]) == 0.5
    print(f"  T22 OK  self_eval scoring math (P(True) + AutoMix) matches")


async def t23_draft_submit_batch() -> None:
    """draft_submit_batch coalesces eligible buffer items into one
    submit_batch RPC. With a 20 ms fake submit RTT and a serialized RPC
    window (pipeline=1), the buffer accumulates during each round trip, so
    later batches MUST carry >1 request — and every request still gets a
    correct cascade Response."""
    d = StubDraftActor("batch", head_decision="SHIP",
                       decode_delay_s=0.001, submit_delay_s=0.02)
    tgt = StubTargetActor("tgt")
    s = V0Scheduler(
        drafts=[d], targets=[tgt], draft_pop_timeout_s=0.005,
        draft_submit_batch=16, draft_submit_pipeline=1,
        # Pin routing: no idle-target spillover — every request must go
        # through the draft so the batch accounting below is exact.
        direct_ratio=0.0,
    )
    await s.start()
    try:
        rs = await asyncio.gather(*(s.submit(f"q{i}") for i in range(40)))
    finally:
        await s.stop()
    assert len(rs) == 40
    bad = [r.verdict for r in rs if r.verdict != "ACCEPT"]
    assert not bad, bad[:3]
    assert d.n_submitted == 40, d.n_submitted
    assert sum(d.batch_sizes) == 40, d.batch_sizes
    assert max(d.batch_sizes) > 1, d.batch_sizes  # coalescing happened
    print(f"  T23 OK  40/40 ACCEPT via {len(d.batch_sizes)} batch RPCs, "
          f"largest={max(d.batch_sizes)}")


async def t24_target_submit_pipeline() -> None:
    """target_submit_pipeline overlaps target dispatch RPCs instead of
    awaiting each serially. With a 30ms fake dispatch RTT on the target and
    every request routed REGEN (force_direct_target so the draft never SHIPs),
    a serial dispatcher can hold only ONE dispatch RPC in flight at a time; a
    pipeline of 4 must overlap several. Assert peak concurrency > 1 with the
    pipeline and exactly 1 without — and that every request still resolves."""
    # Baseline: serial dispatch (pipeline=0) — peak concurrency must be 1.
    tgt_serial = StubTargetActor("tgt-serial", target_delay_s=0.001,
                                 submit_delay_s=0.03, regen_verdict="REGEN")
    s0 = V0Scheduler(
        drafts=[], targets=[tgt_serial], force_direct_target=True,
        target_pop_timeout_s=0.005, adaptive_batch=True,
        adaptive_batch_max=1,  # one item per RPC → concurrency is the only lever
    )
    await s0.start()
    try:
        rs0 = await asyncio.gather(*(s0.submit(f"q{i}") for i in range(12)))
    finally:
        await s0.stop()
    assert len(rs0) == 12 and all(r.verdict == "DIRECT_TARGET" for r in rs0)
    assert tgt_serial._max_concurrent_batches == 1, \
        f"serial path overlapped: {tgt_serial._max_concurrent_batches}"

    # Pipelined: target_submit_pipeline=4 — dispatch RPCs must overlap.
    tgt_pipe = StubTargetActor("tgt-pipe", target_delay_s=0.001,
                               submit_delay_s=0.03, regen_verdict="REGEN")
    s1 = V0Scheduler(
        drafts=[], targets=[tgt_pipe], force_direct_target=True,
        target_pop_timeout_s=0.005, adaptive_batch=True, adaptive_batch_max=1,
        target_submit_pipeline=4,
    )
    await s1.start()
    try:
        rs1 = await asyncio.gather(*(s1.submit(f"q{i}") for i in range(12)))
    finally:
        await s1.stop()
    assert len(rs1) == 12 and all(r.verdict == "DIRECT_TARGET" for r in rs1)
    assert tgt_pipe._max_concurrent_batches > 1, \
        f"pipeline did not overlap: {tgt_pipe._max_concurrent_batches}"
    print(f"  T24 OK  serial peak-concurrency=1, "
          f"pipeline peak-concurrency={tgt_pipe._max_concurrent_batches}; "
          f"12/12 resolved on both")


async def main() -> None:
    tests = [
        ("T1 multi-draft dispatch serves all",
         t1_multi_draft_dispatch_serves_all),
        ("T1b _pick_draft unit", t1b_picker_unit),
        ("T2 calibration reconciles drift", t2_calibration_reconciles_drift),
        ("T5 response labelling", t5_response_labelling),
        ("T6 force_direct_target with empty drafts",
         t6_force_direct_target_no_drafts),
        ("T6b empty targets rejected even with force",
         t6b_force_direct_target_rejects_empty_both),
        ("T6c empty drafts rejected without force",
         t6c_drafts_required_without_force),
        ("T7 encoder pool intercepts draft dispatch",
         t7_encoder_pool_intercepts_draft),
        ("T7b encoder failure surfaces as ERROR",
         t7b_encoder_error_propagates),
        ("T8 per-request force_direct overrides scheduler-wide config",
         t8_force_direct_per_request),
        ("T9 per-request force_cascade overrides force_direct_target",
         t9_force_cascade_per_request),
        ("T10 buffer without scorer serves all",
         t10_buffer_without_scorer_serves_all),
        ("T12 head_cascade=False skips head firing (draft_only_via_scheduler)",
         t12_head_cascade_false_skips_head),
        ("T13 REGEN shares the DIRECT router",
         t13_regen_shares_direct_router),
        ("T14 scorer-buffer partitions by score",
         t14_scorer_buffer_partitions_by_score),
        ("T17 target_engine_send_rps caps rate",
         t17_target_engine_send_rps_caps_rate),
        ("T18 direct-RPC dispatch SHIP", t18_direct_rpc_ship),
        ("T19 direct-RPC dispatch REGEN", t19_direct_rpc_regen),
        ("T20 direct-RPC force-disables batching",
         t20_direct_rpc_force_disables_batching),
        ("T21 self_eval score flows to Response",
         t21_self_eval_score_flows_to_response),
        ("T22 self_eval scoring math", t22_self_eval_scoring_unit),
        ("T23 draft submit_batch coalesces + stays correct",
         t23_draft_submit_batch),
        ("T24 target_submit_pipeline overlaps dispatch RPCs",
         t24_target_submit_pipeline),
    ]
    failures = 0
    for name, fn in tests:
        print(f"[run] {name}")
        try:
            await fn()
        except Exception as e:
            failures += 1
            print(f"  FAIL: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print()
    print(f"{'PASS' if failures == 0 else 'FAIL'} — {len(tests) - failures}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
