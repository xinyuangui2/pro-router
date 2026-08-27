"""MockTargetActor.

A zero-GPU Ray actor that mimics `TargetEngineAsync`'s public surface
enough to satisfy `V0Scheduler`'s init (and its periodic gpu_util /
qsize / pop_finished polls) on hardware that has no large-model node.

Two modes, see the class docstring: the default refuses to serve (for
draft-only runs pinned with `force_cascade=True, force_draft_response=True`),
and `serve_decode=True` -- what `run_pipeline.py --mock-large` uses -- answers
escalations with an empty-text stub so the routing path can run end to end.

This is NOT a behavioral mock — `submit_decode` / `submit_regen`
should never be called in the draft-only configurations this bench uses. They
raise loudly if invoked so a misconfigured routing path is detected
immediately rather than silently losing requests.

Methods mirrored from `TargetEngineAsync`:
  - ping() — readiness handshake (returns "ok")
  - pop_finished(max_n, timeout_s) — always returns [] after waiting
  - gpu_util() — returns a fixed-shape dict the scheduler accepts
  - qsize() — returns {"queued": 0, "in_flight": 0}
  - submit_decode / submit_regen — raise RuntimeError

Resource footprint: 0 GPUs, 1 small CPU. Lands on the head node (or
any CPU resource) so it doesn't compete with the draft for GPU slots.
"""
from __future__ import annotations

import asyncio
import time

import ray


@ray.remote(num_cpus=0, num_gpus=0)
class MockTargetActor:
    """Zero-GPU stand-in for TargetEngineAsync.

    `serve_decode=False` (default, contract): `submit_decode` and
    `submit_regen` raise. The mock is only valid for runs that pin
    force_cascade + force_draft_response so target dispatch must not
    happen; a raise is the loud failure those runs want.

    `serve_decode=True` : the mock behaves as a behavioural
    target — `submit_decode` enqueues a fake REGEN-verdict finished
    item after `decode_delay_s`, and `pop_finished` drains the queue.
    Lets cascade runs go end-to-end without a real target node so a
    routing-side comparison (FIFO vs scorer-ordered buffer) can be run
    on a draft-only cluster. ACCEPT/REGEN verdicts on the cascade arm
    still come from the real head firing inside the draft actor; only
    the post-head REGEN dispatch + the DIRECT bypass go through this
    mock.
    """

    def __init__(
        self,
        serve_decode: bool = False,
        decode_delay_s: float = 0.0,
    ) -> None:
        self._started_t = time.perf_counter()
        self._serve_decode = bool(serve_decode)
        self._decode_delay_s = float(decode_delay_s)
        self._finished_q: asyncio.Queue = asyncio.Queue()
        self._in_flight = 0

    async def ping(self) -> str:
        return "ok"

    async def pop_finished(
        self, max_n: int = 64, timeout_s: float = 0.05,
    ) -> list[dict]:
        if not self._serve_decode:
            # Default mock contract — nothing should ever land in the
            # finished queue. Sleep so the scheduler's _target_pump
            # doesn't busy-spin and return empty.
            if timeout_s > 0:
                await asyncio.sleep(timeout_s)
            return []
        items: list[dict] = []
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

    async def gpu_util(self) -> dict:
        # Match TargetEngineAsync.gpu_util() shape (per-rank list of dicts).
        # The scheduler's direct-route util-ceiling check reads
        # `max(r.get('gpu_util_pct', 0) for r in val)` — return 0% so the
        # cap never triggers. Not consulted in force_cascade=True paths.
        return [{"gpu_util_pct": 0, "memory_used_pct": 0, "rank": 0}]

    async def qsize(self) -> dict:
        return {"queued": 0, "in_flight": self._in_flight}

    async def submit_decode(self, *args, **kwargs) -> None:
        if not self._serve_decode:
            raise RuntimeError(
                "MockTargetActor.submit_decode called — the caller should "
                "be force_cascade=True with force_draft_response=True so "
                "target never receives a request. Routing is misconfigured."
            )
        # Behavioural path: emit a fake REGEN-verdict item. The
        # scheduler's _on_target_finished maps verdict + routing_path
        # into the final response label ('REGEN' when draft-routed,
        # 'DIRECT_TARGET' when direct-routed) so the per-source
        # accounting in the run still captures the routing distinction.
        req_id = args[0] if args else kwargs.get("req_id")
        self._in_flight += 1
        asyncio.create_task(self._drive(req_id))

    async def _drive(self, req_id: str) -> None:
        if self._decode_delay_s > 0:
            await asyncio.sleep(self._decode_delay_s)
        await self._finished_q.put({
            "req_id": req_id,
            "verdict": "REGEN",
            "text": "",
            "n_output_tokens": 0,
            "finish_reason": "stop",
            "completed_t": time.perf_counter(),
        })
        self._in_flight -= 1

    # Alias kept for backward-compat with engine.py's `submit_regen` callers.
    async def submit_regen(self, *args, **kwargs) -> None:
        await self.submit_decode(*args, **kwargs)

    async def submit_decode_batch(self, items: list[dict]) -> None:
        """Mock the batched API. Each item becomes one
        synthesised REGEN finished item, matching `submit_decode`'s
        behaviour."""
        if not self._serve_decode:
            raise RuntimeError(
                "MockTargetActor.submit_decode_batch called — the caller should "
                "be force_cascade=True so target never receives a request."
            )
        for it in items:
            self._in_flight += 1
            asyncio.create_task(self._drive(it["req_id"]))
