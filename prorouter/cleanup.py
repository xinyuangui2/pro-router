"""Run a vLLM worker-cleanup script on every GPU worker node.

Called at the top of bench/smoke scripts (after ray.init, before
engine creation) to prevent orphan vLLM Worker / EngineCore
subprocesses from a prior failed run from holding GPU memory and
OOMing the next engine init.

The pattern: vLLM's `distributed_executor_backend="mp"` spawns
Worker / EngineCore subprocesses that aren't Ray-tracked. When the
parent Ray actor dies (job failure or even normal exit with cleanup
errors), those subprocesses are reparented to PID 1 and keep running,
holding ~18 GB / GPU each. The cleanup script pkills them and waits
for the GPUs to report free. It is not shipped here: point
$PROROUTER_CLEANUP_SCRIPT at your own, or place it at
scripts/cleanup_vllm_workers.sh relative to the Ray working directory.

Idempotent: safe to call when no orphans exist.

Off-node note: in the off-node topology, both the draft node (A10)
and the target node (8× A100) can have orphans. We schedule one
cleanup task per GPU node, pinned via NodeAffinitySchedulingStrategy,
so neither node is missed.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_gpus=0, num_cpus=1)
def _cleanup_on_worker() -> tuple[int, str, str]:
    """Run the cleanup script. Caller pins via scheduling_strategy."""
    candidates = [
        # $PROROUTER_CLEANUP_SCRIPT wins; otherwise look working_dir-relative.
        *( [Path(os.environ["PROROUTER_CLEANUP_SCRIPT"])]
           if os.getenv("PROROUTER_CLEANUP_SCRIPT") else [] ),
        Path("scripts/cleanup_vllm_workers.sh"),
    ]
    for path in candidates:
        if path.exists():
            r = subprocess.run(
                ["bash", str(path)],
                capture_output=True, text=True, timeout=60,
            )
            return r.returncode, r.stdout, r.stderr
    return 127, "", "cleanup script not found in any expected location"


def cleanup_vllm_workers() -> None:
    """Run cleanup_vllm_workers.sh on every GPU node and wait.

    Iterates over every alive node with GPUs and pins one cleanup
    task to each via NodeAffinitySchedulingStrategy. Prints each
    node's stdout inline so the bench log captures the verdict. On
    non-zero rc, prints a warning but does not raise — engine init
    will fail with a clear OOM if GPUs are still held, which is more
    informative than a wrapper exception.
    """
    nodes = [
        n for n in ray.nodes()
        if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0
    ]
    if not nodes:
        print("[cleanup] no GPU nodes found")
        return

    futures = {}
    for n in nodes:
        nid = n["NodeID"]
        addr = n.get("NodeManagerAddress", nid)
        strategy = NodeAffinitySchedulingStrategy(node_id=nid, soft=False)
        futures[addr] = _cleanup_on_worker.options(
            scheduling_strategy=strategy
        ).remote()

    print(f"[cleanup] killing any orphan vLLM workers on {len(futures)} GPU node(s)...")
    for addr, fut in futures.items():
        rc, out, err = ray.get(fut)
        for line in out.splitlines():
            print(f"[cleanup {addr}]   {line}")
        if rc != 0:
            print(f"[cleanup {addr}] WARN: cleanup exited rc={rc}")
            if err:
                for line in err.splitlines()[:10]:
                    print(f"[cleanup-stderr {addr}]   {line}")
