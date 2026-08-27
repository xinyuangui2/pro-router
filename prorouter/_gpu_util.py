"""Per-GPU utilization sampler shared by actors + the probe.

Single source of truth: tries pynvml first (lazy NVML init, ~1 ms/GPU),
falls back to nvidia-smi subprocess (slower, ~50-100 ms). Returns one
dict per visible GPU:
    {gpu, util_pct, mem_pct}
On failure returns a single error dict:
    [{"error": "<TypeName>: <message>"}]

Kept in its own module (no vllm/torch import) so the scheduler probe
can import it without dragging in the engine's heavy dependencies.
"""
from __future__ import annotations


def sample_gpu_util() -> list[dict]:
    """Per-GPU utilization on the current node.

    Returns one dict per visible GPU: {gpu, util_pct, mem_pct}.
    On failure returns [{"error": "..."}].
    """
    try:
        import pynvml
        if not getattr(sample_gpu_util, "_inited", False):
            pynvml.nvmlInit()
            sample_gpu_util._inited = True  # type: ignore[attr-defined]
        n = pynvml.nvmlDeviceGetCount()
        out = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            mi = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append({
                "gpu": i,
                "util_pct": int(u.gpu),
                "mem_pct": int(100 * mi.used / max(mi.total, 1)),
            })
        return out
    except Exception:
        pass
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        out = []
        for line in r.stdout.strip().splitlines():
            parts = [s.strip() for s in line.split(",")]
            if len(parts) < 4:
                continue
            idx, util, mu, mt = parts[:4]
            out.append({
                "gpu": int(idx),
                "util_pct": int(util),
                "mem_pct": int(100 * int(mu) / max(int(mt), 1)),
            })
        return out
    except Exception as e:
        return [{"error": f"gpu_util failed: {type(e).__name__}: {e}"}]
