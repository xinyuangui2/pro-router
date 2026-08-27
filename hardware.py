"""Adapt the runnable pieces to whatever hardware is actually present.

The experiments were developed on a particular fleet (a small-model node and a larger large-model node), but nothing about the method depends on that. This module
keeps the hardware assumptions in one place so the same scripts run on a single
GPU, on a workstation, or on a heterogeneous cluster:

  * tensor-parallel size is derived from the GPUs actually visible,
  * dtype falls back when the device predates bfloat16,
  * node placement constraints are dropped when the cluster does not advertise
    them, instead of leaving actors pending forever,
  * vision-processor knobs are only passed to models that accept them.

Every value can still be pinned explicitly; auto-detection is what happens when
you do not.
"""
from __future__ import annotations

import os

# Claimed per pinned actor. Small on purpose -- see placement_kwargs.
_RESOURCE_SHARE = 0.001

# Model families whose HF processor accepts a `max_pixels` override. Passing it
# to anything else raises inside the processor, so it is opt-in by family.
_PIXEL_BUDGET_FAMILIES = ("qwen2-vl", "qwen2.5-vl", "qwen2_5_vl", "qwen-vl")


def local_gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def largest_power_of_two(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return max(1, p)


def pick_tensor_parallel(requested: int | None, available: int | None = None,
                         label: str = "engine") -> int:
    """Resolve a tensor-parallel size that the engine's node can satisfy.

    `requested` of 0 or None means auto. vLLM requires the world size to divide
    the model's attention-head count, so auto picks a power of two rather than
    an arbitrary GPU count (5 GPUs -> 4, not 5).

    Auto-detection reads the GPUs visible to *this* process. In the common Ray
    topology the driver runs on a CPU-only head node while the engines run on
    GPU workers, so that count is 0 and silently resolving to 1 would put a
    large model on a single device and fail at load with an opaque OOM. When
    the cluster has GPUs the driver cannot see, this refuses to guess.
    """
    avail = local_gpu_count() if available is None else available
    if requested:
        if avail and requested > avail:
            tp = largest_power_of_two(avail)
            print(f"[hardware] {label}: requested tensor-parallel={requested} "
                  f"but only {avail} GPU(s) visible -- falling back to {tp}",
                  flush=True)
            return tp
        return requested

    if not avail:
        cluster = describe_cluster().get("gpus", 0)
        if cluster:
            raise SystemExit(
                f"[hardware] {label}: this process sees no GPUs, but the "
                f"cluster reports {cluster}. Auto tensor-parallel cannot be "
                f"inferred from a CPU-only driver -- pass the size explicitly "
                f"(e.g. --small-tp / --large-tp).")
        print(f"[hardware] {label}: no GPU visible -- tensor-parallel=1",
              flush=True)
        return 1

    tp = largest_power_of_two(avail)
    print(f"[hardware] {label}: auto tensor-parallel={tp} "
          f"({avail} GPU(s) visible)", flush=True)
    return tp


def pick_dtype(requested: str = "auto") -> str:
    """`auto` lets vLLM choose; an explicit bfloat16 is downgraded on pre-Ampere."""
    if requested != "bfloat16":
        return requested
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            print("[hardware] bfloat16 unsupported on this device -- using float16",
                  flush=True)
            return "float16"
    except Exception:
        pass
    return requested


def supports_pixel_budget(model_id: str) -> bool:
    m = (model_id or "").lower().replace("_", "-")
    return any(fam.replace("_", "-") in m for fam in _PIXEL_BUDGET_FAMILIES)


def mm_kwargs(model_id: str, max_pixels: int | None,
              max_images: int = 2) -> dict:
    """Multimodal engine kwargs that are safe for this model family."""
    out: dict = {}
    if max_images:
        out["limit_mm_per_prompt"] = {"image": max_images}
    if max_pixels and supports_pixel_budget(model_id):
        out["mm_processor_kwargs"] = {"max_pixels": max_pixels}
    elif max_pixels:
        print(f"[hardware] {model_id}: ignoring --max-pixels "
              f"(its processor does not take one); pre-resize images instead",
              flush=True)
    return out


def placement_kwargs(resource: str | None, num_gpus: int,
                     label: str = "engine") -> dict:
    """Ray actor options, pinning to a node label only if the cluster has one.

    A cluster that does not advertise the label would otherwise leave the actor
    pending forever, which looks like a hang rather than a config error.

    Only a sliver of the resource is claimed (`_RESOURCE_SHARE`), not a whole
    unit. Ray's implicit per-node resources (`node:<ip>`) have a capacity of
    exactly 1.0, so claiming a full unit lets only ONE actor pin to a given
    node: point both tiers at the same node and the second actor is
    unschedulable forever. GPUs are what actually bound co-location, and
    `num_gpus` already carries that.
    """
    opts: dict = {"num_gpus": num_gpus}
    if not resource:
        return opts
    try:
        import ray
        available = ray.cluster_resources()
        free = ray.available_resources()
    except Exception:
        return opts
    if resource not in available:
        print(f"[hardware] {label}: node resource {resource!r} is not "
              f"advertised by this cluster -- scheduling on any GPU node",
              flush=True)
        return opts
    if free.get(resource, 0.0) < _RESOURCE_SHARE:
        raise SystemExit(
            f"[hardware] {label}: node resource {resource!r} is advertised but "
            f"fully claimed ({free.get(resource, 0.0)} free). An actor pinned to "
            f"it would stay pending forever with no error. Free it, or pin this "
            f"tier to a different node.")
    opts["resources"] = {resource: _RESOURCE_SHARE}
    print(f"[hardware] {label}: pinned to node resource {resource!r}",
          flush=True)
    return opts


def describe_cluster() -> dict:
    try:
        import ray
        res = ray.cluster_resources()
        return {"gpus": int(res.get("GPU", 0)), "cpus": int(res.get("CPU", 0)),
                "node_labels": sorted(k for k in res
                                      if k not in ("CPU", "GPU", "memory",
                                                   "object_store_memory"))}
    except Exception:
        return {"gpus": local_gpu_count(), "cpus": os.cpu_count(),
                "node_labels": []}
