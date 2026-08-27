"""
Standalone encoder actor for the DRAFT engine's vision tower.

Mirror of `prorouter/target_vision.py`, but in the **reverse direction**:
target_vision runs the *target's* ViT on the *draft's* GPU(s) for the
in-engine head-cascade flow. draft_vision runs the
*draft's* ViT on its **own** dedicated GPU and ships post-merger
`image_embeds` to the draft engine over NCCL on a dedicated stream.

Why this exists: found the draft's in-engine ViT is the
saturation bottleneck at sustained Poisson load. Removing it (via
pre-encoded image_embeds shipped over Ray RPC) raised the draft's
saturation ceiling by ~35% and cut p50 latency 4-13×. This module
productionizes that path:

  - GPU↔GPU via NCCL (no CPU round-trip).
  - Receiver runs on a dedicated CUDA stream on the draft, so the
    LM forward can issue concurrently with the embed transfer.

This file is the **producer side** (encoder actor). The consumer side
lives in the vLLM fork at:

  the vLLM fork's draft-offload tree
  (branch: cascade-draft-encoder-offload, off cascade-prod-fixes
   HEAD 2c2e5d79f)

See the design doc for the file:line list of fork-side changes.

Status: skeleton only. The class wires up correctly but the
NCCL send path needs cluster validation. See `## What's prototyped +
what's blocking` in the design doc for the open items.
"""

from __future__ import annotations

from typing import Any

import ray
import torch


DRAFT_VISION_GROUP_NAME = "draft_vision_offload"
ENCODER_RANK = 0
DRAFT_RANK = 1
WORLD_SIZE = 2


@ray.remote(num_gpus=1, max_concurrency=8)
class DraftVisionEncoder:
    """Ray actor that owns one GPU dedicated to the draft's ViT.

    Lifecycle (mirrors `prorouter/engine.py:DraftEngine.setup_collective`):

      enc = DraftVisionEncoder.options(num_gpus=1).remote(
          model_id="Qwen/Qwen2.5-VL-7B-Instruct",
      )
      ray.get(enc.setup_collective.remote(
          group_name="draft_vision_offload",
          rank=0, world_size=2, dst_rank=1,
      ))
      # …draft engine joins the same group as rank=1 from its side…
      ray.get(enc.encode.remote(
          req_id="r-001", mm_hash="r-001",
          image_paths=["/path/img.jpg"],
      ))

    The NCCL send call runs inside a dedicated CUDA stream so back-to-
    back encodes overlap (H2D of next request's pixel_values while
    previous request's ViT forward is still running). The receive
    side on the draft engine also uses a dedicated CUDA stream
    (stream 2 in the design doc) so the LM forward can issue
    concurrently with the embed transfer.
    """

    def __init__(
        self,
        model_id: str,
        dtype: str = "float16",
    ) -> None:
        # `load_target_visual` constructs vLLM's Qwen2_5_VisionTransformer
        # which instantiates CustomOps at __init__; those require a live
        # `set_current_vllm_config` context + initialized dist env. We're
        # a standalone actor (not inside an AsyncLLMEngine), so we set up
        # a minimal TP=1 single-process group here, matching the recipe
        # the standalone encoder probe and `_pre_encode_all` both use.
        import os
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed.parallel_state import (
            init_distributed_environment, initialize_model_parallel,
        )

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        # Pick a free port per-actor so multiple DraftVisionEncoders
        # can colocate on the same node (n_encoders > 1 on a single g5.12).
        # Using setdefault with a fixed port collides whenever a parent
        # process already exported MASTER_PORT — every actor inherits it
        # and torch.distributed.init_process_group EADDRINUSEs the second.
        import socket
        with socket.socket() as _s:
            _s.bind(("", 0))
            _free_port = _s.getsockname()[1]
        os.environ["MASTER_PORT"] = str(_free_port)
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        # Hold the context manager for the actor's lifetime — the
        # CustomOps consult get_current_vllm_config() not just at
        # construction but also at forward time.
        self._vllm_cfg_ctx = set_current_vllm_config(VllmConfig())
        self._vllm_cfg_ctx.__enter__()
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)

        # `load_target_visual` is misleadingly named — it loads the
        # `.visual` submodule of any Qwen2.5-VL checkpoint, target
        # OR draft. used it for the target model; here
        # we pass the DRAFT model id so the encoder produces embeds
        # in the draft LM's hidden dim.
        from prorouter.target_vision import load_target_visual

        torch_dtype = getattr(torch, dtype)
        device = torch.device("cuda:0")
        self._visual = load_target_visual(model_id, device, torch_dtype)
        self._device = device
        self._dtype = torch_dtype

        # Dedicated encode stream — back-to-back encode kernels and
        # the trailing NCCL send all live on this stream so the
        # main thread can return immediately after launching the
        # send without waiting for the ViT to finish.
        self._encode_stream = torch.cuda.Stream(device=device)

        from transformers import AutoProcessor
        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True,
        )

        # Set in setup_collective().
        self._group_name: str | None = None
        self._dst_rank: int = DRAFT_RANK

    def setup_collective(
        self,
        group_name: str = DRAFT_VISION_GROUP_NAME,
        rank: int = ENCODER_RANK,
        world_size: int = WORLD_SIZE,
        dst_rank: int = DRAFT_RANK,
    ) -> dict[str, Any]:
        """Join the NCCL collective group used for the encoder→draft
        embed channel. Blocks until both sides have joined.

        Mirror of `prorouter/engine.py:DraftEngine.setup_collective`. The
        group name must differ from "vision_offload" so the Phase
        C target-ViT channel and this draft-ViT channel can
        coexist on the same cluster.
        """
        import ray.util.collective as col

        col.init_collective_group(
            world_size=world_size,
            rank=rank,
            backend="nccl",
            group_name=group_name,
        )
        self._group_name = group_name
        self._dst_rank = dst_rank
        return {
            "status": "joined",
            "group": group_name,
            "rank": rank,
            "world_size": world_size,
            "dst_rank": dst_rank,
        }

    def encode(
        self,
        req_id: str,
        mm_hash: str,
        image_paths: list[str],
    ) -> dict[str, Any]:
        """Encode one record's images, then ship the post-merger
        `image_embeds` + `image_grid_thw` to the draft engine via
        NCCL p2p.

        The send is fire-and-forget from the caller's perspective:
        we return as soon as the kernels are launched on the encode
        stream (not after they complete). The draft side's stream-2
        recv pump deposits the embeds into the encoder cache keyed
        by `mm_hash`; the draft LM's `_execute_mm_encoder` consumes
        them in place of the local `.visual` forward.

        Header layout (16 longs):
          [mm_hash_lo, mm_hash_hi, mm_hash_lo2, mm_hash_hi2,
           n_image_tokens, hidden_dim, n_images, _pad0,
           _pad1, _pad2, _pad3, _pad4,
           _pad5, _pad6, _pad7, _pad8]

        `mm_hash` is encoded as 4×8 bytes (32 chars) into the first 4
        longs. v0 uses `req_id` as mm_hash for simplicity — see the
        design doc's open question on encoder-cache key plumbing.

        Args:
          req_id: opaque tag for telemetry.
          mm_hash: encoder-cache key the draft engine will use.
          image_paths: paths reachable on the encoder actor's local FS.

        Returns:
          {"req_id": req_id, "mm_hash": mm_hash, "n_image_tokens": int}
        """
        if self._group_name is None:
            raise RuntimeError(
                "setup_collective() must be called before encode()"
            )

        from PIL import Image
        import ray.util.collective as col

        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        proc = self._processor.image_processor(
            images=imgs, return_tensors="pt",
        )
        pv = proc["pixel_values"].to(
            self._device, dtype=self._dtype, non_blocking=True,
        )
        thw = proc["image_grid_thw"].to(self._device, non_blocking=True)

        with torch.cuda.stream(self._encode_stream):
            with torch.inference_mode():
                embeds = self._visual(pv, grid_thw=thw)
            # Some vLLM-internal ViTs return BaseModelOutputWithPooling.
            # `load_target_visual` returns vLLM's
            # Qwen2_5_VisionTransformer which returns a plain Tensor,
            # but be defensive.
            if not isinstance(embeds, torch.Tensor):
                cand = getattr(embeds, "pooler_output", None)
                if cand is None:
                    cand = getattr(embeds, "last_hidden_state", None)
                if cand is None:
                    cand = embeds[0]
                embeds = cand
            embeds = embeds.contiguous()
            thw_c = thw.contiguous()

            header = _build_header(mm_hash, embeds, thw_c)
            col.send(header, self._dst_rank, self._group_name)
            col.send(embeds, self._dst_rank, self._group_name)
            col.send(thw_c, self._dst_rank, self._group_name)

        # No CPU sync — encode_stream owns the dependency chain;
        # the trailing NCCL kernel is enqueued AFTER the forward
        # kernels on the same stream, so it sees the produced data.
        return {
            "req_id": req_id,
            "mm_hash": mm_hash,
            "n_image_tokens": int(embeds.shape[0]),
            "hidden_dim": int(embeds.shape[1]),
            "n_images": int(thw_c.shape[0]),
        }

    def encode_return(
        self,
        req_id: str,
        image_paths: list[str],
    ) -> dict[str, Any]:
        """Run the draft's ViT on this actor's GPU and RETURN
        the embeds + image_grid_thw to the caller via Ray RPC.

        Simpler counterpart to `encode()` (which does NCCL p2p send).
        Used by the bench to measure the production architecture
        without the fork-side recv pump. Production build would
        replace the Ray return with a NCCL p2p send to the draft's
        worker process; this Ray return is the production-proxy that
        lets us measure the win/loss with no fork changes.

        Returns CPU tensors so Ray's pickle path can serialize them
        cheaply (GPU tensors pickle very slowly through Ray).
        """
        from PIL import Image

        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        proc = self._processor.image_processor(
            images=imgs, return_tensors="pt",
        )
        pv = proc["pixel_values"].to(
            self._device, dtype=self._dtype, non_blocking=True,
        )
        thw = proc["image_grid_thw"].to(self._device, non_blocking=True)

        with torch.cuda.stream(self._encode_stream):
            with torch.inference_mode():
                embeds = self._visual(pv, grid_thw=thw)
            if not isinstance(embeds, torch.Tensor):
                cand = getattr(embeds, "pooler_output", None)
                if cand is None:
                    cand = getattr(embeds, "last_hidden_state", None)
                if cand is None:
                    cand = embeds[0]
                embeds = cand

        # Sync the encode_stream (NOT current_stream — the `with`
        # context exited so current_stream reverted to the default).
        # Required so the d2h copy below sees completed kernels.
        self._encode_stream.synchronize()
        embeds_cpu = embeds.detach().to("cpu", non_blocking=False).contiguous()
        thw_cpu = thw.detach().to("cpu", non_blocking=False).contiguous()
        return {
            "req_id": req_id,
            "image_embeds": embeds_cpu,
            "image_grid_thw": thw_cpu,
            "n_image_tokens": int(embeds_cpu.shape[0]),
            "hidden_dim": int(embeds_cpu.shape[1]),
            "n_images": int(thw_cpu.shape[0]),
        }

    def ping(self) -> str:
        return "ok"


HEADER_NLONGS = 16
"""Number of int64s in the per-send header. mm_hash takes the first
4 (32 bytes); shape/count takes the next 3; rest is padding for
forward compatibility."""


def _build_header(
    mm_hash: str,
    embeds: torch.Tensor,
    thw: torch.Tensor,
) -> torch.Tensor:
    """Pack the per-request header. The draft-side recv pump in the
    fork mirrors this layout."""
    raw = mm_hash.encode("utf-8")[:32].ljust(32, b"\0")
    h = [
        int.from_bytes(raw[0:8], "little"),
        int.from_bytes(raw[8:16], "little"),
        int.from_bytes(raw[16:24], "little"),
        int.from_bytes(raw[24:32], "little"),
        int(embeds.shape[0]),  # n_image_tokens
        int(embeds.shape[1]),  # hidden_dim
        int(thw.shape[0]),     # n_images
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
    return torch.tensor(h, dtype=torch.long, device=embeds.device)


def parse_header(header_cpu: torch.Tensor) -> dict[str, Any]:
    """Counterpart for the fork-side recv pump. Reverses `_build_header`.

    `header_cpu` must be a length-`HEADER_NLONGS` int64 tensor on CPU.
    Returns:
      {"mm_hash": str, "n_image_tokens": int, "hidden_dim": int,
       "n_images": int}
    """
    assert header_cpu.numel() == HEADER_NLONGS, (
        f"header should be {HEADER_NLONGS} longs; got {header_cpu.numel()}"
    )
    h = header_cpu.tolist()
    mm_hash_bytes = b"".join(
        int(h[i]).to_bytes(8, "little") for i in range(4)
    ).rstrip(b"\0").decode("utf-8")
    return {
        "mm_hash": mm_hash_bytes,
        "n_image_tokens": int(h[4]),
        "hidden_dim": int(h[5]),
        "n_images": int(h[6]),
    }
