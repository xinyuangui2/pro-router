"""
Standalone loader for the target-VLM's vision tower (ViT + patch merger).

Use case: in the vision-offload pipeline, we run the *target's*
ViT on the *draft's* GPU(s), concurrent with the draft pipeline. The
post-merger image_embeds are shipped to target via Ray (or NCCL); the
target consumes them via vLLM's `multi_modal_data={"image_embeds": ...}`
public API and skips its own ViT forward.

Why standalone: we don't want to load the 72B LLM weights on the draft
node (~140 GB at bfloat16). The Qwen2.5-VL `.visual` submodule
(ViT + merger) is ~1.0 GB at 7B / ~1.8 GB at 72B.

**TP-aware loading (2026-05-12)**: we use vLLM's own
`Qwen2_5_VisionTransformer` class — same architecture as HF's
`Qwen2_5_VLVisionModel`, but with `QKVParallelLinear` /
`MergedColumnParallelLinear` / `RowParallelLinear` in place of plain
`nn.Linear`. These layers consult the active tensor-model-parallel
group and shard their weights across all TP ranks automatically.

Memory math, 7B-VL at TP=4 on 4×A10G:
  - patch_embed + RMSNorms + replicated bits: ~0.05 GB / rank
  - per-block attention (qkv + proj, sharded by TP): ~0.05 GB / rank
  - per-block MLP (gate_up + down, sharded by TP): ~0.21 GB / rank
  - merger (column + row parallel): ~0.01 GB / rank
  - Total: ~0.32 GB / rank, vs the old 1.0 GB on rank-0 only.

Net: KV cache reclaims ~0.7 GB on rank-0 and pays ~0.3 GB on each of
ranks 1–3. Since vLLM sizes KV from `min(free_per_rank)`, the rank-0
relief is the limiting factor — net +0.7 GB usable KV cache per rank.

Loader strategy:
  1. Build a `Qwen2_5_VisionTransformer` on the current `cuda:0` (the
     worker's GPU; each TP rank has its own visible cuda:0). Inside a
     vLLM worker, the TP group is already initialized; QKV / MLP linears
     auto-shard via `get_tensor_model_parallel_world_size()`.
  2. Pull HF safetensors and feed them through vLLM's `load_weights`
     method on `Qwen2_5_VisionTransformer`. That handles
     `gate_proj + up_proj → gate_up_proj` merging and per-rank slicing.
     HF already stores `attn.qkv` as a single fused tensor, so QKV
     loading is a direct shard of the combined weight.

Encoder API: `TargetVisionEncoder.encode(images)` and the
`encode_async_batched_pre_processed(...)` flow used by the head-cascade
engine stay the same shape. The forward call collectively runs across
all 4 TP ranks (NCCL all-reduce inside each block), but the engine
dispatch + payload assembly happens on rank-0 only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TargetVitBatch:
    """Result of one batched target-ViT forward over N requests' images.

    All N requests share ONE CUDA event recorded on the encoder's stream
    after the forward kernels are submitted. Per-request payloads are
    extracted via offsets into the flat image_embeds tensor.

    Fields are GPU-resident until the consumer copies to CPU.
    """
    image_embeds: torch.Tensor       # [total_image_tokens, llm_hidden_dim]
    image_grid_thw: torch.Tensor     # [total_images, 3]
    offsets_tokens: list[int]        # length N+1; req j's slice is
                                     # image_embeds[offsets_tokens[j]
                                     #               : offsets_tokens[j+1]]
    offsets_images: list[int]        # length N+1; req j's grid_thw rows
                                     # image_grid_thw[offsets_images[j]
                                     #                 : offsets_images[j+1]]
    event: torch.cuda.Event          # signals when the forward is done


def load_target_visual(
    target_model_id: str,
    device: torch.device | str | int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """Load a FULL copy of the `.visual` submodule (ViT + patch merger)
    of a Qwen2.5-VL checkpoint onto `device` in `dtype`. Returns the
    submodule in eval mode with no_grad parameters.

    **Implementation (2026-05-13, post-Phase B):** uses vLLM's
    `Qwen2_5_VisionTransformer` class directly, not HF's
    `Qwen2_5_VLForConditionalGeneration().visual`. The target's own vLLM
    engine instantiates THE SAME class internally; matching the encoder
    class on both sides guarantees bit-identical post-merger embeddings.

    Earlier history:
      - (2026-05-12): originally used vLLM's class but
        TP-sharded across the draft's TP=4 ranks. The per-shard matmuls
        were too small on 7B-VL/A10G; NCCL collective latency dominated.
      - Reverted (same day) to HF's `.visual` for DP-per-rank.
      - (2026-05-13): HF's encoder produces post-merger
        embeds that drift numerically from what the 72B-VL LLM expects
        from vLLM's class. 36% of REGENs returned empty/garbage tokens
        (📐-emoji loops, immediate EOS).

    The fix: keep DP-per-rank (no TP collectives in encode), but use
    vLLM's class. To disable internal TP sharding without touching the
    surrounding engine's TP group, we monkey-patch
    `is_vit_use_data_parallel` to True during construction so the
    layer factory wires Column/Row/QKVParallelLinear with `disable_tp=True`.
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig

    from vllm.model_executor.models import vision as _vllm_vision
    from vllm.model_executor.models import qwen2_5_vl as _vllm_qwen2_5_vl
    from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer

    target_device = (
        torch.device(device) if not isinstance(device, torch.device)
        else device
    )

    hf_config = AutoConfig.from_pretrained(
        target_model_id, trust_remote_code=True,
    )
    vision_config = hf_config.vision_config

    # Force the merger / attn / mlp parallel linears to behave as plain
    # nn.Linear (no TP sharding) for our standalone DP-per-rank usage.
    # The surrounding engine's TP group is shared by the draft LLM; we
    # don't want the visual to participate in its collectives.
    orig_is_dp = _vllm_vision.is_vit_use_data_parallel
    _vllm_vision.is_vit_use_data_parallel = lambda: True
    # The qwen2_5_vl module imports the function by name, so patch
    # both call sites to be safe across versions.
    orig_is_dp_qwen = getattr(_vllm_qwen2_5_vl, "is_vit_use_data_parallel", None)
    if orig_is_dp_qwen is not None:
        _vllm_qwen2_5_vl.is_vit_use_data_parallel = lambda: True
    # Construction allocates parameters on the current default dtype.
    # Set torch's default dtype + device so RMSNorm/LayerNorm weights
    # land in the right precision; restore after.
    orig_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(target_device):
            visual = Qwen2_5_VisionTransformer(
                vision_config=vision_config,
                norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
                quant_config=None,
                prefix="visual",
            )
    finally:
        torch.set_default_dtype(orig_default_dtype)
        _vllm_vision.is_vit_use_data_parallel = orig_is_dp
        if orig_is_dp_qwen is not None:
            _vllm_qwen2_5_vl.is_vit_use_data_parallel = orig_is_dp_qwen

    visual = visual.to(device=target_device, dtype=dtype)

    # Stream weights from HF safetensors. We only need keys under
    # `visual.*`. vLLM's `load_weights` expects (name, tensor) pairs
    # WITHOUT the `visual.` prefix (since the module IS the visual).
    ckpt_dir = snapshot_download(
        target_model_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"],
    )
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(
            {weight_map[k] for k in weight_map if k.startswith("visual.")}
        )
    else:
        shards = ["model.safetensors"]

    def _weight_stream():
        for shard in shards:
            path = os.path.join(ckpt_dir, shard)
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    if not key.startswith("visual."):
                        continue
                    sub_key = key[len("visual."):]
                    tensor = f.get_tensor(key).to(
                        device=target_device, dtype=dtype,
                    )
                    yield sub_key, tensor

    loaded = visual.load_weights(_weight_stream())
    if not loaded:
        raise RuntimeError(
            f"No 'visual.*' weights loaded from {target_model_id} "
            f"shards {shards}. Check the checkpoint layout."
        )

    visual.eval()
    for p in visual.parameters():
        p.requires_grad_(False)
    return visual


class TargetVisionEncoder:
    """Wraps a TP-sharded target `.visual` (ViT + merger) and runs
    forward passes for the head-cascade engine.

    Lifecycle:
      enc = TargetVisionEncoder("Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0")
      payload = enc.encode([pil_image])
      # payload = {"image_embeds": Tensor, "image_grid_thw": Tensor}

    **TP coordination:** the underlying
    `Qwen2_5_VisionTransformer` is built with vLLM's TP-parallel linear
    layers. Each `.encode()` / `.encode_async_batched_pre_processed()`
    call executes NCCL collectives across the active TP group. Callers
    MUST invoke these methods in lockstep across all TP ranks with
    matching inputs (vLLM's mirrored `mm_features` in CachedRequestState
    guarantees the inputs are identical across ranks).

    **Stream policy (Phase C step A):** the forward runs on the
    *current* CUDA stream (caller-controlled). With `use_dedicated_stream
    = False` (default), this is whatever stream the engine is on when
    the encode is dispatched — typically the worker's default stream,
    which is also where vLLM's main model forward lives. Encode and
    main forward serialize on that stream → no overlap, but trivially
    correct under NCCL.

    Step B (later): re-enable the dedicated stream via the
    `use_dedicated_stream=True` constructor flag. Requires careful
    coordination so the encode's NCCL collectives don't deadlock against
    the main forward's collectives on the default stream.
    """

    def __init__(
        self,
        target_model_id: str,
        device: torch.device | str | int = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        processor: Any = None,
        use_dedicated_stream: bool = False,
    ) -> None:
        self._device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        self._dtype = dtype
        self._visual = load_target_visual(target_model_id, self._device, dtype)

        if processor is None:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                target_model_id, trust_remote_code=True,
            )
        self._processor = processor

        # Stream policy: see class docstring. The dedicated stream is
        # allocated regardless (cheap) so step B is a one-line flip,
        # but encode methods honor `_use_dedicated_stream` to decide
        # whether to enter it.
        self._stream = torch.cuda.Stream(device=self._device)
        self._use_dedicated_stream = bool(use_dedicated_stream)

        # Cache the spatial merge size for token-count accounting (used
        # when splitting a batched ViT forward back into per-image chunks).
        # HF Qwen2_5_VLVisionModel exposes it via `.config`; vLLM's
        # Qwen2_5_VisionTransformer hoists it to a top-level attribute.
        sm = getattr(self._visual, "spatial_merge_size", None)
        if sm is None:
            sm = getattr(
                getattr(self._visual, "config", None),
                "spatial_merge_size", 2,
            )
        self._spatial_merge_size = int(sm)

        # Output-tensor hidden dim per token. Needed by rank-0 to
        # pre-allocate recv buffers for batched p2p without round-
        # tripping a per-rid shape header. Probe via the visual's
        # config — HF's Qwen2_5_VLVisionModel returns pre-merger
        # `hidden_size` (1280 for 7B); vLLM's Qwen2_5_VisionTransformer
        # returns post-merger `out_hidden_size`. Try the
        # post-merger field first.
        cfg = getattr(self._visual, "config", None)
        out_dim = getattr(cfg, "out_hidden_size", None)
        if out_dim is None:
            out_dim = getattr(self._visual, "out_hidden_size", None)
        if out_dim is None:
            # Fallback to ViT hidden (pre-merger output of HF visual).
            out_dim = getattr(cfg, "hidden_size", None)
        if out_dim is None:
            out_dim = 1280  # last-resort hardcoded for HF Qwen2.5-VL-7B
        self._out_hidden_size = int(out_dim)

    def _stream_ctx(self):
        """Return a context manager for the stream the encode will run
        on. With `use_dedicated_stream=False` (default for Phase C step
        A), this is a no-op — runs on the caller's current stream."""
        if self._use_dedicated_stream:
            return torch.cuda.stream(self._stream)
        import contextlib
        return contextlib.nullcontext()

    @property
    def stream(self) -> torch.cuda.Stream:
        """The dedicated CUDA stream this encoder uses. Caller can
        record events on it / wait for it from another stream."""
        return self._stream

    @property
    def device(self) -> torch.device:
        return self._device

    @torch.no_grad()
    def encode(
        self,
        images: list,
        sync: bool = True,
        to_cpu: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Encode one record's images. Returns
          {"image_embeds": Tensor[N_image_tokens, llm_hidden_dim],
           "image_grid_thw": Tensor[n_images, 3]}.

        sync=True (default): block until the encode completes. sync=False
        returns immediately; caller must `self.stream.synchronize()` (or
        wait via a CUDA event) before reading the output tensors.

        to_cpu=True: move outputs to CPU before returning. Required when
        the result will cross a Ray RPC boundary.
        """
        if not images:
            raise ValueError("encode() requires at least one image")

        inputs = self._processor.image_processor(
            images=images, return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        with self._stream_ctx():
            pv = pixel_values.to(
                self._device, dtype=self._dtype, non_blocking=True,
            )
            gt = image_grid_thw.to(self._device, non_blocking=True)
            embeds = self._visual(pv, grid_thw=gt)
            # HF returns Tensor[total_image_tokens, llm_hidden_dim] (post-merger).

        if sync:
            if self._use_dedicated_stream:
                self._stream.synchronize()
            else:
                torch.cuda.current_stream(self._device).synchronize()

        if to_cpu:
            if not sync:
                (self._stream if self._use_dedicated_stream
                 else torch.cuda.current_stream(self._device)).synchronize()
            embeds = embeds.detach().cpu().contiguous()
            gt = gt.detach().cpu().contiguous()
        else:
            embeds = embeds.detach()
            gt = gt.detach()

        return {
            "image_embeds": embeds,
            "image_grid_thw": gt,
        }

    def encode_batch(
        self,
        images_per_record: list[list],
        sync: bool = True,
        to_cpu: bool = False,
    ) -> list[dict[str, torch.Tensor]]:
        """Batched encode: processes images from N records in one ViT
        forward, then splits the output back per record. Higher GPU
        utilization than calling encode() per record.

        Returns a list of N payloads, one per input record.
        """
        if not images_per_record:
            return []

        flat: list = []
        counts: list[int] = []
        for record_images in images_per_record:
            counts.append(len(record_images))
            flat.extend(record_images)

        if not flat:
            return [
                {"image_embeds": None, "image_grid_thw": None}
                for _ in images_per_record
            ]

        inputs = self._processor.image_processor(
            images=flat, return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        with self._stream_ctx():
            pv = pixel_values.to(
                self._device, dtype=self._dtype, non_blocking=True,
            )
            gt = image_grid_thw.to(self._device, non_blocking=True)
            embeds = self._visual(pv, grid_thw=gt)

        if sync:
            if self._use_dedicated_stream:
                self._stream.synchronize()
            else:
                torch.cuda.current_stream(self._device).synchronize()

        if to_cpu:
            if not sync:
                (self._stream if self._use_dedicated_stream
                 else torch.cuda.current_stream(self._device)).synchronize()
            embeds = embeds.detach().cpu().contiguous()
            gt = gt.detach().cpu().contiguous()
        else:
            embeds = embeds.detach()
            gt = gt.detach()

        # Each image contributes thw[0]*thw[1]*thw[2] / merge_size**2
        # post-merger tokens. Walk the gt rows and slice the flat embeds.
        merge_sq = self._spatial_merge_size ** 2
        per_image_token_counts: list[int] = []
        for i in range(gt.shape[0]):
            thw = gt[i].tolist()
            per_image_token_counts.append(
                int(thw[0] * thw[1] * thw[2] // merge_sq)
            )

        results: list[dict[str, torch.Tensor]] = []
        embed_cursor = 0
        thw_cursor = 0
        for n_images in counts:
            if n_images == 0:
                results.append({"image_embeds": None, "image_grid_thw": None})
                continue
            n_tokens = sum(
                per_image_token_counts[thw_cursor : thw_cursor + n_images]
            )
            results.append({
                "image_embeds": embeds[embed_cursor : embed_cursor + n_tokens],
                "image_grid_thw": gt[thw_cursor : thw_cursor + n_images],
            })
            embed_cursor += n_tokens
            thw_cursor += n_images
        return results

    @torch.no_grad()
    def encode_async_batched_pre_processed(
        self,
        pixel_values_list: list[torch.Tensor],
        image_grid_thw_list: list[torch.Tensor],
    ) -> TargetVitBatch:
        """Run ONE batched target-ViT forward over N requests' images on
        the dedicated stream, return a TargetVitBatch with offsets +
        event. Non-blocking.

        Designed for the in-engine head-cascade flow: vLLM has already
        preprocessed images into (pixel_values, image_grid_thw) and
        stored them in CachedRequestState.mm_features. The caller pulls
        those tensors per OPEN request and hands the list here.

        Semantics:
          - Tensors are concat'd along their respective batch axes.
          - One ViT+merger forward — kernel launches amortized across
            requests, GPU SM utilization improves with N.
          - Per-request slicing is via offsets_tokens / offsets_images
            on the returned batch.
          - Caller waits for completion via `batch.event.synchronize()`
            or polls with `batch.event.query()`.
        """
        assert len(pixel_values_list) == len(image_grid_thw_list)
        assert len(pixel_values_list) > 0
        merge_sq = self._spatial_merge_size ** 2

        # Compute offsets BEFORE the cat. Each request's grid_thw rows
        # are typically tiny (1 row per image) so tolist() is cheap.
        offsets_tokens = [0]
        offsets_images = [0]
        for thw in image_grid_thw_list:
            rows = thw.cpu().tolist() if thw.device.type == "cuda" else thw.tolist()
            n_tokens = sum(t * h * w for t, h, w in rows) // merge_sq
            offsets_tokens.append(offsets_tokens[-1] + int(n_tokens))
            offsets_images.append(offsets_images[-1] + len(rows))

        pvs = [
            pv.to(self._device, dtype=self._dtype, non_blocking=True)
            for pv in pixel_values_list
        ]
        thws = [
            thw.to(self._device, non_blocking=True)
            for thw in image_grid_thw_list
        ]
        flat_pv = torch.cat(pvs, dim=0)
        flat_thw = torch.cat(thws, dim=0)

        with self._stream_ctx():
            embeds = self._visual(flat_pv, grid_thw=flat_thw)

        # HF `Qwen2_5_VLForConditionalGeneration().visual.forward` returns
        # `BaseModelOutputWithPooling(last_hidden_state=<pre-merger trunk
        # output, shape [N_patches, vision_hidden_size]>,
        # pooler_output=<post-merger LLM-dim embeds, shape
        # [N_tokens, out_hidden_size]>)`. We need post-merger embeds for
        # the target's LLM to consume them as `multi_modal_data["image"]
        # ["image_embeds"]`. Earlier this fell through to last_hidden_state
        # first, shipping 1280-dim trunk features to the 8192-dim 72B-VL
        # LLM and triggering a CUDA assert mid-attention (Phase A 2026-05-13).
        if not isinstance(embeds, torch.Tensor):
            cand = getattr(embeds, "pooler_output", None)
            if cand is None:
                cand = getattr(embeds, "last_hidden_state", None)
            if cand is None:
                cand = embeds[0]
            embeds = cand

        event = torch.cuda.Event()
        if self._use_dedicated_stream:
            event.record(self._stream)
        else:
            # Recording without an arg records on the current stream.
            event.record()

        return TargetVitBatch(
            image_embeds=embeds.detach(),
            image_grid_thw=flat_thw,
            offsets_tokens=offsets_tokens,
            offsets_images=offsets_images,
            event=event,
        )
