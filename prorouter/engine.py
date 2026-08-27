"""
Ray actor wrappers around vLLM engines.

Each actor owns one Python process running a vllm.LLM instance. The
orchestrator (in spec_decode.py) calls into these actors via Ray RPC for
the spec-decode control plane (tokens, logprobs, metadata).

For large tensor transfer (projected vision features, ~33 MB/request in
P6), we use ray.util.collective to set up a NCCL group between the two
actors and transfer GPU↔GPU directly — no CPU round-trip.

For P1 (MVP) we run draft + target as two actors, each pinned to its own
GPU(s). No vision offload yet; but the NCCL channel is bootstrapped so
we can validate it works alongside vLLM's internal NCCL (used for TP).

Collective group convention:
  - group_name: "vision_offload"
  - rank 0: DraftEngine
  - rank 1: TargetEngine
  - world_size: 2
"""

from __future__ import annotations

import asyncio
import collections
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any

import ray
import ray.util.collective as col
import torch
from vllm import LLM, SamplingParams


# Collective group convention (see module docstring).
VISION_GROUP_NAME = "vision_offload"
DRAFT_RANK = 0
TARGET_RANK = 1
WORLD_SIZE = 2


from prorouter._gpu_util import sample_gpu_util as _sample_gpu_util  # noqa: F401


def _to_messages(p) -> list[dict]:
    """Normalize an engine input into a chat-template message list.

      str  → [{"role": "user", "content": str}]   (single-turn)
      list → list(p)                              (multi-turn passthrough)

    Multi-turn callers (e.g., MT-Bench T2) build their own message
    list with prior assistant/user turns and pass it as the prompt.
    Returns a fresh list so callers can append additional turns
    (e.g., the verify question on top of a multi-turn prefix) without
    mutating the caller's input.
    """
    if isinstance(p, list):
        return list(p)
    # Use LIST-form text content, not a plain string. The non-Qwen
    # multimodal chat templates (LLaVA-OV, Pixtral) require list-of-typed-
    # blocks for BOTH text and image and silently drop plain-string content;
    # Qwen's template renders list-form text identically. This keeps one
    # template path working for every family + modality.
    return [{"role": "user", "content": [{"type": "text", "text": p}]}]


_CHAT_TEMPLATER_CACHE: dict = {}


def _get_chat_templater(tokenizer):
    """Return an object whose apply_chat_template can render multimodal
    content-lists ({"type":"image"}/{"type":"text"}). The model PROCESSOR's
    template handles this for ALL families; the bare TOKENIZER template does
    NOT for LLaVA-OneVision / Pixtral — only Qwen's permissive template
    tolerates the list, which is why cross-family cascade drafts silently
    produced 0 tokens (apply_chat_template raised on the list, malforming the
    draft prompt). Falls back to the tokenizer if no processor is available
    (text-only / non-MM models)."""
    name = getattr(tokenizer, "name_or_path", None)
    if not name:
        return tokenizer
    cached = _CHAT_TEMPLATER_CACHE.get(name)
    if cached is None:
        try:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
            cached = proc if hasattr(proc, "apply_chat_template") else tokenizer
        except Exception:
            cached = tokenizer
        _CHAT_TEMPLATER_CACHE[name] = cached
    return cached


def _build_mm_request(
    tokenizer, prompt_text: str, image_paths: list[str] | None,
    extra_messages_pre: list[dict] | None = None,
    extra_messages_post: list[dict] | None = None,
):
    """Build a vLLM multimodal request from text + image paths.

    Returns (formatted_prompt_text, multi_modal_data_dict_or_None)
    where multi_modal_data matches vLLM's LLM.generate dict input
    format: {"image": [PIL.Image, ...]}.

    Optional extra_messages_{pre,post} let callers wrap the user
    turn (e.g., judge-verify needs to add an assistant turn for the
    draft response and a follow-up user turn for the verify question).

    Image paths are loaded via PIL.Image.open. Caller is responsible
    for paths being valid on the actor's local filesystem (typically
    a shared volume).
    """
    from PIL import Image

    images = []
    if image_paths:
        for p in image_paths:
            images.append(Image.open(p).convert("RGB"))

    # User-turn content. With images, the chat template needs
    # {"type": "image"} placeholders followed by the text part.
    if images:
        content = [{"type": "image"} for _ in images] + [
            {"type": "text", "text": prompt_text}
        ]
    else:
        content = prompt_text

    messages = []
    if extra_messages_pre:
        messages.extend(extra_messages_pre)
    messages.append({"role": "user", "content": content})
    if extra_messages_post:
        messages.extend(extra_messages_post)

    # Use the PROCESSOR's chat template for multimodal (cross-family safe);
    # the bare tokenizer template raises on content-lists for LLaVA/Pixtral.
    templater = _get_chat_templater(tokenizer) if images else tokenizer
    formatted = templater.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    multi_modal_data = {"image": images} if images else None
    return formatted, multi_modal_data


def _build_mm_prompt_from_count(
    tokenizer, prompt_text: str, image_count: int,
    extra_messages_pre: list[dict] | None = None,
    extra_messages_post: list[dict] | None = None,
) -> str:
    """Like _build_mm_request, but takes the image *count* instead of
    paths — used by the vision-offload consumers on the target
    actor, which receive pre-encoded image_embeds and never need to
    open the source image files. The chat template still needs the
    correct number of `{"type": "image"}` placeholder slots so the
    tokenizer emits the matching <|image_pad|> markers; vLLM then
    expands each <|image_pad|> using the supplied `image_grid_thw`."""
    if image_count > 0:
        content = [{"type": "image"} for _ in range(image_count)] + [
            {"type": "text", "text": prompt_text}
        ]
    else:
        content = prompt_text

    messages = []
    if extra_messages_pre:
        messages.extend(extra_messages_pre)
    messages.append({"role": "user", "content": content})
    if extra_messages_post:
        messages.extend(extra_messages_post)

    # Same cross-family fix as _build_mm_request. The cascade draft
    # submit path (DraftEngineAsync.submit, image_paths + image_embeds
    # branches) renders the prompt HERE; the bare tokenizer template RAISES
    # ('can only concatenate str (not list)') on the {type:image}/{type:text}
    # content-list for LLaVA-OneVision / Pixtral — only Qwen's permissive
    # template tolerates it. That malformed the draft prompt → 0-token gen →
    # ship=0 on every cross-family cascade. The model PROCESSOR's template
    # renders the right per-family image markers; for Qwen it is byte-identical
    # to the tokenizer template (verified), so the switch is safe for all.
    templater = _get_chat_templater(tokenizer) if image_count > 0 else tokenizer
    return templater.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )


def _resolve_bit_token_ids(tokenizer) -> tuple[int, int]:
    """Resolve the (id_for_"1", id_for_"0") pair used by bit-mode verify.

    BPE tokenizers (Qwen2/Llama family) encode bare digits "1" and "0"
    as single tokens — we just take the first encoded id and assert
    single-token. Cached on the tokenizer so repeated judge calls
    don't pay the encode cost.
    """
    cached = getattr(tokenizer, "_v0_bit_ids", None)
    if cached is not None:
        return cached
    one_ids = tokenizer.encode("1", add_special_tokens=False)
    zero_ids = tokenizer.encode("0", add_special_tokens=False)
    assert len(one_ids) == 1 and len(zero_ids) == 1, (
        f"bit-mode verify needs '1' and '0' to encode as single tokens; "
        f"got {one_ids=} {zero_ids=}"
    )
    pair = (one_ids[0], zero_ids[0])
    tokenizer._v0_bit_ids = pair
    return pair


def _parse_bit_verdict(text: str) -> str:
    """Parse a bit-mode verifier output. Looks at the first non-space
    character: '1' → ACCEPT, anything else → REJECT (conservative,
    same fallback as the word-mode parser when ambiguous)."""
    s = text.strip()
    if not s:
        return "REJECT"
    return "ACCEPT" if s[0] == "1" else "REJECT"


def _ensure_all_gpus_visible() -> None:
    # Ray actors created with num_gpus=0 (colocation pattern) get
    # CUDA_VISIBLE_DEVICES stripped, so vLLM's mp workers can't enumerate
    # devices. Restore it from nvidia-smi before LLM() runs.
    current = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if current:
        return
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        indices = [x.strip() for x in r.stdout.strip().splitlines() if x.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return
    if indices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(indices)


@dataclass
class DraftStep:
    """Result of one K-step draft generation starting from a committed prefix."""

    token_ids: list[int]          # the K draft tokens (may be < K if draft hit EOS)
    hit_eos: bool


@dataclass
class VerifyResult:
    """Result of target's verify forward over prefix + draft_token_ids.

    `draft_top1_ids` has exactly K entries (one per draft token position),
    each being target's argmax prediction at that position. Computed by
    taking the last K entries of vLLM's prompt_logprobs output — this is
    robust to vLLM's multimodal handling (which may skip logprobs at
    image-placeholder positions in the prefix, since draft tokens are
    always text and live at the end of the prompt).
    """

    draft_top1_ids: list[int | None]    # target argmax at each draft position (len K)
    bonus_token_id: int                 # target's own first generated token (beyond prompt)


@ray.remote(num_gpus=1)
class DraftEngine:
    """Owns a vllm.LLM running the draft model.

    The actor is kept alive across spec-decode rounds so vLLM's prefix cache
    can amortize repeated prefills.
    """

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        limit_mm_per_prompt: dict[str, int] | None = None,
        distributed_executor_backend: str | None = None,
        extract_hidden_states_layer: int | None = None,  # N-from-end
        enforce_eager: bool = False,
        logprobs_mode: str | None = None,
        head_checkpoint_path: str | None = None,
        head_tau_table_path: str | None = None,
        async_scheduling: bool = False,
    ) -> None:
        # `distributed_executor_backend="mp"` is required when this Ray
        # actor owns >1 GPU; vLLM's default Ray backend would conflict
        # with the outer actor's GPU accounting.
        # `enable_prefix_caching=True` is load-bearing for V1: the
        # target's verify→continue chain reuses KV across the two
        # vLLM calls via token-id prefix match.
        _ensure_all_gpus_visible()
        # Same external-head wiring as DraftEngineAsync: set
        # VLLM_EXTRACT_HIDDEN_STATES_LAYER before engine init so the
        # fork's gpu_model_runner installs the per-step persistent
        # buffer + post-forward hook for layer-N hidden states.
        # SamplingParams.extract_hidden_states=True opts requests in.
        # Used by T P4 to score val records offline with the
        # head and report a per-source AUROC.
        # NOTE: the cluster's vLLM image is `cascade-prod-fixes` @
        # 2c2e5d79f — i.e. WITHOUT the extract_mutates_args
        # fix. Under graph_async the Inductor DCE zeros the extract
        # buffer (head scores collapse to sigmoid(MLP(0))=0.5016).
        # Callers wanting non-degenerate hidden states must pass
        # `enforce_eager=True` until the fix is baked into the image.
        if extract_hidden_states_layer is not None:
            import os as _os
            _os.environ["VLLM_EXTRACT_HIDDEN_STATES_LAYER"] = str(
                extract_hidden_states_layer,
            )
        # In-engine cascade head on the sync engine. Same env-var
        # wiring as DraftEngineAsync — must be set in this process before
        # LLM() boots so get_in_engine_head() finds them.
        if head_checkpoint_path:
            import os as _os
            _os.environ["VLLM_CASCADE_ATTN_POOL_CKPT"] = head_checkpoint_path
        if head_tau_table_path:
            import os as _os
            _os.environ["VLLM_CASCADE_ATTN_POOL_TAU"] = head_tau_table_path
        llm_kwargs = dict(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            limit_mm_per_prompt=limit_mm_per_prompt or {"image": 1},
            enforce_eager=enforce_eager,
        )
        if distributed_executor_backend is not None:
            llm_kwargs["distributed_executor_backend"] = distributed_executor_backend
        if logprobs_mode is not None:
            llm_kwargs["logprobs_mode"] = logprobs_mode
        if async_scheduling:
            # axis isolation only — default (False) keeps the
            # engine's own default scheduling.
            llm_kwargs["async_scheduling"] = True
        self._llm = LLM(**llm_kwargs)

    def measure_forward_time(self, n_tokens: int = 64, n_iter: int = 3) -> float:
        """Generate `n_tokens` from a short prompt; return mean ms/token.

        Used by bench_colocation to compare isolated vs concurrent
        per-forward wall time.
        """
        import time

        prompt = "Hello."
        tokenizer = self._llm.get_tokenizer()
        ids = tokenizer.encode(prompt)
        sp = SamplingParams(max_tokens=n_tokens, temperature=0.0)

        # One warmup iteration
        self._llm.generate({"prompt_token_ids": ids}, sp, use_tqdm=False)

        total_ms = 0.0
        for _ in range(n_iter):
            t0 = time.perf_counter()
            self._llm.generate({"prompt_token_ids": ids}, sp, use_tqdm=False)
            total_ms += (time.perf_counter() - t0) * 1000.0
        return total_ms / n_iter / n_tokens

    # ------------------------------------------------------------------
    # NCCL data plane for GPU-to-GPU tensor transfer (projected vision
    # features in P6). ray.util.collective wraps NCCL; see module
    # docstring for the group conventions.
    # ------------------------------------------------------------------

    def setup_collective(
        self,
        rank: int = DRAFT_RANK,
        world_size: int = WORLD_SIZE,
        group_name: str = VISION_GROUP_NAME,
    ) -> None:
        """Initialize NCCL collective group. Call once after actor creation.

        The call blocks until both actors (draft + target) have joined the
        group. Bootstrap typically takes ~1 s.
        """
        col.init_collective_group(
            world_size=world_size,
            rank=rank,
            backend="nccl",
            group_name=group_name,
        )

    def send_tensor(
        self,
        shape: tuple[int, ...],
        dtype_str: str = "bfloat16",
        dst_rank: int = TARGET_RANK,
        group_name: str = VISION_GROUP_NAME,
    ) -> bool:
        """Send a deterministic test tensor to `dst_rank`. Used for
        validating the NCCL channel before we plug in real ViT features.

        Returns True on success. For P6 this becomes `send_vision_features`
        which takes a real tensor from the ViT forward.
        """
        dtype = getattr(torch, dtype_str)
        t = torch.arange(
            int(torch.tensor(shape).prod().item()), dtype=dtype, device="cuda"
        ).reshape(shape).contiguous()

        col.send(t, dst_rank, group_name)
        return True

    # ------------------------------------------------------------------
    # Spec-decode control plane.
    # ------------------------------------------------------------------

    def batch_generate_text(
        self,
        prompt_token_ids_list: list[list[int]],
        max_tokens: int,
    ) -> dict[str, float]:
        """Parallel to TargetEngine.batch_generate_target_only, used for
        draft-only bs sweeps. Measures draft's batched throughput.
        """
        import time

        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        requests = [{"prompt_token_ids": ids} for ids in prompt_token_ids_list]

        if prompt_token_ids_list:
            self._llm.generate(requests[:1], sp, use_tqdm=False)

        t0 = time.perf_counter()
        outputs = self._llm.generate(requests, sp, use_tqdm=False)
        wall_s = time.perf_counter() - t0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        return {
            "n_requests": float(len(outputs)),
            "total_tokens": float(total_tokens),
            "wall_s": float(wall_s),
            "tokens_per_s": float(total_tokens / wall_s) if wall_s > 0 else 0.0,
        }

    def keep_busy_for_seconds(
        self,
        duration_seconds: float = 30.0,
        prefill_len: int = 256,
        bs: int = 1,
    ) -> dict[str, float]:
        """Loop generate() calls back-to-back for `duration_seconds`.

        Each iteration uses fresh random-token prompts (defeats prefix
        cache) so each call is real prefill + 1 decode work on the GPU.
        No sleep between iterations — the only gap is Python+vLLM scheduler
        overhead between back-to-back generate() calls.

        This is the "sustained background load" pattern: aux process
        keeps the GPU partition busy throughout the entire concurrent
        main-process measurement, so the measured request_OH reflects
        worst-case sustained contention rather than burst contention.

        Returns:
          iters: number of generate() calls completed
          wall_s: total elapsed wall time
          mean_iter_ms: average per-iter wall (lower = aux gets more SMs)
          throughput_iters_per_s: iters / wall_s
        """
        import random
        import time

        rng = random.Random(0xBEEF)
        sp = SamplingParams(max_tokens=1, temperature=0.0)

        iters = 0
        total_iter_ms = 0.0
        start = time.perf_counter()
        while time.perf_counter() - start < duration_seconds:
            requests = []
            for _ in range(bs):
                ids = [rng.randint(100, 100_000) for _ in range(prefill_len)]
                requests.append({"prompt_token_ids": ids})
            t0 = time.perf_counter()
            self._llm.generate(requests, sp, use_tqdm=False)
            total_iter_ms += (time.perf_counter() - t0) * 1000.0
            iters += 1

        wall_s = time.perf_counter() - start
        return {
            "iters": float(iters),
            "wall_s": float(wall_s),
            "mean_iter_ms": total_iter_ms / iters if iters > 0 else 0.0,
            "throughput_iters_per_s": iters / wall_s if wall_s > 0 else 0.0,
            "prefill_len": float(prefill_len),
            "bs": float(bs),
        }

    def batch_measure_prefill_time(
        self,
        prefill_len: int = 1024,
        bs: int = 1,
        n_iter: int = 3,
    ) -> dict[str, float]:
        """Measure batched fresh-prefill time: submit `bs` requests each
        of `prefill_len` fresh (non-cached) tokens, max_tokens=1. vLLM
        prefills the whole prompt and decodes 1 bonus token.

        Each iter uses random-but-distinct token IDs so prefix caching
        cannot hit across iters — this measures the "new request enters
        the serving system" kernel shape, which dominates TTFT in the
        encode+prefill phase.

        Used by bench_pipeline_overlap to test whether prefill-shape
        work can overlap with concurrent decode under MPS (the proposed
        pipelined VLM-serving mechanism).
        """
        import random
        import time

        def build_requests(iter_seed: int) -> list[dict]:
            rng = random.Random(iter_seed * 1_000_003 + 42)
            reqs = []
            for b in range(bs):
                # Token IDs in a safe range (100..100000) to avoid special
                # tokens. Random per (iter, b) so nothing shared with the
                # prefix cache across iters or requests.
                ids = [rng.randint(100, 100_000) for _ in range(prefill_len)]
                reqs.append({"prompt_token_ids": ids})
            return reqs

        sp = SamplingParams(max_tokens=1, temperature=0.0)

        # Warmup with its own seed, distinct from timed iters (seeds 1..n_iter).
        self._llm.generate(build_requests(0), sp, use_tqdm=False)

        total_ms = 0.0
        for i in range(n_iter):
            requests = build_requests(i + 1)
            t0 = time.perf_counter()
            self._llm.generate(requests, sp, use_tqdm=False)
            total_ms += (time.perf_counter() - t0) * 1000.0

        mean_ms = total_ms / n_iter
        return {
            "wall_ms_per_batch": mean_ms,
            "bs": float(bs),
            "prefill_len": float(prefill_len),
        }

    def batch_measure_forward_time_with_prefix(
        self,
        prefix_len: int = 256,
        n_tokens: int = 64,
        bs: int = 1,
        n_iter: int = 3,
    ) -> dict[str, float]:
        """Measure batched forward time with a matched `prefix_len`-token
        prompt per request. Submits bs distinct requests, each generates
        n_tokens tokens sequentially (so n_tokens decode forwards on the
        batched engine). Returns per-forward wall time at this (bs, prefix_len).

        Used by bench_bubble_pipelined_sweep to anchor draft's per-step
        cost at production-sized prefixes (not the 2-token toy prompt
        that measure_forward_time uses).
        """
        import time

        tokenizer = self._llm.get_tokenizer()
        base = tokenizer.encode("Hello. ")
        while len(base) < prefix_len:
            base = base + base
        prefix_ids = base[:prefix_len]

        # Build bs distinct prompts (slight tail perturbation)
        requests = []
        for b in range(bs):
            p = list(prefix_ids)
            p[-1] = (p[-1] + b * 13) % 150000
            requests.append({"prompt_token_ids": p})

        sp = SamplingParams(max_tokens=n_tokens, temperature=0.0)

        # Warmup (primes prefix cache + CUDA graphs)
        self._llm.generate(requests, sp, use_tqdm=False)

        total_ms = 0.0
        total_tokens = 0
        for _ in range(n_iter):
            t0 = time.perf_counter()
            outputs = self._llm.generate(requests, sp, use_tqdm=False)
            total_ms += (time.perf_counter() - t0) * 1000.0
            total_tokens += sum(len(o.outputs[0].token_ids) for o in outputs)

        mean_wall_ms = total_ms / n_iter
        mean_tokens = total_tokens / n_iter
        # Per forward, all bs requests produce 1 token → n_tokens forwards total.
        per_forward_ms = mean_wall_ms / n_tokens if n_tokens > 0 else 0.0
        return {
            "wall_ms": mean_wall_ms,
            "total_tokens": float(mean_tokens),
            "bs": float(bs),
            "prefix_len": float(prefix_len),
            "n_tokens": float(n_tokens),
            "per_forward_ms": per_forward_ms,
            "aggregate_tok_per_s": (
                mean_tokens * 1000.0 / mean_wall_ms if mean_wall_ms > 0 else 0.0
            ),
        }

    def generate_continuation(
        self,
        prompt_token_ids: list[int],
        max_tokens: int,
        multi_modal_data: dict[str, Any] | None = None,
    ) -> DraftStep:
        """Generate up to `max_tokens` continuation tokens from the given prefix."""
        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )

        req = {"prompt_token_ids": prompt_token_ids}
        if multi_modal_data is not None:
            req["multi_modal_data"] = multi_modal_data

        outputs = self._llm.generate(req, sp, use_tqdm=False)
        out = outputs[0].outputs[0]

        # "stop" means an actual stop token (EOS or custom); "length" means
        # we simply hit max_tokens=K, which happens on every normal round.
        # Only the former is a signal to stop the outer spec-decode loop.
        return DraftStep(
            token_ids=list(out.token_ids),
            hit_eos=out.finish_reason == "stop",
        )

    def batch_generate_continuation(
        self,
        prompt_token_ids_list: list[list[int]],
        max_tokens: int,
        ignore_eos: bool = True,
    ) -> list[list[int]]:
        """Batched generation given token-ID prefixes. Returns per-request
        generated token IDs (length == max_tokens when ignore_eos=True).

        Used by Y1 (whole-sequence spec decode bench). Drafts a fixed-length
        completion for each request in a single batched vLLM call so the
        scheduler can overlap them; max_tokens is enforced uniformly so
        per-request walls are comparable.
        """
        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            ignore_eos=ignore_eos,
        )
        requests = [{"prompt_token_ids": list(ids)} for ids in prompt_token_ids_list]
        outputs = self._llm.generate(requests, sp, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outputs]

    def generate_text(
        self,
        user_prompts: list,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate text responses to a batch of user prompts. Applies
        the model's chat template (Qwen2.5 instruct format) so this
        matches normal API serving behavior. Returns decoded text only.

        Each entry of `user_prompts` may be a string (single-turn) or
        a list of message dicts (multi-turn — e.g., MT-Bench T2).
        """
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        tokenizer = self._llm.get_tokenizer()
        formatted = [
            tokenizer.apply_chat_template(
                _to_messages(p),
                add_generation_prompt=True,
                tokenize=False,
            )
            for p in user_prompts
        ]
        outputs = self._llm.generate(formatted, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def generate_text_with_logprobs(
        self,
        user_prompts: list[str],
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> list[dict]:
        """Generate responses with per-token logprobs.

        Used by the offline draft + calibration experiment to test
        whether draft confidence (mean / min log-probability) correlates
        with the judge's ACCEPT verdict — i.e., whether a confidence
        threshold is a viable way to skip target verification entirely.

        Returns one dict per prompt with:
          - text: generated text
          - n_output_tokens: number of generated tokens
          - mean_logprob: average log-probability of generated tokens
          - min_logprob: minimum (worst) per-token log-probability
        """
        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=1,  # chosen-token logprob is always included
        )
        tokenizer = self._llm.get_tokenizer()
        formatted = [
            tokenizer.apply_chat_template(
                _to_messages(p),
                add_generation_prompt=True,
                tokenize=False,
            )
            for p in user_prompts
        ]
        outputs = self._llm.generate(formatted, sp, use_tqdm=False)

        results: list[dict] = []
        for o in outputs:
            completion = o.outputs[0]
            token_logprobs: list[float] = []
            # completion.logprobs is list[dict[token_id, Logprob] | None],
            # one entry per output token. The chosen token's logprob is
            # always included when SamplingParams.logprobs is set.
            if completion.logprobs is not None:
                for token_id, lp_dict in zip(
                    completion.token_ids, completion.logprobs
                ):
                    if lp_dict is None:
                        continue
                    lp = lp_dict.get(token_id)
                    if lp is not None:
                        token_logprobs.append(float(lp.logprob))
            n_tokens = len(token_logprobs)
            mean_lp = sum(token_logprobs) / n_tokens if n_tokens > 0 else 0.0
            min_lp = min(token_logprobs) if token_logprobs else 0.0
            results.append({
                "text": completion.text,
                "n_output_tokens": n_tokens,
                "mean_logprob": mean_lp,
                "min_logprob": min_lp,
            })
        return results

    def generate_text_mm(
        self,
        records: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> list[str]:
        """Multimodal variant of generate_text. Each record:
          {"prompt": str, "images": list[str] | None}
        where images are paths reachable from this actor's filesystem.
        Returns list[str] of decoded responses (parallel to records)."""
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        tokenizer = self._llm.get_tokenizer()
        requests = []
        for r in records:
            text, mm = _build_mm_request(
                tokenizer, r["prompt"], r.get("images"),
            )
            req = {"prompt": text}
            if mm is not None:
                req["multi_modal_data"] = mm
            requests.append(req)
        outputs = self._llm.generate(requests, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def generate_text_mm_with_hidden_states(
        self,
        records: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        logprobs: int | None = None,
    ) -> list[dict]:
        """Multimodal generate that emits the last-position hidden state
        (cut-1) per request, optionally with logprob stats too. Each
        record: {"prompt": str, "images": list[str] | None}. Requires
        the cascade-prod-fixes fork (SamplingParams.extract_hidden_states
        + RequestOutput.hidden_states attached). Used by T P4/P7 to
        score val/test records with the head offline.

        If `logprobs` is set (>=2), each result also carries the four
        output-confidence stats Gatekeeper / output-conf gates threshold
        (mean_logprob, min_logprob, mean_max_prob, neg_mean_entropy),
        so one pass produces inputs for both head and Gatekeeper.

        Returns one dict per record:
          {text, n_output_tokens, finish_reason, hidden_states:
           torch.Tensor[hidden_dim, fp16] | None, mean_logprob?,
           min_logprob?, mean_max_prob?, neg_mean_entropy?}"""
        import math
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        sp.extract_hidden_states = True
        if logprobs is not None:
            sp.logprobs = logprobs
        tokenizer = self._llm.get_tokenizer()
        requests = []
        for r in records:
            text, mm = _build_mm_request(
                tokenizer, r["prompt"], r.get("images"),
            )
            req = {"prompt": text}
            if mm is not None:
                req["multi_modal_data"] = mm
            requests.append(req)
        outputs = self._llm.generate(requests, sp, use_tqdm=False)
        results: list[dict] = []
        for o in outputs:
            completion = o.outputs[0]
            hs = getattr(o, "hidden_states", None)
            row = {
                "text": completion.text,
                "n_output_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "hidden_states": hs,
            }
            if logprobs is not None and completion.logprobs is not None:
                chosen_lps: list[float] = []
                max_probs: list[float] = []
                entropies: list[float] = []
                for tok_id, lp_dict in zip(
                    completion.token_ids, completion.logprobs,
                ):
                    if not lp_dict:
                        continue
                    lp = lp_dict.get(tok_id)
                    if lp is not None:
                        chosen_lps.append(float(lp.logprob))
                    probs = [math.exp(float(x.logprob))
                             for x in lp_dict.values()]
                    if probs:
                        max_probs.append(max(probs))
                        z = sum(probs)
                        if z > 0 and len(probs) > 1:
                            entropies.append(
                                -sum((p / z) * math.log(p / z)
                                     for p in probs if p > 0)
                            )
                row["mean_logprob"] = (sum(chosen_lps) / len(chosen_lps)
                                       if chosen_lps else None)
                row["min_logprob"] = (min(chosen_lps) if chosen_lps
                                      else None)
                row["mean_max_prob"] = (sum(max_probs) / len(max_probs)
                                        if max_probs else None)
                row["neg_mean_entropy"] = (-sum(entropies) / len(entropies)
                                           if entropies else None)
            results.append(row)
        return results

    def generate_text_mm_per_token_logprobs(
        self,
        records: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        logprobs: int = 20,
    ) -> list[dict]:
        """Like generate_text_mm_with_logprobs but emits the FULL
        per-token sequence of logprob features instead of aggregating.

        Returns one dict per record:
          {text, n_output_tokens, finish_reason,
           per_token: list[[chosen_logprob, max_softmax_prob,
                            neg_entropy, position_fraction]]}
        Used by T P16 to train sequence models (transformer,
        BiLSTM, 1D CNN) on the per-token logprob trajectory."""
        import math
        sp = SamplingParams(
            max_tokens=max_tokens, temperature=temperature,
            logprobs=logprobs,
        )
        tokenizer = self._llm.get_tokenizer()
        requests = []
        for r in records:
            text, mm = _build_mm_request(
                tokenizer, r["prompt"], r.get("images"),
            )
            req = {"prompt": text}
            if mm is not None:
                req["multi_modal_data"] = mm
            requests.append(req)
        outputs = self._llm.generate(requests, sp, use_tqdm=False)
        results: list[dict] = []
        for o in outputs:
            completion = o.outputs[0]
            per_token: list[list[float]] = []
            if completion.logprobs is not None:
                n = len(completion.token_ids)
                for idx, (tok_id, lp_dict) in enumerate(
                    zip(completion.token_ids, completion.logprobs),
                ):
                    if not lp_dict:
                        continue
                    lp = lp_dict.get(tok_id)
                    chosen = (float(lp.logprob) if lp is not None
                              else 0.0)
                    probs = [math.exp(float(o.logprob))
                             for o in lp_dict.values()]
                    if not probs:
                        continue
                    max_p = max(probs)
                    z = sum(probs)
                    ent = (
                        -sum((p / z) * math.log(p / z)
                             for p in probs if p > 0)
                        if z > 0 and len(probs) > 1 else 0.0
                    )
                    pos_frac = (idx + 1) / max(1, n)
                    per_token.append([chosen, max_p, -ent, pos_frac])
            results.append({
                "text": completion.text,
                "n_output_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "per_token": per_token,
            })
        return results

    def _run_in_engine_head_batch(
        self,
        records: list[dict],
        max_tokens: int = 512,
    ) -> list[dict]:
        """equivalence check helper. Runs each record
        through the engine with BOTH `emit_per_token_feature_seq=True`
        AND `in_engine_cascade_head=True`, so the returned
        CompletionOutput carries both:
          - per_token_features (the features the gate is run on)
          - head_decision (the in-engine verdict dict)
        Caller can then recompute the driver-side verdict from the
        same features and confirm equivalence.

        Requires DraftEngine to have been booted with the cascade head
        env vars set (the in-engine head loads once at boot).
        """
        sp = SamplingParams(
            max_tokens=max_tokens, temperature=0.0,
        )
        # Forks may not expose both fields; tolerate missing attrs.
        for attr in ("emit_per_token_feature_seq", "in_engine_cascade_head"):
            try:
                setattr(sp, attr, True)
            except Exception:
                pass
        tokenizer = self._llm.get_tokenizer()
        requests = []
        for r in records:
            text, mm = _build_mm_request(
                tokenizer, r["prompt"], r.get("images"),
            )
            req = {"prompt": text}
            if mm is not None:
                req["multi_modal_data"] = mm
            sp_per = SamplingParams(
                max_tokens=max_tokens, temperature=0.0,
            )
            for attr in ("emit_per_token_feature_seq",
                         "in_engine_cascade_head"):
                try:
                    setattr(sp_per, attr, True)
                except Exception:
                    pass
            try:
                setattr(sp_per, "cascade_source", r.get("source"))
            except Exception:
                pass
            req["sampling_params"] = sp_per
            requests.append(req)
        # Pass per-request SamplingParams.
        results: list[dict] = []
        outputs = self._llm.generate(
            [{k: v for k, v in r.items() if k != "sampling_params"}
             for r in requests],
            [r["sampling_params"] for r in requests],
            use_tqdm=False,
        )
        for o in outputs:
            completion = o.outputs[0]
            results.append({
                "text": completion.text,
                "n_output_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "per_token_features": getattr(
                    completion, "per_token_features", None,
                ),
                "head_decision": getattr(
                    completion, "head_decision", None,
                ),
            })
        return results

    def generate_text_mm_per_token_topk_values(
        self,
        records: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        logprobs: int = 20,
    ) -> list[dict]:
        """Emit raw top-K logprob VALUES per token, with no softmax-side
        feature math. Pairs with logprobs_mode="raw_logits" to get top-K
        raw logits (caller computes logit-space features); equally works
        with the default raw_logprobs mode for symmetric comparison.

        Returns one dict per record:
          {text, n_output_tokens, finish_reason,
           per_token: list[{"chosen": float, "topk": list[float] sorted
                            descending, "chosen_in_topk": bool}]}
        """
        sp = SamplingParams(
            max_tokens=max_tokens, temperature=temperature,
            logprobs=logprobs,
        )
        tokenizer = self._llm.get_tokenizer()
        requests = []
        for r in records:
            text, mm = _build_mm_request(
                tokenizer, r["prompt"], r.get("images"),
            )
            req = {"prompt": text}
            if mm is not None:
                req["multi_modal_data"] = mm
            requests.append(req)
        outputs = self._llm.generate(requests, sp, use_tqdm=False)
        results: list[dict] = []
        for o in outputs:
            completion = o.outputs[0]
            per_token: list[dict] = []
            if completion.logprobs is not None:
                for tok_id, lp_dict in zip(
                    completion.token_ids, completion.logprobs,
                ):
                    if not lp_dict:
                        continue
                    chosen_lp = lp_dict.get(tok_id)
                    chosen_val = (float(chosen_lp.logprob)
                                  if chosen_lp is not None
                                  else float("-inf"))
                    topk = sorted(
                        (float(o.logprob) for o in lp_dict.values()),
                        reverse=True,
                    )
                    per_token.append({
                        "chosen": chosen_val,
                        "topk": topk,
                        "chosen_in_topk": chosen_lp is not None,
                    })
            results.append({
                "text": completion.text,
                "n_output_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "per_token": per_token,
            })
        return results

    def generate_text_mm_inline_aggregate(
        self,
        records: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        logprobs: int = 20,
    ) -> list[dict]:
        """T P17 validation: opt into SamplingParams.
        emit_aggregate_logprob_stats=True. Returns both the inline
        aggregate (from CompletionOutput.aggregate_logprob_stats) AND
        the driver-side per-token sequence so the caller can compare
        them numerically. Requires worker vLLM on lp-classifier-inline
        @ >= a021fde6a."""
        import math
        sp = SamplingParams(
            max_tokens=max_tokens, temperature=temperature,
            logprobs=logprobs,
            emit_aggregate_logprob_stats=True,
        )
        tokenizer = self._llm.get_tokenizer()
        requests = []
        for r in records:
            text, mm = _build_mm_request(
                tokenizer, r["prompt"], r.get("images"),
            )
            req = {"prompt": text}
            if mm is not None:
                req["multi_modal_data"] = mm
            requests.append(req)
        outputs = self._llm.generate(requests, sp, use_tqdm=False)
        results: list[dict] = []
        for o in outputs:
            completion = o.outputs[0]
            per_token: list[list[float]] = []
            if completion.logprobs is not None:
                n = len(completion.token_ids)
                for idx, (tok_id, lp_dict) in enumerate(
                    zip(completion.token_ids, completion.logprobs),
                ):
                    if not lp_dict:
                        continue
                    lp = lp_dict.get(tok_id)
                    chosen = (float(lp.logprob) if lp is not None
                              else 0.0)
                    probs = [math.exp(float(o.logprob))
                             for o in lp_dict.values()]
                    if not probs:
                        continue
                    max_p = max(probs)
                    z = sum(probs)
                    ent = (
                        -sum((p / z) * math.log(p / z)
                             for p in probs if p > 0)
                        if z > 0 and len(probs) > 1 else 0.0
                    )
                    pos_frac = (idx + 1) / max(1, n)
                    per_token.append([chosen, max_p, -ent, pos_frac])
            results.append({
                "text": completion.text,
                "n_output_tokens": len(completion.token_ids),
                "inline_stats": getattr(
                    completion, "aggregate_logprob_stats", None,
                ),
                "per_token": per_token,
            })
        return results

    def generate_text_mm_with_logprobs(
        self,
        records: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        logprobs: int = 20,
    ) -> list[dict]:
        """Multimodal generate that also emits per-token logprob stats.
        Each record: {"prompt": str, "images": list[str] | None}.
        With logprobs>=2 the returned dict carries the four output-
        confidence stats baselines threshold:
          text, n_output_tokens, mean_logprob, min_logprob,
          mean_max_prob, neg_mean_entropy.
        max_softmax (mean_max_prob) + neg predictive entropy
        (neg_mean_entropy) are the Gatekeeper-rule signals; mirrors
        the async submit() path's stat computation."""
        import math
        sp = SamplingParams(
            max_tokens=max_tokens, temperature=temperature, logprobs=logprobs,
        )
        tokenizer = self._llm.get_tokenizer()
        requests = []
        for r in records:
            text, mm = _build_mm_request(
                tokenizer, r["prompt"], r.get("images"),
            )
            req = {"prompt": text}
            if mm is not None:
                req["multi_modal_data"] = mm
            requests.append(req)
        outputs = self._llm.generate(requests, sp, use_tqdm=False)
        results: list[dict] = []
        for o in outputs:
            completion = o.outputs[0]
            chosen_lps: list[float] = []
            max_probs: list[float] = []
            entropies: list[float] = []
            if completion.logprobs is not None:
                for tok_id, lp_dict in zip(
                    completion.token_ids, completion.logprobs,
                ):
                    if not lp_dict:
                        continue
                    lp = lp_dict.get(tok_id)
                    if lp is not None:
                        chosen_lps.append(float(lp.logprob))
                    probs = [math.exp(float(o.logprob))
                             for o in lp_dict.values()]
                    if probs:
                        max_probs.append(max(probs))
                        z = sum(probs)
                        if z > 0 and len(probs) > 1:
                            entropies.append(
                                -sum((p / z) * math.log(p / z)
                                     for p in probs if p > 0)
                            )
            n = len(completion.token_ids)
            results.append({
                "text": completion.text,
                "n_output_tokens": n,
                "finish_reason": completion.finish_reason,
                "mean_logprob": (sum(chosen_lps) / len(chosen_lps)
                                 if chosen_lps else None),
                "min_logprob": (min(chosen_lps) if chosen_lps else None),
                "mean_max_prob": (sum(max_probs) / len(max_probs)
                                  if max_probs else None),
                "neg_mean_entropy": (-sum(entropies) / len(entropies)
                                     if entropies else None),
            })
        return results

    # ------------------------------------------------------------------
    # vision offload — run target's ViT+merger on the draft GPU(s)
    # so the target can skip its own ViT forward (consume image_embeds
    # via vLLM's public multi_modal_data API).
    # ------------------------------------------------------------------

    def init_target_vision(
        self,
        target_model_id: str,
        dtype: str = "bfloat16",
    ) -> dict[str, Any]:
        """Lazy-load the target VLM's `.visual` submodule (ViT + merger)
        onto this draft actor's GPU. Returns a small status dict for
        bench-side asserts. Idempotent; repeat calls reuse the loaded
        encoder.

        The encoder runs on its own CUDA stream so its forward overlaps
        with the draft pipeline's vLLM forwards on the actor's main
        stream. Memory cost: ~1.8 GB for the 72B-VL visual on bf16
        (~0.45 GB / GPU at TP=4)."""
        from prorouter.target_vision import TargetVisionEncoder

        if getattr(self, "_target_vision", None) is not None:
            current_id = getattr(self, "_target_vision_model_id", None)
            if current_id == target_model_id:
                return {
                    "status": "already_loaded",
                    "model_id": current_id,
                }
            raise RuntimeError(
                f"target vision already initialized with {current_id}; "
                f"refusing to reload as {target_model_id}"
            )

        torch_dtype = getattr(torch, dtype)
        # Default to cuda:0 — vLLM's mp backend pins vLLM workers to
        # other devices already; cuda:0 here is the actor's "main"
        # control GPU and the only one we can issue python-side
        # tensor ops on without colliding with vLLM's TP workers.
        device = torch.device("cuda:0")
        self._target_vision = TargetVisionEncoder(
            target_model_id=target_model_id,
            device=device,
            dtype=torch_dtype,
        )
        self._target_vision_model_id = target_model_id
        return {
            "status": "loaded",
            "model_id": target_model_id,
            "device": str(device),
            "dtype": dtype,
        }

    def precompute_target_vision_mm(
        self,
        records: list[dict],
        sync: bool = True,
    ) -> list[dict | None]:
        """Run the target's ViT+merger over each record's images on
        this draft actor. Returns a per-record payload usable by
        TargetEngine.*_mm_with_embeds:

          [
            {"image_embeds": cpu_tensor[N_image_tokens, llm_hidden_dim],
             "image_grid_thw": cpu_tensor[n_images, 3]},
            None,  # for records with no images
            ...
          ]

        Tensors are on CPU + contiguous so they serialize cleanly through
        the Ray object store on the way to the target actor.

        sync=True (default) blocks the actor thread until the encoder
        stream finishes. The two-stream overlap with vLLM's draft
        pipeline is achieved at the *caller* level: kick this off
        on a separate Ray task, run draft generation on another, then
        gather both. (Future: a fused method that runs draft + vision
        on one actor and returns both at once.)
        """
        if getattr(self, "_target_vision", None) is None:
            raise RuntimeError(
                "init_target_vision() must be called before "
                "precompute_target_vision_mm()"
            )

        from PIL import Image as _PIL_Image

        # Stage 1: load images per record. Empty-image records yield None.
        images_per_record: list[list] = []
        non_empty_indices: list[int] = []
        for i, r in enumerate(records):
            paths = r.get("images") or []
            if not paths:
                images_per_record.append([])
                continue
            imgs = [_PIL_Image.open(p).convert("RGB") for p in paths]
            images_per_record.append(imgs)
            non_empty_indices.append(i)

        if not non_empty_indices:
            return [None] * len(records)

        # Stage 2: batched encode (one ViT forward across all records'
        # images), then split.
        non_empty_images = [images_per_record[i] for i in non_empty_indices]
        encoded = self._target_vision.encode_batch(
            non_empty_images, sync=sync, to_cpu=True,
        )

        # Stage 3: weave back into per-record list with Nones for empty.
        out: list[dict | None] = [None] * len(records)
        for orig_i, payload in zip(non_empty_indices, encoded):
            out[orig_i] = payload
        return out

    def ping(self) -> str:
        return "ok"


@ray.remote(num_gpus=1)
class TargetEngine:
    """Owns a vllm.LLM running the target model.

    Exposes a `verify` method that runs one prefill forward over
    (prefix + draft_token_ids) with prompt_logprobs=1, plus produces one
    bonus token beyond the prompt.
    """

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        limit_mm_per_prompt: dict[str, int] | None = None,
        distributed_executor_backend: str | None = None,
        max_num_seqs: int | None = None,
        enable_prefix_caching: bool = True,
    ) -> None:
        # See DraftEngine.__init__ for the rationale on `mp` backend
        # and `enable_prefix_caching=True` (V1 verify→continue KV reuse).
        # Prefix caching is disabled for the multi-image MuirBench target
        # pass — it gives no benefit for independent one-shot generation and its
        # mm_receiver_cache can assert on multi-image batches (F1).
        _ensure_all_gpus_visible()
        llm_kwargs = dict(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            limit_mm_per_prompt=limit_mm_per_prompt or {"image": 1},
            # Required for the vision-offload path
            # (generate_text_mm_with_embeds / batch_judge_verify_binary_mm_with_embeds):
            # vllm rejects pre-encoded `image_embeds` without this flag.
            enable_mm_embeds=True,
        )
        if distributed_executor_backend is not None:
            llm_kwargs["distributed_executor_backend"] = distributed_executor_backend
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs
        self._llm = LLM(**llm_kwargs)

    def measure_forward_time(self, n_tokens: int = 64, n_iter: int = 3) -> float:
        """Generate `n_tokens` from a short prompt; return mean ms/token.

        Mirror of DraftEngine.measure_forward_time, for isolated vs
        concurrent forward-time comparison.
        """
        import time

        prompt = "Hello."
        tokenizer = self._llm.get_tokenizer()
        ids = tokenizer.encode(prompt)
        sp = SamplingParams(max_tokens=n_tokens, temperature=0.0)

        self._llm.generate({"prompt_token_ids": ids}, sp, use_tqdm=False)

        total_ms = 0.0
        for _ in range(n_iter):
            t0 = time.perf_counter()
            self._llm.generate({"prompt_token_ids": ids}, sp, use_tqdm=False)
            total_ms += (time.perf_counter() - t0) * 1000.0
        return total_ms / n_iter / n_tokens

    # ------------------------------------------------------------------
    # NCCL data plane — receiving side. See DraftEngine.setup_collective
    # for the group conventions.
    # ------------------------------------------------------------------

    def setup_collective(
        self,
        rank: int = TARGET_RANK,
        world_size: int = WORLD_SIZE,
        group_name: str = VISION_GROUP_NAME,
    ) -> None:
        """Initialize NCCL collective group. Call once after actor creation."""
        col.init_collective_group(
            world_size=world_size,
            rank=rank,
            backend="nccl",
            group_name=group_name,
        )

    def recv_tensor_and_checksum(
        self,
        shape: tuple[int, ...],
        dtype_str: str = "bfloat16",
        src_rank: int = DRAFT_RANK,
        group_name: str = VISION_GROUP_NAME,
    ) -> dict[str, float]:
        """Receive a tensor from `src_rank` and return its checksum.

        Used to validate the NCCL channel end-to-end. The expected input is
        the deterministic tensor produced by DraftEngine.send_tensor(shape).

        Returns {"sum": float, "first": float, "last": float, "numel": int}.
        The caller compares to the known expected values to confirm
        GPU→GPU transfer was lossless.

        For P6 this becomes `recv_vision_features_and_verify`, which
        receives the projected feature tensor, wraps it in
        multi_modal_data={"image_embeds": ...}, and passes it through to
        vLLM's generate (leveraging the `image_embeds` public API we
        confirmed in the fork-action-items investigation).
        """
        dtype = getattr(torch, dtype_str)
        buf = torch.empty(shape, dtype=dtype, device="cuda")
        col.recv(buf, src_rank, group_name)

        return {
            "sum": float(buf.sum().item()),
            "first": float(buf.flatten()[0].item()),
            "last": float(buf.flatten()[-1].item()),
            "numel": int(buf.numel()),
        }

    # ------------------------------------------------------------------
    # Spec-decode control plane.
    # ------------------------------------------------------------------

    def verify(
        self,
        prefix_token_ids: list[int],
        draft_token_ids: list[int],
        multi_modal_data: dict[str, Any] | None = None,
    ) -> VerifyResult:
        """Run one target forward over (prefix + draft), return per-position top-1 + bonus."""
        full_ids = list(prefix_token_ids) + list(draft_token_ids)

        sp = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=1,  # top-1 logprob at each prompt position
        )

        req = {"prompt_token_ids": full_ids}
        if multi_modal_data is not None:
            req["multi_modal_data"] = multi_modal_data

        outputs = self._llm.generate(req, sp, use_tqdm=False)
        ro = outputs[0]

        # prompt_logprobs is aligned with prompt tokens; each entry is either
        # None (no distribution — typically position 0) or a dict
        # {token_id: Logprob}. Draft tokens are always text and live at the
        # end of the prompt, so we take the last K entries. This is robust
        # even if vLLM omits entries for image-placeholder positions in the
        # prefix (current behavior in 0.19.1 per our investigation).
        K = len(draft_token_ids)
        tail = ro.prompt_logprobs[-K:] if K > 0 else []

        draft_top1_ids: list[int | None] = []
        for entry in tail:
            if entry is None:
                draft_top1_ids.append(None)
                continue
            best_id = max(entry.items(), key=lambda kv: kv[1].logprob)[0]
            draft_top1_ids.append(best_id)

        bonus_token_id = ro.outputs[0].token_ids[0]

        return VerifyResult(
            draft_top1_ids=draft_top1_ids,
            bonus_token_id=bonus_token_id,
        )

    def generate_target_only(
        self,
        prompt_token_ids: list[int],
        max_tokens: int,
        multi_modal_data: dict[str, Any] | None = None,
    ) -> list[int]:
        """Baseline: run target alone (no spec decode) and return output tokens."""
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)

        req = {"prompt_token_ids": prompt_token_ids}
        if multi_modal_data is not None:
            req["multi_modal_data"] = multi_modal_data

        outputs = self._llm.generate(req, sp, use_tqdm=False)
        return list(outputs[0].outputs[0].token_ids)

    def measure_verify_time(
        self,
        prefix_len: int = 256,
        K: int = 4,
        n_iter: int = 5,
        use_prompt_logprobs: bool = True,
    ) -> float:
        """Measure mean ms per verify forward. Shape matches bubble-pipelined
        scheduler's operational call: prefill K+1 positions past a realistic
        prefix, with prompt_logprobs=1.

        Unlike measure_forward_time (pure decode loop), this captures the
        verify kernel pattern: prefill-heavy forward over (prefix + K) plus
        one decode step for the bonus token. Relevant contention
        characteristics differ from decode-vs-decode.

        Args:
          prefix_len: length of the "realistic prefix" to prefix-cache; ~256
            matches our typical spec-decode workload mid-request.
          K: number of draft tokens to verify per forward.
          n_iter: iterations averaged (plus 1 warmup).
        """
        import time

        tokenizer = self._llm.get_tokenizer()

        # Build a prefix of approximately the target length. "Hello. " is
        # ~3 tokens so we repeat it to reach prefix_len.
        base = tokenizer.encode("Hello. ")
        while len(base) < prefix_len:
            base = base + base
        prefix_ids = base[:prefix_len]

        # Pick K plausible draft tokens. Doesn't matter for timing; use the
        # first K token IDs that definitely exist.
        draft_tokens = [100, 200, 300, 400, 500, 600, 700, 800][:K]

        full_ids = list(prefix_ids) + list(draft_tokens)

        # Pass use_prompt_logprobs=False to measure the "fix A ceiling": what
        # verify would cost if vLLM's LM-head-per-position tax were removed.
        # Not functional — returns no scoring logits — but isolates the
        # underlying transformer+decode cost from the tax.
        sp_kwargs = dict(max_tokens=1, temperature=0.0)
        if use_prompt_logprobs:
            sp_kwargs["prompt_logprobs"] = 1
        sp = SamplingParams(**sp_kwargs)

        # Warmup (prime prefix cache + CUDA graphs)
        self._llm.generate({"prompt_token_ids": full_ids}, sp, use_tqdm=False)

        total_ms = 0.0
        for _ in range(n_iter):
            t0 = time.perf_counter()
            self._llm.generate({"prompt_token_ids": full_ids}, sp, use_tqdm=False)
            total_ms += (time.perf_counter() - t0) * 1000.0
        return total_ms / n_iter

    def batch_measure_verify_time(
        self,
        prefix_len: int = 256,
        K: int = 4,
        bs: int = 1,
        n_iter: int = 3,
        use_prompt_logprobs: bool = True,
        vary_tails: bool = True,
    ) -> dict[str, float]:
        """Measure batched verify: submit `bs` requests each of shape
        (prefix_len + K), max_tokens=1.

        vLLM's continuous batching processes all bs requests in the same
        forward, so wall time is per-BATCH not per-request.

        `vary_tails=True` (default) — the K tail tokens change every
        iteration. The shared `prefix_len`-token prefix is prefix-cached
        (enable_prefix_caching=True on the engine), but the K tail tokens
        are fresh each iter, so vLLM must prefill K new positions and
        decode 1 bonus = K+1 new forward positions per request per iter.
        This matches the shape of a real verify forward in spec decode:
        cached prefix + K draft tokens + 1 bonus, all attended against
        the cached prefix KV.

        `vary_tails=False` — old behavior. Same prompts across iters, so
        after the warmup iter everything is cached and each timed iter
        collapses to one decode forward + scheduler round-trip. Kept for
        back-compat and to reproduce the pre-fix measurement.

        Returns {"wall_ms_per_batch": per-batch wall ms, "bs", "prefix_len",
        "K", "vary_tails"}.
        """
        import time

        tokenizer = self._llm.get_tokenizer()
        base = tokenizer.encode("Hello. ")
        while len(base) < prefix_len:
            base = base + base
        prefix_ids = base[:prefix_len]

        def build_requests(iter_seed: int) -> list[dict]:
            # bs distinct tails per iter; iter_seed rotates tails across iters
            # so the K tail positions aren't prefix-cached from warmup/prior.
            reqs = []
            for b in range(bs):
                tail = [
                    (100 + k * 100 + b * 13 + iter_seed * 7919) % 150000
                    for k in range(K)
                ]
                reqs.append({"prompt_token_ids": list(prefix_ids) + tail})
            return reqs

        sp_kwargs = dict(max_tokens=1, temperature=0.0)
        if use_prompt_logprobs:
            sp_kwargs["prompt_logprobs"] = 1
        sp = SamplingParams(**sp_kwargs)

        # Warmup — primes the prefix cache for the shared `prefix_len` prefix
        # and primes CUDA graphs at (bs, K+1) shape. Warmup uses iter_seed=0,
        # which is distinct from the timed iters (seeds 1..n_iter).
        warmup_seed = 0 if not vary_tails else -1
        self._llm.generate(build_requests(warmup_seed), sp, use_tqdm=False)

        total_ms = 0.0
        for i in range(n_iter):
            iter_seed = (i + 1) if vary_tails else 0
            requests = build_requests(iter_seed)
            t0 = time.perf_counter()
            self._llm.generate(requests, sp, use_tqdm=False)
            total_ms += (time.perf_counter() - t0) * 1000.0

        mean_ms = total_ms / n_iter
        return {
            "wall_ms_per_batch": mean_ms,
            "bs": float(bs),
            "prefix_len": float(prefix_len),
            "K": float(K),
            "vary_tails": float(vary_tails),
        }

    def batch_generate_target_only(
        self,
        prompt_token_ids_list: list[list[int]],
        max_tokens: int,
    ) -> dict[str, float]:
        """Submit a list of prompts to vLLM at once. vLLM's continuous
        batching serves them concurrently; we measure aggregate throughput.

        Returns {"n_requests", "total_tokens", "wall_s", "tokens_per_s",
                 "first_preview"}.
        """
        import time

        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        requests = [{"prompt_token_ids": ids} for ids in prompt_token_ids_list]

        # Warmup (single request) to ensure CUDA graphs compiled.
        if prompt_token_ids_list:
            self._llm.generate(requests[:1], sp, use_tqdm=False)

        t0 = time.perf_counter()
        outputs = self._llm.generate(requests, sp, use_tqdm=False)
        wall_s = time.perf_counter() - t0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        first_preview = (
            self._llm.get_tokenizer().decode(outputs[0].outputs[0].token_ids)[:80]
            if outputs else ""
        )
        return {
            "n_requests": float(len(outputs)),
            "total_tokens": float(total_tokens),
            "wall_s": float(wall_s),
            "tokens_per_s": float(total_tokens / wall_s) if wall_s > 0 else 0.0,
            "first_preview": first_preview,
        }

    def batch_measure_prefill_time(
        self,
        prefill_len: int = 1024,
        bs: int = 1,
        n_iter: int = 3,
    ) -> dict[str, float]:
        """Same as DraftEngine.batch_measure_prefill_time — fresh-prefill
        microbench. Used by bench_pipeline_overlap for the target side
        of prefill-vs-decode MPS overlap testing.
        """
        import random
        import time

        def build_requests(iter_seed: int) -> list[dict]:
            rng = random.Random(iter_seed * 1_000_003 + 42)
            reqs = []
            for b in range(bs):
                ids = [rng.randint(100, 100_000) for _ in range(prefill_len)]
                reqs.append({"prompt_token_ids": ids})
            return reqs

        sp = SamplingParams(max_tokens=1, temperature=0.0)
        self._llm.generate(build_requests(0), sp, use_tqdm=False)

        total_ms = 0.0
        for i in range(n_iter):
            requests = build_requests(i + 1)
            t0 = time.perf_counter()
            self._llm.generate(requests, sp, use_tqdm=False)
            total_ms += (time.perf_counter() - t0) * 1000.0

        mean_ms = total_ms / n_iter
        return {
            "wall_ms_per_batch": mean_ms,
            "bs": float(bs),
            "prefill_len": float(prefill_len),
        }

    def batch_measure_request_time(
        self,
        prefill_len: int = 256,
        n_decode_tokens: int = 128,
        bs: int = 1,
        n_iter: int = 3,
    ) -> dict[str, float]:
        """Measure end-to-end (fresh prefill + n_decode_tokens decode) wall
        time per iter. Each iter uses random-token prompts so prefix
        caching cannot short-circuit prefill — this is the REAL production
        workload (TTFT + TPOT), not a cached-decode microbench.

        Used by bench_pipeline_overlap's "main process" side: the target
        doing actual user-request serving (prefill the pre-pruned prompt,
        then decode the response). Does NOT include vision encoder work —
        that's the concurrent aux process's job in the paper's proposed
        architecture.

        Returns:
          wall_ms_per_batch: total wall (prefill + all decodes) per batch
          aggregate_tok_per_s: bs * n_decode_tokens * 1000 / wall
          wall_per_request_ms: wall_ms / bs
        """
        import random
        import time

        def build_requests(iter_seed: int) -> list[dict]:
            rng = random.Random(iter_seed * 2_000_029 + 17)
            reqs = []
            for b in range(bs):
                ids = [rng.randint(100, 100_000) for _ in range(prefill_len)]
                reqs.append({"prompt_token_ids": ids})
            return reqs

        sp = SamplingParams(max_tokens=n_decode_tokens, temperature=0.0)
        self._llm.generate(build_requests(0), sp, use_tqdm=False)

        total_ms = 0.0
        total_tokens = 0
        for i in range(n_iter):
            requests = build_requests(i + 1)
            t0 = time.perf_counter()
            outputs = self._llm.generate(requests, sp, use_tqdm=False)
            total_ms += (time.perf_counter() - t0) * 1000.0
            total_tokens += sum(len(o.outputs[0].token_ids) for o in outputs)

        mean_wall_ms = total_ms / n_iter
        mean_tokens = total_tokens / n_iter
        return {
            "wall_ms_per_batch": mean_wall_ms,
            "bs": float(bs),
            "prefill_len": float(prefill_len),
            "n_decode_tokens": float(n_decode_tokens),
            "aggregate_tok_per_s": (
                mean_tokens * 1000.0 / mean_wall_ms if mean_wall_ms > 0 else 0.0
            ),
            "wall_per_request_ms": mean_wall_ms / bs if bs > 0 else 0.0,
        }

    def batch_measure_forward_time_with_prefix(
        self,
        prefix_len: int = 256,
        n_tokens: int = 64,
        bs: int = 1,
        n_iter: int = 3,
    ) -> dict[str, float]:
        """Same as DraftEngine.batch_measure_forward_time_with_prefix — used
        for target-only decode baselines at matched prefix.
        """
        import time

        tokenizer = self._llm.get_tokenizer()
        base = tokenizer.encode("Hello. ")
        while len(base) < prefix_len:
            base = base + base
        prefix_ids = base[:prefix_len]

        requests = []
        for b in range(bs):
            p = list(prefix_ids)
            p[-1] = (p[-1] + b * 13) % 150000
            requests.append({"prompt_token_ids": p})

        sp = SamplingParams(max_tokens=n_tokens, temperature=0.0)
        self._llm.generate(requests, sp, use_tqdm=False)

        total_ms = 0.0
        total_tokens = 0
        for _ in range(n_iter):
            t0 = time.perf_counter()
            outputs = self._llm.generate(requests, sp, use_tqdm=False)
            total_ms += (time.perf_counter() - t0) * 1000.0
            total_tokens += sum(len(o.outputs[0].token_ids) for o in outputs)

        mean_wall_ms = total_ms / n_iter
        mean_tokens = total_tokens / n_iter
        per_forward_ms = mean_wall_ms / n_tokens if n_tokens > 0 else 0.0
        return {
            "wall_ms": mean_wall_ms,
            "total_tokens": float(mean_tokens),
            "bs": float(bs),
            "prefix_len": float(prefix_len),
            "n_tokens": float(n_tokens),
            "per_forward_ms": per_forward_ms,
            "aggregate_tok_per_s": (
                mean_tokens * 1000.0 / mean_wall_ms if mean_wall_ms > 0 else 0.0
            ),
        }

    def batch_verify_with_results(
        self,
        prefix_token_ids_list: list[list[int]],
        draft_token_ids_list: list[list[int]],
    ) -> list[dict[str, Any]]:
        """Batched verify: bs prompts, each appended with its own draft tokens.
        Returns per-request top-1 IDs at each draft position plus the bonus token.

        Used by Y1 (whole-sequence spec decode bench). Lets the bench measure
        the bs-batched verify wall in a single vLLM call (which is what
        production spec decode would do) rather than bs sequential .verify()
        calls.

        Returns: list of dicts, each {"draft_top1_ids": [...], "bonus_token_id": int}.
        """
        assert len(prefix_token_ids_list) == len(draft_token_ids_list)

        sp = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=1,
        )

        requests = [
            {"prompt_token_ids": list(prefix) + list(drafts)}
            for prefix, drafts in zip(prefix_token_ids_list, draft_token_ids_list)
        ]
        outputs = self._llm.generate(requests, sp, use_tqdm=False)

        results: list[dict[str, Any]] = []
        for ro, drafts in zip(outputs, draft_token_ids_list):
            K = len(drafts)
            tail = ro.prompt_logprobs[-K:] if K > 0 else []
            top1_ids: list[int | None] = []
            for entry in tail:
                if entry is None:
                    top1_ids.append(None)
                    continue
                best_id = max(entry.items(), key=lambda kv: kv[1].logprob)[0]
                top1_ids.append(best_id)
            bonus = ro.outputs[0].token_ids[0] if ro.outputs[0].token_ids else None
            results.append({
                "draft_top1_ids": top1_ids,
                "bonus_token_id": bonus,
            })
        return results

    def batch_generate_continuation(
        self,
        prompt_token_ids_list: list[list[int]],
        max_tokens: int,
        ignore_eos: bool = True,
    ) -> list[list[int]]:
        """Batched generation given token-ID prefixes (bs requests).

        Used by Y1 to time both Path B (full-N decode) and Path A's resume
        phase (decode the suffix from break point). Returns per-request
        generated token IDs. With ignore_eos=True, every request emits
        exactly max_tokens tokens, so per-request walls are comparable.
        """
        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            ignore_eos=ignore_eos,
        )
        requests = [{"prompt_token_ids": list(ids)} for ids in prompt_token_ids_list]
        outputs = self._llm.generate(requests, sp, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outputs]

    def tokenize(self, text: str) -> list[int]:
        """Use target's tokenizer (same for the Qwen2.5-VL family)."""
        return self._llm.get_tokenizer().encode(text)

    def detokenize(self, token_ids: list[int]) -> str:
        return self._llm.get_tokenizer().decode(token_ids)

    def eos_token_id(self) -> int:
        return self._llm.get_tokenizer().eos_token_id

    def generate_text(
        self,
        user_prompts: list[str],
        max_tokens: int = 128,
        temperature: float = 0.0,
        ignore_eos: bool = False,
    ) -> list[str]:
        """Generate text responses to a batch of user prompts. Applies
        the model's chat template (Qwen2.5 instruct format). Returns
        decoded text only. For Y2 judge-verify the prompts already
        carry the verify template; the chat template wraps that as
        a user turn so the target replies in assistant-format.

        ignore_eos=True forces decoding all max_tokens regardless of
        natural EOS — used by long-N benches to isolate the decode
        wall as a function of N when the prompt mix would normally
        EOS-terminate well before max_tokens.
        """
        sp = SamplingParams(
            max_tokens=max_tokens, temperature=temperature,
            ignore_eos=ignore_eos,
        )
        tokenizer = self._llm.get_tokenizer()
        formatted = [
            tokenizer.apply_chat_template(
                _to_messages(p),
                add_generation_prompt=True,
                tokenize=False,
            )
            for p in user_prompts
        ]
        outputs = self._llm.generate(formatted, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def generate_text_detailed(
        self,
        user_prompts: list,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> list[dict]:
        """Same as generate_text but returns text + n_output_tokens per
        prompt. Each entry of `user_prompts` may be a str (single-turn)
        or a list of message dicts (multi-turn)."""
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        tokenizer = self._llm.get_tokenizer()
        formatted = [
            tokenizer.apply_chat_template(
                _to_messages(p),
                add_generation_prompt=True,
                tokenize=False,
            )
            for p in user_prompts
        ]
        outputs = self._llm.generate(formatted, sp, use_tqdm=False)
        return [
            {"text": o.outputs[0].text,
             "n_output_tokens": len(o.outputs[0].token_ids)}
            for o in outputs
        ]

    def batch_judge_verify_binary(
        self,
        query_prompts: list[str],
        draft_responses: list[str],
        max_judge_tokens: int = 16,
        verify_template: str | None = None,
        bit_mode: bool = False,
    ) -> list[dict]:
        """V0 judge-verify primitive. Three-turn chat:
          [user: query]
          [assistant: draft_response]
          [user: VERIFY_QUESTION_TEMPLATE_BINARY]
          [assistant: ACCEPT|REJECT]

        Used by V0: ACCEPT → ship draft; REJECT → target.generate_text
        on the clean prompt. No TRUNCATE-continue, no substring matching.

        max_judge_tokens=16 is enough for the one-or-two-token verdict.
        Parse is trivial: search for ACCEPT/REJECT; default to PARSE_ERROR
        on ambiguity (caller treats as REJECT — pay the regen tax rather
        than ship a wrong answer).

        verify_template: optional override of the third-turn user text.
        Defaults to VERIFY_QUESTION_TEMPLATE_BINARY. Pair with the
        DIGIT/COMPACT variants from probe_judge_verify to A/B Lever 1/2.

        bit_mode: when True, force decode to a single token in {"1","0"}
        via SamplingParams.allowed_token_ids and max_tokens=1. Verify
        becomes "prefill + exactly 1 step" regardless of how the
        tokenizer chunks the word verdicts. Pair with
        VERIFY_QUESTION_TEMPLATE_DIGIT (or any prompt asking for 1/0)
        so the response framing matches the constraint.

        Returns:
          list[dict]: [{verdict: "ACCEPT" | "REJECT" | "PARSE_ERROR",
                        raw: str (truncated to 200 chars)}]
        """
        from prorouter.probe_judge_verify import VERIFY_QUESTION_TEMPLATE_BINARY

        template = verify_template or VERIFY_QUESTION_TEMPLATE_BINARY
        tokenizer = self._llm.get_tokenizer()
        if bit_mode:
            id_one, id_zero = _resolve_bit_token_ids(tokenizer)
            sp = SamplingParams(
                max_tokens=1, temperature=0.0,
                allowed_token_ids=[id_one, id_zero],
            )
        else:
            sp = SamplingParams(max_tokens=max_judge_tokens, temperature=0.0)
        formatted = []
        for q, r in zip(query_prompts, draft_responses):
            # query_prompt may be a string (single-turn) or a list of
            # message dicts (multi-turn — e.g., MT-Bench T2). Either
            # way, append (assistant: draft, user: verify-question).
            messages = _to_messages(q) + [
                {"role": "assistant", "content": r},
                {"role": "user", "content": template},
            ]
            formatted.append(tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            ))
        outputs = self._llm.generate(formatted, sp, use_tqdm=False)

        results: list[dict] = []
        for o in outputs:
            text = o.outputs[0].text.strip()
            if bit_mode:
                results.append({
                    "verdict": _parse_bit_verdict(text),
                    "raw": text[:200],
                })
                continue
            # Trivial parse. Recognize UNSURE (3-way verifier output)
            # before falling through to ACCEPT/REJECT detection, since
            # an UNSURE response is independently meaningful and should
            # not be confused with PARSE_ERROR.
            upper = text.upper()
            if "UNSURE" in upper and "ACCEPT" not in upper and "REJECT" not in upper:
                verdict = "UNSURE"
            elif "ACCEPT" in upper and "REJECT" not in upper:
                verdict = "ACCEPT"
            elif "REJECT" in upper and "ACCEPT" not in upper:
                verdict = "REJECT"
            elif "ACCEPT" in upper:
                # Both present — pick whichever appears first (the
                # token-by-token decode emits the verdict first).
                verdict = "ACCEPT" if upper.find("ACCEPT") < upper.find("REJECT") else "REJECT"
            else:
                verdict = "PARSE_ERROR"
            results.append({"verdict": verdict, "raw": text[:200]})
        return results

    def generate_text_mm(
        self,
        records: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> list[str]:
        """Multimodal target.generate_text. Records:
          {"prompt": str, "images": list[str] | None}
        Used by VLM benchmarks for the target_only baseline and the
        REJECT-path regen on multimodal prompts."""
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        tokenizer = self._llm.get_tokenizer()
        requests = []
        for r in records:
            text, mm = _build_mm_request(
                tokenizer, r["prompt"], r.get("images"),
            )
            req = {"prompt": text}
            if mm is not None:
                req["multi_modal_data"] = mm
            requests.append(req)
        outputs = self._llm.generate(requests, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def batch_judge_verify_binary_mm(
        self,
        records: list[dict],
        draft_responses: list[str],
        max_judge_tokens: int = 16,
        verify_format: str = "accept_reject",
        verify_template: str | None = None,
        bit_mode: bool = False,
    ) -> list[dict]:
        """Multimodal V0 binary judge. Each record needs:
          {"prompt": str, "images": list[str] | None}
        plus the parallel draft_responses list. Builds the 3-turn
        chat (user(image+question), assistant(draft), user(verify-Q))
        with images attached only to the first user turn — the verify
        question is text-only.

        Two ways to control the verify call (verify_template + bit_mode
        wins when explicitly set; verify_format is the legacy keyword
        kept for run_mmmu_v0.py callers):

        verify_format (legacy):
          "accept_reject" (default) — ACCEPT/REJECT word template
          "digit"                   — 1/0 word template, soft parse
                                       (cheaper decode without
                                       allowed_token_ids constraint)

        verify_template (str): override the verify question. Pair with
        the COMPACT or DIGIT templates from probe_judge_verify to
        ablate Lever 1 / Lever 2 cleanly.

        bit_mode (bool): force decode to a single token in {"1","0"}
        via SamplingParams.allowed_token_ids and max_tokens=1. Verify
        becomes "prefill + exactly 1 step." Pair with DIGIT (or any
        prompt asking for 1/0).

        Returns:
          [{"verdict": "ACCEPT"|"REJECT"|"PARSE_ERROR", "raw": str}]
        """
        from prorouter.probe_judge_verify import (
            VERIFY_QUESTION_TEMPLATE_BINARY,
            VERIFY_QUESTION_TEMPLATE_DIGIT,
        )

        # Resolve the template + decode params:
        # 1. If verify_template explicitly set, that wins (new API).
        # 2. Else fall back to legacy verify_format mapping.
        if verify_template is not None:
            template = verify_template
            digit_legacy = False
        elif verify_format == "digit":
            template = VERIFY_QUESTION_TEMPLATE_DIGIT
            digit_legacy = True
        else:
            template = VERIFY_QUESTION_TEMPLATE_BINARY
            digit_legacy = False

        tokenizer = self._llm.get_tokenizer()
        if bit_mode:
            # Hard constraint: exactly 1 decode step yielding "1" or "0".
            id_one, id_zero = _resolve_bit_token_ids(tokenizer)
            sp = SamplingParams(
                max_tokens=1, temperature=0.0,
                allowed_token_ids=[id_one, id_zero],
            )
        elif digit_legacy:
            # Cluster's legacy soft-digit path: short cap, soft parse.
            sp = SamplingParams(max_tokens=min(max_judge_tokens, 4),
                                temperature=0.0)
        else:
            sp = SamplingParams(max_tokens=max_judge_tokens, temperature=0.0)
        requests = []
        for r, draft in zip(records, draft_responses):
            # Pre-build messages: user(image+question), assistant(draft).
            # The verify question is appended as the final user turn
            # (without images — verify Q is text-only).
            from PIL import Image
            images = []
            if r.get("images"):
                for p in r["images"]:
                    images.append(Image.open(p).convert("RGB"))
            user1_content = (
                [{"type": "image"} for _ in images] + [
                    {"type": "text", "text": r["prompt"]}
                ]
                if images else r["prompt"]
            )
            messages = [
                {"role": "user", "content": user1_content},
                {"role": "assistant", "content": draft},
                {"role": "user", "content": template},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            req = {"prompt": formatted}
            if images:
                req["multi_modal_data"] = {"image": images}
            requests.append(req)
        outputs = self._llm.generate(requests, sp, use_tqdm=False)

        results = []
        for o in outputs:
            text = o.outputs[0].text.strip()
            if bit_mode:
                results.append({
                    "verdict": _parse_bit_verdict(text),
                    "raw": text[:200],
                })
                continue
            if digit_legacy:
                # Cluster's soft digit parser (max_tokens up to 4):
                # allow leading whitespace + bare "1"/"0".
                t = text.strip()
                if t.startswith("1") or " 1 " in f" {t} " or t == "1":
                    verdict = "ACCEPT"
                elif t.startswith("0") or " 0 " in f" {t} " or t == "0":
                    verdict = "REJECT"
                else:
                    found = None
                    for ch in t:
                        if ch in "01":
                            found = ch
                            break
                    if found == "1":
                        verdict = "ACCEPT"
                    elif found == "0":
                        verdict = "REJECT"
                    else:
                        verdict = "PARSE_ERROR"
                results.append({"verdict": verdict, "raw": text[:200]})
                continue
            upper = text.upper()
            if "UNSURE" in upper and "ACCEPT" not in upper and "REJECT" not in upper:
                verdict = "UNSURE"
            elif "ACCEPT" in upper and "REJECT" not in upper:
                verdict = "ACCEPT"
            elif "REJECT" in upper and "ACCEPT" not in upper:
                verdict = "REJECT"
            elif "ACCEPT" in upper:
                verdict = ("ACCEPT" if upper.find("ACCEPT") <
                           upper.find("REJECT") else "REJECT")
            else:
                verdict = "PARSE_ERROR"
            results.append({"verdict": verdict, "raw": text[:200]})
        return results

    def batch_judge_verify_merged(
        self,
        query_prompts: list,
        draft_responses: list[str],
        max_regen_tokens: int = 1024,
    ) -> list[dict]:
        """Merged verify+regen primitive (text-only). One target call:
        the model either replies with 'YES' (ACCEPT, draft is shipped)
        or writes the corrected response directly (REGEN, output IS the
        V0 response).

        Returns:
          [{"verdict": "ACCEPT"|"REGEN", "text": str, "raw": str}, ...]
        """
        from prorouter.probe_judge_verify import (
            VERIFY_QUESTION_TEMPLATE_MERGED, parse_merged_verdict,
        )

        sp = SamplingParams(max_tokens=max_regen_tokens, temperature=0.0)
        tokenizer = self._llm.get_tokenizer()
        formatted = []
        for q, r in zip(query_prompts, draft_responses):
            messages = _to_messages(q) + [
                {"role": "assistant", "content": r},
                {"role": "user", "content": VERIFY_QUESTION_TEMPLATE_MERGED},
            ]
            formatted.append(tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            ))
        outputs = self._llm.generate(formatted, sp, use_tqdm=False)

        results: list[dict] = []
        for o, draft in zip(outputs, draft_responses):
            text = o.outputs[0].text
            verdict, regen_text = parse_merged_verdict(text)
            v0_text = draft if verdict == "ACCEPT" else regen_text
            results.append({
                "verdict": verdict,
                "text": v0_text,
                "raw": text[:1000],  # keep the raw output for inspection
            })
        return results

    def batch_judge_verify_merged_mm(
        self,
        records: list[dict],
        draft_responses: list[str],
        max_regen_tokens: int = 1024,
    ) -> list[dict]:
        """Multimodal merged verify+regen. Each record:
          {"prompt": str, "images": list[str] | None}
        Same return shape as batch_judge_verify_merged.

        Verify question is text-only (images attached only to the
        first user turn).
        """
        from prorouter.probe_judge_verify import (
            VERIFY_QUESTION_TEMPLATE_MERGED, parse_merged_verdict,
        )

        sp = SamplingParams(max_tokens=max_regen_tokens, temperature=0.0)
        tokenizer = self._llm.get_tokenizer()
        requests = []
        for r, draft in zip(records, draft_responses):
            from PIL import Image
            images = []
            if r.get("images"):
                for p in r["images"]:
                    images.append(Image.open(p).convert("RGB"))
            user1_content = (
                [{"type": "image"} for _ in images] + [
                    {"type": "text", "text": r["prompt"]}
                ]
                if images else r["prompt"]
            )
            messages = [
                {"role": "user", "content": user1_content},
                {"role": "assistant", "content": draft},
                {"role": "user", "content": VERIFY_QUESTION_TEMPLATE_MERGED},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            req = {"prompt": formatted}
            if images:
                req["multi_modal_data"] = {"image": images}
            requests.append(req)
        outputs = self._llm.generate(requests, sp, use_tqdm=False)

        results: list[dict] = []
        for o, draft in zip(outputs, draft_responses):
            text = o.outputs[0].text
            verdict, regen_text = parse_merged_verdict(text)
            v0_text = draft if verdict == "ACCEPT" else regen_text
            results.append({
                "verdict": verdict,
                "text": v0_text,
                "raw": text[:1000],
            })
        return results

    # ------------------------------------------------------------------
    # vision-offload consumers — use pre-encoded image_embeds from
    # the draft actor's TargetVisionEncoder. Skip target's own ViT
    # forward (`multi_modal_data={"image_embeds": ...}` public API path,
    # confirmed in vllm_fork_action_items.md §2).
    # ------------------------------------------------------------------

    def generate_text_mm_with_embeds(
        self,
        records: list[dict],
        vision_payloads: list[dict | None],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> list[str]:
        """variant of generate_text_mm. Each record:
          {"prompt": str, "images": list[str] | None}
        and the parallel `vision_payloads[i]` is one of:
          - None (no images for this record)
          - {"image_embeds": Tensor[N_image_tokens, llm_hidden_dim],
             "image_grid_thw": Tensor[n_images, 3]}
            with tensors on CPU as serialized through Ray.

        Builds the chat-templated prompt using `{"type": "image"}`
        placeholders matching the count from records[i]["images"], then
        passes `image_embeds`+`image_grid_thw` via multi_modal_data so
        vLLM skips its ViT forward and consumes the pre-encoded tokens
        directly."""
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        tokenizer = self._llm.get_tokenizer()
        requests = []
        for r, payload in zip(records, vision_payloads):
            n_images = len(r.get("images") or [])
            text = _build_mm_prompt_from_count(
                tokenizer, r["prompt"], n_images,
            )
            req = {"prompt": text}
            if n_images and payload is not None:
                req["multi_modal_data"] = {
                    "image": {
                        "image_embeds": payload["image_embeds"],
                        "image_grid_thw": payload["image_grid_thw"],
                    }
                }
            requests.append(req)
        outputs = self._llm.generate(requests, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def batch_judge_verify_binary_mm_with_embeds(
        self,
        records: list[dict],
        draft_responses: list[str],
        vision_payloads: list[dict | None],
        max_judge_tokens: int = 16,
        verify_format: str = "accept_reject",
        verify_template: str | None = None,
        bit_mode: bool = False,
    ) -> list[dict]:
        """variant of batch_judge_verify_binary_mm. Same 3-turn
        chat (user(image+question), assistant(draft), user(verify-Q))
        but consumes pre-encoded `image_embeds` instead of raw images.

        See generate_text_mm_with_embeds for the payload contract.
        """
        from prorouter.probe_judge_verify import (
            VERIFY_QUESTION_TEMPLATE_BINARY,
            VERIFY_QUESTION_TEMPLATE_DIGIT,
        )

        if verify_template is not None:
            template = verify_template
            digit_legacy = False
        elif verify_format == "digit":
            template = VERIFY_QUESTION_TEMPLATE_DIGIT
            digit_legacy = True
        else:
            template = VERIFY_QUESTION_TEMPLATE_BINARY
            digit_legacy = False

        tokenizer = self._llm.get_tokenizer()
        if bit_mode:
            id_one, id_zero = _resolve_bit_token_ids(tokenizer)
            sp = SamplingParams(
                max_tokens=1, temperature=0.0,
                allowed_token_ids=[id_one, id_zero],
            )
        elif digit_legacy:
            sp = SamplingParams(
                max_tokens=min(max_judge_tokens, 4), temperature=0.0,
            )
        else:
            sp = SamplingParams(max_tokens=max_judge_tokens, temperature=0.0)

        requests = []
        for r, draft, payload in zip(records, draft_responses, vision_payloads):
            n_images = len(r.get("images") or [])
            user1_content = (
                [{"type": "image"} for _ in range(n_images)] + [
                    {"type": "text", "text": r["prompt"]}
                ]
                if n_images else r["prompt"]
            )
            messages = [
                {"role": "user", "content": user1_content},
                {"role": "assistant", "content": draft},
                {"role": "user", "content": template},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            req = {"prompt": formatted}
            if n_images and payload is not None:
                req["multi_modal_data"] = {
                    "image": {
                        "image_embeds": payload["image_embeds"],
                        "image_grid_thw": payload["image_grid_thw"],
                    }
                }
            requests.append(req)
        outputs = self._llm.generate(requests, sp, use_tqdm=False)

        results: list[dict] = []
        for o in outputs:
            text = o.outputs[0].text.strip()
            if bit_mode:
                results.append({
                    "verdict": _parse_bit_verdict(text),
                    "raw": text[:200],
                })
                continue
            if digit_legacy:
                t = text.strip()
                if t.startswith("1") or " 1 " in f" {t} " or t == "1":
                    verdict = "ACCEPT"
                elif t.startswith("0") or " 0 " in f" {t} " or t == "0":
                    verdict = "REJECT"
                else:
                    found = None
                    for ch in t:
                        if ch in "01":
                            found = ch
                            break
                    if found == "1":
                        verdict = "ACCEPT"
                    elif found == "0":
                        verdict = "REJECT"
                    else:
                        verdict = "PARSE_ERROR"
                results.append({"verdict": verdict, "raw": text[:200]})
                continue
            upper = text.upper()
            if "UNSURE" in upper and "ACCEPT" not in upper and "REJECT" not in upper:
                verdict = "UNSURE"
            elif "ACCEPT" in upper and "REJECT" not in upper:
                verdict = "ACCEPT"
            elif "REJECT" in upper and "ACCEPT" not in upper:
                verdict = "REJECT"
            elif "ACCEPT" in upper:
                verdict = (
                    "ACCEPT" if upper.find("ACCEPT") < upper.find("REJECT")
                    else "REJECT"
                )
            else:
                verdict = "PARSE_ERROR"
            results.append({"verdict": verdict, "raw": text[:200]})
        return results

    def ping(self) -> str:
        return "ok"


# ---------------------------------------------------------------------------
# V0 async actors (PR 1+2) — vLLM AsyncLLMEngine + per-actor request queues.
#
# Pattern (head scheduler):
#   actor.submit(req_id, prompt, ...)            → returns immediately
#   actor.pop_finished(max_n, timeout_s)         → drains finished_q
#
# The actor's background tasks drive vLLM's continuous batching: each
# submit() spawns a coroutine that calls AsyncLLMEngine.generate() and
# pushes the final RequestOutput to finished_q on completion.
#
# max_concurrency=16 lets the actor accept ≥1 submit + ≥1 pop_finished
# concurrently while background drive tasks run on the same event loop.
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1, max_concurrency=16)
class DraftEngineAsync:
    """V0 draft actor — wraps vLLM's AsyncLLMEngine.

    Public methods (all async):
      - submit(req_id, prompt, max_tokens)  → fire-and-forget; result
                                              lands in finished_q
      - pop_finished(max_n, timeout_s)      → drain finished_q
      - qsize()                              → introspection (for tests/metrics)
      - ping()                               → readiness probe

    No backpressure: submit always queues. KV pool on the draft GPU is
    plentiful at TP=4 7B (vs target's 72B), so we don't admission-gate
    at the draft tier.
    """

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        limit_mm_per_prompt: dict[str, int] | None = None,
        distributed_executor_backend: str | None = None,
        # --- head-cascade config (fork: cascade-prod-fixes) -------------
        head_cascade: bool = False,
        head_checkpoint_path: str | None = None,
        head_tau_table_path: str | None = None,
        extract_hidden_states_layer: int | None = None,  # N-from-end (=7 for idx 20 on 28-layer 7B-VL)
        head_cascade_log_scores: bool = False,
        # Cascade ran under enforce_eager=True + async_scheduling=False
        # historically because the fork couldn't graph-capture the
        # extract write and async mode left valid_sampled_token_ids empty.
        # Both were claimed fixed (cudagraph: 2026-05-13 incidental, async:
        # Phase B `_bookkeeping_sync` materialize-for-cascade), but
        # (2026-05-21) saw a ship_rate collapse 64.4% → 18.7% on
        # long-CoT under graph_async — defaults were reverted to legacy
        # (eager, sync) pending re-test on the head + τ.
        # 2× g5.12 system test (2026-05-28) flips the defaults back
        # to graph_async to re-validate on the mixed workload
        # (`bench_vlm_test_repath.json`). If ship rate holds at ≈0.605 on
        # that cell, the long-CoT regression was workload-specific and
        # graph_async
        # is the right default; if it collapses again, revert this commit.
        # `run_throughput_bench.py --draft-engine-mode` still accepts
        # the four-mode opt-in.
        cascade_enforce_eager: bool = False,
        cascade_async_scheduling: bool = True,
        enable_mm_embeds: bool = False,
        enable_prefix_caching: bool = True,
        # VLLM defaults are max_num_batched_tokens=2048
        # (one prefill chunk + few decodes per step) and max_num_seqs=256.
        # C3b showed throughput saturates at concurrency 128 — 256 adds
        # latency without lifting r/s, so per-step compute on the 2048
        # budget is the bottleneck. Bumping max_num_batched_tokens lets
        # each step pack more prefill alongside decode, reducing the
        # prefill-vs-decode interference visible in that cell's drain (46 r/s)
        # vs C3b closed-loop (35.6 r/s) gap.
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        # logit-features path — when "raw_logits", vLLM returns
        # raw logits (no log_softmax) in completion.logprobs; _drive
        # produces logit-gap features [t1-t2, t1-t5, t1-t20, pos_frac]
        # instead of the softmax features [chosen_lp, max_p,
        # neg_entropy, pos_frac]. Pair with a gate trained on the
        # matching schema (see logit-feature retrofit).
        logprobs_mode: str | None = None,
        # Paths to the in-engine attn_pool head ckpt +
        # tau table. When both are set, vLLM loads the head at engine
        # init (via env vars) and per-request opt-in via
        # SamplingParams.in_engine_cascade_head=True yields
        # CompletionOutput.head_decision. Brings the gate inside the
        # engine — one engine.generate call returns text AND verdict.
        in_engine_cascade_head_ckpt: str | None = None,
        in_engine_cascade_head_tau: str | None = None,
        # Bound Qwen2.5-VL ViT tokens. Multi-image MileBench
        # records at full res blow up to tens of thousands of vision
        # tokens → the draft is pinned in image-prefill, not decode.
        # {"max_pixels": 1280*28*28} caps each image at ~1280 tokens.
        mm_processor_kwargs: dict | None = None,
        # inline self-eval baseline. None (default) → off,
        # Zero cost. "ptrue"/"automix" → after each draft generation, run the
        # self-eval pass(es) on THIS engine (blocking the request's completion)
        # and attach item["self_eval_score"]. Faithful serving-path overhead of
        # the post-hoc baselines. Score is logged; routing is unaffected
        # (the cell config still owns SHIP/REGEN).
        inline_self_eval: str | None = None,
        # network-latency resilience probe: sleep this long (one-way)
        # at serving-RPC entry AND before pop_finished returns, emulating a
        # slow scheduler↔actor link. asyncio.sleep → concurrent requests
        # overlap their delays (models wire latency, not serialization).
        rpc_fake_latency_ms: float = 0.0,
        # actor self-admit (default OFF → behavior unchanged). When on,
        # submit/submit_batch just buffer the request and return stats; a
        # background loop admits from the buffer into vLLM every
        # actor_admit_interval_ms, bounded by actor_admit_max_inflight (a
        # local KV-safe concurrency cap). Moves admission control off the
        # wire and into the actor — dispatch stops being gated by RTT-inflated
        # in-flight, and over-admission preemption is bounded.
        actor_self_admit: bool = False,
        actor_admit_interval_ms: float = 5.0,
        actor_admit_max_inflight: int = 256,
        # draft KV-token gate: reserve estimated prefill KV per admitted
        # request and hold admission at this token threshold (mirror of the
        # target's kv_pool_threshold). 0 = auto-derive 0.85×(actual KV pool)
        # when self-admit is on; <0 = disable (count cap only).
        actor_admit_kv_threshold: int = 0,
    ) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from transformers import AutoTokenizer

        _ensure_all_gpus_visible()
        self._logprobs_mode = logprobs_mode or "raw_logprobs"
        # set env vars so vLLM's OutputProcessor loads
        # The in-engine head at AsyncLLMEngine.from_engine_args below.
        # Use _os local alias to dodge the `import os` later in this
        # function (which would make `os` a local name and shadow the
        # module-level import here).
        import os as _os
        if in_engine_cascade_head_ckpt and in_engine_cascade_head_tau:
            _os.environ["VLLM_CASCADE_ATTN_POOL_CKPT"] = (
                in_engine_cascade_head_ckpt
            )
            _os.environ["VLLM_CASCADE_ATTN_POOL_TAU"] = (
                in_engine_cascade_head_tau
            )
        self._in_engine_cascade_head = bool(
            in_engine_cascade_head_ckpt and in_engine_cascade_head_tau
        )

        # Cascade env vars MUST be set before AsyncLLMEngine init —
        # the fork's gpu_model_runner reads them at engine bring-up to
        # wire the trained classifier head into Qwen2Model.forward.
        # for the deployable
        # head + τ JSON. NOTE: layer is N-from-end (=7 → idx 20 on
        # 28-layer Qwen2.5-VL-7B), NOT the absolute index.
        self._head_cascade_enabled = bool(head_cascade)
        import os
        if head_cascade:
            if head_checkpoint_path:
                os.environ["VLLM_HEAD_CHECKPOINT_PATH"] = head_checkpoint_path
            if head_tau_table_path:
                os.environ["VLLM_HEAD_TAU_TABLE_PATH"] = head_tau_table_path
            if head_cascade_log_scores:
                os.environ["VLLM_HEAD_CASCADE_LOG_SCORES"] = "1"
        # external head: allow extract_hidden_states_layer to
        # be set WITHOUT head_cascade=True. The engine still installs
        # the per-step persistent buffer + writes layer-N hidden states
        # at logits_indices positions, but the INLINE head firing path
        # is skipped (because VLLM_HEAD_CHECKPOINT_PATH is unset → the
        # fork's _load_cascade_head no-ops). The draft can then run
        # in graph_async with cudagraph capturing the buffer write
        # natively; decisions are made externally via CascadeHeadActor
        # consuming RequestOutput.hidden_states post-generation.
        if extract_hidden_states_layer is not None:
            os.environ["VLLM_EXTRACT_HIDDEN_STATES_LAYER"] = str(extract_hidden_states_layer)

        engine_args = AsyncEngineArgs(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            limit_mm_per_prompt=limit_mm_per_prompt or {"image": 1},
        )
        if mm_processor_kwargs:
            engine_args.mm_processor_kwargs = mm_processor_kwargs
        if enable_mm_embeds:
            engine_args.enable_mm_embeds = True
        if max_num_batched_tokens is not None:
            engine_args.max_num_batched_tokens = max_num_batched_tokens
        if max_num_seqs is not None:
            engine_args.max_num_seqs = max_num_seqs
        if logprobs_mode is not None:
            engine_args.logprobs_mode = logprobs_mode
        if head_cascade or extract_hidden_states_layer is not None:
            # defaults below preserve the legacy cell (eager+sync);
            # `--engine-mode graph_only / async_only / graph_async` in
            # sys4_quality_at_scale.py flips them via these kwargs after
            # the fork-side fixes (cudagraph: 2026-05-13 incidental,
            # async: Phase B `_bookkeeping_sync` materialize-for-
            # cascade).
            #
            # also honor these kwargs when extract_hidden_states_
            # Layer is set without head_cascade — the external-head
            # probe needs to choose engine mode explicitly (graph_async
            # is the win condition; eager_baseline is the apples-to-
            # apples reference).
            engine_args.enforce_eager = cascade_enforce_eager
            engine_args.async_scheduling = cascade_async_scheduling
        if distributed_executor_backend is not None:
            engine_args.distributed_executor_backend = (
                distributed_executor_backend
            )
        self._llm = AsyncLLMEngine.from_engine_args(engine_args)

        # Tokenizer for chat-template formatting. Loaded separately
        # because AsyncLLMEngine's tokenizer accessor is async-only in
        # some vLLM versions.
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Internal queues + task tracking. Ray sets up the actor's
        # event loop before __init__, so creating asyncio.Queue here
        # is safe.
        self._finished_q: asyncio.Queue = asyncio.Queue()
        # Keep references to in-flight drive tasks so they aren't GC'd
        # before completion. add_done_callback removes when done.
        self._tasks: set = set()

        # ----- closed-loop controller: piggybacked engine stats -----
        # pop_finished() stamps item["stats"] (load + a cached gpu_util +
        # ship_rate_ma) on every drained item so the scheduler's controller
        # gets them on the response path with NO extra RPC. ship_rate_ma is
        # an EMA over THIS draft's own head SHIP/REGEN verdicts — i.e.
        # ship_rate_per_cascade by construction, since the draft only ever
        # sees cascade-routed requests. All inert unless the scheduler's
        # controller reads item["stats"].
        # inline self-eval baseline config.
        if inline_self_eval not in (None, "ptrue", "automix"):
            raise ValueError(
                f"inline_self_eval={inline_self_eval!r}; expected one of "
                "None / 'ptrue' / 'automix'"
            )
        self._inline_self_eval = inline_self_eval
        self._rpc_lat_s = float(rpc_fake_latency_ms) / 1000.0
        self._self_eval_seq = 0
        if inline_self_eval:
            print(f"[DraftEngineAsync] inline_self_eval={inline_self_eval} "
                  f"(blocking self-eval pass(es) on the draft engine after "
                  f"each generation)", flush=True)
        # APC state must be verifiable from the boot log (probe
        # Records repeat across arms — later arms would read falsely fast
        # off a warm prefix cache).
        print(f"[DraftEngineAsync] enable_prefix_caching="
              f"{enable_prefix_caching}", flush=True)

        self._emit_stats = False  # off by default → zero cost on non-controller runs
        self._ship_ma: float | None = None
        self._ship_ma_alpha = 0.02
        self._gpu_util_cache: float | None = None
        self._gpu_util_cache_t = 0.0
        self._gpu_util_cache_ttl_s = 0.5

        # actor self-admit state (inert unless actor_self_admit).
        self._self_admit = bool(actor_self_admit)
        self._admit_interval_s = float(actor_admit_interval_ms) / 1000.0
        self._admit_max_inflight = int(actor_admit_max_inflight)
        self._pending: collections.deque = collections.deque()
        self._admit_task = None
        # draft KV-token gate (mirror of target's kv_pool_threshold).
        # Draft is prefill-dominated (head fires at ~1 output token), so the
        # per-request KV cost is the PREFILL token count (text + expanded
        # visual tokens) — which text-only estimation misses. We learn it from
        # completed requests' actual prompt_token_ids (p90), reserving
        # max_model_len during warmup. Reserved on admit, freed on finish.
        self._max_model_len = int(max_model_len)
        self._kv_in_flight = 0
        self._prefill_samples: collections.deque = collections.deque(maxlen=256)
        self._prefill_warmup = 16
        _pool = self._read_kv_pool_tokens()
        if actor_admit_kv_threshold > 0:
            self._kv_pool_threshold = int(actor_admit_kv_threshold)
        elif actor_admit_kv_threshold == 0 and self._self_admit and _pool:
            self._kv_pool_threshold = int(0.85 * _pool)   # auto-derive
        else:
            self._kv_pool_threshold = 0                    # off (count cap only)
        if self._self_admit:
            print(f"[DraftEngineAsync] actor_self_admit ON "
                  f"(interval={actor_admit_interval_ms}ms, "
                  f"max_inflight={self._admit_max_inflight}, "
                  f"kv_pool={_pool}, kv_threshold={self._kv_pool_threshold})",
                  flush=True)

    def _read_kv_pool_tokens(self):
        """Best-effort read of the actual GPU KV pool (tokens) for the
        auto-threshold. Returns None if the vLLM internals aren't reachable
        (then the KV gate stays off unless an explicit threshold is passed)."""
        for path in (("engine", "cache_config"),
                     ("engine", "engine", "cache_config"),
                     ("llm_engine", "cache_config")):
            obj = getattr(self, "_llm", None)
            try:
                for a in path:
                    obj = getattr(obj, a)
                nb = getattr(obj, "num_gpu_blocks", None)
                bs = getattr(obj, "block_size", None)
                if nb and bs:
                    return int(nb) * int(bs)
            except Exception:
                continue
        return None

    def _record_prefill(self, n: int) -> None:
        """remember an observed prefill length (prompt_token_ids) so the
        KV-cost estimate self-calibrates per family (llava anyres ≫ qwen)."""
        if n and n > 0:
            self._prefill_samples.append(int(n))

    def _draft_kv_cost(self) -> int:
        """per-request KV reservation. p90(observed prefill)+64 once
        warm; conservative max_model_len during warmup so we never over-admit
        before we've learned the family's real footprint."""
        if len(self._prefill_samples) >= self._prefill_warmup:
            s = sorted(self._prefill_samples)
            return s[int(len(s) * 0.9)] + 64
        return self._max_model_len

    # ---------------- actor self-admit (shared shape w/ target) --------
    # Both actors use the IDENTICAL _ensure_admit_loop + _admit_loop below and
    # differ only in two tiny hooks: _admit_ok() (may we admit one more?) and
    # _admit_entry(entry) (drive one buffered request into vLLM). Keep the two
    # implementations in sync so the design reads the same on both sides.
    def _ensure_admit_loop(self) -> None:
        """Lazily start the background admission loop (the event loop is
        running by the time the first RPC arrives)."""
        if self._admit_task is None or self._admit_task.done():
            self._admit_task = asyncio.create_task(self._admit_loop())

    async def _admit_loop(self) -> None:
        """Drain the pending buffer into vLLM while the actor has local
        headroom (_admit_ok). This moves admission control off the wire and
        into the actor, so dispatch never gates on RTT-inflated in-flight and
        over-admission is bounded. One loop per actor; sleeps between passes."""
        while True:
            while self._pending and self._admit_ok():
                await self._admit_entry(self._pending.popleft())
            await asyncio.sleep(self._admit_interval_s)

    def _admit_ok(self) -> bool:
        """Draft: bound concurrent in-flight by BOTH a concurrency cap and a
        KV-token budget (symmetric with the target). len(_tasks) is real
        admitted load; _kv_in_flight is the reserved prefill KV. The KV gate
        is what stops llava's heavy anyres prefills from over-admitting where
        a count-only cap would."""
        if len(self._tasks) >= self._admit_max_inflight:
            return False
        if self._kv_pool_threshold > 0:
            return self._kv_in_flight + self._draft_kv_cost() <= self._kv_pool_threshold
        return True

    async def _admit_entry(self, entry) -> None:
        """Draft: an entry is (args, kwargs) for _submit_impl (one request).
        Reserve KV cost on admit; _drive frees it (and records the real
        prefill length) on finish."""
        args, kwargs = entry
        cost = self._draft_kv_cost() if self._kv_pool_threshold > 0 else 0
        self._kv_in_flight += cost
        try:
            await self._submit_impl(*args, kv_cost=cost, **kwargs)
        except Exception as e:
            self._kv_in_flight = max(0, self._kv_in_flight - cost)
            await self._finished_q.put({
                "req_id": kwargs.get("req_id"),
                "error": f"{type(e).__name__}: {e}",
                "completed_t": time.perf_counter(),
            })

    async def submit(self, *args, **kwargs) -> None:
        """Single-request RPC shim over `_submit_impl` (which documents
        the arguments). Pays the fake wire latency once per request —
        the pre-behavior every existing caller relies on.

        self-admit: buffer the request and return stats immediately;
        the background loop feeds vLLM. Otherwise (default) admit inline."""
        if self._rpc_lat_s:
            await asyncio.sleep(self._rpc_lat_s)  # fake wire latency
        if self._self_admit:
            self._pending.append((args, kwargs))
            self._ensure_admit_loop()
            return self._current_stats()
        await self._submit_impl(*args, **kwargs)

    async def submit_batch(self, items: list[dict]) -> None:
        """wire-latency fix: N submits in ONE RPC, so a wire round
        trip amortizes over the whole batch instead of gating dispatch at
        1/RTT per request (the target path has had this shape since
's `submit_decode_batch`; the draft path was still
        per-request).

        Each `items[i]` is a kwargs dict for `_submit_impl`. A failing
        item must not sink its batchmates, so per-item errors are pushed
        to finished_q in the same shape `_drive` uses — the scheduler's
        draft pump already routes that to `_respond_error`."""
        if self._rpc_lat_s:
            await asyncio.sleep(self._rpc_lat_s)  # fake wire latency
        if self._self_admit:
            for it in items:
                self._pending.append(((), it))
            self._ensure_admit_loop()
            return self._current_stats()
        for it in items:
            try:
                await self._submit_impl(**it)
            except Exception as e:
                await self._finished_q.put({
                    "req_id": it.get("req_id"),
                    "error": f"{type(e).__name__}: {e}",
                    "completed_t": time.perf_counter(),
                })

    def _build_prompt_payload(self, prompt, image_paths, image_embeds,
                              image_grid_thw):
        """the CPU/IO-heavy prompt+image preprocessing, factored out so
        `_submit_impl` can run it in a worker thread (asyncio.to_thread) instead
        of blocking the actor's event loop. Returns (formatted, mm_data,
        _se_images). Pure function of its args + read-only actor state
        (tokenizer, self-eval flag) — safe to run off-loop."""
        _se_images = None
        if image_embeds is not None and image_grid_thw is not None:
            # Pre-encoded path — skip draft ViT, hand vllm the embeds
            # directly. image_grid_thw shape [N, 3] gives the image count.
            n_images = (
                int(image_grid_thw.shape[0])
                if hasattr(image_grid_thw, "shape") else 1
            )
            formatted = _build_mm_prompt_from_count(
                self._tokenizer,
                prompt_text=prompt if isinstance(prompt, str)
                else prompt[-1]["content"],
                image_count=n_images,
            )
            mm_data = {
                "image": {
                    "image_embeds": image_embeds,
                    "image_grid_thw": image_grid_thw,
                }
            }
        elif image_paths:
            # Multi-modal: chat template needs N image placeholders.
            formatted = _build_mm_prompt_from_count(
                self._tokenizer,
                prompt_text=prompt if isinstance(prompt, str) else prompt[-1]["content"],
                image_count=len(image_paths),
            )
            from PIL import Image
            images = [Image.open(p).convert("RGB") for p in image_paths]
            mm_data = {"image": images if len(images) > 1 else images[0]}
            if self._inline_self_eval:
                _se_images = images
        else:
            # render via the model PROCESSOR's template — the same
            # cross-family fix applied to the image branches above.
            # The llava-ov TOKENIZER template raises on list-form text
            # content ('can only concatenate str (not "list") to str') and
            # pixtral's tokenizer ships no chat template at all; the
            # processor template renders list-form for all three families
            # (Qwen output verified byte-identical to the tokenizer's).
            formatted = _get_chat_templater(self._tokenizer).apply_chat_template(
                _to_messages(prompt),
                add_generation_prompt=True,
                tokenize=False,
            )
            mm_data = None
        return formatted, mm_data, _se_images

    async def _submit_impl(
        self,
        req_id: str,
        prompt,                 # str or list[dict] (multi-turn)
        max_tokens: int = 256,
        temperature: float = 0.0,
        ignore_eos: bool = False,
        image_path: str | None = None,
        head_cascade: bool | None = None,
        image_paths: list[str] | None = None,
        image_embeds=None,
        image_grid_thw=None,
        extract_hidden_states: bool = False,
        logprobs: int | None = None,
        emit_per_token_feature_seq: bool = False,
        in_engine_cascade_head: bool = False,
        cascade_source: str | None = None,
        kv_cost: int = 0,   # reserved draft KV (freed in _drive on finish)
    ) -> None:
        """Queue a request. Returns immediately. Result is pushed to
        finished_q when complete:
          {req_id, text, n_output_tokens, finish_reason, completed_t,
           head_decision?}
        On failure:
          {req_id, error: "<TypeName>: <message>", completed_t}

        `prompt` may be a string (single-turn) or a list of message
        dicts (multi-turn — e.g., MT-Bench T2 conversation history).

        `image_path` (single, legacy) or `image_paths` (multi) load
        PIL.Images and attach them as
        `multi_modal_data={"image": [PIL, ...]}` for VLM requests.

        `image_embeds` + `image_grid_thw` provide pre-encoded
        image features. When set, the draft's own ViT is skipped (vllm
        consumes the embeds directly via `multi_modal_data={"image":
        {"image_embeds": ..., "image_grid_thw": ...}}`). Used by the
        encoder-isolation bench to separate ViT time from decode time,
        and by the (potential) future remote-encoder offload.

        `head_cascade` (optional) overrides the actor-level default.
        When True, SamplingParams.head_cascade is set per-request so
        the fork's classifier head fires at cuts 0.0/1.0 and yields
        `head_decision` ∈ {SHIP, REGEN} on the final output. Requires
        the actor to have been started with `head_cascade=True`.

        Fake wire latency is paid by the RPC wrappers (`submit` /
        `submit_batch`), NOT here — a batch pays it once.
        """
        # Normalize single→list.
        if image_paths is None and image_path is not None:
            image_paths = [image_path]

        # PIL decode + chat-template render are CPU/IO-heavy and were
        # running INLINE on the actor's asyncio loop, inside the submit path —
        # So they blocked vLLM feed (_drive spawn) and drain (pop_finished)
        # while they ran. Heaviest for llava's anyres images, which fits the
        # only-llava stall. Offload to a worker thread: PIL and the HF tokenizer
        # both release the GIL, so the event loop is genuinely free during it.
        formatted, mm_data, _se_images = await asyncio.to_thread(
            self._build_prompt_payload,
            prompt, image_paths, image_embeds, image_grid_thw,
        )

        use_cascade = (head_cascade
                       if head_cascade is not None
                       else self._head_cascade_enabled)

        sp = SamplingParams(
            max_tokens=max_tokens, temperature=temperature,
            ignore_eos=ignore_eos,
        )
        if use_cascade:
            # SamplingParams field exposed by the cascade-prod-fixes
            # fork. Per-request opt-in keeps non-cascade callers free
            # of the head + extract overhead.
            sp.head_cascade = True
        # external head: opt-in to last-position hidden state
        # extraction without firing the inline head. Engine writes
        # `RequestOutput.hidden_states` (CPU tensor) per step; latest
        # step semantics mean the FINAL yield holds the cut-1 hidden
        # state suitable for an external head decision.
        if extract_hidden_states:
            sp.extract_hidden_states = True
        # output-confidence gate (A): request per-token logprobs so
        # _drive can attach mean/min chosen-token logprob to the finished
        # item. The scheduler's gate thresholds that confidence to decide
        # SHIP/REGEN — the output-confidence signal A/B against the head.
        if logprobs is not None:
            sp.logprobs = logprobs
        # lp-classifier-inline fork field. Triggers
        # the inline per-token feature-seq accumulator in
        # gpu_model_runner; output_processor populates
        # CompletionOutput.per_token_features. Stays a no-op on forks
        # that don't have the field.
        if emit_per_token_feature_seq:
            try:
                sp.emit_per_token_feature_seq = True
            except Exception:
                pass
        # in-engine cascade head opt-in. When set, the
        # engine returns CompletionOutput.head_decision dict with the
        # SHIP/REGEN verdict; _drive forwards it via item["head_decision"]
        # in the same shape as the cascade-prod-fixes fork's hidden-state
        # head path. cascade_source picks the per-source τ from the
        # tau-table loaded at engine boot.
        if in_engine_cascade_head:
            try:
                sp.in_engine_cascade_head = True
                if cascade_source is not None:
                    sp.cascade_source = cascade_source
            except Exception:
                pass

        self_eval_ctx = None
        if self._inline_self_eval:
            self_eval_ctx = {
                "question": (prompt if isinstance(prompt, str)
                             else prompt[-1]["content"]),
                "images": _se_images,
            }

        task = asyncio.create_task(
            self._drive(req_id, formatted, sp, mm_data, self_eval_ctx,
                        kv_cost=kv_cost)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _drive(
        self, req_id: str, formatted: str, sp: SamplingParams,
        mm_data: dict | None = None,
        self_eval_ctx: dict | None = None,
        kv_cost: int = 0,
    ) -> None:
        """Drive one request through AsyncLLMEngine and push to finished_q.

        if kv_cost was reserved at admission, free it on completion
        (finally) and record the real prefill length so the estimate
        self-calibrates."""
        # draft-side actor-clock admit + finish timestamps,
        # mirroring TargetEngineAsync._drive_regen. Lets us compute
        # draft's admit-rate and finish-rate on the actor's clock to
        # See if draft has the same cascade-coupling penalty target
        # has (or a different shape).
        t_actor_start = time.perf_counter()
        try:
            prompt_arg = (
                {"prompt": formatted, "multi_modal_data": mm_data}
                if mm_data is not None
                else formatted
            )
            async for output in self._llm.generate(
                prompt=prompt_arg,
                sampling_params=sp,
                request_id=req_id,
            ):
                if output.finished:
                    if kv_cost:
                        ptoks = getattr(output, "prompt_token_ids", None)
                        if ptoks is not None:
                            self._record_prefill(len(ptoks))
                    t_actor_finish = time.perf_counter()
                    completion = output.outputs[0]
                    item = {
                        "req_id": req_id,
                        "text": completion.text,
                        "n_output_tokens": len(completion.token_ids),
                        "finish_reason": completion.finish_reason,
                        "completed_t": t_actor_finish,
                        "draft_admit_actor_t": t_actor_start,
                        "draft_finish_actor_t": t_actor_finish,
                    }
                    # Cascade-prod-fixes fork: head_decision is set on
                    # the final yield when SamplingParams.head_cascade
                    # was True. Forward to the scheduler so it can
                    # route SHIP → respond, REGEN → target.submit_regen.
                    hd = getattr(output, "head_decision", None)
                    if hd is not None:
                        item["head_decision"] = hd
                    # (lp-classifier-inline fork): the
                    # in-engine attn_pool head writes its verdict to
                    # CompletionOutput.head_decision as a dict
                    # {"verdict": "SHIP"|"REGEN", "score": float, "tau":
                    # float, "source": str|None}. Normalize to the same
                    # string the cascade-prod-fixes path uses so the
                    # downstream scheduler doesn't need to branch.
                    comp_hd = getattr(completion, "head_decision", None)
                    if comp_hd is not None and "head_decision" not in item:
                        if isinstance(comp_hd, dict):
                            item["head_decision"] = comp_hd.get("verdict")
                            item["head_decision_score"] = comp_hd.get("score")
                            item["head_decision_tau"] = comp_hd.get("tau")
                        else:
                            item["head_decision"] = comp_hd
                    # external head: when extract_hidden_states
                    # was set on the request, RequestOutput.hidden_states
                    # holds the layer-N hidden state at the LAST decoded
                    # position of the final step (latest-step semantics
                    # in outputs.py:add). External CascadeHeadActor
                    # consumes this to make the cut-1 SHIP/REGEN call.
                    hs = getattr(output, "hidden_states", None)
                    if hs is not None:
                        item["hidden_states"] = hs
                    # output-confidence stats: when the request was
                    # submitted with logprobs, summarize the draft's per-
                    # token distribution into the confidence stats the
                    # scheduler gate can threshold (all oriented higher =
                    # more confident). With logprobs=1 only the chosen-token
                    # stats are meaningful; with logprobs>=2 we also get the
                    # Gatekeeper-rule stats (max-softmax + neg predictive
                    # entropy, top-k approximation). Mirrors / extends the
                    # sync engine's generate_text_with_logprobs.
                    # fast path: if the lp-classifier-inline fork
                    # emitted CompletionOutput.per_token_features directly
                    # (SamplingParams.emit_per_token_feature_seq=True),
                    # consume that and skip the driver-side Python loop
                    # over per-step logprob dicts. Same numerical schema
                    # as the driver-side build below (validated by the
                    # cascade_lp_classifier.compute_feature_seq_from_sample_logprobs
                    # reference). Avoids per-step detokenization +
                    # Logprob-dict construction (the ~3-5% saving we
                    # measured in py-spy).
                    ptf = getattr(completion, "per_token_features", None)
                    if ptf is not None:
                        item["per_token_features"] = ptf
                    lps = getattr(completion, "logprobs", None)
                    if lps is not None:
                        import math
                        chosen_lps: list[float] = []
                        max_probs: list[float] = []
                        entropies: list[float] = []
                        # also build per-token features
                        # [chosen_lp, max_p, neg_entropy, pos_frac]
                        # consumed by the CpuTransformerRouter gate. Same
                        # math as the offline feature extractor in
                        # verifier/src/extract_features.py, so the trained
                        # head sees inputs identical to its training data.
                        # logit-features: when the engine was booted
                        # in logprobs_mode="raw_logits", `lp.logprob`
                        # values are raw LOGITS (not log_softmax) — exp()
                        # would overflow. We then build a different schema
                        # of LOGIT-GAP features [t1-t2, t1-t5, t1-t20,
                        # pos_frac] used by the logit-trained attn_pool.
                        per_token: list[list[float]] = []
                        token_ids_list = list(completion.token_ids)
                        n_tok_total = max(1, len(token_ids_list))
                        _logit_mode = (self._logprobs_mode == "raw_logits")
                        for idx, (tok_id, lp_dict) in enumerate(
                            zip(token_ids_list, lps)
                        ):
                            if not lp_dict:
                                continue
                            lp = lp_dict.get(tok_id)
                            chosen_lp = (float(lp.logprob) if lp is not None
                                          else 0.0)
                            if lp is not None:
                                chosen_lps.append(chosen_lp)
                            if _logit_mode:
                                # Logit-gap schema: top-K values are raw
                                # logits; the gap to top-1 measures
                                # peakedness. Skip the legacy
                                # max_probs/entropies columns (gate doesn't
                                # use them in logit mode).
                                sorted_lps = sorted(
                                    (float(o.logprob)
                                     for o in lp_dict.values()),
                                    reverse=True,
                                )
                                if not sorted_lps:
                                    continue
                                t1 = sorted_lps[0]
                                t2 = (sorted_lps[1] if len(sorted_lps) > 1
                                      else t1)
                                tm = (sorted_lps[4] if len(sorted_lps) >= 5
                                      else sorted_lps[-1])
                                tw = (sorted_lps[19] if len(sorted_lps) >= 20
                                      else sorted_lps[-1])
                                pos_frac = (idx + 1) / n_tok_total
                                per_token.append([t1 - t2, t1 - tm,
                                                   t1 - tw, pos_frac])
                                continue
                            # Top-k distribution at this position (chosen
                            # token plus the alternatives vLLM returned).
                            probs = [math.exp(float(o.logprob))
                                     for o in lp_dict.values()]
                            if probs:
                                max_p = max(probs)
                                max_probs.append(max_p)
                                z = sum(probs)
                                ent = 0.0
                                if z > 0 and len(probs) > 1:
                                    # Entropy of the truncated, renormalized
                                    # top-k distribution — a standard proxy
                                    # for full predictive entropy (vLLM
                                    # doesn't emit full-vocab logprobs cheaply).
                                    ent = -sum((p / z) * math.log(p / z)
                                               for p in probs if p > 0)
                                    entropies.append(ent)
                                pos_frac = (idx + 1) / n_tok_total
                                per_token.append([chosen_lp, max_p,
                                                   -ent, pos_frac])
                        if chosen_lps:
                            item["mean_logprob"] = sum(chosen_lps) / len(chosen_lps)
                            item["min_logprob"] = min(chosen_lps)
                        if max_probs:
                            item["mean_max_prob"] = sum(max_probs) / len(max_probs)
                        if entropies:
                            item["neg_mean_entropy"] = -sum(entropies) / len(entropies)
                        if per_token and "per_token_features" not in item:
                            # Driver-side fallback when the fork didn't
                            # emit per_token_features (older fork, or
                            # request didn't opt in via
                            # emit_per_token_feature_seq).
                            item["per_token_features"] = per_token
                    # inline self-eval — issue the self-eval
                    # generation(s) to THIS engine now, blocking the request's
                    # completion (faithful serving-path semantics). Attach the
                    # Score; routing is unaffected.
                    if self._inline_self_eval and self_eval_ctx is not None:
                        try:
                            _se_t0 = time.perf_counter()
                            score = await self._run_self_eval(
                                self_eval_ctx["question"],
                                completion.text,
                                self_eval_ctx.get("images"),
                            )
                            item["self_eval_score"] = score
                            item["self_eval_method"] = self._inline_self_eval
                            item["self_eval_ms"] = (
                                (time.perf_counter() - _se_t0) * 1000.0
                            )
                        except Exception as e:
                            item["self_eval_error"] = f"{type(e).__name__}: {e}"
                    await self._finished_q.put(item)
                    return
        except Exception as e:
            await self._finished_q.put({
                "req_id": req_id,
                "error": f"{type(e).__name__}: {e}",
                "completed_t": time.perf_counter(),
                "draft_admit_actor_t": t_actor_start,
            })
        finally:
            # free the KV reserved for this request at admission.
            if kv_cost:
                self._kv_in_flight = max(0, self._kv_in_flight - kv_cost)

    def _build_self_eval_prompt(self, text: str, images):
        """Build the (formatted_prompt, mm_data) for a self-eval pass.

        ordering: the verifier text FIRST, then the image placeholder(s)
        — matches sys44_verifier_pass._msgs so an inline score reproduces the
        offline score. Text-only when no images are available (e.g. the pre-
        encoded-embeds submit path, which doesn't carry PILs)."""
        if images:
            content = ([{"type": "text", "text": text}]
                       + [{"type": "image"} for _ in images])
            templater = _get_chat_templater(self._tokenizer)
            formatted = templater.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True, tokenize=False,
            )
            mm_data = {"image": images if len(images) > 1 else images[0]}
        else:
            formatted = self._tokenizer.apply_chat_template(
                _to_messages(text), add_generation_prompt=True, tokenize=False,
            )
            mm_data = None
        return formatted, mm_data

    async def _run_self_eval(self, question: str, answer: str, images) -> float:
        """run one P(True) or k=8 AutoMix self-eval pass on
        this engine and return the score. Blocks the caller (_drive)."""
        from prorouter import self_eval as SE

        if self._inline_self_eval == "ptrue":
            text = SE.ptrue_text(question, answer)
            sp = SamplingParams(
                max_tokens=SE.PTRUE_MAX_TOKENS, temperature=0.0,
                logprobs=SE.PTRUE_LOGPROBS,
            )
        else:  # automix
            text = SE.automix_text(question, answer)
            sp = SamplingParams(
                max_tokens=SE.AUTOMIX_MAX_TOKENS,
                temperature=SE.AUTOMIX_TEMPERATURE, n=SE.AUTOMIX_K,
            )
        formatted, mm_data = self._build_self_eval_prompt(text, images)
        prompt_arg = (
            {"prompt": formatted, "multi_modal_data": mm_data}
            if mm_data is not None else formatted
        )
        self._self_eval_seq += 1
        se_req_id = f"se_{self._self_eval_seq}_{uuid.uuid4().hex[:8]}"
        final = None
        async for output in self._llm.generate(
            prompt=prompt_arg, sampling_params=sp, request_id=se_req_id,
        ):
            if output.finished:
                final = output
                break
        if final is None:
            return 0.5
        if self._inline_self_eval == "ptrue":
            return SE.score_ptrue(final.outputs[0])
        return SE.score_automix(final.outputs)

    async def pop_finished(
        self, max_n: int = 0, timeout_s: float = 0.05,
    ) -> list[dict]:
        """Drain finished items. Blocks up to timeout_s for the first item,
        then non-blocking drains the rest.

        max_n <= 0 (the default) drains EVERYTHING currently finished —
        a finished item is already done, so capping the drain only defers it to
        the next poll, i.e. another whole RTT. At high RTT with the serial pump
        (one pop per round trip) a finite cap silently throttles the return path
        to max_n/RTT. Pass max_n > 0 only to bound the RPC response size."""
        if self._rpc_lat_s:
            await asyncio.sleep(self._rpc_lat_s)  # fake wire latency
        items: list[dict] = []
        try:
            first = await asyncio.wait_for(
                self._finished_q.get(), timeout=timeout_s,
            )
            items.append(first)
        except asyncio.TimeoutError:
            return items
        while max_n <= 0 or len(items) < max_n:
            try:
                items.append(self._finished_q.get_nowait())
            except asyncio.QueueEmpty:
                break
        # when the scheduler's controller is enabled (set_emit_stats),
        # update the ship-rate EMA from this draft's own head verdicts and
        # stamp current engine stats on each item, so the controller gets them
        # For free (no extra RPC). Default off → byte-identical to pre-.
        if self._emit_stats:
            for it in items:
                hd = it.get("head_decision")
                if hd in ("SHIP", "REGEN"):
                    x = 1.0 if hd == "SHIP" else 0.0
                    self._ship_ma = (
                        x if self._ship_ma is None
                        else self._ship_ma_alpha * x
                        + (1.0 - self._ship_ma_alpha) * self._ship_ma
                    )
            stats = self._current_stats()
            for it in items:
                it["stats"] = stats
        return items

    def set_emit_stats(self, on: bool = True) -> None:
        """enable/disable the piggybacked-stats path (off by default)."""
        self._emit_stats = bool(on)

    def set_ship_ma_alpha(self, alpha: float) -> None:
        """tune the ship-rate EMA smoothing (effective window ≈ 1/α)."""
        self._ship_ma_alpha = max(1e-4, min(1.0, float(alpha)))

    def _cached_gpu_util(self) -> float | None:
        """Max per-GPU util%, refreshed at most every _gpu_util_cache_ttl_s
        so NVML never lands on the per-request critical path."""
        now = time.perf_counter()
        if (
            self._gpu_util_cache is None
            or now - self._gpu_util_cache_t >= self._gpu_util_cache_ttl_s
        ):
            try:
                samples = _sample_gpu_util()
                self._gpu_util_cache = max(
                    (s.get("util_pct", 0) for s in samples), default=0,
                )
            except Exception:
                pass
            self._gpu_util_cache_t = now
        return self._gpu_util_cache

    def _current_stats(self) -> dict:
        """snapshot for the scheduler controller (piggybacked).
        `pending` = self-admit buffer depth (0 when off)."""
        return {
            "in_flight": len(self._tasks),
            "finished_q": self._finished_q.qsize(),
            "pending": len(self._pending),
            "kv_in_flight": self._kv_in_flight,
            "kv_threshold": self._kv_pool_threshold,
            "ship_rate_ma": self._ship_ma,
            "gpu_util": self._cached_gpu_util(),
            "t": time.perf_counter(),
        }

    async def qsize(self) -> dict[str, int]:
        """Inspection: current finished_q depth and in-flight task count."""
        return {
            "finished": self._finished_q.qsize(),
            "in_flight": len(self._tasks),
        }

    async def score_self_eval_for_validation(
        self, question: str, answer: str, image_paths: list[str] | None = None,
    ) -> float:
        """gate helper — run the SHIPPING self-eval scoring path
        (_run_self_eval → _build_self_eval_prompt → engine.generate → score)
        on a FIXED (question, answer, images), so the ±0.02 match against the
        offline score is independent of draft-answer reproduction. Boot
        the actor with inline_self_eval set to the method under test."""
        from PIL import Image
        images = (
            [Image.open(p).convert("RGB") for p in image_paths]
            if image_paths else None
        )
        return await self._run_self_eval(question, answer, images)

    async def gpu_util(self) -> list[dict]:
        """Per-GPU utilization on this draft actor's node (one entry
        per visible GPU). Used by bench scripts to characterize draft
        GPU load. See `_sample_gpu_util` for return schema."""
        return _sample_gpu_util()

    async def init_embeds_cache(
        self, cache_path: str, transfer_mode: str = "cpu_pinned",
    ) -> dict:
        """isolation-bench helper. Load a pre-encoded
        draft-ViT (7B-VL) embeds cache directly inside the draft
        actor process, so the bench can dispatch by cache-key without
        serializing 11 MB tensors per request through Ray RPC.

        Mirror of `TargetEngineAsync.init_embeds_cache`. Default
        `transfer_mode='cpu_pinned'` (matches's lesson: actor-
        side GPU allocation on a TP=4 single-node setup competes with
        Worker GPUs for memory; let vLLM do H2D inside the worker).
        `gpu_side_stream` is available but historically OOMs on
        single-node TP setups.

        Cache format: a `.pt` dict from a pre-encoding pass over the
        *draft* model (`Qwen/Qwen2.5-VL-7B-Instruct`), keyed by record id
        ({"rec_id": {"image_embeds": tensor, "image_grid_thw": tensor},
        ...}). The 7B-VL ViT produces embeddings of shape
        [N_image_tokens, 3584] — different from the 72B-VL (shape
        [N_image_tokens, 8192]). The cache MUST be encoded with the
        same model the draft engine is running.

        Returns: {n_entries, mb_in_cache, transfer_mode, transfer_stream}.
        """
        import torch as _torch
        if transfer_mode not in ("gpu_side_stream", "cpu_pinned"):
            raise ValueError(
                f"transfer_mode={transfer_mode!r}; expected one of "
                "{'gpu_side_stream', 'cpu_pinned'}"
            )
        self._embeds_transfer_mode = transfer_mode
        if not hasattr(self, "_embeds_cache"):
            self._embeds_cache: dict = {}
        if transfer_mode == "gpu_side_stream":
            if not hasattr(self, "_embeds_transfer_stream"):
                try:
                    self._embeds_device = _torch.device("cuda:0")
                    self._embeds_transfer_stream = _torch.cuda.Stream(
                        device=self._embeds_device,
                    )
                except Exception:
                    self._embeds_device = None
                    self._embeds_transfer_stream = None
        else:
            self._embeds_device = None
            self._embeds_transfer_stream = None
        loaded = _torch.load(
            cache_path, map_location="cpu", weights_only=False,
        )
        if not isinstance(loaded, dict):
            raise TypeError(
                f"embeds cache at {cache_path} is not a dict "
                f"(got {type(loaded).__name__})"
            )
        if "image_embeds" in loaded:
            raise TypeError(
                "embeds cache is a single-entry dict; expected a dict "
                "of {rec_id: {image_embeds, image_grid_thw}}."
            )
        n = 0
        total_bytes = 0
        for k, v in loaded.items():
            if not isinstance(v, dict):
                continue
            ie = v.get("image_embeds")
            it = v.get("image_grid_thw")
            if ie is None or it is None:
                continue
            try:
                if hasattr(ie, "pin_memory"):
                    ie = ie.pin_memory()
                if hasattr(it, "pin_memory"):
                    it = it.pin_memory()
            except Exception:
                pass
            self._embeds_cache[k] = {
                "image_embeds": ie,
                "image_grid_thw": it,
            }
            n += 1
            total_bytes += (
                ie.element_size() * ie.numel()
                + it.element_size() * it.numel()
            )
        return {
            "n_entries": n,
            "mb_in_cache": total_bytes / (1024 * 1024),
            "transfer_mode": self._embeds_transfer_mode,
            "transfer_stream": (
                "allocated" if self._embeds_transfer_stream is not None
                else ("disabled" if self._embeds_transfer_mode == "cpu_pinned"
                      else "unavailable")
            ),
        }

    async def submit_by_cache_key(
        self, req_id: str, cache_key: str,
        prompt, max_tokens: int = 256,
        temperature: float = 0.0,
        ignore_eos: bool = False,
        head_cascade: bool | None = None,
    ) -> None:
        """isolation-bench dispatch path. Look up cached embeds
        in the draft actor's local memory by `cache_key`, then call
        `self.submit(...)` with `image_embeds` / `image_grid_thw`
        attached directly — no Ray pickle of the 11 MB tensor per
        request.

        Default `transfer_mode='cpu_pinned'`: pass the CPU pinned
        tensors straight to `submit()` (which already accepts them
        via kwargs); vLLM does H2D inside the worker. Mirror
        of `TargetEngineAsync.submit_regen_by_cache_key` minus the
        `_drive_regen` event-sync since DraftEngineAsync's `_drive`
        doesn't currently take an `embed_ready_event`. For the
        cpu_pinned production-proxy path that doesn't matter.
        """
        cache = getattr(self, "_embeds_cache", None)
        if cache is None:
            await self._finished_q.put({
                "req_id": req_id,
                "error": "embeds_cache not initialized — call init_embeds_cache() first",
                "completed_t": time.perf_counter(),
            })
            return
        entry = cache.get(cache_key)
        if entry is None:
            await self._finished_q.put({
                "req_id": req_id,
                "error": f"cache miss for key {cache_key!r}",
                "completed_t": time.perf_counter(),
            })
            return

        embeds = entry["image_embeds"]
        thw = entry["image_grid_thw"]
        # For cpu_pinned (default), pass CPU pinned tensors directly —
        # vLLM does its own H2D inside the worker process. For
        # gpu_side_stream, do the actor-side copy on the dedicated
        # stream so the side-stream-overlap-with-LM pattern matches
        #'s NCCL design.
        if (self._embeds_transfer_mode == "gpu_side_stream"
                and self._embeds_transfer_stream is not None):
            import torch as _torch
            with _torch.cuda.stream(self._embeds_transfer_stream):
                embeds = embeds.to(self._embeds_device, non_blocking=True)
                thw = thw.to(self._embeds_device, non_blocking=True)
                # Note: DraftEngineAsync's _drive doesn't take
                # embed_ready_event today; for gpu_side_stream we sync
                # the side stream here. Cheap (~0.3 ms) and bullet-
                # proof for the test. In production, plumb event into
                # _drive instead.
                self._embeds_transfer_stream.synchronize()

        # Delegate to submit() — it already handles the image_embeds
        # path via mm_data and runs the cascade head if enabled.
        await self.submit(
            req_id=req_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=ignore_eos,
            head_cascade=head_cascade,
            image_embeds=embeds,
            image_grid_thw=thw,
        )

    async def ping(self) -> str:
        return "ok"


@ray.remote(num_gpus=1, max_concurrency=16)
class TargetEngineAsync:
    """V0 target actor — wraps vLLM's AsyncLLMEngine with two-queue dispatch.

    Public methods (all async):
      - submit_verify(req_id, prompt, draft_response, max_tokens)
            → queues verify; on ACCEPT pushes draft to finished_q;
              on REJECT internally routes to regen_q
      - submit_regen(req_id, prompt, max_tokens)
            → queues regen directly (e.g. external skip-verify request)
      - pop_finished(max_n, timeout_s)
            → drains finished_q
      - qsize() → introspection
      - ping() → readiness probe (also kicks off the dispatch loop)

    Internals:
      - verify_q: incoming verify requests
      - regen_q:  regen requests (from REJECT routing or submit_regen)
      - finished_q: final results (ACCEPT-ship + REGEN completions)
      - _dispatch_loop: verify-first picker + length-aware KV admission

    Picker policy (FIFO inside each queue):
      1. If verify_q has items → dispatch one (cheap, ~16-tok response).
      2. Else if regen_q has items AND kv_in_flight + est_cost ≤
         kv_pool_threshold → dispatch one.
      3. Else wait on _wakeup (set by submit / completion).

    KV admission uses a rough word-count estimate (×1.5 tokens/word).
    The point isn't precision — it's keeping vLLM out of the preemption
    regime where wall scales super-linearly with batch size. Tune
    kv_pool_threshold per deployment.
    """

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        limit_mm_per_prompt: dict[str, int] | None = None,
        distributed_executor_backend: str | None = None,
        kv_pool_threshold: int = 200_000,
        max_num_batched_tokens: int | None = None,
        regen_priority: bool = False,
        mm_processor_kwargs: dict | None = None,
        # opt-in: estimate a regen's KV footprint from the live
        # per-source p90 of OBSERVED output lengths instead of the worst-case
        # `max_tokens`. Off by default → byte-identical max_tokens reservation.
        # vLLM PagedAttention preemption (recompute/swap) is the backstop on the
        # rare under-prediction, so this only tunes the admission heuristic to
        # be realistic (less over-reservation → higher KV utilization).
        ma_length_gating: bool = False,
        # c18x9: allow disabling APC when the workload cycles/duplicates
        # prompts (cache hits inflate measured throughput). Default True.
        enable_prefix_caching: bool = True,
        # network-latency resilience probe — see DraftEngineAsync.
        rpc_fake_latency_ms: float = 0.0,
        # actor self-admit (default OFF → unchanged). See DraftEngineAsync.
        actor_self_admit: bool = False,
        actor_admit_interval_ms: float = 5.0,
        actor_admit_max_inflight: int = 256,
    ) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from transformers import AutoTokenizer

        _ensure_all_gpus_visible()

        engine_args = AsyncEngineArgs(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            limit_mm_per_prompt=limit_mm_per_prompt or {"image": 1},
            # Required for the isolation bench paths
            # (`submit_regen_by_cache_key` + the per-request image_embeds
            # variant of `submit_regen`): vLLM rejects pre-encoded
            # `image_embeds` in multi_modal_data without this flag.
            enable_mm_embeds=True,
        )
        if mm_processor_kwargs:
            engine_args.mm_processor_kwargs = mm_processor_kwargs
        # Per-step matmul budget. Default (None) leaves it at vLLM's
        # built-in (V1: typically max(2048, max_model_len) heuristic).
        # Setting explicitly is the lever for the step-density experiment:
        # raising it lets more verify-prefill chunks fit per step;
        # lowering it forces verify chunks to split, leaving room for
        # decoders to ride along with smaller per-step compute load.
        if max_num_batched_tokens is not None:
            engine_args.max_num_batched_tokens = max_num_batched_tokens
        if distributed_executor_backend is not None:
            engine_args.distributed_executor_backend = (
                distributed_executor_backend
            )
        self._llm = AsyncLLMEngine.from_engine_args(engine_args)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Queues + signaling.
        self._verify_q: asyncio.Queue = asyncio.Queue()
        self._regen_q: asyncio.Queue = asyncio.Queue()
        self._finished_q: asyncio.Queue = asyncio.Queue()
        self._wakeup = asyncio.Event()
        self._stop = asyncio.Event()

        # KV admission accounting.
        self._kv_in_flight = 0
        self._kv_pool_threshold = kv_pool_threshold
        # MA-length gating state: per-source ring buffers of observed
        # output-token counts. `_regen_cost` reserves prompt + p90(observed)
        # (capped at max_tokens, floored at max_tokens until WARMUP samples) so
        # the admission heuristic reflects real lengths, not the worst case.
        self._ma_length_gating = bool(ma_length_gating)
        self._rpc_lat_s = float(rpc_fake_latency_ms) / 1000.0
        self._len_samples: dict[str, list[int]] = {}
        self._len_window = 256          # per-source ring size
        self._len_warmup = 16           # samples before trusting the estimate
        self._len_quantile = 0.90       # p90 — safety margin vs the mean
        # closed-loop controller piggyback (see DraftEngineAsync).
        # pop_finished stamps item["stats"] incl. KV proximity so the
        # controller's KV-guard throttle never needs a separate RPC. Off by
        # default (set_emit_stats) → zero cost on non-controller runs.
        self._emit_stats = False
        self._gpu_util_cache: float | None = None
        self._gpu_util_cache_t = 0.0
        self._gpu_util_cache_ttl_s = 0.5
        # Picker policy: False (default) = verify-first (cheap, frees a
        # request); True = regen-first (dispatch regens as soon as KV
        # pool admits, before draining verify_q). Regen-first reduces
        # the regen-tail-drain at end-of-run by interleaving regen with
        # the verify burst instead of clumping it after.
        self._regen_priority = regen_priority

        # Drive task tracking + dispatch loop (lazy-started — Ray's actor
        # event loop may not be running yet at __init__ time).
        self._tasks: set = set()
        self._dispatch_task: asyncio.Task | None = None

        # actor self-admit — SAME design as DraftEngineAsync (default
        # OFF → unchanged; the existing _regen_q/_dispatch_loop path still
        # serves when off). When on, submit_decode_batch buffers to _pending
        # and returns stats; the shared _admit_loop drains _pending into vLLM
        # bounded by _admit_ok. The two hooks (_admit_ok / _admit_entry) below
        # mirror the draft's — the only per-actor difference.
        self._self_admit = bool(actor_self_admit)
        self._admit_interval_s = float(actor_admit_interval_ms) / 1000.0
        self._admit_max_inflight = int(actor_admit_max_inflight)
        self._pending: collections.deque = collections.deque()
        self._admit_task = None
        if self._self_admit:
            print(f"[TargetEngineAsync] actor_self_admit ON "
                  f"(interval={actor_admit_interval_ms}ms, "
                  f"max_inflight={self._admit_max_inflight})", flush=True)

    # ---------------- actor self-admit (shared shape w/ draft) ---------
    # Identical _ensure_admit_loop + _admit_loop as DraftEngineAsync; only
    # _admit_ok / _admit_entry differ (target adds KV-token accounting).
    def _ensure_admit_loop(self) -> None:
        """Lazily start the background admission loop."""
        if self._admit_task is None or self._admit_task.done():
            self._admit_task = asyncio.create_task(self._admit_loop())

    async def _admit_loop(self) -> None:
        """Drain the pending buffer into vLLM while the actor has local
        headroom (_admit_ok). See DraftEngineAsync._admit_loop."""
        while True:
            while self._pending and self._admit_ok():
                await self._admit_entry(self._pending.popleft())
            await asyncio.sleep(self._admit_interval_s)

    def _admit_ok(self) -> bool:
        """Target: concurrency cap (symmetric with draft) AND the target's
        native KV-token headroom, so per-request cost is still respected."""
        return (len(self._tasks) < self._admit_max_inflight
                and self._kv_in_flight < self._kv_pool_threshold)

    async def _admit_entry(self, entry) -> None:
        """Target: an entry is one normalized regen req dict. Mirror the
        dispatch loop's try_regen KV accounting: reserve cost, drive regen
        (which frees the cost on finish)."""
        cost = self._regen_cost(
            entry["prompt"], entry["max_tokens"], entry.get("source"),
        )
        self._kv_in_flight += cost
        self._spawn(self._drive_regen(entry, cost))

    # ---------------- internals ----------------

    @staticmethod
    def _est_tokens(text: str) -> int:
        """Rough token count from word count. ~1.5 tokens/word + floor."""
        return int(len(text.split()) * 1.5) + 10

    def _verify_cost(self, prompt: str, draft: str) -> int:
        """Verify KV: prompt + draft (assistant turn) + verify-Q template
        + ~16-token response. +80 floor covers chat template overhead."""
        return self._est_tokens(prompt) + self._est_tokens(draft) + 80

    def _regen_cost(self, prompt: str, max_tokens: int,
                    source: str | None = None) -> int:
        """Regen KV reservation: prompt + an output-length estimate + 30.

        Default: worst-case `max_tokens` (conservative; same for every regen so
        admission cannot distinguish cascade-REGEN from DIRECT —).

 `ma_length_gating`: once a source has ≥ `_len_warmup` observed
        output lengths, reserve `p90(observed)` instead of `max_tokens` (capped
        at max_tokens, never above). p90 (not the mean) keeps a safety margin
        against the long-output tail; vLLM PagedAttention preemption is the
        backstop on the rare under-prediction. Less over-reservation → higher
        KV utilization → more concurrent regens admitted."""
        out_est = max_tokens
        if self._ma_length_gating:
            est = self._len_estimate(source)
            if est is not None:
                out_est = min(max_tokens, est)
        return self._est_tokens(prompt) + out_est + 30

    def _len_estimate(self, source: str | None) -> int | None:
        """p90 of observed output lengths for `source` (None until warmup)."""
        key = source or "__default__"
        s = self._len_samples.get(key)
        if not s or len(s) < self._len_warmup:
            return None
        ss = sorted(s)
        idx = min(len(ss) - 1, int(self._len_quantile * len(ss)))
        return ss[idx]

    def _record_len(self, source: str | None, n_tokens: int) -> None:
        """Record an observed output length into the source's ring buffer."""
        if not self._ma_length_gating or not n_tokens:
            return
        key = source or "__default__"
        buf = self._len_samples.setdefault(key, [])
        buf.append(int(n_tokens))
        if len(buf) > self._len_window:
            del buf[0]

    def set_ma_length_gating(self, on: bool = True) -> None:
        """toggle MA-length KV gating at runtime (off by default)."""
        self._ma_length_gating = bool(on)

    def _ensure_dispatch(self) -> None:
        """Start the dispatch loop on first use (lazy)."""
        if self._dispatch_task is None or self._dispatch_task.done():
            self._dispatch_task = asyncio.create_task(self._dispatch_loop())

    # ---------------- public API ----------------

    async def submit_verify(
        self, req_id: str, prompt: str, draft_response: str,
        max_tokens: int = 1024,
        ignore_eos: bool = False,
        skip_regen_on_reject: bool = False,
        verify_template: str | None = None,
        bit_mode: bool = False,
    ) -> None:
        """Queue a verify request. ACCEPT → ship draft via finished_q;
        REJECT → internally routed to regen_q (no head round-trip).

        ignore_eos applies only to the regen path on REJECT; verify
        itself is fixed-cost (16 tokens, or 1 in bit_mode).

        skip_regen_on_reject=True short-circuits the REJECT→regen
        path: REJECT becomes a finished item directly. Used by the
        verify-only throughput ceiling bench.

        verify_template / bit_mode: same semantics as the sync
        TargetEngine.batch_judge_verify_binary path; flow into
        _drive_verify so the verify decode either runs the default
        word-form output or the constrained 1-step bit form.
        """
        self._ensure_dispatch()
        await self._verify_q.put({
            "req_id": req_id,
            "prompt": prompt,
            "draft_response": draft_response,
            "max_tokens": max_tokens,
            "ignore_eos": ignore_eos,
            "skip_regen_on_reject": skip_regen_on_reject,
            "verify_template": verify_template,
            "bit_mode": bit_mode,
        })
        self._wakeup.set()

    async def submit_decode(
        self, req_id: str, prompt: str, max_tokens: int = 1024,
        ignore_eos: bool = False,
        image_path: str | None = None,
        image_paths: list[str] | None = None,
        *,
        image_embeds=None,
        image_grid_thw=None,
    ) -> None:
        """Queue a full prefill+decode request at target.

        Used by BOTH the DIRECT path (request bypasses draft) AND the
        cascade-REGEN path (head said REGEN, draft's output discarded
        and target re-decodes from scratch). Target sees no
        distinction between the two — same call, no length_hint,
        identical KV-admission estimate. The verdict tag is decided
        upstream by V0Scheduler based on routing_path.

        Previously named `submit_regen` (kept as a class-level alias
        below for any callers that haven't migrated).

        Image input variants (mutually exclusive):
          - `image_embeds` + `image_grid_thw` set — pre-encoded path
            shipped via Ray pickle per request. Used by the
            isolation bench's `target_only_no_vit` cell (biased upper
            bound on encoder-offload value: costs ~3-5 ms per request in
            Ray pickle overhead).
          - `image_path` / `image_paths` set — raw path(s), target runs
            its own ViT. Production cascade REGEN path.
          - none — text-only.
        For the cleaner side-stream measurement (no per-request pickle),
        use `init_embeds_cache` + `submit_decode_by_cache_key`."""
        if self._rpc_lat_s:
            await asyncio.sleep(self._rpc_lat_s)  # fake wire latency
        # Normalize: image_paths is the multi-image form; image_path
        # is the single-image legacy form. If both set, image_paths wins.
        if image_paths is None and image_path is not None:
            image_paths = [image_path]
        req = {
            "req_id": req_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "ignore_eos": ignore_eos,
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
            "image_path": image_path,
            "image_paths": image_paths,
        }
        if self._self_admit:
            self._pending.append(req)   # background _admit_loop drives it
            self._ensure_admit_loop()
            return self._current_stats()
        self._ensure_dispatch()
        await self._regen_q.put(req)
        self._wakeup.set()

    async def submit_regen(self, *args, **kwargs) -> None:
        """Deprecated — kept for callers that still invoke the old
        name via `target.submit_regen.remote(...)`. Forwards to
        `submit_decode`."""
        return await self.submit_decode(*args, **kwargs)

    async def submit_decode_batch(self, items: list[dict]) -> None:
        """batched submission — one Ray RPC, N enqueues to
        regen_q. Reduces per-REGEN RPC overhead in the cascade path
        (cascade-REGEN otherwise pays one RPC to draft + one to
        target per request; batching lets multiple REGEN dispatches
        share a target RPC).

        Each `items[i]` is a dict with the same keys as
        `submit_decode` args: req_id, prompt, max_tokens, ignore_eos,
        image_path / image_paths, image_embeds, image_grid_thw.
        Image-path normalization (single → list) is applied per item,
        matching `submit_decode`."""
        if self._rpc_lat_s:
            await asyncio.sleep(self._rpc_lat_s)  # fake wire latency
        for it in items:
            image_paths = it.get("image_paths")
            image_path  = it.get("image_path")
            if image_paths is None and image_path is not None:
                image_paths = [image_path]
            req = {
                "req_id":         it["req_id"],
                "prompt":         it["prompt"],
                "max_tokens":     it.get("max_tokens", 1024),
                "ignore_eos":     it.get("ignore_eos", False),
                "image_embeds":   it.get("image_embeds"),
                "image_grid_thw": it.get("image_grid_thw"),
                "image_path":     image_path,
                "image_paths":    image_paths,
                # carry source for per-source MA-length KV gating.
                "source":         it.get("source"),
            }
            if self._self_admit:
                self._pending.append(req)   # background _admit_loop drives it
            else:
                await self._regen_q.put(req)
        if self._self_admit:
            self._ensure_admit_loop()
            return self._current_stats()
        self._ensure_dispatch()
        self._wakeup.set()

    async def init_embeds_cache(
        self, cache_path: str, transfer_mode: str = "gpu_side_stream",
    ) -> dict:
        """isolation-bench helper. Load a pre-encoded embeds
        cache directly inside the target actor process, so the bench
        can dispatch by cache-key without serializing 11 MB tensors
        per request through Ray RPC.

        Cache format: a `.pt` dict from a pre-encoding pass over the draft
        model, keyed by record id ({"rec_id": {"image_embeds": tensor,
        "image_grid_thw": tensor}, ...}).

        Embeds are kept on **CPU pinned memory** so the per-request
        side-stream copy in `submit_regen_by_cache_key` can run as a
        non-blocking PCIe DMA, overlapped with LM decode on the default
        stream. Also allocates `_embeds_transfer_stream` — the dedicated
        CUDA stream that does CPU→GPU transfers in parallel with decode.
        In production, this same stream will be the NCCL recv stream
        for encoder offload; the per-request transfer pattern
        is identical (`side stream produces` → `event.record()` →
        `default stream wait_event` → `vLLM consumes GPU tensor`).

        Returns: {"n_entries": int, "mb_in_cache": float}."""
        import torch as _torch
        if transfer_mode not in ("gpu_side_stream", "cpu_pinned"):
            raise ValueError(
                f"transfer_mode={transfer_mode!r}; expected one of "
                "{'gpu_side_stream', 'cpu_pinned'}"
            )
        self._embeds_transfer_mode = transfer_mode
        if not hasattr(self, "_embeds_cache"):
            self._embeds_cache: dict = {}
        # Side stream for per-request CPU→GPU transfers (production:
        # NCCL recv stream). Allocate on the actor's primary CUDA
        # device. Cached on first call. Only used when
        # transfer_mode='gpu_side_stream'; cpu_pinned skips the actor's
        # GPU copy entirely and lets vLLM do H2D inside the worker
        # process (closer to the production NCCL-into-worker
        # pattern + avoids competing with vLLM's KV pool on the same
        # physical GPU).
        if transfer_mode == "gpu_side_stream":
            if not hasattr(self, "_embeds_transfer_stream"):
                try:
                    self._embeds_device = _torch.device("cuda:0")
                    self._embeds_transfer_stream = _torch.cuda.Stream(
                        device=self._embeds_device,
                    )
                except Exception:
                    # Fallback for non-CUDA test environments.
                    self._embeds_device = None
                    self._embeds_transfer_stream = None
        else:
            # cpu_pinned: no actor-side GPU allocation.
            self._embeds_device = None
            self._embeds_transfer_stream = None
        loaded = _torch.load(
            cache_path, map_location="cpu", weights_only=False,
        )
        if not isinstance(loaded, dict):
            raise TypeError(
                f"embeds cache at {cache_path} is not a dict "
                f"(got {type(loaded).__name__})"
            )
        if "image_embeds" in loaded:
            raise TypeError(
                "embeds cache is a single-entry dict; expected a dict "
                "of {rec_id: {image_embeds, image_grid_thw}}."
            )
        n = 0
        total_bytes = 0
        for k, v in loaded.items():
            if not isinstance(v, dict):
                continue
            ie = v.get("image_embeds")
            it = v.get("image_grid_thw")
            if ie is None or it is None:
                continue
            # Pin embeds to page-locked memory for fast PCIe DMA on the
            # side stream. grid_thw is tiny — pin too, no harm.
            try:
                if hasattr(ie, "pin_memory"):
                    ie = ie.pin_memory()
                if hasattr(it, "pin_memory"):
                    it = it.pin_memory()
            except Exception:
                # Fallback if CUDA not available or pin fails.
                pass
            self._embeds_cache[k] = {
                "image_embeds": ie,
                "image_grid_thw": it,
            }
            n += 1
            total_bytes += (
                ie.element_size() * ie.numel()
                + it.element_size() * it.numel()
            )
        return {
            "n_entries": n,
            "mb_in_cache": total_bytes / (1024 * 1024),
            "transfer_mode": self._embeds_transfer_mode,
            "transfer_stream": (
                "allocated" if self._embeds_transfer_stream is not None
                else ("disabled" if self._embeds_transfer_mode == "cpu_pinned"
                      else "unavailable")
            ),
        }

    async def submit_regen_by_cache_key(
        self, req_id: str, cache_key: str,
        prompt: str, max_tokens: int = 1024,
        ignore_eos: bool = False,
    ) -> None:
        """isolation-bench dispatch path with side-stream
        CPU→GPU transfer.

        Per-request flow:
          1. Look up `cache_key` → CPU pinned embeds.
          2. On `_embeds_transfer_stream`, kick off non-blocking copy
             to GPU (`cpu.to(device, non_blocking=True)`).
          3. Record a CUDA event on the transfer stream right after
             the copy is enqueued.
          4. Pass GPU tensors + event handle into the regen queue.
          5. `_drive_regen` issues `default_stream.wait_event(event)`
             before vLLM consumes the embeds → guaranteed ordering
             with zero blocking on the bench thread.

        This mirrors what the future NCCL recv path will do:
        encoder GPU produces, side stream recvs, event records, LM
        consumes after wait_event. The same instrumentation hooks
        carry over.

        Requires `init_embeds_cache(...)` to have been called first.
        Errors finished_q with verdict=ERROR if cache_key missing.
        """
        self._ensure_dispatch()
        cache = getattr(self, "_embeds_cache", None)
        if cache is None:
            await self._finished_q.put({
                "req_id": req_id,
                "verdict": "ERROR",
                "stage": "submit_regen_by_cache_key",
                "error": "embeds_cache not initialized — call init_embeds_cache() first",
                "completed_t": time.perf_counter(),
            })
            return
        entry = cache.get(cache_key)
        if entry is None:
            await self._finished_q.put({
                "req_id": req_id,
                "verdict": "ERROR",
                "stage": "submit_regen_by_cache_key",
                "error": f"cache miss for key {cache_key!r}",
                "completed_t": time.perf_counter(),
            })
            return

        cpu_embeds = entry["image_embeds"]
        cpu_thw = entry["image_grid_thw"]
        transfer_stream = self._embeds_transfer_stream
        device = self._embeds_device
        embed_ready_event = None
        if transfer_stream is not None and device is not None:
            import torch as _torch
            # Side-stream non-blocking copy. Mirrors the NCCL recv
            # pattern: side stream owns the producer dependency, event
            # marks "data is ready on GPU", default stream waits when
            # vLLM is about to consume. Allocates GPU memory in the
            # actor process — requires `gpu_memory_utilization` < ~0.80
            # so the actor's per-request copies don't compete with the
            # vLLM worker's KV pool on the same physical GPU 0.
            with _torch.cuda.stream(transfer_stream):
                gpu_embeds = cpu_embeds.to(device, non_blocking=True)
                gpu_thw = cpu_thw.to(device, non_blocking=True)
                embed_ready_event = _torch.cuda.Event()
                embed_ready_event.record(transfer_stream)
        else:
            # cpu_pinned mode (or non-CUDA test env): pass pinned CPU
            # tensors straight to vLLM. The worker process does its
            # own H2D into worker-owned GPU memory, so the actor's
            # GPU 0 stays unused. Closer to the production
            # NCCL-into-worker pattern (encoder GPU sends directly
            # to target worker GPU, not via the orchestrator).
            gpu_embeds = cpu_embeds
            gpu_thw = cpu_thw

        await self._regen_q.put({
            "req_id": req_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "ignore_eos": ignore_eos,
            "image_embeds": gpu_embeds,
            "image_grid_thw": gpu_thw,
            "embed_ready_event": embed_ready_event,
        })
        self._wakeup.set()

    async def pop_finished(
        self, max_n: int = 0, timeout_s: float = 0.05,
    ) -> list[dict]:
        """Drain finished items. Each item:
          {req_id, verdict ∈ {ACCEPT, REGEN, ERROR}, text, ...}

        max_n <= 0 (default) drains everything finished — see the draft
        actor's pop_finished for why a finite cap throttles the return path at
        high RTT. Pass max_n > 0 only to bound the RPC response size."""
        if self._rpc_lat_s:
            await asyncio.sleep(self._rpc_lat_s)  # fake wire latency
        items: list[dict] = []
        try:
            first = await asyncio.wait_for(
                self._finished_q.get(), timeout=timeout_s,
            )
            items.append(first)
        except asyncio.TimeoutError:
            return items
        while max_n <= 0 or len(items) < max_n:
            try:
                items.append(self._finished_q.get_nowait())
            except asyncio.QueueEmpty:
                break
        # stamp current engine stats (incl. KV proximity) on each item
        # for the scheduler controller — free on the response path. Off by
        # default → byte-identical to pre-.
        if self._emit_stats:
            stats = self._current_stats()
            for it in items:
                it["stats"] = stats
        return items

    def set_emit_stats(self, on: bool = True) -> None:
        """enable/disable the piggybacked-stats path (off by default)."""
        self._emit_stats = bool(on)

    def _cached_gpu_util(self) -> float | None:
        """Max per-GPU util%, refreshed at most every _gpu_util_cache_ttl_s
        so NVML never lands on the per-request critical path."""
        now = time.perf_counter()
        if (
            self._gpu_util_cache is None
            or now - self._gpu_util_cache_t >= self._gpu_util_cache_ttl_s
        ):
            try:
                samples = _sample_gpu_util()
                self._gpu_util_cache = max(
                    (s.get("util_pct", 0) for s in samples), default=0,
                )
            except Exception:
                pass
            self._gpu_util_cache_t = now
        return self._gpu_util_cache

    def _current_stats(self) -> dict:
        """snapshot for the scheduler controller (piggybacked)."""
        return {
            "in_flight": len(self._tasks),
            "finished_q": self._finished_q.qsize(),
            "pending": len(self._pending),
            "kv_in_flight": self._kv_in_flight,
            "kv_threshold": self._kv_pool_threshold,
            "gpu_util": self._cached_gpu_util(),
            "t": time.perf_counter(),
        }

    async def qsize(self) -> dict:
        return {
            "verify": self._verify_q.qsize(),
            "regen": self._regen_q.qsize(),
            "finished": self._finished_q.qsize(),
            "in_flight": len(self._tasks),
            "kv_in_flight": self._kv_in_flight,
            "kv_threshold": self._kv_pool_threshold,
        }

    async def gpu_util(self) -> list[dict]:
        """Per-GPU utilization on this target actor's node (one entry
        per visible GPU). Used by bench scripts to characterize target
        GPU load — the central question in checking whether V0
        underutilizes target. See `_sample_gpu_util` for return schema."""
        return _sample_gpu_util()

    async def ping(self) -> str:
        self._ensure_dispatch()
        return "ok"

    # ---------------- dispatch loop ----------------

    async def _dispatch_loop(self) -> None:
        """Picker with KV-aware admission for regen.

        Default (regen_priority=False): verify-first. Verifies are cheap
        and free a request; we drain verify_q before any regen.

        regen_priority=True: dispatch regens as soon as KV admits, before
        draining verify_q. Verifies still dispatch when no regen is
        pending or when regen is KV-blocked. Hypothesis: interleaving
        regens with the verify burst eliminates the regen-tail drain.
        """
        pending_regen: dict | None = None
        while not self._stop.is_set():
            verify_dispatched = False
            regen_dispatched = False

            def try_verify() -> bool:
                try:
                    req = self._verify_q.get_nowait()
                except asyncio.QueueEmpty:
                    return False
                cost = self._verify_cost(req["prompt"], req["draft_response"])
                self._kv_in_flight += cost
                self._spawn(self._drive_verify(req, cost))
                return True

            def try_regen() -> bool:
                nonlocal pending_regen
                if pending_regen is None:
                    try:
                        pending_regen = self._regen_q.get_nowait()
                    except asyncio.QueueEmpty:
                        return False
                cost = self._regen_cost(
                    pending_regen["prompt"],
                    pending_regen["max_tokens"],
                    pending_regen.get("source"),
                )
                if self._kv_in_flight + cost > self._kv_pool_threshold:
                    return False  # KV-blocked; hold pending_regen
                self._kv_in_flight += cost
                req = pending_regen
                pending_regen = None
                self._spawn(self._drive_regen(req, cost))
                return True

            if self._regen_priority:
                regen_dispatched = try_regen()
                if not regen_dispatched:
                    verify_dispatched = try_verify()
            else:
                verify_dispatched = try_verify()
                if not verify_dispatched:
                    regen_dispatched = try_regen()

            if verify_dispatched or regen_dispatched:
                continue

            # Wait for new submission or KV freed by completion.
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=0.05)
                self._wakeup.clear()
            except asyncio.TimeoutError:
                pass

    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _drive_verify(self, req: dict, cost: int) -> None:
        """Run verify; ACCEPT ships draft, REJECT routes to regen_q.

        req['prompt'] may be a string or a list[dict] of messages
        (multi-turn). _to_messages handles both."""
        try:
            from prorouter.probe_judge_verify import (
                VERIFY_QUESTION_TEMPLATE_BINARY,
            )
            template = (
                req.get("verify_template") or VERIFY_QUESTION_TEMPLATE_BINARY
            )
            bit_mode = bool(req.get("bit_mode"))
            messages = _to_messages(req["prompt"]) + [
                {"role": "assistant", "content": req["draft_response"]},
                {"role": "user", "content": template},
            ]
            formatted = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            if bit_mode:
                id_one, id_zero = _resolve_bit_token_ids(self._tokenizer)
                sp = SamplingParams(
                    max_tokens=1, temperature=0.0,
                    allowed_token_ids=[id_one, id_zero],
                )
            else:
                sp = SamplingParams(max_tokens=16, temperature=0.0)
            async for out in self._llm.generate(
                prompt=formatted,
                sampling_params=sp,
                request_id=f"verify-{req['req_id']}",
            ):
                if not out.finished:
                    continue
                text = out.outputs[0].text.strip()
                if bit_mode:
                    verdict = _parse_bit_verdict(text)
                else:
                    upper = text.upper()
                    if "ACCEPT" in upper and "REJECT" not in upper:
                        verdict = "ACCEPT"
                    elif "REJECT" in upper and "ACCEPT" not in upper:
                        verdict = "REJECT"
                    elif "ACCEPT" in upper:
                        verdict = ("ACCEPT" if upper.find("ACCEPT") <
                                   upper.find("REJECT") else "REJECT")
                    else:
                        # Conservative: ambiguous → REJECT (pay regen tax,
                        # don't ship a wrong answer).
                        verdict = "REJECT"

                # Free verify KV before downstream regen accounting.
                self._kv_in_flight -= cost
                self._wakeup.set()

                if verdict == "ACCEPT":
                    await self._finished_q.put({
                        "req_id": req["req_id"],
                        "verdict": "ACCEPT",
                        "text": req["draft_response"],
                        "judge_raw": text[:200],
                        "completed_t": time.perf_counter(),
                    })
                elif req.get("skip_regen_on_reject"):
                    # Verify-only ceiling bench: on REJECT, emit a
                    # finished item directly without regen. Lets the
                    # bench measure pure verify throughput.
                    await self._finished_q.put({
                        "req_id": req["req_id"],
                        "verdict": "REJECT",
                        "text": "",
                        "judge_raw": text[:200],
                        "completed_t": time.perf_counter(),
                    })
                else:
                    # Internal route: REJECT → own regen queue.
                    await self._regen_q.put({
                        "req_id": req["req_id"],
                        "prompt": req["prompt"],
                        "max_tokens": req["max_tokens"],
                        "ignore_eos": req.get("ignore_eos", False),
                    })
                    self._wakeup.set()
                return
        except Exception as e:
            self._kv_in_flight -= cost
            self._wakeup.set()
            await self._finished_q.put({
                "req_id": req["req_id"],
                "verdict": "ERROR",
                "stage": "verify",
                "error": f"{type(e).__name__}: {e}",
                "completed_t": time.perf_counter(),
            })

    def _build_regen_payload(self, prompt, image_paths):
        """CPU/IO-heavy target preprocessing (PIL decode + chat-template
        render) for the raw-image and text-only regen paths, factored out so
        `_drive_regen` can run it via asyncio.to_thread instead of blocking the
        actor event loop. The pre-encoded (image_embeds) path is NOT routed
        here — it stays inline in `_drive_regen` for its CUDA stream wait."""
        prompt_text = (prompt if isinstance(prompt, str) else prompt[-1]["content"])
        if image_paths:
            from PIL import Image
            images = [Image.open(p).convert("RGB") for p in image_paths]
            formatted = _build_mm_prompt_from_count(
                self._tokenizer, prompt_text=prompt_text,
                image_count=len(images),
            )
            return formatted, {"image": images}
        formatted = self._tokenizer.apply_chat_template(
            _to_messages(prompt),
            add_generation_prompt=True,
            tokenize=False,
        )
        return formatted, None

    async def _drive_regen(self, req: dict, cost: int) -> None:
        """Run regen; push completion to finished_q.

        req['prompt'] may be a string or list[dict] (multi-turn).

        Three image-input paths:
          (a) req['image_embeds'] + req['image_grid_thw'] set —
              pre-encoded path. Skips target's own ViT, consumes
              embeds via vLLM's public multi_modal_data API. Used by
              the isolation bench (`target_only_cached`) and
              the future encoder-offload pipeline (NCCL-delivered
              embeds will land here).
          (b) req['image_paths'] / req['image_path'] set — raw path(s),
              target runs its own ViT. The production cascade REGEN
              path.
          (c) neither — text-only request.
        """
        # engine-side instrumentation — actor's perf_counter at
        # the moment _drive_regen actually starts (i.e. after the
        # dispatch_loop's KV-admission gate let this request through).
        # The matching `t_actor_finish` is stamped when vLLM emits
        # out.finished. Together they give us admit-rate and
        # finish-rate ON THE ACTOR'S CLOCK, independent of bench-side
        # `pop_finished` polling artefacts. Per-request fields land on
        # the finished_q item → propagate to the Response via
        # _on_target_finished.
        t_actor_start = time.perf_counter()
        try:
            image_embeds = req.get("image_embeds")
            image_grid_thw = req.get("image_grid_thw")
            image_path = req.get("image_path")
            image_paths = req.get("image_paths")
            # Normalize single→list.
            if image_paths is None and image_path is not None:
                image_paths = [image_path]
            if image_embeds is not None and image_grid_thw is not None:
                # Pre-encoded path — vLLM's public image_embeds API.
                # See for the bit-identical-class requirement
                # (load_target_visual must use vLLM's
                # Qwen2_5_VisionTransformer, not HF's).
                #
                # isolation bench: when `embed_ready_event` is
                # set on the req, the embed tensors were produced by a
                # side-stream non-blocking copy (or in production, a
                # side-stream NCCL recv). Make the default stream wait
                # on that event before vLLM consumes — guarantees the
                # tensors are fully resident on GPU before any LM op
                # reads them, with zero blocking on this Python thread
                # (the wait is enqueued onto the GPU stream).
                ev = req.get("embed_ready_event")
                if ev is not None:
                    import torch as _torch
                    try:
                        _torch.cuda.current_stream().wait_event(ev)
                    except Exception:
                        # Best-effort. If the wait fails (e.g. CPU
                        # tests, mocked CUDA), continue — the tensors
                        # are still valid; we just lose the explicit
                        # ordering guarantee.
                        pass
                n_images = (
                    int(image_grid_thw.shape[0])
                    if hasattr(image_grid_thw, "shape") else 1
                )
                prompt_text = req["prompt"] if isinstance(req["prompt"], str) \
                    else req["prompt"][-1]["content"]
                # only the CUDA wait_event above must stay on the actor
                # thread (thread-local stream); the tokenize can go off-loop too.
                formatted = await asyncio.to_thread(
                    _build_mm_prompt_from_count,
                    self._tokenizer, prompt_text, n_images,
                )
                mm_data = {
                    "image": {
                        "image_embeds": image_embeds,
                        "image_grid_thw": image_grid_thw,
                    }
                }
            else:
                # PIL decode + chat-template render are CPU/IO-heavy;
                # run them in a worker thread so they don't block the target
                # actor's event loop (vLLM feed/drain + pop_finished). The
                # pre-encoded branch above stays inline — its CUDA stream
                # Wait_event must run on the actor thread.
                formatted, mm_data = await asyncio.to_thread(
                    self._build_regen_payload, req["prompt"], image_paths,
                )
            sp = SamplingParams(
                max_tokens=req["max_tokens"], temperature=0.0,
                ignore_eos=req.get("ignore_eos", False),
            )
            prompt_arg = (
                {"prompt": formatted, "multi_modal_data": mm_data}
                if mm_data is not None
                else formatted
            )
            async for out in self._llm.generate(
                prompt=prompt_arg,
                sampling_params=sp,
                request_id=f"regen-{req['req_id']}",
            ):
                if not out.finished:
                    continue
                t_actor_finish = time.perf_counter()
                completion = out.outputs[0]
                self._kv_in_flight -= cost
                self._wakeup.set()
                # feed the observed output length back into the
                # per-source ring so future _regen_cost reservations track
                # reality (no-op unless ma_length_gating is on).
                self._record_len(req.get("source"), len(completion.token_ids))
                await self._finished_q.put({
                    "req_id": req["req_id"],
                    "verdict": "REGEN",
                    "text": completion.text,
                    "n_output_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "completed_t": t_actor_finish,
                    "target_admit_actor_t": t_actor_start,
                    "target_finish_actor_t": t_actor_finish,
                })
                return
        except Exception as e:
            self._kv_in_flight -= cost
            self._wakeup.set()
            await self._finished_q.put({
                "req_id": req["req_id"],
                "verdict": "ERROR",
                "stage": "regen",
                "error": f"{type(e).__name__}: {e}",
                "completed_t": time.perf_counter(),
                "target_admit_actor_t": t_actor_start,
            })
