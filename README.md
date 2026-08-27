# Pro-Router

Reference implementation of **Pro-Router: Token-Aware Progressive Model Routing
with Adaptive Edge-Cloud Collaboration for Efficient Multimodal LLM Inference**.

Serving a large multimodal model is expensive, and most requests never needed
it. Pro-Router puts a small model on cheap edge hardware, lets it answer first,
and decides per request whether to **ship** that answer or **escalate** to the
large cloud model — using a signal the small model has already produced for
free.

```
                    ┌─────────────┐
   request ────────▶│ pre-scorer  │  TF-IDF over the prompt, no generation yet
                    └──────┬──────┘  orders the queue by likely shippability
                           ▼
                    ┌─────────────┐
                    │  scheduler  │  sorted buffer + escalation buffer
                    └──┬───────┬──┘  dispatch sized by each device's measured rate
                       ▼       ▼
              ┌────────────┐  ┌────────────┐
              │ small model│  │ large model│
              │   (edge)   │  │  (cloud)   │
              └──────┬─────┘  └─────▲──────┘
                     ▼              │
              ┌────────────┐        │
              │  verifier  │────────┘  escalate
              │ 67k params │
              └──────┬─────┘
                     ▼ ship
```

The **verifier** is a 2-layer transformer over the per-token sampling
distribution the small model emitted while decoding. It reads a `[T, 4]`
sequence — one row per generated token, columns `[chosen_logprob, max_prob,
neg_entropy, position_frac]` — and outputs one logit for "this answer is
correct". 67,329 parameters, runs on the CPU, in parallel with decoding.

That is the whole point. Every competing signal either runs a side model on the
GPU (RouteLLM's 278M router, FrugalGPT's DistilBERT) or spends extra whole
generation passes (P(True), AutoMix). Reading a byproduct of decoding costs
**2–3 ms per request** instead of 47–119 ms or seconds-to-minutes.

## Results

From the paper. Three small models routed against a shared Qwen2.5-VL-72B
target, over 15 benchmarks spanning single-image, multi-image and text.

| | Qwen2.5-VL-7B | LLaVA-OV-7B | Pixtral-12B |
|---|---|---|---|
| Verifier AUROC (pooled test) | 0.805 | 0.753 | 0.685 |

| | |
|---|---|
| Routing signal latency | 2–3 ms, **19–28× faster** than the cheapest baseline |
| End-to-end throughput vs. best baseline signal | **1.16–1.28×** (up to 2.57× vs. the weakest) |
| vs. the same policy as a Ray Serve deployment graph | **1.77–1.79×** |
| Throughput retained at 1000 ms one-way network latency | **90–96%** |
| Largest deployment (3 cloud + 4 edge devices) | **5.8–6.9×** a single cloud device |

Ablations, on the same deployment:

| variant | AUROC (Qwen) | throughput | latency |
|---|---|---|---|
| **Pro-Router (full)** | **0.805** | **124 req/s** | **754 ms** |
| hidden states instead of sampling features (3584 dims/token) | 0.758 | 95 req/s | 932 ms |
| quantile summary + MLP instead of the sequence transformer | 0.691 | 118 req/s | 783 ms |
| without batched dispatch | 0.805 | 28 req/s | 1038 ms |
| without the KV-cache admission gate | 0.805 | 112 req/s | 1634 ms |

Both pipeline optimizations are load-bearing: one remote call per request
collapses throughput 4.4×, and removing the admission gate more than doubles
latency through KV-cache preemption.

## Repository layout

```
verifier/            offline half: train the verifier, calibrate thresholds, evaluate
  run_all.sh           the five-stage pipeline, end to end
  src/                 build records, extract features, judge, train, score, eval
  build_tau_table.py   per-source threshold table
prorouter/           serving half: the Ray package
  scheduler.py         sorted buffer, escalation buffer, rate-sized dispatch
  engine.py            small/large model actors, KV-gated admission
  gate.py              the ship-or-escalate decision
  pre_router.py        pre-scorer features at request time
  head_model.py        the verifier network
  head_arch.py         ablation architectures (BiLSTM, CNN1D, AttnPool)
  tests/               23 scheduler tests, no GPU, no Ray
run_pipeline.py      run the two-tier pipeline over a record set
run_throughput_bench.py  throughput / saturation / latency-injection harness
train_prescorer.py   train the TF-IDF pre-scorer
weights/             trained verifier + threshold table
vllm_patch/          optional: compute the features inside the vLLM sampler
```

## Install

```bash
pip install -r requirements.txt
export HF_TOKEN=...          # dataset download
export OPENAI_API_KEY=...    # judge labels
```

Python 3.9+, `vllm==0.19.1`, `ray[default]`. `hardware.py` autodetects GPU
count, dtype and actor placement, so the same commands run on one GPU or across
a multi-node cluster.

## Train and evaluate the verifier

```bash
cd verifier && ./run_all.sh runs
```

Five stages: build records from the manifest, generate with the small model and
capture per-token features (**needs a GPU**), label with the judge, train the
head, evaluate. The benchmark is a five-source multimodal mix — ChartQA, DocVQA,
MathVista, MMBench, MMMU — pinned by dataset-native record id in
`bench_mix_manifest.json` (679 val / 682 test). No dataset content is
redistributed; stage 1 fetches it from the original sources.

Results land in `runs/results/verifier_accuracy.json`. `SMOKE=1` caps stage 2 at
32 records per split for a fast path through all five stages.

Then train the pre-scorer on the same split:

```bash
python train_prescorer.py \
    --answers verifier/runs/features/bench_val_answers.jsonl \
    --labels  verifier/runs/labels/val.json \
    --out     weights/prescorer.pkl
```

## Run the pipeline

```bash
python run_pipeline.py --records bench_test.jsonl \
  --small-model Qwen/Qwen2.5-VL-7B-Instruct \
  --large-model Qwen/Qwen2.5-VL-72B-Instruct \
  --head weights/head.pt --tau weights/tau.json \
  --scorer weights/prescorer.pkl --concurrency 256
```

Needs a running Ray cluster. For a GPU-light smoke test that still routes every
request through the verifier:

```bash
python run_pipeline.py --records bench_test.jsonl --mock-large --force-cascade
```

**`--mock-large` on its own is not a smoke test.** The stub reports 0% GPU
utilization and an empty queue, so backpressure spills every request past the
small model and the verifier never executes — while the run still reports 0
errors. `--force-cascade` is what routes through it. This cost three external
test rounds to find.

Throughput and scaling numbers come from the bench harness, which drives one
isolated configuration per invocation:

```bash
python run_throughput_bench.py --cell cascade_gate \
  --records bench_test.jsonl --label my_run --out-dir runs/
```

`--cell target_only` and `--cell draft_only` measure each tier alone, so the
pipeline's throughput can be checked against the sum of its parts.

## Placement on a cluster

`prorouter/launcher.py` holds two node-level resource keys, both empty by
default so placement is unconstrained until you name a resource your cluster
actually advertises:

```bash
DRAFT_RESOURCE=<small-model node label> \
TARGET_RESOURCE=<large-model node label> python run_pipeline.py ...
```

`--small-resource` / `--large-resource` override them for a single run. The
paper's deployment is one 8×A100 node for the 72B model and N 4×A10G nodes for
the small tier.

## Thresholds are per-deployment

`weights/tau.json` was calibrated for one model pair on one benchmark mix.
**It does not transfer.** Reused elsewhere it sits below the whole score
distribution, the router ships ~93% of requests, and the ship rate degenerates
to the small model's base rate — which any head reproduces. Recalibrate on your
own validation split with `verifier/build_tau_table.py` or
`verifier/src/calibrate_tau.py`. `run_pipeline.py` warns per source when the
escalation rate leaves the 5–95% band.

## Tests

```bash
python -m prorouter.tests.test_scheduler_unit
```

23 scheduler tests covering dispatch, calibration drift, escalation routing,
backpressure and the admission gate. Stub actors implement the subset of the
actor interface the scheduler uses, so no GPU and no Ray are required.

## Naming

The paper and the code grew up at different times, so some names differ:

| paper | code |
|---|---|
| token-aware verifier | confidence head, `head`, `gate` |
| ship / escalate | `ACCEPT` / `REGEN` |
| small model (edge) | `draft` |
| large model (cloud) | `target` |
| routing | `cascade` |

## Citing

See [CITATION.cff](CITATION.cff).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The vLLM patch
derives from Apache-2.0 code; no dataset content or third-party model weights
are redistributed.
