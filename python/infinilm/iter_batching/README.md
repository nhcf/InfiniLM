# iter_batching: Iterative Scheduling + Selective Batching for InfiniLM

A non-invasive enhancement package: **no existing InfiniLM file is modified**.
It implements two scheduling capabilities from Orca / Sarathi-Serve:

1. **Iteration-level scheduling (per-request phasing)**: the scheduling output
   carries the token range each request computes in this step (`spans`),
   replacing the batch-level `is_prefill`;
2. **Selective batching**:
   - chunked prefill — long prompts are sliced by the token budget instead of
     occupying a whole step;
   - mixed batching — decode requests take the budget first
     (decode-maximal), and the remaining budget is filled with prefill chunks,
     so decodes no longer stall during prefill.

## Enabling

```bash
# Master environment switch (off by default; when off, behavior is identical to the original engine).
export INFINILM_ITER_BATCHING=1
```

Importing `infinilm.iter_batching` after `import infinilm` activates it
automatically (the benchmark script `mytest/bench/mmlu_baseline.py` already
includes this import).
Programmatic API: `infinilm.iter_batching.enable(**overrides)` / `disable()`.

## Parameters (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `INFINILM_ITER_BATCHING` | `0` | Master switch |
| `INFINILM_IB_CHUNKED_PREFILL` | `1` | Chunked prefill switch |
| `INFINILM_IB_MIXED_BATCH` | `1` | Mixed batching switch (relies on chunked prefill) |
| `INFINILM_IB_MAX_BATCHED_TOKENS` | `8192` | Per-step token budget (chunk size cap) |
| `INFINILM_IB_MIXED_STEP_TOKENS` | `0` | Prefill token cap within a mixed step; `0` = unlimited (default, best throughput / TTFT tail in measurements) |
| `INFINILM_IB_PREFILL_PRIORITY` | `decode-first` | Mixed-batch budget policy |
| `INFINILM_IB_FALLBACK` | `warn` | Handling of unsupported configs: `warn` falls back / `error` refuses |

## Deployment prerequisite (important)

**The MATE flash-prefill library is required** (`LD_LIBRARY_PATH` pointing to
`.infini-mate*/lib`). In mixed batching, decode tokens also go through the
prefill operator path; the slow prefill kernel of the old library makes mixed
batching a net loss. `enable()` probes the library version at startup and
handles the old library per `INFINILM_IB_FALLBACK`.

## Limitations (first version)

- Text models only (`BasicLLMProcessor` family, e.g. Qwen3 / LLaMA);
- Mamba / linear-attention models, multimodal, speculative decoding, PP>1:
  automatically detected and fallen back to the original scheduling (or
  refused at startup with `fallback=error`);
- Mixed batches miss the decode graph and run eager (prefill already runs
  eager, so the impact is limited);
- Sampling parameters are still per-step global scalars (a pre-existing
  constraint).

## Tuning advice (measured at c16/c64/c128)

- Concurrency >= 64: enable directly — TTFT P50 drops 30–58%, throughput and
  ITL median unchanged;
- Concurrency <= 16: the gain does not cover the mixed-step bridging overhead
  (TTFT +~0.1 s); prefer `INFINILM_IB_MIXED_BATCH=0` (chunked prefill only)
  or leave it off;
- The ITL P99 tail grows at c128 (~66 -> 173 ms, P50 unchanged). If your SLO
  bounds P99, tune `INFINILM_IB_MIXED_STEP_TOKENS` (start from 2048); the tail
  is measured insensitive to the cap — the real trade-off is between TTFT and
  throughput.

## Validation

`mytest/diagnostics/iter_batching_20260921/`:

- `mixed_batch_probe.py`: kernel-level mixed-batch numerical probe (max diff
  0.0156 BF16);
- `e2e_regression.py`: token-by-token comparison of 12 greedy outputs (12/12
  identical with chunking off; with chunking on, differences are pre-existing
  batch-shape numerical noise, and first-token invariance is verified);
- `itl_stall_test.py`: instrumented verification that mixed batching
  eliminates decode stalls;
- `test_guards.py`: unit tests for config / fallback guards (no GPU needed).
