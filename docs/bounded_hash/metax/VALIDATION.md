# Historical validation on MetaX C550

The 2026-11-17 performance runs used Qwen3-8B, BF16 weights and KV, and eight
MetaX C550 devices with one full TP1 replica per device. Qwen3-32B remained
blocked by TP1 memory capacity and did not complete the target experiment.

## Source correspondence

The release starts at the old B checkout
`82b5687a513778eaec4f3f23664bf4a2189847af`. That commit descends from the
[frozen InfiniLM upstream base](https://github.com/InfiniTensor/InfiniLM/commit/df54bf0e621595cea9dd57eb4ebb508918127967)
and includes the original [V3 integration](https://github.com/InfiniTensor/InfiniLM/pull/616),
MetaX SDK paths and build identity, Torch-first bootstrap, and explicit stop-string
handling when EOS is ignored. The V3 chain is attributed in the
[routing NOTICE](../../../python/infinilm/routing/bounded_hash/NOTICE).

The later D/B batch reused the earlier B native build and its independent
calibration. Its frozen Python source adds `round_robin_router.py` and changes
`bounded_hash_config.py` and `bounded_hash_gateway.py`. The scheduler,
`prefix_lru` allocator, backend contract, and V3 routing modules match B.
The dependent InfiniOps source is recoverable with the three ordered patches.
The [source manifest](sources.json) records the historical run identifiers,
frozen evidence and runtime source digests, public dependency pins, and
tested trees.

This release adds deployment documentation, placeholder configurations, an
adaptation of the original local build-identity generator, and CPU regression
sources. The release checks covered source equality, patch application and
static syntax. There was no new CPU test-suite execution, dependency install,
build, model run, GPU test, benchmark, or calibration for the release commit.

## D/B paired experiment with shared prefix-LRU

Each policy ran three units with 64 learning and 128 measured requests per
unit. Both used concurrency 16, 8,192 input tokens, a 7,936-token shared prefix,
32 output tokens, and the same `prefix_lru` backend with 256-token pages,
32,768 KV tokens and a three-request cap per replica. The formal target prefix
started cold each unit. Separate preflight warmup left unrelated idle pages
resident.

| Policy | Measured output tok/s | Measured TTFT p95 (s) | Learning + measured output tok/s | Learning + measured TTFT p95 (s) |
| --- | ---: | ---: | ---: | ---: |
| D, RR | 340.060 | 0.960 | 314.958 | 1.339 |
| B, V3 | 505.066 | 0.834 | 446.880 | 1.431 |

V3 measured throughput rose 48.52% across the paired units. V3's p95 TTFT
**increased 6.86%** when the learning phase was included. Throughput is total
output tokens over total phase wall time. The measured gain is for the complete
policies, including one contract RPC for RR and eight for V3. The result covers
three paired rounds on one workload and server.

## Earlier A/B/vLLM batch

This separately frozen batch compared A (native InfiniLM + RR), B (InfiniLM
V3 + prefix-LRU), and C (vLLM-MetaX + least-active). Its B result belongs to
this batch.

| Method | Measured output tok/s | Measured TTFT p95 (s) | Learning + measured output tok/s | Learning + measured TTFT p95 (s) |
| --- | ---: | ---: | ---: | ---: |
| A | 293.588 | 1.203 | 282.406 | 1.313 |
| B | 518.625 | 0.833 | 447.367 | 1.374 |
| C | 388.960 | 1.268 | 353.902 | 1.283 |

B's p95 TTFT with learning was higher than A's and C's. The earlier
B/C result is a whole-stack comparison at fixed cache capacity and concurrency.
The vLLM-MetaX cache was capped at 128 blocks of its 797-block natural profile.
Native InfiniLM logged device-graph warnings despite equivalent D/B launch
flags. Effective graph execution was unconfirmed across the earlier stacks.
The later D/B batch gives no paired comparison to vLLM.

## Correctness evidence and open limits

- Historical runs recorded passes for the native MetaX operator probe, selected
  single-device Qwen3-8B cold/warm and EOS cases, and the two-replica V3
  contract, retirement, and fault/restart lifecycle. In the D/B batch, selected
  RR, V3, prefix-LRU and policy-selection CPU regressions passed. The audit
  also checked all 576 timed RR routes against first-eligible cyclic admission.
- The independent audit recomputed D/B throughput and client latency from
  1,152 observations, checked request and SSE integrity, and independently
  rebuilt the 192 unique workload prompts with the tokenizer. Forty-eight
  D/B preflights returned matching generated token IDs. The earlier selected
  cold/warm target also had matching IDs.
- The 1,152 timed D/B responses retained no generated token IDs, logits, or
  independent references. Some paired response texts differed. Full timed
  generation and cache semantics remain unverified. The old C timed token-ID
  payloads were null despite the preservation label. Those C records support
  usage-based metrics only.
- Cache hits are scheduler-reported counts. D and B use different boundaries
  for the diagnostic `queue_ms`, so compare client TTFT or end-to-end latency.
  Phase wall time was derived from the client's saved phase boundaries.
- The old snapshot packaging copied the blocked 32B model/config metadata into
  the later run's source snapshot. The frozen D/B configuration and all 48
  actual backend launches identify the completed 8B run. The source lock's
  `base_git_status` label recorded the D/B copy, as noted by the original
  correction. The audit established the three-file runtime delta. Historical
  imports lacked a complete module and environment inventory, and the model
  manifest did not hash all tensor payload bytes.
