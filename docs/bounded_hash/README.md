# Bounded-hash routing on MetaX C550

The InfiniLM gateway supports explicit `bounded_hash_v3` and `round_robin`
policies for independent TP1 replicas of a dense text model. The regular
inference server keeps its default cache policy and only starts the bounded-hash
backend when given `--bounded-hash-backend-key-env`. Both gateway policies
require `prefix_lru` and reserve the full page-rounded prompt and output budget.

- [Build the pinned native stack and prepare the deployment identity](metax/BUILD.md)
- [Historical execution evidence and audit limits](metax/VALIDATION.md)
- [Source and dependency digests](metax/sources.json)

## Launch

Start each replica with the same model, build, and execution settings and one
visible physical device. Use different listener and master ports on a shared
host. The following is one replica of the historical Qwen3-8B launch shape.
Set `MODEL_DIR`, `GPU_INDEX`, `BACKEND_PORT`, and `MASTER_PORT` for the deployment.
Supply `INFINILM_REPLICA_KEY` through the deployment's secret store, and set
`INFINILM_BOUNDED_HASH_BUILD_MANIFEST` to the generated local manifest.

```bash
export MACA_VISIBLE_DEVICES="$GPU_INDEX"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export INFINILM_PRELOAD_TORCH=1
export INFINILM_MAX_NUM_BATCHED_TOKENS=8192
python -m infinilm.server.inference_server \
    --device metax --model "$MODEL_DIR" --dtype bfloat16 --tp 1 \
    --host 127.0.0.1 --port "$BACKEND_PORT" --master-port "$MASTER_PORT" \
    --enable-paged-attn --attn flash-attn --enable-graph \
    --num-blocks 128 --block-size 256 --max-batch-size 3 \
    --max-cache-len 8224 --max-new-tokens 32 --ignore-eos \
    --temperature 0 --top-p 1 --top-k 1 \
    --bounded-hash-backend-key-env INFINILM_REPLICA_KEY \
    --bounded-hash-cache-policy prefix_lru --bounded-hash-max-inflight 3 \
    --bounded-hash-hardware-id metax-runtime-verified-C550
```

Copy the [V3 example](gateway.example.json) or the [RR example](gateway.rr.example.json)
to a deployment configuration. Replace the placeholder origins, configure
backend secret environment-variable names, and set `enabled` to `true`. The
examples list two placeholder origins. The performance runs used eight replicas.

Both policies compare the configured compatibility values against the live
backend contract. Populate the `compatibility.identity` in your deployment file
from the live backend's authenticated `POST /internal/bounded-hash/contract`
response for a supported request. Use the response's `identity` object and the
actual page size, capacity, cache policy, and concurrency. The contract body
contains `request` and `prefix_tokens`.

- V3 uses [cost.template.json](cost.template.json). Supply independent deployment
  measurements for the Prefill coefficients and Decode samples. The historical
  calibration hash identifies the run input. It cannot populate a new build's
  deployment identity.
- RR uses [compatibility.template.json](compatibility.template.json). Supply the
  same compatibility fields. RR has no phase-cost model and calls the selected
  replica's contract after reserving capacity.

The examples specify 8,192 prompt tokens, a 7,936-token shared prefix, 32 output
tokens, 256-token pages, 32,768 tokens of KV capacity and at most three requests
per replica. The full reservation is 8,448 tokens. Align this contract and the
execution settings to the deployment before enabling it.

```bash
python -m infinilm.server.bounded_hash_gateway --config "$GATEWAY_CONFIG"
```

Run one gateway worker and one gateway for the replica pool. The gateway owns
admission state under its configured `state_lock`. Send text requests to
`POST /v1/chat/completions` with an explicit positive `max_tokens`. The gateway
uses the backend's tokenizer contract and forwards one generation attempt.
Admission releases follow backend request retirement. An uncertain generation
failure quarantines that backend until its instance changes.

The V3 path uses 128 virtual nodes, up to two persistent owners, a skew of
1.25, and instance-scoped residency observations with a five-second hint TTL
in these examples. Its active reuse wait uses a fixed deadline bounded by
1,000 ms and 25% of the predicted cold-to-warm Prefill savings. RR selects the
first healthy replica with enough capacity from a fixed cyclic cursor. These
policies share the scheduler, allocator, backend contract, and SSE path.
Routing code attribution is in the [routing NOTICE](../../python/infinilm/routing/bounded_hash/NOTICE).
