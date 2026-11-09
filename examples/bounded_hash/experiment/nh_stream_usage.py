"""CPU cache telemetry and opt-in cache control for existing nhcf/InfiniLM.

Adds exact token usage and observed cache residency to SSE responses. Cache
snapshots are consistent at observation time, not promises of future residency.
All InfiniLM policies use the same wrapper. No tensor reads, GPU synchronization,
model changes, or batch-scheduler changes are introduced. The matrix wrapper
additionally installs cache_policy.py: native mode preserves the allocator;
prefix_lru mode changes idle-block eviction only, between drained test cells.

The request's optional private ``_fourway_prefix`` metadata is verified against
the engine's actual prompt tokens. Only first real text and final usage events
carry a cache snapshot; ordinary token events carry the generation count.
"""
from collections import OrderedDict
from contextvars import ContextVar
import functools
import hashlib
import json
import threading


CURRENT = ContextVar('fourway_request', default=None)


class PrefixObservations:
    """A bounded directory of known prefixes; residency comes from BlockManager.

    ``lock`` protects both the directory and scheduler CPU cache operations.
    The full returned map replaces previous observations at the receiver. Zero
    means a known prefix is no longer resident; an omitted prefix is unknown.
    """

    def __init__(self, max_entries=256):
        if max_entries < 1:
            raise ValueError('max_entries must be positive')
        self.lock = threading.RLock()
        self.max_entries = max_entries
        self.known = OrderedDict()
        self.epoch = 0

    def register(self, req, metadata, block_size):
        if metadata is None:
            return
        if not isinstance(metadata, dict):
            raise ValueError('_fourway_prefix must be an object')
        tokens, declared = metadata.get('tokens'), metadata.get('digest')
        if (type(tokens) is not int or tokens <= 0 or tokens % block_size
                or tokens >= req.get_prompt_length()):
            raise ValueError('_fourway_prefix tokens must be an aligned proper prefix')
        actual = hashlib.sha256(json.dumps(
            [int(t) for t in req.prompt_token_ids[:tokens]],
            ensure_ascii=False, separators=(',', ':'),
        ).encode()).hexdigest()
        if not isinstance(declared, str) or declared != actual:
            raise ValueError('_fourway_prefix digest does not match actual prompt tokens')
        count = tokens // block_size
        hashes = tuple(req.block_hashes[:count])
        if len(hashes) != count or any(not h for h in hashes):
            raise ValueError('Engine did not initialize the declared prefix block hashes')
        with self.lock:
            self.known.pop(actual, None)
            self.known[actual] = hashes
            while len(self.known) > self.max_entries:
                self.known.popitem(last=False)

    def snapshot(self, scheduler):
        with self.lock:
            manager = scheduler.cache_manager
            prefixes = {}
            for digest, hashes in self.known.items():
                resident = 0
                if scheduler.enable_prefix_caching:
                    for block_hash in hashes:
                        ids = manager.hash_to_block_ids.get(block_hash, ())
                        # Inspect only. get_computed_blocks() would PIN blocks,
                        # perturb ownership, and must not be used by telemetry.
                        present = any(
                            block_id in manager.used_block_ids
                            and manager.blocks[block_id].hash == block_hash
                            for block_id in ids
                        )
                        if not present:
                            break
                        resident += manager.block_size
                prefixes[digest] = resident
            self.epoch += 1
            return {'epoch': self.epoch, 'prefixes': prefixes}


def install(server_module, Scheduler):
    if getattr(Scheduler, '_fourway_cache_telemetry_installed', False):
        raise RuntimeError('Cache telemetry is already installed')
    old_init = Scheduler.__init__
    old_add = server_module.AsyncLLMEngine.add_chat_request
    old_schedule = Scheduler.schedule
    old_stream = server_module.InferenceServer._stream_chat

    @functools.wraps(old_init)
    def init(self, *args, **kwargs):
        self._fourway_observations = PrefixObservations()
        old_init(self, *args, **kwargs)

    @functools.wraps(old_add)
    def add(self, *args, **kwargs):
        req = old_add(self, *args, **kwargs)
        CURRENT.set(req)
        return req

    @functools.wraps(old_schedule)
    def schedule(self, *args, **kwargs):
        with self._fourway_observations.lock:
            result = old_schedule(self, *args, **kwargs)
            if result is not None and result.is_prefill:
                for req in result.scheduled_requests:
                    if not hasattr(req, '_fourway_cached'):
                        # Capture initial hit length before decode mutates it.
                        req._fourway_cached = int(req.num_local_cached_tokens)
                        req._fourway_cache_enabled = bool(self.enable_prefix_caching)
                        req._fourway_prefills = 0
                    req._fourway_prefills += 1
            return result

    def locked(method):
        @functools.wraps(method)
        def call(self, *args, **kwargs):
            with self._fourway_observations.lock:
                return method(self, *args, **kwargs)
        return call

    @functools.wraps(old_stream)
    async def stream(self, request_id, data, http_request):
        token = CURRENT.set(None)
        upstream = old_stream(self, request_id, data, http_request)
        observed_first_text = False
        try:
            async for event in upstream:
                req = CURRENT.get()
                if event.strip() == 'data: [DONE]':
                    if req is not None:
                        scheduler = self.engine.engine.scheduler
                        state = scheduler._fourway_observations
                        p, o = req.get_prompt_length(), req.get_num_generated_tokens()
                        usage = dict(prompt_tokens=p, completion_tokens=o, total_tokens=p + o,
                                     prompt_tokens_details={'cached_tokens': getattr(req, '_fourway_cached', None)})
                        info = dict(generated_tokens=o,
                                    prefix_caching=getattr(req, '_fourway_cache_enabled', None),
                                    prefill_events=getattr(req, '_fourway_prefills', 0),
                                    cache_policy=getattr(scheduler.cache_manager, '_matrix_mode', 'native'),
                                    idle_evictions=getattr(scheduler.cache_manager, '_matrix_evictions', 0),
                                    cache_snapshot=state.snapshot(scheduler))
                        yield 'data: ' + json.dumps(dict(id=request_id, choices=[], usage=usage,
                                                        x_infinilm=info)) + '\n\n'
                    yield event
                elif req is not None and event.startswith('data: '):
                    obj = json.loads(event[6:])
                    info = {'generated_tokens': req.get_num_generated_tokens()}
                    real_text = (not obj.get('error') and info['generated_tokens'] > 0 and any(
                        choice.get('delta', {}).get('content')
                        and choice.get('finish_reason') not in ('error', 'timeout', 'cancelled', 'canceled')
                        for choice in obj.get('choices', [])
                    ))
                    if real_text and not observed_first_text:
                        scheduler = self.engine.engine.scheduler
                        state = scheduler._fourway_observations
                        with state.lock:
                            state.register(req, data.get('_fourway_prefix'), scheduler.block_size)
                            info['cache_snapshot'] = state.snapshot(scheduler)
                        observed_first_text = True
                    obj['x_infinilm'] = info
                    yield 'data: ' + json.dumps(obj, ensure_ascii=False) + '\n\n'
                else:
                    yield event
        finally:
            try:
                await upstream.aclose()
            finally:
                CURRENT.reset(token)

    Scheduler.__init__ = init
    Scheduler.schedule = schedule
    # These CPU methods cover BlockManager mutations in the full-replica
    # execution path. Nested calls are permitted by RLock. GPU execute_model()
    # lives in LLMEngine.step() BETWEEN these calls and is never locked here.
    for name in ('commit_computed_tokens', 'complete_requests',
                 'update_waiting_for_remote_kv', 'update_from_output'):
        if hasattr(Scheduler, name):
            setattr(Scheduler, name, locked(getattr(Scheduler, name)))
    server_module.AsyncLLMEngine.add_chat_request = add
    server_module.InferenceServer._stream_chat = stream
    Scheduler._fourway_cache_telemetry_installed = True


def main():
    import os
    if any(k in os.environ for k in ('INFINILM_DIAGNOSTICS_DIR', 'INFINILM_DIAGNOSTICS_LOGITS', 'INFINILM_DIAGNOSTICS_KV')):
        raise RuntimeError('Remove tensor diagnostics variables before timing')
    from infinilm.server import inference_server
    from infinilm.llm.scheduler import Scheduler
    if os.environ.get('FOURWAY_EXPECT_NATIVE_DIR'):
        from pathlib import Path
        import _infinilm
        expected = Path(os.environ['FOURWAY_EXPECT_NATIVE_DIR']).resolve()
        actual = Path(_infinilm.__file__).resolve()
        if actual not in {p.resolve() for p in expected.glob('_infinilm*.so')}:
            raise RuntimeError('Unexpected NH native binding: ' + str(actual) + '; expected under ' + str(expected))
        print('FOURWAY_NATIVE_BINDING=' + str(actual), flush=True)
    install(inference_server, Scheduler)
    from infinilm.llm.cache_manager import BlockManager
    from cache_policy import install_control
    install_control(inference_server, BlockManager)
    inference_server.main()


if __name__ == '__main__':
    main()
