"""CPU-only tests of observed cache residency and stream metadata contracts."""
import asyncio
import hashlib
import json
import threading
import types
import unittest

from nh_stream_usage import CURRENT, PrefixObservations, install


def digest(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode()).hexdigest()


def request(start=0):
    class Request:
        prompt_token_ids = tuple(range(start, start + 16))
        block_hashes = tuple(bytes([start + i + 1]) for i in range(4))
        num_local_cached_tokens = 4
        generated = 0

        def get_prompt_length(self):
            return len(self.prompt_token_ids)

        def get_num_generated_tokens(self):
            return self.generated
    return Request()


def manager(req):
    return types.SimpleNamespace(
        block_size=4,
        blocks=[types.SimpleNamespace(hash=h, ref_count=1) for h in req.block_hashes],
        used_block_ids=set(range(4)),
        hash_to_block_ids={h: {i} for i, h in enumerate(req.block_hashes)},
    )


def fake_server(req, *, evict=True):
    class Scheduler:
        def __init__(self):
            self.block_size = 4
            self.enable_prefix_caching = True
            self.cache_manager = manager(req)

        def schedule(self):
            return types.SimpleNamespace(is_prefill=True, scheduled_requests=[req])

        def commit_computed_tokens(self, *args):
            # Real code publishes complete blocks here, before a text event.
            pass

        def complete_requests(self, *args):
            self.cache_manager.hash_to_block_ids.pop(req.block_hashes[0], None)

        def update_from_output(self, *args):
            pass

    class AsyncLLMEngine:
        def __init__(self, scheduler):
            self.engine = types.SimpleNamespace(scheduler=scheduler)

        def add_chat_request(self):
            self.engine.scheduler.schedule()
            return req

    class InferenceServer:
        def __init__(self, scheduler):
            self.engine = AsyncLLMEngine(scheduler)
            self.closed = False

        async def _stream_chat(self, request_id, data, http_request):
            self.engine.add_chat_request()
            try:
                yield 'data: ' + json.dumps({'choices': [{'delta': {'role': 'assistant'}}]}) + '\n\n'
                # Force concurrent collectors to interleave across ContextVars.
                await asyncio.sleep(0)
                req.generated = 1
                self.engine.engine.scheduler.commit_computed_tokens(req, 16)
                yield 'data: ' + json.dumps({'choices': [{'delta': {'content': 'a'}}]}) + '\n\n'
                req.generated = 2
                req.num_local_cached_tokens = 16
                yield 'data: ' + json.dumps({'choices': [{'delta': {'content': 'b'}}]}) + '\n\n'
                req.generated = 3
                if evict:
                    self.engine.engine.scheduler.complete_requests([req])
                yield 'data: ' + json.dumps({'choices': [{'delta': {}, 'finish_reason': 'length'}]}) + '\n\n'
                yield 'data: [DONE]\n\n'
            finally:
                self.closed = True

    module = types.SimpleNamespace(AsyncLLMEngine=AsyncLLMEngine, InferenceServer=InferenceServer)
    install(module, Scheduler)
    scheduler = Scheduler()
    return InferenceServer(scheduler), scheduler


class CacheDirectoryTests(unittest.TestCase):
    def test_snapshot_inspects_residency_without_pinning(self):
        req = request()
        scheduler = types.SimpleNamespace(cache_manager=manager(req), enable_prefix_caching=True)
        state = PrefixObservations()
        key = digest(req.prompt_token_ids[:8])
        state.register(req, {'digest': key, 'tokens': 8}, 4)
        refs = [block.ref_count for block in scheduler.cache_manager.blocks]
        self.assertEqual(state.snapshot(scheduler), {'epoch': 1, 'prefixes': {key: 8}})
        self.assertEqual([b.ref_count for b in scheduler.cache_manager.blocks], refs)
        # Later blocks alone cannot make a valid contiguous prefix.
        scheduler.cache_manager.hash_to_block_ids.pop(req.block_hashes[0])
        self.assertEqual(state.snapshot(scheduler), {'epoch': 2, 'prefixes': {key: 0}})

    def test_actual_prompt_digest_and_alignment_are_checked(self):
        state, req = PrefixObservations(), request()
        for metadata in ({'digest': '0' * 64, 'tokens': 8},
                         {'digest': digest(req.prompt_token_ids[:7]), 'tokens': 7},
                         {'digest': digest(req.prompt_token_ids), 'tokens': 16},
                         {'digest': 'wrong', 'tokens': True}, 'not-an-object'):
            with self.assertRaises(ValueError):
                state.register(req, metadata, 4)
        self.assertFalse(state.known)

    def test_registry_is_bounded_and_unknown_is_not_resident(self):
        state = PrefixObservations(max_entries=2)
        keys = []
        for offset in (0, 10, 20):
            req = request(offset)
            key = digest(req.prompt_token_ids[:8])
            keys.append(key)
            state.register(req, {'digest': key, 'tokens': 8}, 4)
        self.assertEqual(list(state.known), keys[1:])
        req = request(30)
        scheduler = types.SimpleNamespace(cache_manager=manager(req), enable_prefix_caching=True)
        self.assertEqual(set(state.snapshot(scheduler)['prefixes'].values()), {0})

    def test_scheduler_mutation_and_snapshot_share_lock(self):
        req = request()
        server, scheduler = fake_server(req)
        state = scheduler._fourway_observations
        key = digest(req.prompt_token_ids[:8])
        state.register(req, {'digest': key, 'tokens': 8}, 4)
        read_started, read_done = threading.Event(), threading.Event()
        result = []

        def reader():
            read_started.set()
            result.append(state.snapshot(scheduler))
            read_done.set()

        with state.lock:
            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            self.assertTrue(read_started.wait(1))
            self.assertFalse(read_done.wait(0.03))
            # Nested scheduling methods must not deadlock under the same RLock.
            scheduler.complete_requests([req])
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0]['prefixes'][key], 0)


class StreamTelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_real_text_and_final_snapshot_exact_usage(self):
        req = request()
        server, scheduler = fake_server(req)
        key = digest(req.prompt_token_ids[:8])
        data = {'_fourway_prefix': {'digest': key, 'tokens': 8}}
        events = [event async for event in server._stream_chat('req-1', data, None)]
        objects = [json.loads(e[6:]) for e in events if e.strip() != 'data: [DONE]']
        self.assertNotIn('cache_snapshot', objects[0]['x_infinilm'])
        self.assertEqual(objects[1]['x_infinilm']['cache_snapshot']['prefixes'], {key: 8})
        self.assertNotIn('cache_snapshot', objects[2]['x_infinilm'])
        usage = objects[-1]
        self.assertEqual(usage['usage'], {'prompt_tokens': 16, 'completion_tokens': 3,
                                         'total_tokens': 19, 'prompt_tokens_details': {'cached_tokens': 4}})
        self.assertEqual(usage['x_infinilm']['cache_snapshot'], {'epoch': 2, 'prefixes': {key: 0}})
        self.assertEqual(usage['x_infinilm']['prefill_events'], 1)
        self.assertTrue(server.closed)
        self.assertIsNone(CURRENT.get())

    async def test_digest_mismatch_closes_upstream_and_publishes_nothing(self):
        req = request()
        server, scheduler = fake_server(req)
        received = []
        with self.assertRaisesRegex(ValueError, 'actual prompt tokens'):
            async for event in server._stream_chat('bad', {'_fourway_prefix': {'digest': 'bad', 'tokens': 8}}, None):
                received.append(event)
        self.assertEqual(len(received), 1)  # Only the role event got through.
        self.assertTrue(server.closed)
        self.assertFalse(scheduler._fourway_observations.known)
        self.assertIsNone(CURRENT.get())

    async def test_direct_probe_without_private_metadata_is_supported(self):
        server, scheduler = fake_server(request())
        events = [event async for event in server._stream_chat('probe', {}, None)]
        final = json.loads(events[-2][6:])
        self.assertEqual(final['x_infinilm']['cache_snapshot']['prefixes'], {})
        self.assertEqual(final['usage']['completion_tokens'], 3)

    async def test_distinct_stream_contexts_do_not_mix_token_usage(self):
        a, _ = fake_server(request(0), evict=False)
        b, _ = fake_server(request(20), evict=False)

        async def collect(server, offset):
            req = request(offset)
            data = {'_fourway_prefix': {'digest': digest(req.prompt_token_ids[:8]), 'tokens': 8}}
            return [e async for e in server._stream_chat(str(offset), data, None)]

        results = await asyncio.gather(collect(a, 0), collect(b, 20))
        for events, offset in zip(results, (0, 20)):
            final = json.loads(events[-2][6:])
            self.assertEqual(final['id'], str(offset))
            self.assertEqual(list(final['x_infinilm']['cache_snapshot']['prefixes']),
                             [digest(request(offset).prompt_token_ids[:8])])
        self.assertIsNone(CURRENT.get())


if __name__ == '__main__':
    unittest.main()
