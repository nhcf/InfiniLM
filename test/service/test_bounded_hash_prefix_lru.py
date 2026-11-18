"""CPU regressions for the prefix-LRU scheduler and gateway retirement."""

import sys
import unittest
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import httpx


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "python"


def load_test_modules():
    """Use the checkout's Python modules without importing the native engine."""
    packages = {}
    for name in ("infinilm", "infinilm.llm", "infinilm.processors", "infinilm.routing", "infinilm.server"):
        package = ModuleType(name)
        package.__path__ = [str(PACKAGE_ROOT.joinpath(*name.split(".")))]
        packages[name] = package
    with patch.dict(sys.modules, packages):
        return (
            import_module("infinilm.llm.cache_manager"),
            import_module("infinilm.llm.prefix_cache"),
            import_module("infinilm.llm.request"),
            import_module("infinilm.llm.sampling_params"),
            import_module("infinilm.llm.scheduler"),
            import_module("infinilm.server.bounded_hash_backend"),
            import_module("infinilm.server.bounded_hash_gateway"),
        )


cache_manager, prefix_cache, request, sampling, scheduler, backend, gateway = load_test_modules()
BlockManager = cache_manager.BlockManager
hash_block_tokens = prefix_cache.hash_block_tokens
FinishReason = request.FinishReason
InferenceRequest = request.InferenceRequest
SamplingParams = sampling.SamplingParams
Scheduler = scheduler.Scheduler


class AllocatorAndScheduler(unittest.TestCase):
    def test_snapshot_does_not_pin_and_live_pages_survive_idle_eviction(self):
        manager = BlockManager(3, 256)
        manager.configure_routing_cache_policy('prefix_lru')
        blocks, _ = manager.allocate_slots(512)
        hashes = [hash_block_tokens([1] * 256), hash_block_tokens([2] * 256)]
        manager.publish_computed_blocks(blocks, hashes, 0, 512)
        manager.free_blocks(blocks)
        pinned, _ = manager.get_computed_blocks(hashes[:1], 256)
        observations = backend.PrefixObservations('instance')
        observations.register('prefix', hashes)
        before = [block.ref_count for block in manager.blocks]
        observations.publish(manager)
        counts = observations.snapshot()['cpu_allocator']
        self.assertEqual(counts['total_blocks'], 3)
        self.assertEqual(counts['in_use_ref_positive_blocks'], 1)
        self.assertEqual(counts['reference_count_sum'], 1)
        self.assertEqual(counts['idle_resident_ref0_blocks'], 1)
        self.assertEqual(before, [block.ref_count for block in manager.blocks])
        allocated, _ = manager.allocate_slots(512)
        self.assertNotIn(pinned[0], allocated)
        self.assertEqual(manager.blocks[pinned[0]].hash, hashes[0])
        manager.free_blocks(pinned)
        manager.free_blocks(allocated)
        self.assertTrue(manager.try_free_blocks(3))
        self.assertEqual(len(manager.free_block_ids), 3)
        self.assertFalse(manager.used_block_ids)
        self.assertFalse(manager.hash_to_block_ids)

    def test_actual_scheduler_freezes_initial_hit_and_retires_cancel_once(self):
        scheduler = Scheduler(max_batch_size=1, num_blocks=4, block_size=256, enable_prefix_caching=True)
        scheduler.cache_manager.configure_routing_cache_policy('prefix_lru')
        observer = backend.PrefixObservations('scheduler-fixture')
        scheduler.bounded_hash_observer = observer
        tokens = [1] * 256 + [2] * 256 + [3]
        first = InferenceRequest('first', prompt_token_ids=tokens, sampling_params=SamplingParams(max_tokens=1))
        scheduler.add_request(first)
        scheduler.schedule()
        self.assertEqual(first.initial_local_cached_tokens, 0)
        scheduler.commit_computed_tokens(first, len(tokens))
        first.mark_finished(FinishReason.LENGTH)
        scheduler.complete_requests([first])
        second = InferenceRequest('second', prompt_token_ids=tokens, sampling_params=SamplingParams(max_tokens=1))
        scheduler.add_request(second)
        scheduler.schedule()
        self.assertEqual(second.initial_local_cached_tokens, 512)
        second.num_local_cached_tokens = 513
        second.capture_initial_cached_tokens()
        self.assertEqual(second.initial_local_cached_tokens, 512)
        second.mark_canceled()
        scheduler.complete_requests([second])
        refs = [block.ref_count for block in scheduler.cache_manager.blocks]
        scheduler.complete_requests([second])
        self.assertEqual(refs, [block.ref_count for block in scheduler.cache_manager.blocks])
        self.assertTrue(all(value == 0 for value in refs))
        self.assertEqual(second._paged_completion_state, 'done')
        self.assertEqual(observer.snapshot()['cpu_allocator']['reference_count_sum'], 0)
        self.assertEqual(observer.snapshot()['cpu_allocator']['in_use_ref_positive_blocks'], 0)


class GatewayRetirement(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.gateway = gateway.Gateway(SimpleNamespace(urls=['http://cpu-fixture'], phase_model={
            'schema_version': 1, 'prefill_coef': [1, 1, 0], 'decode_step_ms': {'1': 1}, 'decode_reference_context_tokens': 1000},
            routing_policy='bounded_hash_v3',
            max_inflight=1, kv_tokens=1024, page_size=256, skew=1.25, virtual_nodes=128,
            cache_ttl_s=5, queue_timeout_s=5, warm_wait_ms=1000, keys=['fixture']))
        self.gateway.router.observe_backend(0, 'instance')
        self.item = {'id': 'test', 'key': 'test', 'prompt_tokens': 256, 'output_tokens': 1, 'shared_tokens': 0, 'prefix_digest': None}
        self.contracts = [{'backend_instance_id': 'instance', 'token_digest': 'digest', 'shared_tokens': 0}]
        self.posts = 0

    def tearDown(self):
        self.gateway.router.close()

    async def run_stream(self, raw, close_early=False):
        def handler(request):
            self.posts += 1
            return httpx.Response(200, headers={'X-InfiniLM-Backend-Instance': 'instance'}, text=raw)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            self.gateway.client = client
            stream = self.gateway.generate({}, self.item, self.contracts)
            if close_early:
                await anext(stream)
                await stream.aclose()
            else:
                async for _ in stream:
                    pass

    async def test_client_cancel_after_admission_quarantines_without_post_replay(self):
        await self.run_stream('data: {"choices":[{"delta":{"content":"x"}}]}\n\n', close_early=True)
        self.assertEqual(self.posts, 1)
        self.assertEqual(self.gateway.router.reserved, [0])
        self.assertFalse(self.gateway.router.active)
        self.assertEqual(self.gateway.quarantined, {0: 'instance'})

    async def test_incomplete_response_releases_once_and_quarantines(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            await self.run_stream('data: {"choices":[]}\n\n')
        self.assertEqual(self.posts, 1)
        self.assertEqual(self.gateway.router.reserved, [0])
        self.assertFalse(self.gateway.router.active)
        self.assertFalse(self.gateway.router.workers[0].healthy)

    async def test_successful_stream_reports_admission_without_counting_it_as_content(self):
        raw = ('data: {"id":"result","choices":[{"delta":{"content":"x"},"finish_reason":"length"}]}\n\n'
               'data: {"id":"result","choices":[],"usage":{"prompt_tokens":256,"completion_tokens":1}}\n\n'
               'data: [DONE]\n\n')
        observed = []
        def handler(request):
            self.posts += 1
            return httpx.Response(200, headers={'X-InfiniLM-Backend-Instance': 'instance'}, text=raw)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            self.gateway.client = client
            async for event in self.gateway.generate({}, self.item, self.contracts):
                observed.append(event)
        admission = observed[-1]
        self.assertFalse(admission['choices'])
        self.assertEqual(admission['x_admission']['backend'], 0)
        self.assertEqual(admission['x_admission']['peak_reserved_tokens'], [512])
        self.assertEqual(admission['x_admission']['peak_inflight'], [1])
        self.assertEqual(self.posts, 1)
        self.assertEqual(self.gateway.router.reserved, [0])
        self.assertTrue(self.gateway.router.workers[0].healthy)
