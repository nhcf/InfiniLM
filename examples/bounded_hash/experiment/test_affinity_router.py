"""CPU-only admission, accounting, cache-observation and routing regressions.

Run: python -m unittest -v test_affinity_router.py
These validate protocol invariants, not GPU performance predictions.
"""
import asyncio
import time
import unittest

from affinity_router import PrefillModel, Router


def item(i=0, key=None, prompt=8192, output=64, shared=7680, digest='doc'):
    return dict(id=str(i), key=str(i) if key is None else key,
                prompt_tokens=prompt, output_tokens=output,
                shared_tokens=shared, prefix_digest=digest)


def event(epoch, prefixes, content='x', generated=1):
    return dict(choices=[dict(delta=dict(content=content))],
                x_infinilm=dict(generated_tokens=generated,
                                cache_snapshot=dict(epoch=epoch, prefixes=prefixes)))


def router(policy='bounded', count=4, **kwargs):
    return Router(['http://worker'+str(i) for i in range(count)], policy, **kwargs)


def owner(r, key):
    return r.index[next(r.ring.walk(key)).member_id]


def key_for(r, worker):
    # Test fixture only: production workloads must never select documents by
    # their hash destinations, which would bias a benchmark.
    for i in range(10000):
        key = 'fixture-'+str(i)
        if owner(r, key) == worker:
            return key
    raise AssertionError('Could not construct deterministic fixture')


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_least_active_rotates_equal_loads(self):
        r = router('least_active')
        leases = [await r.acquire(item(i)) for i in range(8)]
        self.assertEqual([x.worker for x in leases], [0, 1, 2, 3, 0, 1, 2, 3])
        for x in leases:
            r.release(x, True)
        self.assertEqual(r.snapshot()['active'], [0]*4)

    async def test_outside_scope_is_exact_least_active_even_without_profile(self):
        for policy in ('bounded', 'cost_bounded'):
            for p, o, shared in ((2048, 64, 1536), (8192, 512, 7680),
                                 (8192, 64, 0), (8192, 64, 4096)):
                r = router(policy)
                baseline = router('least_active')
                for i in range(8):
                    request = item(i, prompt=p, output=o, shared=shared)
                    a, b = await r.acquire(request), await baseline.acquire(request)
                    self.assertEqual(a.worker, b.worker)
                    self.assertEqual(a.route['branch'], 'least_active_outside_affinity')
                    self.assertNotIn('bounded_candidates', a.route)
                    if i % 2:
                        r.release(a, True)
                        baseline.release(b, True)

    async def test_missing_profile_is_explicit_least_active(self):
        r = router('cost_bounded')
        leases = [await r.acquire(item(i)) for i in range(4)]
        self.assertEqual([x.worker for x in leases], [0, 1, 2, 3])
        self.assertTrue(all(x.route['branch'] == 'least_active_missing_cost_profile'
                            for x in leases))

    async def test_cold_owner_learning_and_near_tie_affinity(self):
        for policy in ('bounded', 'cost_bounded'):
            r = router(policy, model=PrefillModel((1, 100, 0), {1: 10, 2: 12, 3: 15}))
            key = key_for(r, 3)
            for i in range(4):
                lease = await r.acquire(item(i, key=key))
                self.assertEqual(lease.worker, 3)
                self.assertEqual(lease.estimated_cached, 0)
                r.release(lease, True)

    async def test_page_rounded_kv_limits_three_8192_requests(self):
        r = router('least_active', count=1, max_inflight=4)
        leases = [await r.acquire(item(i)) for i in range(3)]
        self.assertEqual(r.reserved, [3*8448])
        waiting = asyncio.create_task(r.acquire(item(4)))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        r.release(leases[0], True)
        new = await asyncio.wait_for(waiting, .5)
        self.assertEqual(r.reserved, [3*8448])
        self.assertGreaterEqual(new.route['gateway_wait_s'], 0)
        self.assertAlmostEqual(new.route['gateway_wait_ms']/1000,
                               new.route['gateway_wait_s'])
        for lease in [*leases[1:], new]:
            r.release(lease, True)
        self.assertEqual(r.reserved, [0])

    async def test_hard_inflight_cap_independent_of_kv(self):
        r = router('least_active', count=1, max_inflight=1, kv_tokens=100000)
        first = await r.acquire(item())
        waiting = asyncio.create_task(r.acquire(item(1)))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        r.release(first, True)
        second = await asyncio.wait_for(waiting, .5)
        self.assertEqual(r.snapshot()['active'], [1])
        r.release(second, True)

    async def test_fifo_queue_and_cancelled_head_do_not_leak_capacity(self):
        r = router('least_active', count=1, max_inflight=1)
        first = await r.acquire(item())
        cancelled = asyncio.create_task(r.acquire(item(1)))
        next_waiter = asyncio.create_task(r.acquire(item(2)))
        last_waiter = asyncio.create_task(r.acquire(item(3)))
        await asyncio.sleep(0)
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled
        self.assertEqual(r.snapshot()['queue_waiters'], 2)
        r.release(first, True)
        second = await asyncio.wait_for(next_waiter, .5)
        self.assertFalse(last_waiter.done())
        r.release(second, True)
        last = await asyncio.wait_for(last_waiter, .5)
        r.release(last, True)
        self.assertEqual([h['request_id'] for h in r.history], ['0', '2', '3'])
        self.assertEqual(r.snapshot()['queue_cancellations'], 1)
        self.assertEqual(r.snapshot()['queue_waiters'], 0)

    async def test_capacity_timeout_does_not_select_or_reserve(self):
        r = router('least_active', count=1, max_inflight=1, queue_timeout_s=.015)
        first = await r.acquire(item())
        with self.assertRaises(TimeoutError):
            await r.acquire(item(1))
        self.assertEqual(len(r.history), 1)
        self.assertEqual(r.snapshot()['queue_waiters'], 0)
        self.assertEqual(r.snapshot()['queue_timeouts'], 1)
        self.assertEqual(r.reserved, [8448])
        r.release(first, True)

    async def test_release_once_and_foreign_lease_rejected(self):
        r, foreign = router(), router()
        lease = await r.acquire(item())
        with self.assertRaises(RuntimeError):
            foreign.release(lease, True)
        r.release(lease, True)
        with self.assertRaises(RuntimeError):
            r.release(lease, True)
        with self.assertRaises(RuntimeError):
            r.progress(lease, event(1, {}))
        self.assertEqual(sum(r.snapshot()['selections']), 1)
        self.assertEqual(sum(r.snapshot()['active']), 0)

    async def test_first_content_removes_pending_prefill_only_once(self):
        r = router(count=1)
        lease = await r.acquire(item())
        r.progress(lease, event(1, {'doc': 7680}, content=''))
        self.assertFalse(lease.started)
        self.assertEqual(len(r._pending(0)), 1)
        r.progress(lease, event(2, {'doc': 7680}))
        self.assertTrue(lease.started)
        self.assertEqual(r._pending(0), [])
        r.progress(lease, event(3, {'doc': 7680}, generated=3))
        self.assertEqual(lease.produced, 3)
        r.release(lease, True)

    async def test_snapshot_epoch_admission_invalidation_and_expiry(self):
        r = router(count=1, cache_ttl_s=1)
        lease = await r.acquire(item())
        r.progress(lease, event(3, {'doc': 7680}))
        r.progress(lease, event(2, {}))
        self.assertEqual(r._cached(0, 'doc', 7680, time.monotonic()), 7680)
        r.release(lease, True)
        hit = await r.acquire(item(1))
        self.assertEqual(hit.estimated_cached, 7680)
        # Full conservative KV reservation is unchanged even on a proven hit.
        self.assertEqual(hit.reserved_tokens, 8448)
        self.assertEqual(r.cache[0]['prefixes'], {})
        r.progress(hit, event(3, {'doc': 7680}))
        self.assertEqual(r.cache[0]['prefixes'], {})  # same epoch cannot refresh TTL
        r.progress(hit, event(4, {'doc': 7680}))
        r.release(hit, True)
        r.cache[0]['observed_at'] -= 2
        expired = await r.acquire(item(2))
        self.assertEqual(expired.estimated_cached, 0)
        r.release(expired, True)

    async def test_other_final_snapshot_cannot_hide_pending_cold_prefill(self):
        r = router(count=1)
        old = await r.acquire(item())
        r.progress(old, event(1, {'doc': 7680}))
        pending = await r.acquire(item(1, digest='new'))
        r.progress(old, event(2, {'doc': 7680}, content='', generated=64))
        self.assertEqual(r._cached(0, 'doc', 7680, time.monotonic()), 0)
        r.progress(pending, event(3, {'new': 7680}))
        self.assertEqual(r._cached(0, 'doc', 7680, time.monotonic()), 0)
        self.assertEqual(r._cached(0, 'new', 7680, time.monotonic()), 7680)
        r.release(old, True)
        r.release(pending, True)

    async def test_pending_prefill_guard_spills_within_relative_bound(self):
        for policy in ('bounded', 'cost_bounded'):
            r = router(policy, model=PrefillModel((1, 100, 0), {1: 10, 2: 12, 3: 15}))
            leases = [await r.acquire(item(i)) for i in range(4)]
            self.assertEqual(sorted(x.worker for x in leases), [0, 1, 2, 3])
            blocked_worker = leases[0].worker
            for lease in leases[1:]:
                r.progress(lease, event(1, {}))
            new = await r.acquire(item('new', key=key_for(r, blocked_worker)))
            self.assertNotEqual(new.worker, blocked_worker)
            self.assertIn('prefill_spill', new.route['branch'])
            self.assertLessEqual(new.route['active_before'][new.worker],
                                 new.route['count_limit_before'])
            self.assertEqual(new.route['pending_uncached_tokens'][blocked_worker], 8192)
            if policy == 'cost_bounded':
                self.assertEqual(new.route['decode_step_ms'][new.worker], 10)
                self.assertEqual(new.route['decode_step_ms'][blocked_worker], 0)
            for lease in [*leases, new]:
                r.release(lease, True)

    async def test_known_secondary_cache_beats_cold_owner(self):
        for policy in ('bounded', 'cost_bounded'):
            r = router(policy, model=PrefillModel((1, 100, 0), {1: 10, 2: 12, 3: 15}))
            warm = await r.acquire(item('warm', key=key_for(r, 2)))
            r.progress(warm, event(1, {'doc': 7680}))
            r.release(warm, True)
            new = await r.acquire(item('new', key=key_for(r, 0)))
            self.assertEqual(new.preferred, 0)
            self.assertEqual(new.worker, 2)
            self.assertEqual(new.estimated_cached, 7680)
            r.release(new, True, {'prompt_tokens_details': {'cached_tokens': 0}})
            self.assertEqual(r.snapshot()['cache_hint_overestimates'], 1)

    async def test_caps_and_count_bound_hold_under_mixed_release_order(self):
        r = router('bounded', count=8)
        active = []
        for i in range(80):
            if len(active) >= 16:
                lease = active.pop((i*7) % len(active))
                r.release(lease, True)
            lease = await r.acquire(item(i, key='one-hot-document'))
            active.append(lease)
            route = lease.route
            self.assertLessEqual(route['active_before'][lease.worker], route['count_limit_before'])
            self.assertTrue(all(w.active_load <= 3 for w in r.workers))
            self.assertTrue(all(v <= 32768 for v in r.reserved))
            if i % 3:
                r.progress(lease, event(i+1, {'doc': 7680}))
        for lease in active:
            r.release(lease, True)
        self.assertEqual(sum(r.reserved), 0)
        self.assertEqual(sum(w.active_load for w in r.workers), 0)
        self.assertTrue(all(h['released'] for h in r.history))

    async def test_invalid_request_and_no_healthy_backend_do_not_leak(self):
        r = router()
        for request in (item(prompt=0, shared=0), item(output=-1), item(shared=9000),
                        item(prompt=40000, shared=7680)):
            with self.assertRaises(ValueError):
                await r.acquire(request)
        for worker in r.workers:
            worker.healthy = False
        with self.assertRaises(RuntimeError):
            await r.acquire(item())
        self.assertEqual(r.history, [])
        self.assertEqual(r.snapshot()['queue_waiters'], 0)


class ModelTests(unittest.TestCase):
    def test_profile_has_prefill_and_one_decode_step_units(self):
        m = PrefillModel.from_profile({'prefill_coef': [1, 100, 1],
                                      'decode_step_ms': {'1': 10, '3': 14}})
        self.assertGreater(m.prefill(8192, 0), m.prefill(8192, 7680))
        self.assertAlmostEqual(m.prefill(8192, 7680), 1+51.2+(512**2+2*512*7680)/1e6)
        self.assertEqual(m.decode_step(0, 8192), 0)
        self.assertEqual(m.decode_step(2, 8192), 12)
        self.assertEqual(m.decode_step(3, 8192), 14)
        self.assertEqual(m.decode_step(3, 2048), 14)  # measured-context model, no false generalization

    def test_invalid_cost_profile_rejected(self):
        for coeff, steps in (((1, 2), {1: 2}), ((1, -1, 0), {1: 2}),
                             ((1, 2, 0), {}), ((1, 2, 0), {0: 2}),
                             ((1, float('nan'), 0), {1: 2})):
            with self.assertRaises(ValueError):
                PrefillModel(coeff, steps)


if __name__ == '__main__':
    unittest.main()
