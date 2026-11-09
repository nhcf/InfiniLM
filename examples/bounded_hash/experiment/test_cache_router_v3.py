"""Deterministic CPU admission tests; no GPU speed claim or simulated throughput."""
import asyncio
import copy
import math
import time
import unittest

from cache_router import Router as V2
from cache_router_v3 import Router
from test_phase_router import MODEL, item, place, admit, warm


class WarmWaitTests(unittest.IsolatedAsyncioTestCase):
    def router(self, cls=Router, **kw):
        r = cls(['a', 'b'], phase_model=MODEL, max_inflight=1, queue_timeout_s=1, **kw)
        self.addCleanup(self.cleanup, r)
        return r

    def cleanup(self, r):
        for lease in list(r.active.values()):
            r.release(lease, True)
        r.close()
        self.assertEqual(sum(r.reserved), 0)

    def setup_warm_nonowner(self, r, *, age=0, matching=False, produced=63):
        target = item('incoming', 'prefix')
        place(r, target, 0)
        old = admit(r, item('decoding', 'prefix' if matching else 'different'), 1, started=True)
        old.produced = produced
        warm(r, 1, target, age=age)
        return target, old

    def ticket_plan(self, r, target):
        contract = r._validate_item(target)
        ticket = dict(item=target, contract=contract, bypasses=0)
        plan = r._preview(target, contract, [0], time.monotonic())
        return ticket, plan

    async def test_nonowner_cached_document_waits_and_reuses_after_release(self):
        r = self.router(warm_wait_ms=100)
        target, old = self.setup_warm_nonowner(r)
        task = asyncio.create_task(r.acquire(target))
        await asyncio.sleep(.005)
        self.assertFalse(task.done())
        self.assertEqual(r.workers[1].active_load, 1)
        self.assertEqual(r.reserved[1], 8448)
        r.release(old, True)
        new = await task
        self.assertEqual(new.worker, 1)
        self.assertEqual(new.estimated_cached, 7936)
        self.assertEqual(new.reserved_tokens, 8448)
        self.assertNotIn(1, r.owners[('prefix', 7936)])
        info = new.route['v3_warm_wait']
        self.assertTrue(info['waited_for_warm_copy'])
        self.assertEqual(info['first_candidates'][0]['evidence'], 'fresh_cache_snapshot')
        self.assertEqual(info['first_candidates'][0]['decoding'][0]['digest'], 'different')
        self.assertEqual(info['final_worker'], 1)
        self.assertIsNone(r._retry)
        self.assertEqual(r._wait_evidence, {})

    async def test_original_v2_does_not_wait_for_same_nonowner_snapshot(self):
        r = self.router(cls=V2)
        target, old = self.setup_warm_nonowner(r)
        new = await r.acquire(target)
        self.assertEqual(new.worker, 0)
        self.assertEqual(new.estimated_cached, 0)
        self.assertFalse(new.route['owner_wait_requested'])

    async def test_fixed_deadline_is_not_refreshed_by_rechecks(self):
        r = self.router(warm_wait_ms=100)
        target, old = self.setup_warm_nonowner(r)
        ticket, plan = self.ticket_plan(r, target)
        now = time.monotonic()
        self.assertTrue(r._should_defer(ticket, plan, now))
        deadline = ticket['v3_wait_deadline']
        self.assertTrue(r._should_defer(ticket, plan, now+.01))
        self.assertEqual(ticket['v3_wait_deadline'], deadline)
        self.assertFalse(r._should_defer(ticket, plan, deadline+.001))
        self.assertEqual(r._wait_evidence[target['id']]['stop_reason'], 'deadline')

    async def test_expiry_spills_without_exceeding_absolute_wait(self):
        r = self.router(warm_wait_ms=20)
        target, old = self.setup_warm_nonowner(r)
        before = time.monotonic()
        lease = await r.acquire(target)
        elapsed = time.monotonic()-before
        self.assertGreaterEqual(elapsed, .018)
        self.assertLess(elapsed, .3)
        self.assertEqual(lease.worker, 0)
        self.assertTrue(lease.route['v3_warm_wait']['waited_for_warm_copy'])
        self.assertEqual(lease.route['v3_warm_wait']['stop_reason'], 'deadline')
        self.assertEqual(r._wait_evidence, {})
        self.assertIsNone(r._retry)

    async def test_expired_snapshot_without_live_matching_decode_does_not_wait(self):
        r = self.router(warm_wait_ms=100)
        target, old = self.setup_warm_nonowner(r, age=6)
        lease = await r.acquire(target)
        self.assertEqual(lease.worker, 0)
        self.assertFalse(lease.route['v3_warm_wait']['waited_for_warm_copy'])
        self.assertEqual(lease.route['v3_warm_wait']['last_candidates'], [])

    async def test_pending_prefill_invalidates_candidate_even_with_snapshot(self):
        r = self.router(warm_wait_ms=100)
        target, old = self.setup_warm_nonowner(r)
        old.started = False
        lease = await r.acquire(target)
        self.assertEqual(lease.worker, 0)
        self.assertFalse(lease.route['v3_warm_wait']['waited_for_warm_copy'])

    async def test_long_release_prediction_does_not_hold_free_card(self):
        r = self.router(warm_wait_ms=20)
        target, old = self.setup_warm_nonowner(r, produced=0)
        lease = await r.acquire(target)
        info = lease.route['v3_warm_wait']
        self.assertEqual(lease.worker, 0)
        self.assertFalse(info['waited_for_warm_copy'])
        self.assertGreater(info['last_candidates'][0]['release_eta_ms'], 20)

    async def test_cancellation_removes_ticket_evidence_and_timer(self):
        r = self.router(warm_wait_ms=100)
        target, old = self.setup_warm_nonowner(r)
        task = asyncio.create_task(r.acquire(target))
        await asyncio.sleep(.005)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(r._waiters), 0)
        self.assertEqual(r._wait_evidence, {})
        self.assertIsNone(r._retry)
        self.assertEqual(len(r.active), 1)

    async def test_active_match_is_recorded_as_prediction_not_observed_cache(self):
        r = self.router(warm_wait_ms=100)
        target, old = self.setup_warm_nonowner(r, age=6, matching=True)
        ticket, plan = self.ticket_plan(r, target)
        self.assertTrue(r._should_defer(ticket, plan, time.monotonic()))
        evidence = r._wait_evidence[target['id']]['first_candidates'][0]
        self.assertEqual(evidence['evidence'], 'active_matching_decode_prediction')
        self.assertEqual(evidence['observed_cached_tokens'], 0)
        self.assertEqual(r._cached(1, 'prefix', 7936, time.monotonic()), 0)
        self.assertEqual(r.reserved[1], 8448)

    async def test_fairness_bypass_forces_admission(self):
        r = self.router(warm_wait_ms=100)
        target, old = self.setup_warm_nonowner(r)
        ticket, plan = self.ticket_plan(r, target)
        ticket['bypasses'] = r.max_bypasses
        self.assertFalse(r._should_defer(ticket, plan, time.monotonic()))
        self.assertEqual(r._wait_evidence[target['id']]['stop_reason'], 'fairness')

    async def test_unhealthy_cache_copy_is_never_waited_for(self):
        r = self.router(warm_wait_ms=100)
        target, old = self.setup_warm_nonowner(r)
        r.workers[1].healthy = False
        lease = await r.acquire(target)
        self.assertEqual(lease.worker, 0)
        self.assertFalse(lease.route['v3_warm_wait']['waited_for_warm_copy'])

    async def test_wait_disabled_preserves_v2_immediate_choice(self):
        r = self.router(warm_wait_ms=0)
        target, old = self.setup_warm_nonowner(r)
        lease = await r.acquire(target)
        self.assertEqual(lease.worker, 0)
        self.assertEqual(lease.route['v3_warm_wait']['stop_reason'], 'disabled')

    async def test_existing_fast_warm_choice_is_not_delayed(self):
        r = self.router(warm_wait_ms=100)
        target, old = self.setup_warm_nonowner(r)
        warm(r, 0, target)
        lease = await r.acquire(target)
        self.assertEqual(lease.worker, 0)
        self.assertEqual(lease.estimated_cached, 7936)
        self.assertFalse(lease.route['v3_warm_wait']['waited_for_warm_copy'])

    async def test_preview_does_not_change_placement_or_capacity(self):
        r = self.router(warm_wait_ms=100)
        target, old = self.setup_warm_nonowner(r)
        before = copy.deepcopy((r.owners, r.owner_bytes, r.reserved, r.history))
        self.ticket_plan(r, target)
        self.assertEqual(before, (r.owners, r.owner_bytes, r.reserved, r.history))

    async def test_count_bound_only_exclusion_does_not_enable_new_wait(self):
        r = Router(['a', 'b'], phase_model=MODEL, max_inflight=3, warm_wait_ms=100)
        self.addCleanup(self.cleanup, r)
        target, old = self.setup_warm_nonowner(r)
        contract = r._validate_item(target)
        ticket = dict(item=target, contract=contract, bypasses=0)
        # A valid hard-capacity candidate is never considered by the new wait.
        plan = r._preview(target, contract, [0, 1], time.monotonic())
        self.assertFalse(r._should_defer(ticket, plan, time.monotonic()))

    async def test_failed_release_invalidates_cache_and_rechecks_choice(self):
        r = self.router(warm_wait_ms=100)
        target, old = self.setup_warm_nonowner(r)
        task = asyncio.create_task(r.acquire(target))
        await asyncio.sleep(.005)
        # Failure invalidates cached content, so released capacity alone cannot
        # authorize a fabricated cache hit or keep a stale wait promise.
        r.release(old, False)
        lease = await task
        self.assertEqual(lease.estimated_cached, 0)
        self.assertLessEqual(max(r.reserved), 32768)
        self.assertLessEqual(max(w.active_load for w in r.workers), 1)

    async def test_first_release_must_free_the_full_incoming_reservation(self):
        r = Router(['a', 'b'], phase_model=MODEL, kv_tokens=9000,
                   max_inflight=3, warm_wait_ms=100)
        self.addCleanup(self.cleanup, r)
        target = item('incoming', 'prefix')
        place(r, target, 0)
        for i in range(3):
            old = admit(r, item('short'+str(i), p=2048, shared=1536), 1, started=True)
            old.produced = 63
        warm(r, 1, target)
        lease = await r.acquire(target)
        info = lease.route['v3_warm_wait']
        self.assertEqual(lease.worker, 0)
        self.assertFalse(info['waited_for_warm_copy'])
        self.assertFalse(info['last_candidates'][0]['fits_after_first_release'])
        self.assertEqual(lease.reserved_tokens, 8448)

    async def test_pending_selected_owner_can_wait_for_ready_full_replica(self):
        r = Router(['a', 'b'], phase_model=MODEL, max_inflight=3,
                   queue_timeout_s=1, warm_wait_ms=100)
        self.addCleanup(self.cleanup, r)
        target = item('incoming', 'prefix')
        place(r, target, 0)
        admit(r, item('owner_pending', 'different'), 0)
        old = []
        for i in range(3):
            previous = admit(r, item('replica_decode'+str(i)), 1, started=True)
            previous.produced = 63
            old.append(previous)
        warm(r, 1, target)
        task = asyncio.create_task(r.acquire(target))
        await asyncio.sleep(.005)
        self.assertFalse(task.done())
        r.release(old[0], True)
        lease = await task
        self.assertEqual(lease.worker, 1)
        self.assertEqual(lease.estimated_cached, 7936)
        self.assertTrue(lease.route['v3_warm_wait']['waited_for_warm_copy'])
        self.assertLessEqual(lease.route['active_before'][lease.worker]+1,
                             lease.route['projected_count_limit'])

    async def test_timeout_cleans_wait_without_consuming_extra_reservation(self):
        r = Router(['a', 'b'], phase_model=MODEL, max_inflight=1,
                   queue_timeout_s=.01, warm_wait_ms=100)
        self.addCleanup(self.cleanup, r)
        target, old = self.setup_warm_nonowner(r)
        with self.assertRaises(TimeoutError):
            await r.acquire(target)
        self.assertEqual(r._wait_evidence, {})
        self.assertEqual(len(r._waiters), 0)
        self.assertIsNone(r._retry)
        self.assertEqual(sum(r.reserved), 8448)

    async def test_concurrent_arrivals_respect_cap_and_fully_drain(self):
        r = Router(list('abcdefgh'), phase_model=MODEL, queue_timeout_s=3, warm_wait_ms=10)
        self.addCleanup(self.cleanup, r)
        async def request(i):
            lease = await r.acquire(item(str(i), 'hot' if i%4 == 0 else 'doc'+str(i%8)))
            self.assertLessEqual(max(r.reserved), 32768)
            self.assertLessEqual(max(w.active_load for w in r.workers), 3)
            if not lease.route['count_bound_relaxed']:
                self.assertLessEqual(lease.route['active_before'][lease.worker]+1,
                                     lease.route['projected_count_limit'])
            await asyncio.sleep(.001)
            r.release(lease, True)
        await asyncio.gather(*(request(i) for i in range(64)))
        self.assertEqual(len(r.history), 64)
        self.assertEqual(r._wait_evidence, {})
        self.assertEqual(sum(r.reserved), 0)
        self.assertIsNone(r._retry)
        self.assertTrue(all(e['released'] for e in r.history))
        self.assertEqual(r.snapshot()['policy'], 'cache_bounded_v3')

    async def test_invalid_wait_budget_is_rejected(self):
        for value in (-1, 2001, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Router(['a'], phase_model=MODEL, warm_wait_ms=value)


if __name__ == '__main__':
    unittest.main()
