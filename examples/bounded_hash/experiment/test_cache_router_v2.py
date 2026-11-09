"""Queue/ownership/prediction mechanics; these are not GPU performance tests."""
import asyncio
import copy
import math
import time
import unittest

from cache_router import Router
from test_phase_router import MODEL, item, place, admit, warm


class CacheRouterTests(unittest.IsolatedAsyncioTestCase):
    def router(self, backends=('a','b'), **kw):
        return Router(list(backends),phase_model=MODEL,**kw)

    def drain(self, router):
        for lease in list(router.active.values()):
            router.release(lease,True)
        router.close()
        self.assertEqual(sum(router.reserved),0)

    async def test_serial_pending_prefix_is_forecast_once_without_fabricating_hit(self):
        r=self.router();x=item('a','same');place(r,x,0)
        first=admit(r,x,0)
        second=admit(r,item('b','same'),0)
        c=r._forecast(0,r._validate_item(item('new','same')),0)
        self.assertEqual(c['coalesced_pending_requests'],1)
        self.assertEqual(c['predicted_cached_tokens'],7936)
        self.assertAlmostEqual(c['pending_prefill_ms'],r.model.prefill(8192,0)+r.model.prefill(8192,7936))
        self.assertEqual(r._cached(0,'same',7936,time.monotonic()),0)
        self.assertEqual(first.estimated_cached,0);self.assertEqual(second.estimated_cached,0)
        self.drain(r)

    async def test_active_shared_prefix_survives_other_pending_only_as_prediction(self):
        r=self.router();target=item('new','same');place(r,target,0)
        admit(r,item('decode','same'),0,started=True)
        admit(r,item('pending','other'),0)
        warm(r,0,target)
        c=r._forecast(0,r._validate_item(target),r._cached(0,'same',7936,time.monotonic()))
        self.assertEqual(c['observed_cached_tokens'],0)
        self.assertEqual(c['predicted_cached_tokens'],7936)
        self.drain(r)

    async def test_unrelated_prefix_never_gets_coalescing_credit(self):
        r=self.router();admit(r,item('old','different'),0,started=True)
        c=r._forecast(0,r._validate_item(item('new','same')),0)
        self.assertEqual(c['predicted_cached_tokens'],0)
        self.drain(r)

    async def test_coalescing_ablation_preserves_observed_cache_contract(self):
        r=self.router(coalescing=False);admit(r,item('old','same'),0,started=True)
        c=r._forecast(0,r._validate_item(item('new','same')),0)
        self.assertEqual(c['predicted_cached_tokens'],0)
        self.drain(r)

    async def test_warm_known_owner_and_full_kv_reservation(self):
        r=self.router();target=item('new','same');place(r,target,0);warm(r,0,target)
        lease=await r.acquire(target)
        self.assertEqual(lease.worker,0);self.assertEqual(lease.estimated_cached,7936)
        self.assertEqual(lease.reserved_tokens,8448)
        self.assertFalse(lease.route['predicted_reuse_is_verified_hit'])
        self.assertEqual(lease.route['queue_score_ms'],lease.route['scores'][lease.worker])
        self.drain(r)

    async def test_old_owner_hysteresis_survives_one_unknown_background_prefill(self):
        r=self.router();target=item('new','same');place(r,target,0)
        admit(r,item('pending'),0);admit(r,item('decode'),1,started=True)
        lease=await r.acquire(target)
        self.assertEqual(lease.worker,0)
        self.assertEqual(lease.estimated_cached,0)
        self.assertGreater(lease.route['retention_credit_ms'],0)
        self.assertFalse(lease.route['retention_is_verified_hit'])
        self.drain(r)

    async def test_two_heavy_prefills_escape_without_changing_owner(self):
        r=self.router();target=item('new','same');place(r,target,0)
        admit(r,item('pending1'),0);admit(r,item('pending2'),0)
        admit(r,item('decode1'),1,started=True);admit(r,item('decode2'),1,started=True)
        lease=await r.acquire(target)
        self.assertEqual(lease.worker,1)
        self.assertEqual(r.owners[('same',7936)],[0])
        self.assertEqual(lease.route['branch'],'cache_overload_escape')
        self.drain(r)

    async def test_secondary_preview_is_pure_and_commit_has_two_owner_cap(self):
        r=self.router();target=item('new','same');place(r,target,0)
        r.recent.extend([('same',7936)]*16)
        before=(copy.deepcopy(r.owners),list(r.owner_bytes),len(r.history))
        plan=r._preview(target,r._validate_item(target),[1],time.monotonic())
        self.assertEqual(before,(r.owners,r.owner_bytes,len(r.history)))
        self.assertEqual(plan['commit_secondary'],1)
        lease=r._commit(target,r._validate_item(target),[1],0,plan)
        self.assertEqual(r.owners[('same',7936)],[0,1])
        self.assertTrue(lease.route['secondary_committed']);self.drain(r)

    async def test_nearly_finished_owner_is_waited_for_and_release_wakes_queue(self):
        r=self.router(owner_wait_ms=100,max_inflight=1,queue_timeout_s=1)
        target=item('new','same');place(r,target,0)
        old=admit(r,item('old','same'),0,started=True);old.produced=63
        task=asyncio.create_task(r.acquire(target));await asyncio.sleep(.01)
        self.assertFalse(task.done());self.assertGreater(r.deferred_decisions,0)
        r.release(old,True)
        new=await task
        self.assertEqual(new.worker,0);self.assertTrue(new.route['owner_wait_requested'])
        self.drain(r)

    async def test_owner_wait_expires_and_spills_without_a_release(self):
        r=self.router(owner_wait_ms=20,max_inflight=1,queue_timeout_s=1)
        target=item('new','same');place(r,target,0)
        old=admit(r,item('old','same'),0,started=True);old.produced=63
        began=time.monotonic();lease=await r.acquire(target)
        self.assertEqual(lease.worker,1)
        self.assertGreaterEqual(time.monotonic()-began,.018)
        self.assertLess(time.monotonic()-began,.3)
        self.assertIsNone(r._retry);self.drain(r)

    async def test_long_remaining_decode_does_not_hold_available_worker(self):
        r=self.router(owner_wait_ms=20,max_inflight=1)
        target=item('new','same');place(r,target,0)
        admit(r,item('old','same'),0,started=True)
        lease=await r.acquire(target)
        self.assertEqual(lease.worker,1);self.assertFalse(lease.route['owner_wait_requested'])
        self.drain(r)

    async def test_cancellation_during_owner_wait_leaves_no_ticket_or_timer(self):
        r=self.router(owner_wait_ms=500,max_inflight=1)
        target=item('new','same');place(r,target,0)
        old=admit(r,item('old','same'),0,started=True);old.produced=63
        task=asyncio.create_task(r.acquire(target));await asyncio.sleep(.005);task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(len(r._waiters),0);self.assertIsNone(r._retry)
        self.drain(r)

    async def test_capacity_timeout_and_failure_release_leave_no_reservations(self):
        r=self.router(backends=('a',),max_inflight=1,queue_timeout_s=.01)
        lease=await r.acquire(item('running'))
        with self.assertRaises(TimeoutError):await r.acquire(item('waiter'))
        r.release(lease,False)
        self.assertEqual(sum(r.reserved),0);self.assertEqual(len(r._waiters),0);r.close()

    async def test_64_requests_respect_count_and_kv_bounds_then_drain(self):
        r=self.router(backends=tuple('abcdefgh'),owner_wait_ms=0,queue_timeout_s=5)
        async def request(i):
            lease=await r.acquire(item(str(i),'hot' if i%4==0 else f'doc{i%8}'))
            self.assertLessEqual(max(r.reserved),32768)
            self.assertLessEqual(max(x.active_load for x in r.workers),3)
            e=lease.route
            if not e['count_bound_relaxed']:
                self.assertLessEqual(e['active_before'][lease.worker]+1,e['projected_count_limit'])
            await asyncio.sleep(.001);r.release(lease,True)
        await asyncio.gather(*(request(i) for i in range(64)))
        self.assertEqual(len(r.history),64);self.assertEqual(len(r._waiters),0)
        self.assertEqual(sum(r.reserved),0);self.assertTrue(all(x['released'] for x in r.history))
        self.assertTrue(all(len(owners)<=2 for owners in r.owners.values()))
        self.assertLessEqual(max(x['queue_bypasses'] for x in r.history),8)
        r.close()

    async def test_no_prefix_has_no_owner_and_long_output_is_supported(self):
        r=self.router()
        ordinary=item('cold',p=2048,o=512,shared=0);ordinary['prefix_digest']=''
        lease=await r.acquire(ordinary)
        self.assertFalse(r.owners);self.assertEqual(lease.estimated_cached,0)
        r.release(lease,True)
        target=item('long',p=2048,o=512,shared=1536);place(r,target,1);warm(r,1,target)
        lease=await r.acquire(target)
        self.assertEqual(lease.worker,1);self.assertEqual(lease.reserved_tokens,2560)
        self.drain(r)


if __name__=='__main__':unittest.main()
