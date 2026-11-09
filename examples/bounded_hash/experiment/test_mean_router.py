"""CPU-only checks of guarded escape, queue consistency and lease safety."""
import asyncio
import copy
import time
import unittest

from affinity_router import Router as BaseRouter
from mean_router import Router
from stable_router import Router as StableRouter


def item(identifier, key=None):
    key = key or str(identifier)
    return dict(id=str(identifier), key=key, prefix_digest=key,
                prompt_tokens=8192, output_tokens=64, shared_tokens=7936)


def place(router, value, worker):
    key = (value['prefix_digest'], value['shared_tokens'])
    router.owners[key] = [worker]
    router.home[key] = worker
    router.owner_bytes[worker] += value['shared_tokens']


def admit_at(router, value, worker, warm=0, started=False):
    if warm:
        router.cache[worker].update(epoch=1, observed_at=time.monotonic(),
                                    prefixes={value['prefix_digest']:warm})
    lease = BaseRouter._select(router, value, router._validate_item(value), [worker], 0)
    lease.started = started
    return lease


class MeanRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_heavy_prefills_allow_escape_without_changing_primary(self):
        r = Router(['a','b'], 'mean_bounded')
        target = item('target'); place(r,target,0)
        background = [admit_at(r,item('pending1'),0), admit_at(r,item('pending2'),0),
                      admit_at(r,item('decoding1'),1,started=True),
                      admit_at(r,item('decoding2'),1,started=True)]
        lease = await r.acquire(target)
        self.assertEqual(lease.worker,1)
        self.assertEqual(lease.preferred,0)
        self.assertEqual(lease.route['branch'],'mean_overload_escape')
        self.assertEqual(lease.route['stable_score_tokens'],24576)
        self.assertEqual(lease.route['routing_only_retention_credit_tokens'],7936)
        self.assertEqual(lease.route['effective_stable_score_tokens'],16640)
        self.assertEqual(lease.route['selected_score_tokens'],8192)
        self.assertEqual(r.owners[(target['prefix_digest'],7936)],[0])
        for x in [lease]+background:r.release(x,True)
        self.assertEqual(r.snapshot()['active'],[0,0])

    async def test_unknown_owner_with_one_heavy_prefill_keeps_affinity(self):
        r = Router(['a','b'], 'mean_bounded')
        target=item('target');place(r,target,0)
        pending=admit_at(r,item('pending'),0)
        decoding=admit_at(r,item('decoding'),1,started=True)
        lease=await r.acquire(target)
        self.assertEqual(lease.worker,0)
        self.assertEqual(lease.route['branch'],'mean_owner')
        self.assertEqual(lease.route['stable_score_tokens'],16384)
        self.assertEqual(lease.route['routing_only_retention_credit_tokens'],7936)
        self.assertEqual(lease.route['effective_stable_score_tokens'],8448)
        # The credit affects neither cache telemetry nor the conservative lease.
        self.assertEqual(lease.estimated_cached,0)
        self.assertEqual(lease.route['estimated_cached_tokens'],0)
        self.assertEqual(lease.route['selected_cached_hint'],0)
        self.assertEqual(lease.reserved_tokens,8448)
        self.assertFalse(lease.route['retention_credit_is_verified_hit'])
        self.assertFalse(lease.route['queue_score_uses_retention_credit'])
        self.assertEqual(lease.route['queue_score_tokens'],16384)
        for x in (lease,pending,decoding):r.release(x,True)

    async def test_small_advantage_keeps_stable_owner(self):
        r = Router(['a','b'], 'mean_bounded')
        target = item('target'); place(r,target,0)
        pending = admit_at(r,item('pending'),0,warm=7936)
        decoding = admit_at(r,item('decoding'),1,started=True)
        lease = await r.acquire(target)
        self.assertEqual(lease.worker,0)
        self.assertEqual(lease.route['branch'],'mean_owner')
        self.assertEqual(lease.route['stable_score_tokens'],8448)
        for x in (lease,pending,decoding):r.release(x,True)

    async def test_preview_has_no_ownership_side_effects(self):
        r = Router(['a','b'], 'mean_bounded')
        target = item('target'); place(r,target,0)
        r.recent.extend([(target['prefix_digest'],7936)]*16)
        before=(copy.deepcopy(r.owners),list(r.owner_bytes),len(r.history))
        plan=r._preview(target,r._validate_item(target),[1],time.monotonic())
        self.assertEqual(plan['commit_secondary'],1)
        self.assertEqual(before,(r.owners,r.owner_bytes,len(r.history)))
        lease=r._select(target,r._validate_item(target),[1],0)
        self.assertEqual(r.owners[(target['prefix_digest'],7936)],[0,1])
        self.assertTrue(lease.route['secondary_committed'])
        r.release(lease,True)

    async def test_queue_uses_cost_of_actual_destination_not_cheapest_unused_worker(self):
        r=Router(['a','b'],'mean_bounded',queue_timeout_s=2)
        a,b=item('first','a-prefix'),item('second','b-prefix')
        place(r,a,0);place(r,b,1)
        pending=admit_at(r,item('pending'),0,warm=7936)
        decoding=admit_at(r,item('decoding'),1,started=True)
        r._dispatching=True
        first=asyncio.create_task(r.acquire(a));second=asyncio.create_task(r.acquire(b))
        await asyncio.sleep(.001)
        start=len(r.history)
        r._dispatching=False;r._dispatch()
        x,y=await asyncio.gather(first,second)
        self.assertEqual(r.history[start]['request_id'],'second')
        self.assertEqual(r.history[start]['chosen'],1)
        self.assertEqual(r.history[start]['queue_score_tokens'],8192)
        for e in r.history[start:]:
            self.assertEqual(e['queue_selected_worker'],e['chosen'])
            self.assertEqual(e['queue_score_tokens'],e['scores'][e['chosen']])
        for lease in (x,y,pending,decoding):r.release(lease,True)

    async def test_short_requests_keep_least_active_behavior(self):
        r=Router(list('abcd'),'mean_bounded');baseline=Router(list('abcd'),'least_active')
        active=[]
        for i in range(8):
            x=item(i);x.update(prompt_tokens=2048,output_tokens=512,shared_tokens=1536)
            a,b=await r.acquire(x),await baseline.acquire(x)
            self.assertEqual(a.worker,b.worker)
            self.assertEqual(a.route['branch'],'least_active_outside_affinity')
            active.append((a,b))
        self.assertEqual(r.owners,{})
        for a,b in active:r.release(a,True);baseline.release(b,True)

    async def test_queue_cancel_timeout_and_64_requests_leave_no_reservations(self):
        r=Router(['a'],'mean_bounded',max_inflight=1,queue_timeout_s=.02)
        running=await r.acquire(item('running'))
        cancelled=asyncio.create_task(r.acquire(item('cancelled')))
        await asyncio.sleep(.001);cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):await cancelled
        with self.assertRaises(TimeoutError):await r.acquire(item('timeout'))
        r.release(running,True)
        self.assertEqual(r.snapshot()['queue_waiters'],0)
        self.assertEqual(r.snapshot()['queue_cancellations'],1)
        self.assertEqual(r.snapshot()['queue_timeouts'],1)
        r=Router(list('abcdefgh'),'mean_bounded',queue_timeout_s=5)
        async def request(i):
            lease=await r.acquire(item(i,'hot' if i%4==0 else f'doc{i%16}'))
            self.assertLessEqual(max(w.active_load for w in r.workers),3)
            self.assertLessEqual(max(r.reserved),32768)
            await asyncio.sleep(.001)
            r.release(lease,True)
        await asyncio.gather(*(request(i) for i in range(64)))
        self.assertEqual(r.snapshot()['active'],[0]*8)
        self.assertEqual(r.snapshot()['kv_reserved_tokens'],[0]*8)
        self.assertEqual(r.snapshot()['queue_waiters'],0)
        self.assertEqual(len(r.history),64)
        self.assertTrue(all(e['released'] for e in r.history))
        self.assertTrue(all(len(x)<=2 for x in r.owners.values()))
        self.assertLessEqual(max(e['queue_bypasses'] for e in r.history),8)

    async def test_waiting_cold_request_is_served_after_eight_bypasses(self):
        r=Router(['a'],'mean_bounded',max_inflight=1,queue_timeout_s=2)
        running=await r.acquire(item('running','old'))
        cold=asyncio.create_task(r.acquire(item('cold','cold')))
        warm=[asyncio.create_task(r.acquire(item(f'w{i}','warm'))) for i in range(12)]
        await asyncio.sleep(.001)
        for index in range(13):
            r.cache[0].update(epoch=index+1,observed_at=time.monotonic(),prefixes={'warm':7936})
            r.release(running,True)
            await asyncio.sleep(.001)
            ready=[t for t in [cold]+warm if t.done() and not t.result().released]
            self.assertEqual(len(ready),1)
            running=ready[0].result()
            if index==8:self.assertEqual(running.route['request_id'],'cold')
        r.release(running,True)
        self.assertEqual(cold.result().route['queue_bypasses'],8)
        self.assertTrue(all(t.done() for t in [cold]+warm))
        self.assertEqual(r.snapshot()['active'],[0])

    async def test_legacy_policy_routing_remains_unchanged(self):
        for policy in ['least_active','round_robin','bounded','stable_bounded']:
            actual=Router(list('abcd'),policy);expected=StableRouter(list('abcd'),policy)
            for i in range(12):
                a,b=await actual.acquire(item(i)),await expected.acquire(item(i))
                self.assertEqual((a.worker,a.preferred,a.route['branch']),
                                 (b.worker,b.preferred,b.route['branch']))
                actual.release(a,True);expected.release(b,True)


if __name__=='__main__':unittest.main()
