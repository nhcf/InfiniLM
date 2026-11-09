"""CPU mechanics tests. These do not prove throughput or latency improvement."""
import asyncio
import copy
from pathlib import Path
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if not (HERE/'mean_router.py').is_file():
    sys.path.append(str(HERE.parent/'sources/report/mean_prefix97_20261022T070505_725816Z/source'))

from affinity_router import Router as BaseRouter
from mean_router import Router as MeanRouter
from phase_router import PhaseModel, Router


MODEL = dict(schema_version=1, prefill_coef=[0, 10, 0],
             decode_step_ms={'1': 1, '2': 2, '3': 3},
             decode_reference_context_tokens=8192, calibration_id='CPU_TEST_ONLY')


def item(identifier, key=None, p=8192, o=64, shared=7936):
    key = key or str(identifier)
    return dict(id=str(identifier), key=key, prefix_digest=key,
                prompt_tokens=p, output_tokens=o, shared_tokens=shared)


def place(router, value, worker):
    key = (value['prefix_digest'], value['shared_tokens'])
    router.owners[key] = [worker]
    router.home[key] = worker
    router.owner_bytes[worker] += value['shared_tokens']


def admit(router, value, worker, started=False):
    lease = BaseRouter._select(router, value, router._validate_item(value), [worker], 0)
    lease.started = started
    return lease


def warm(router, worker, value, tokens=None, age=0):
    router.cache[worker].update(epoch=100, observed_at=time.monotonic()-age,
        prefixes={value['prefix_digest']: value['shared_tokens'] if tokens is None else tokens})


class ModelTests(unittest.TestCase):
    def test_rejects_invalid_calibration(self):
        invalid = [None, {}, dict(MODEL, schema_version=2),
                   dict(MODEL, prefill_coef=[0,-1,0]),
                   dict(MODEL, prefill_coef=[0,float('nan'),0]),
                   dict(MODEL, prefill_coef=[0,0,0]),
                   dict(MODEL, decode_step_ms={'1': 0}),
                   dict(MODEL, decode_step_ms={'1':1,'3':3}),
                   dict(MODEL, decode_reference_context_tokens=None)]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PhaseModel(value)

    def test_context_and_prefill_models_are_monotone(self):
        m=PhaseModel(dict(MODEL,prefill_coef=[1,10,1]))
        self.assertLess(m.prefill(8192,7936), m.prefill(8192,0))
        self.assertLess(m.decode_step(1,8192), m.decode_step(1,16384))
        self.assertLessEqual(m.decode_step(1,32768),m.decode_step(1,100000))
        self.assertEqual(m.decode_step(0,0),0)
        with self.assertRaises(ValueError):m.decode_step(4,32768)

    def test_model_fingerprint_includes_calibration(self):
        a=PhaseModel(MODEL)
        self.assertEqual(a.fingerprint,PhaseModel(copy.deepcopy(MODEL)).fingerprint)
        self.assertNotEqual(a.fingerprint,PhaseModel(dict(MODEL,calibration_id='other')).fingerprint)

    def test_decode_context_grid_preserves_fixed_cost_and_interpolates(self):
        grid={str(b):[dict(context_tokens=2048*b,step_ms=10*b),
                      dict(context_tokens=8192*b,step_ms=16*b)] for b in (1,2,3)}
        model=PhaseModel(dict(MODEL,decode_context_samples=grid))
        self.assertEqual(model.decode_step(1,1024),10)
        self.assertEqual(model.decode_step(1,5120),13)
        self.assertEqual(model.decode_step(2,10240),26)
        self.assertEqual(model.decode_step(1,1000000),64)
        bad=copy.deepcopy(grid);bad['1'][1]['context_tokens']=2048
        with self.assertRaises(ValueError):PhaseModel(dict(MODEL,decode_context_samples=bad))


class PhaseRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_warm_owner_not_excluded_by_old_relative_request_count_gate(self):
        r=Router(['a','b'],phase_model=MODEL)
        old=MeanRouter(['a','b'],'mean_bounded')
        target=item('target')
        for x in (r,old):
            place(x,target,0)
            admit(x,item('background'),0,started=True)
            warm(x,0,target)
        new_lease,old_lease=await r.acquire(target),await old.acquire(target)
        self.assertEqual(new_lease.worker,0)
        self.assertEqual(old_lease.worker,1)
        self.assertEqual(new_lease.estimated_cached,7936)
        self.assertEqual(new_lease.reserved_tokens,8448)
        self.assertLess(new_lease.route['scores'][0],new_lease.route['scores'][1])
        for x in (r,old):
            for lease in list(x.active.values()):x.release(lease,True)

    async def test_decode_step_pressure_changes_cold_target_placement(self):
        model=dict(MODEL,decode_step_ms={'1':100,'2':200,'3':300})
        r=Router(['a','b'],phase_model=model,skew=1.0,cost_margin_ms=0)
        target=item('target');place(r,target,0)
        background=admit(r,item('decode'),0,started=True)
        lease=await r.acquire(target)
        self.assertEqual(lease.worker,1)
        self.assertEqual(lease.route['phase_components'][0]['active_decode_requests'],1)
        self.assertEqual(lease.route['phase_components'][0]['decode_interference_ms'],100)
        self.assertEqual(lease.route['decode_rounds_charged'],1)
        for x in (lease,background):r.release(x,True)

    async def test_total_decode_context_affects_ranking(self):
        r=Router(['a','b'],phase_model=MODEL,skew=1,cost_margin_ms=0,kv_tokens=65536)
        target=item('target');place(r,target,1)
        short=admit(r,item('short',p=4096,shared=3840),0,started=True)
        long=admit(r,item('long',p=16384,shared=16128),1,started=True)
        lease=await r.acquire(target)
        c=lease.route['phase_components']
        self.assertLess(c[0]['decode_interference_ms'],c[1]['decode_interference_ms'])
        self.assertEqual(lease.worker,0)
        for x in (lease,short,long):r.release(x,True)

    async def test_long_output_short_input_is_affinity_eligible(self):
        r=Router(['a','b'],phase_model=MODEL)
        target=item('longout',p=2048,o=512,shared=1536);place(r,target,1)
        warm(r,1,target)
        lease=await r.acquire(target)
        self.assertEqual(lease.worker,1)
        self.assertEqual(lease.estimated_cached,1536)
        self.assertEqual(lease.reserved_tokens,2560)
        r.release(lease,True)

    async def test_no_prefix_uses_phase_pressure_without_owner_registration(self):
        r=Router(['a','b'],phase_model=MODEL,skew=1,cost_margin_ms=0)
        background=admit(r,item('pending'),0)
        target=item('ordinary',p=2048,o=512,shared=0);target['prefix_digest']=''
        lease=await r.acquire(target)
        self.assertEqual(lease.worker,1)
        self.assertEqual(lease.estimated_cached,0)
        self.assertNotIn(('',0),r.owners)
        for x in (lease,background):r.release(x,True)

    async def test_pending_and_ttl_cannot_fake_cache_hits(self):
        for pending in (False,True):
            r=Router(['a'],phase_model=MODEL,cache_ttl_s=1)
            target=item('target');place(r,target,0)
            background=admit(r,item('pending'),0) if pending else None
            warm(r,0,target,age=0 if pending else 2)
            lease=await r.acquire(target)
            self.assertEqual(lease.estimated_cached,0)
            self.assertEqual(lease.route['selected_cached_hint'],0)
            self.assertEqual(lease.reserved_tokens,8448)
            r.release(lease,True)
            if background:r.release(background,True)

    async def test_capacity_filter_applies_before_cost_ranking(self):
        r=Router(['a','b'],phase_model=MODEL,max_inflight=3,kv_tokens=9000)
        first=await r.acquire(item('first'))
        target=item('target');place(r,target,first.worker)
        warm(r,first.worker,target)
        lease=await r.acquire(target)
        self.assertNotEqual(lease.worker,first.worker)
        self.assertNotIn(first.worker,lease.route['capacity_candidates'])
        for x in (first,lease):r.release(x,True)
        r=Router(['a','b'],phase_model=MODEL,max_inflight=1)
        first=await r.acquire(item('first'));first.started=True
        target=item('target');place(r,target,first.worker);warm(r,first.worker,target)
        lease=await r.acquire(target)
        self.assertNotEqual(first.worker,lease.worker)
        for x in (first,lease):r.release(x,True)

    async def test_preview_pure_and_queue_uses_actual_destination_score(self):
        r=Router(['a','b'],phase_model=MODEL)
        target=item('target');place(r,target,0);warm(r,0,target)
        before=(copy.deepcopy(r.owners),list(r.reserved),list(r.owner_bytes),len(r.history))
        plan=r._preview(target,r._validate_item(target),[0,1],time.monotonic())
        self.assertEqual(before,(r.owners,r.reserved,r.owner_bytes,len(r.history)))
        lease=await r.acquire(target)
        self.assertEqual(lease.route['queue_score_ms'],lease.route['scores'][lease.worker])
        self.assertNotIn('queue_score_tokens',lease.route)
        self.assertEqual(lease.route['queue_policy'],'phase_bounded_actual_destination')
        self.assertEqual(plan['chosen'],lease.worker)
        r.release(lease,True)

    async def test_cancel_timeout_and_concurrent_release_conserve_resources(self):
        r=Router(['a'],phase_model=MODEL,max_inflight=1,queue_timeout_s=.02)
        first=await r.acquire(item('first'))
        cancel=asyncio.create_task(r.acquire(item('cancel')))
        await asyncio.sleep(.001);cancel.cancel()
        with self.assertRaises(asyncio.CancelledError):await cancel
        with self.assertRaises(TimeoutError):await r.acquire(item('timeout'))
        r.release(first,True)
        self.assertEqual(r.snapshot()['active'],[0])
        self.assertEqual(r.snapshot()['kv_reserved_tokens'],[0])
        self.assertEqual(r.snapshot()['queue_waiters'],0)
        r=Router(list('abcdefgh'),phase_model=MODEL,queue_timeout_s=5)
        async def request(i):
            lease=await r.acquire(item(i,'hot' if i%4==0 else f'doc{i%16}'))
            self.assertLessEqual(max(w.active_load for w in r.workers),3)
            self.assertLessEqual(max(r.reserved),32768)
            await asyncio.sleep(.001)
            r.release(lease,True)
        await asyncio.gather(*(request(i) for i in range(64)))
        self.assertEqual(r.snapshot()['active'],[0]*8)
        self.assertEqual(r.snapshot()['kv_reserved_tokens'],[0]*8)
        self.assertTrue(all(x['released'] for x in r.history))

    async def test_waiting_cold_request_is_forced_after_eight_bypasses(self):
        r=Router(['a'],phase_model=MODEL,max_inflight=1,queue_timeout_s=2)
        running=await r.acquire(item('running','old'))
        cold=asyncio.create_task(r.acquire(item('cold','cold')))
        warm_tasks=[asyncio.create_task(r.acquire(item(f'w{i}','warm'))) for i in range(12)]
        await asyncio.sleep(.001)
        for index in range(13):
            warm(r,0,item('hint','warm'))
            r.release(running,True)
            await asyncio.sleep(.001)
            ready=[t for t in [cold]+warm_tasks if t.done() and not t.result().released]
            self.assertEqual(len(ready),1)
            running=ready[0].result()
            if index==8:self.assertEqual(running.route['request_id'],'cold')
        r.release(running,True)
        self.assertEqual(cold.result().route['queue_bypasses'],8)
        self.assertTrue(all(t.done() for t in [cold]+warm_tasks))


if __name__=='__main__':unittest.main()
