"""CPU checks of cache ownership, controlled routing and C16/C32/C64 pairing.

These are correctness tests, not GPU performance claims.
"""
import asyncio
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from cache_policy import install_manager, reset_idle, EXPECTED_SHA
from nh_stream_usage import PrefixObservations
from stable_router import Router
from matrix import paired_rows, make_plan, configuration, audit, run_cell
from test_gateway_integration import FakeEngine, make_item
from aiohttp import ClientSession, ClientTimeout


def manager_class(patched=True):
    path = Path(__file__).parent/'vendor/nhcf_cache/cache_manager.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_SHA
    name = '_test_matrix_cache_'+str(len(sys.modules))
    stub = types.ModuleType('infinilm.llm.prefix_cache')
    stub.BlockHash, stub.EMPTY_BLOCK_HASH = bytes, b''
    with patch.dict(sys.modules, {stub.__name__:stub}):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    sys.modules[name] = module
    if patched:
        install_manager(module.BlockManager)
    return module.BlockManager


def replay(cls, mode, count=90):
    bm=cls(128,256)
    if mode is not None:
        bm._matrix_mode=mode
    hits=[]
    for step in range(count):
        hashes=[hashlib.blake2b(f'd{step%3}:p{i}'.encode(),digest_size=16).digest() for i in range(30)]
        hashes += [hashlib.blake2b(f'r{step}:s{i}'.encode(),digest_size=16).digest() for i in range(2)]
        table,hit=bm.get_computed_blocks(hashes,8191);hits.append(hit)
        table,_=bm.allocate_slots(8192-hit,hit,table)
        bm.publish_computed_blocks(table,hashes,hit//256,8192)
        for length in range(8193,8256):
            table,_=bm.append_slot(table,length)
        bm.free_blocks(table)
        assert all(b.ref_count==0 for b in bm.blocks)
    return hits,bm


class CachePolicyTests(unittest.TestCase):
    def test_native_mode_preserves_original_eviction_trace(self):
        original,_=replay(manager_class(False),None)
        native,_=replay(manager_class(),'native')
        self.assertEqual(native,original)

    def test_three_document_working_set_retains_all_reusable_prefixes(self):
        original,_=replay(manager_class(False),None)
        optimized,bm=replay(manager_class(),'prefix_lru')
        self.assertGreater(original[3:].count(0),0)
        self.assertEqual(optimized[:3],[0,0,0])
        self.assertEqual(optimized[3:],[7680]*87)
        self.assertGreater(bm._matrix_evictions,0)

    def test_shared_blocks_stay_pinned_until_last_release(self):
        bm=manager_class()(5,4);bm._matrix_mode='prefix_lru'
        a,_=bm.allocate_slots(8);bm.publish_computed_blocks(a,[b'a',b'b'],0,8)
        b,hit=bm.get_computed_blocks([b'a',b'b'],8)
        self.assertEqual(hit,8)
        bm.free_blocks(a)
        self.assertTrue(all(bm.blocks[i].ref_count==1 for i in b))
        self.assertFalse(bm.try_free_blocks(5))
        self.assertTrue(all(i in bm.used_block_ids for i in b))
        bm.free_blocks(b)
        self.assertTrue(bm.try_free_blocks(5))
        self.assertFalse(bm.hash_to_block_ids)

    def test_retouch_removes_cached_blocks_from_evictable_queue(self):
        bm=manager_class()(3,4);bm._matrix_mode='prefix_lru'
        a,_=bm.allocate_slots(8);bm.publish_computed_blocks(a,[b'a',b'b'],0,8);bm.free_blocks(a)
        reused,hit=bm.get_computed_blocks([b'a',b'b'],8)
        self.assertEqual(hit,8);self.assertFalse(bm._matrix_idle)
        bm.try_free_blocks(2)
        self.assertTrue(all(bm.blocks[i].ref_count==1 for i in reused))

    def test_idle_reset_refuses_live_ownership_or_waiters(self):
        bm=manager_class()(8,4)
        s=types.SimpleNamespace(cache_manager=bm,_fourway_observations=PrefixObservations(),
                                waiting_queue=types.SimpleNamespace(sync_q=queue.Queue()),
                                running_queue=types.SimpleNamespace(sync_q=queue.Queue()))
        table,_=bm.allocate_slots(4)
        self.assertFalse(reset_idle(s,'prefix_lru')['success'])
        bm.free_blocks(table);s.waiting_queue.sync_q.put(object())
        self.assertFalse(reset_idle(s,'prefix_lru')['success'])
        s.waiting_queue.sync_q.get()
        self.assertFalse(reset_idle(s,'prefix_lru',http_active=1)['success'])
        self.assertTrue(reset_idle(s,'prefix_lru')['success'])
        self.assertIs(s.cache_manager,bm)
        self.assertEqual(bm._matrix_mode,'prefix_lru')
        self.assertEqual(len(bm.free_block_ids),8)
        self.assertTrue(reset_idle(s,'native')['success'])


def item(i,key=None):
    return dict(id=str(i),key=key or f'doc{i}',prefix_digest=key or f'digest{i}',
                prompt_tokens=8192,output_tokens=64,shared_tokens=7680)


class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_affinity_queue_uses_available_resident_prefix(self):
        r=Router(['a'],'stable_bounded',max_inflight=1,queue_timeout_s=2)
        running=await r.acquire(item('running','old'))
        cold=asyncio.create_task(r.acquire(item('cold','cold')))
        warm=asyncio.create_task(r.acquire(item('warm','warm')))
        await asyncio.sleep(.001)
        r.cache[0].update(epoch=1,observed_at=time.monotonic(),prefixes={'warm':7680})
        r.release(running,True)
        chosen=await warm
        self.assertEqual(chosen.route['queue_position_on_admission'],1)
        self.assertFalse(cold.done())
        r.release(chosen,True)
        old=await cold
        self.assertEqual(old.route['queue_bypasses'],1)
        r.release(old,True)

    async def test_affinity_queue_aging_and_cancel_have_no_lease_leak(self):
        r=Router(['a'],'stable_bounded',max_inflight=1,queue_timeout_s=2)
        running=await r.acquire(item('running','old'))
        cold=asyncio.create_task(r.acquire(item('cold','cold')))
        warm=[asyncio.create_task(r.acquire(item(f'w{i}','warm'))) for i in range(12)]
        cancelled=asyncio.create_task(r.acquire(item('cancel','x')))
        await asyncio.sleep(.001);cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):await cancelled
        for index in range(13):
            r.cache[0].update(epoch=index+1,observed_at=time.monotonic(),prefixes={'warm':7680})
            r.release(running,True)
            await asyncio.sleep(.001)
            ready=[t for t in [cold]+warm if t.done() and not t.result().released]
            self.assertEqual(len(ready),1)
            running=ready[0].result()
            if index==8:self.assertEqual(running.route['request_id'],'cold')
        r.release(running,True)
        self.assertTrue(all(t.done() for t in [cold]+warm))
        self.assertLessEqual(max(e['queue_bypasses'] for e in r.history),8)
        self.assertEqual(r.snapshot()['active'],[0])
        self.assertEqual(r.snapshot()['queue_waiters'],0)
        self.assertEqual(r.snapshot()['queue_cancellations'],1)

    async def test_round_robin_rotates_and_skips_full_workers(self):
        r=Router(['a','b'],'round_robin',max_inflight=1,queue_timeout_s=2)
        a,b=await r.acquire(item(1)),await r.acquire(item(2))
        self.assertEqual([a.worker,b.worker],[0,1])
        task=asyncio.create_task(r.acquire(item(3)));await asyncio.sleep(.001)
        self.assertFalse(task.done())
        r.release(b,True);c=await task
        self.assertEqual(c.worker,1)
        self.assertEqual(c.route['branch'],'round_robin_capacity_eligible')
        r.release(a,True);r.release(c,True)

    async def test_primary_placement_is_stable_and_prefix_bytes_bounded(self):
        r=Router(list('abcdefgh'),'stable_bounded')
        home={}
        for cycle in range(3):
            for k in range(16):
                lease=await r.acquire(item(f'{cycle}-{k}',f'doc{k}'))
                home.setdefault(k,lease.preferred)
                self.assertEqual(home[k],lease.preferred)
                r.release(lease,True)
        self.assertLessEqual(max(r.owner_bytes),3*7680)
        self.assertEqual(sum(r.owner_bytes),16*7680)

    async def test_outside_long_prefix_uses_least_active(self):
        r=Router(list('abcd'),'stable_bounded');base=Router(list('abcd'),'least_active')
        for i in range(8):
            x=item(i);x.update(prompt_tokens=2048,shared_tokens=1536,output_tokens=512)
            a,b=await r.acquire(x),await base.acquire(x)
            self.assertEqual(a.worker,b.worker)
        self.assertEqual(r.owners,{})

    async def test_c16_c32_c64_drain_without_exceeding_capacity(self):
        for concurrency in (16,32,64):
            r=Router(list('abcdefgh'),'stable_bounded',queue_timeout_s=5)
            async def request(i):
                lease=await r.acquire(item(i,'hot' if i%4==0 else f'doc{i%16}'))
                self.assertLessEqual(max(w.active_load for w in r.workers),3)
                self.assertLessEqual(max(r.reserved),32768)
                await asyncio.sleep(.001)
                r.release(lease,True)
            await asyncio.gather(*(request(i) for i in range(concurrency)))
            snap=r.snapshot()
            self.assertEqual(snap['active'],[0]*8)
            self.assertEqual(snap['kv_reserved_tokens'],[0]*8)
            self.assertEqual(snap['queue_waiters'],0)
            self.assertEqual(len(snap['events']),concurrency)
            self.assertTrue(all(len(v)<=2 for v in r.owners.values()))


class ReportTests(unittest.TestCase):
    def test_pairs_match_concurrency_not_last_row(self):
        rows=[]
        for c in (16,32,64):
            for arm,gain in [('nh_lru_stable_bounded',.9),('vllm_least_active',1)]:
                rows.append(dict(arm=arm,seed=1,concurrency=c,phase='measured',valid=True,
                                 output_tok_s=c*gain,ttft_p95_ms=c*100/gain))
        pairs=paired_rows(rows)
        self.assertEqual(len(pairs),3)
        self.assertTrue(all(abs(p['throughput_ratio']-.9)<1e-9 for p in pairs))
        rows[-1]['valid']=False
        self.assertEqual(len(paired_rows(rows)),2)

    def test_exact_model_and_small_matrix(self):
        plan=make_plan(configuration())
        self.assertEqual(plan['model'],'/mnt/zj-data/models/Qwen3-32B/')
        self.assertEqual(plan['concurrency'],[16,32,64])
        self.assertEqual(plan['cells'],15)
        self.assertEqual(plan['total_learning_and_measured_requests'],2400)


class MatrixIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_cell_stream_audit_and_reset_with_stable_router(self):
        engines=[await FakeEngine(i).start() for i in range(2)]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);case=dict(folder='case',seed=1)
            work=root/'workloads/case';work.mkdir(parents=True)
            def workload(prefix,n):
                rows=[]
                for i in range(n):
                    x=make_item(f'{prefix}{i}');x.update(key=f'key{i%3}',prefix_digest=f'prefix{i%3}')
                    rows.append(x)
                return rows
            for phase,n in [('learn',4),('steady',16)]:
                (work/(phase+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in workload(phase,n)))
            (work/'smoke.json').write_text('[]')
            pool=types.SimpleNamespace(engine='nh',urls=[e.url for e in engines],models=['actual-model-0','actual-model-1'],
                                       folder=root/'pool',assert_alive=lambda:None)
            async def reset(client,pool,mode,folder):
                for e in engines:
                    e.prefixes.clear();e.cache_mode=mode
            async def smoke(client,pool,probes,destination=None):
                destination.write_text('[]')
            try:
                async with ClientSession(timeout=ClientTimeout(total=5)) as client:
                    with patch('matrix.reset_pool',reset),patch('matrix.common.smoke_pool',smoke),patch('sys.stdout',io.StringIO()):
                        # Bind a real gateway on an ephemeral port; run_cell
                        # normally uses its fixed configured port.
                        import socket
                        with socket.socket() as s:
                            s.bind(('127.0.0.1',0));port=s.getsockname()[1]
                        result=await run_cell(dict(port=port,queue_timeout_s=3),root,pool,case,16,
                                              ('nh_lru_stable_bounded','nh','stable_bounded','prefix_lru'),client)
                self.assertEqual(len(result),3)
                self.assertTrue(all(r['valid'] for r in result))
                self.assertTrue(all(r['route_valid'] for r in result))
                self.assertEqual(result[1]['requests'],16)
            finally:
                for engine in engines:await engine.close()

    async def test_orchestration_runs_15_cells_with_four_model_pool_starts(self):
        import matrix
        launches=[];closes=[];visited=[]
        class Pool:
            def __init__(self,cfg,engine,folder):self.engine,self.folder=engine,folder
            async def start(self):launches.append((self.engine,str(self.folder)))
            async def close(self):closes.append((self.engine,str(self.folder)))
        async def preflight(cfg,root):
            return {'nh_files':{'python/infinilm/llm/cache_manager.py':EXPECTED_SHA}}
        async def build(argv,env,cwd,path,timeout):
            work=path.parent/'workloads';work.mkdir()
            (work/'manifest.json').write_text(json.dumps({'cells':[dict(folder='case',seed=33959)]}))
        async def cell(cfg,root,pool,case,c,arm,client):
            visited.append((arm[0],c))
            rows=[dict(arm=arm[0],seed=33959,concurrency=c,phase=phase,valid=True,
                       output_tok_s=10,ttft_p95_ms=100) for phase in ('learn','measured','learn_plus_measured')]
            folder=root/'cells'/f'c{c}_{arm[0]}';folder.mkdir(parents=True)
            (folder/'cell_statistics.json').write_text(json.dumps(rows))
            return rows
        with tempfile.TemporaryDirectory() as temp:
            cfg=configuration()
            with patch('matrix.rt.Pool',Pool),patch('matrix.common.preflight',preflight),patch('matrix.common.run_logged',build),patch('matrix.run_cell',cell):
                result=await matrix.run(cfg,Path(temp))
            self.assertEqual(set(visited),{(arm[0],c) for arm in matrix.ARMS for c in (16,32,64)})
            self.assertEqual(len(visited),15)
            self.assertEqual([x[0] for x in launches],['nh','vllm','vllm','vllm'])
            self.assertEqual(launches,closes)
            self.assertTrue(result['complete'])


if __name__=='__main__':
    unittest.main()
