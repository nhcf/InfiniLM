"""CPU-only protocol tests: fair pairing, mean criterion, orchestration and SSE audit."""
import io
import json
from pathlib import Path
import socket
import tempfile
import types
import unittest
from unittest.mock import patch

from aiohttp import ClientSession,ClientTimeout
import mean_suite as suite
from cache_policy import EXPECTED_SHA
from test_gateway_integration import FakeEngine,make_item


def row(arm,profile='prefix97',seed=33959,c=16,phase='measured',throughput=100,mean=100,p95=100,valid=True):
    return dict(arm=arm,profile=profile,seed=seed,concurrency=c,phase=phase,valid=valid,
                output_tok_s=throughput,request_s=throughput/64,ttft_mean_ms=mean,ttft_p95_ms=p95,
                ttft_p99_ms=p95+1,ordered_request_contract_sha256=f'{profile}-{seed}-{c}-{phase}')


class ProtocolTests(unittest.TestCase):
    def test_plan_fixed_32b_and_compact_counts(self):
        plan=suite.make_plan(suite.parse_args(['plan']),suite.configuration())
        self.assertEqual(plan['model'],'/mnt/zj-data/models/Qwen3-32B/')
        self.assertEqual(plan['shared_tokens'],7936)
        self.assertEqual(plan['cells'],15)
        self.assertEqual(plan['total_learning_and_measured_requests'],3520)
        self.assertEqual(plan['model_pool_starts'],4)
        self.assertEqual(plan['max_backend_inflight'],24)
        quick=suite.make_plan(suite.parse_args(['run','--concurrency','16','--seeds','33959','33960','--profile','replay94']),suite.configuration())
        self.assertEqual(quick['shared_tokens'],7680)
        self.assertEqual(quick['cells'],10)
        self.assertEqual(quick['model_pool_starts'],3)

    def test_pairing_keeps_profile_seed_concurrency_and_phase(self):
        records=[]
        for profile in suite.PROFILES:
            for seed in (33959,33960):
                for c in (16,32,64):
                    for phase in suite.PHASES:
                        records += [row(suite.CANDIDATE,profile,seed,c,phase,95,80,200),row('vllm_least_active',profile,seed,c,phase,100,100,100)]
        pairs=suite.paired_rows(records)
        self.assertEqual(len(pairs),36)
        self.assertTrue(all(p['mean_goal_met'] for p in pairs))
        self.assertTrue(all(not p['historical_p95_goal_met'] for p in pairs))
        self.assertTrue(all(not p['throughput_faster'] for p in pairs))
        records[-1]['valid']=False
        self.assertEqual(len(suite.paired_rows(records)),35)

    def test_no_pair_for_different_workload_order(self):
        a,b=row(suite.CANDIDATE),row('vllm_least_active')
        b['ordered_request_contract_sha256']='different'
        with self.assertRaisesRegex(ValueError,'different ordered workload'):suite.paired_rows([a,b])
        with self.assertRaisesRegex(ValueError,'Duplicate'):suite.paired_rows([a,a])

    def test_primary_mean_goal_is_strict_but_tail_is_retained(self):
        a=row(suite.CANDIDATE,throughput=95,mean=100,p95=1000);b=row('vllm_least_active')
        self.assertFalse(suite.paired_rows([a,b])[0]['mean_goal_met'])
        a['ttft_mean_ms']=99
        pair=suite.paired_rows([a,b])[0]
        self.assertTrue(pair['mean_goal_met']);self.assertEqual(pair['ttft_p95_ratio'],10)
        a['output_tok_s']=94.99
        self.assertFalse(suite.paired_rows([a,b])[0]['mean_goal_met'])

    def test_partial_or_failed_matrix_cannot_qualify(self):
        plan=suite.make_plan(suite.parse_args(['plan','--concurrency','16']),suite.configuration())
        records=[row(arm[0],phase=phase,mean=80 if arm[0]==suite.CANDIDATE else 100) for arm in suite.ARMS for phase in suite.PHASES]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);result=suite.report(root,records,plan)
            self.assertTrue(result['complete']);self.assertTrue(result['all_requested_mean_goals_met'])
            self.assertFalse(result['statistical_significance_claimed'])
            records[0]['valid']=False;result=suite.report(root,records,plan)
            self.assertFalse(result['complete']);self.assertFalse(result['all_requested_mean_goals_met'])

    def test_cache_hit_threshold_uses_selected_profile_and_unknown_stays_unknown(self):
        def record(cached):
            details={} if cached is None else {'cached_tokens':cached}
            return dict(success=True,usage={'prompt_tokens_details':details},ttft_ms=10,e2e_ms=20)
        records=[record(7680),record(7936)];routes=[dict(gateway_wait_ms=1)]*2
        self.assertEqual(suite.stats_extra(records,routes,7936)['full_prefix_hit_fraction'],.5)
        self.assertEqual(suite.stats_extra(records,routes,7680)['full_prefix_hit_fraction'],1)
        self.assertIsNone(suite.stats_extra(records+[record(None)],routes,7936)['full_prefix_hit_fraction'])

    def test_ordered_subset_hash_covers_order_and_output_contract(self):
        a,b=make_item('a'),make_item('b');h=suite.workload_selection([a,b])['ordered_request_contract_sha256']
        self.assertNotEqual(h,suite.workload_selection([b,a])['ordered_request_contract_sha256'])
        b['output_tokens']=32
        self.assertNotEqual(h,suite.workload_selection([a,b])['ordered_request_contract_sha256'])


class OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_matrix_reuses_nh_but_starts_fresh_vllm_for_every_seed_and_c(self):
        launches=[];closes=[];visited=[]
        class Pool:
            def __init__(self,cfg,engine,folder):self.engine,self.folder=engine,folder
            async def start(self):launches.append((self.engine,str(self.folder)))
            async def close(self):closes.append((self.engine,str(self.folder)))
        async def preflight(cfg,root):return {'nh_files':{'python/infinilm/llm/cache_manager.py':EXPECTED_SHA}}
        async def build(argv,env,cwd,path,timeout):
            opts=json.loads(argv[-1]);work=path.parent/'workloads';work.mkdir()
            self.assertEqual(opts['shared_tokens'],7936);self.assertEqual(opts['requests'],256)
            (work/'manifest.json').write_text(json.dumps({'cells':[dict(folder=f'seed{s}',seed=s) for s in opts['seeds']]}))
        async def cell(cfg,root,pool,case,c,arm,client,plan):
            visited.append((arm[0],case['seed'],c));records=[row(arm[0],plan['profile'],case['seed'],c,phase) for phase in suite.PHASES]
            folder=root/'cells'/f'{case["seed"]}_{c}_{arm[0]}';folder.mkdir(parents=True)
            (folder/'cell_statistics.json').write_text(json.dumps(records));return records
        cfg=suite.configuration();plan=suite.make_plan(suite.parse_args(['run','--seeds','33959','33960']),cfg)
        with tempfile.TemporaryDirectory() as temp:
            with patch('mean_suite.rt.Pool',Pool),patch('mean_suite.common.preflight',preflight),patch('mean_suite.common.run_logged',build),patch('mean_suite.run_cell',cell),patch('sys.stdout',io.StringIO()):
                result=await suite.run(cfg,Path(temp),plan)
        self.assertEqual(set(visited),{(a[0],s,c) for a in suite.ARMS for s in (33959,33960) for c in (16,32,64)})
        self.assertEqual(len(visited),30);self.assertEqual([e for e,_ in launches],['nh']+['vllm']*6)
        self.assertEqual(launches,closes);self.assertTrue(result['complete'])
        self.assertFalse(result['all_requested_mean_goals_met'])

    async def test_mean_cell_real_http_prefix97_audit(self):
        engines=[await FakeEngine(i).start() for i in range(2)]
        try:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);case=dict(folder='case',seed=1);work=root/'workloads/case';work.mkdir(parents=True)
                for phase,n in [('learn',64),('steady',16)]:
                    items=[]
                    for i in range(n):
                        x=make_item(f'{phase}{i}');x.update(key=f'key{i%3}',prefix_digest=f'prefix{i%3}',shared_tokens=7936,prompt_sha256=f'prompt-{phase}-{i}');items.append(x)
                    (work/(phase+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in items))
                (work/'smoke.json').write_text('[]')
                pool=types.SimpleNamespace(engine='nh',urls=[e.url for e in engines],models=['actual-model-0','actual-model-1'],folder=root/'pool',assert_alive=lambda:None)
                async def reset(client,pool,mode,folder):
                    for e in engines:e.prefixes.clear();e.cache_mode=mode
                async def smoke(client,pool,probes,destination=None):destination.write_text('[]')
                with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
                plan=suite.make_plan(suite.parse_args(['run','--concurrency','16','--seeds','1']),suite.configuration())
                async with ClientSession(timeout=ClientTimeout(total=8)) as client:
                    with patch('mean_suite.previous.reset_pool',reset),patch('mean_suite.common.smoke_pool',smoke),patch.dict(suite.MEASURED,{16:16}),patch('sys.stdout',io.StringIO()):
                        result=await suite.run_cell(dict(port=port,queue_timeout_s=5),root,pool,case,16,(suite.CANDIDATE,'nh','mean_bounded','prefix_lru'),client,plan)
                self.assertEqual([r['requests'] for r in result],[64,16,80])
                self.assertTrue(all(r['valid'] and r['route_valid'] and r['cache_policy_valid'] and r['cohort_count_valid'] and r['cohort_contract_valid'] for r in result))
                self.assertTrue(all(r.get('ordered_request_contract_sha256') for r in result))
                self.assertEqual(result[1]['shared_tokens'],7936)
                selection=json.loads(next((root/'cells').glob('*/workload_selection.json')).read_text())
                self.assertEqual(selection['learn']['requests'],64);self.assertEqual(selection['measured']['requests'],16)
        finally:
            for engine in engines:await engine.close()


if __name__=='__main__':unittest.main()
