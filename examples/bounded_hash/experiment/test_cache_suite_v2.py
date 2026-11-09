import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import cache_suite as suite
import cache_workload as workload
from cache_router import Router
from gateway import Gateway
from streaming import request_one
from test_gateway_integration import FakeEngine, make_item, payload_signature
from test_phase_router import MODEL


class SuiteTests(unittest.TestCase):
    def test_prediction_error_is_reported_separately_from_cache_hint_error(self):
        records=[dict(success=True,usage=dict(prompt_tokens=8192,completion_tokens=32,
            prompt_tokens_details=dict(cached_tokens=0)),ttft_ms=10.,e2e_ms=50.)]
        routes=[dict(predicted_cached_tokens=7936,estimated_cached_tokens=0,observed_cached_tokens=0,
            gateway_wait_ms=1.,hint_overestimated=False,owner_wait_requested=True)]
        x=suite.extra_stats(records,routes,7936)
        self.assertEqual(x['cache_hint_overestimates'],0)
        self.assertEqual(x['predicted_reuse_admissions'],1)
        self.assertEqual(x['predicted_reuse_overestimates'],1)
        self.assertEqual(x['owner_wait_admissions'],1)

    def test_summary_ignores_report_zip_and_handles_failed_metrics(self):
        from summarize_cache import latest_run,summarize
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'cache_v2_run_001';run.mkdir()
            (root/'cache_v2_run_999_report.zip').write_bytes(b'fixture')
            (run/'status.json').write_text(json.dumps(dict(status='FAILED')))
            (run/'summary.csv').write_text('profile,seed,concurrency,arm,phase,output_tok_s,ttft_mean_ms,ttft_p95_ms,failures,valid\nreuse8,33959,16,nh_cache_bounded_v2,measured,,,,1,False\n')
            self.assertEqual(latest_run(root),run)
            text=summarize(run)
            self.assertIn('STATUS=FAILED',text);self.assertIn('tok/s=NA',text)

    def test_default_is_three_favorable_c16_cells(self):
        a=suite.arguments([]);p=suite.make_plan(a,suite.configuration(a.config))
        self.assertEqual(a.mode,'plan');self.assertEqual(p['cells'],3)
        self.assertEqual(p['concurrency'],[16]);self.assertEqual(list(p['profiles']),['reuse8'])
        self.assertEqual(p['total_learning_plus_measured_requests'],576)
        self.assertEqual([x[0] for x in p['arms']],['nh_cache_bounded_v2','nh_native_rr','vllm_least_active'])
        self.assertFalse(p['performance_win_claimed']);self.assertEqual(p['gpu_jobs_started'],0)

    def test_per_request_output_field_is_distinct_from_total(self):
        records=[dict(success=True,usage=dict(prompt_tokens=8192,completion_tokens=32),ttft_ms=10.,e2e_ms=50.) for _ in range(3)]
        r=suite.phase_statistics(records,2.,dict(output_tokens=32))
        self.assertEqual(r['output_tokens_per_request'],32);self.assertEqual(r['total_output_tokens'],96)
        self.assertEqual(r['output_tok_s'],48.)

    def test_invalid_cohort_suppresses_throughput(self):
        records=[dict(success=False)]
        r=suite.phase_statistics(records,2.,dict(output_tokens=32))
        self.assertFalse(r['valid']);self.assertIsNone(r['output_tok_s'])

    def test_complete_report_is_not_a_performance_win(self):
        a=suite.arguments(['plan','--arms','nh_cache_bounded_v2','vllm_least_active'])
        p=suite.make_plan(a,suite.configuration(a.config));rows=[]
        for arm,t,m in [('nh_cache_bounded_v2',90.,1500.),('vllm_least_active',100.,1000.)]:
            for phase in ('learn','measured','learn_plus_measured'):
                rows.append(dict(profile='reuse8',seed=33959,concurrency=16,arm=arm,phase=phase,valid=True,
                    output_tok_s=t,ttft_mean_ms=m,ttft_p95_ms=m,ttft_p99_ms=m,e2e_mean_ms=3000.,
                    failures=0,ordered_request_contract_sha256='paired'))
        with tempfile.TemporaryDirectory() as d:
            r=suite.report(Path(d),rows,p)
            self.assertTrue(r['complete']);self.assertFalse(r['performance_win_claimed'])
            self.assertEqual(r['candidate_both_metrics_better_count'],0)
            rows[3]['ordered_request_contract_sha256']='different'
            with self.assertRaisesRegex(ValueError,'cohorts'):suite.report(Path(d),rows,p)

    def test_request_order_has_no_hash_ring_dependency(self):
        seq=workload.arrivals(8,128,.25,33959)
        self.assertEqual(seq,workload.arrivals(8,128,.25,33959))
        self.assertEqual(len(seq),128);self.assertEqual(len(set(seq)),8)
        for i in (0,64):
            self.assertEqual(max(seq[i:i+64].count(k) for k in range(8)),16)
        self.assertNotEqual(seq,workload.arrivals(8,128,.25,33960))

    def test_workload_counts_prefix_identity_and_untimed_warmup(self):
        # Structural fixture only: real token lengths are checked with the
        # installed tokenizer during workload build and exact engine usage.
        def doc(tok,seed,name,shared):return f'{seed}-{name}'
        def row(tok,doc,ident,p,o,shared):
            h=lambda x:hashlib.sha256(x.encode()).hexdigest()
            return dict(id=ident,key=h(doc),prefix_digest=h(doc+str(shared)),
                first_block_digest=h(doc),prompt_sha256=h(doc+ident),prompt_tokens=p,
                output_tokens=o,shared_tokens=shared,body={})
        def calibration(tok,out,used):out.mkdir(parents=True)
        with tempfile.TemporaryDirectory() as d,patch.object(workload,'document',doc),patch.object(workload,'row',row),patch.object(workload,'build_calibration',calibration):
            root=Path(d)
            m=workload.build(None,root,['reuse8','reuse16','cold'],[33959],64)
            self.assertEqual(len(m['cells']),3)
            for c in m['cells']:
                self.assertEqual(c['files']['learn']['count'],64)
                self.assertEqual(c['files']['measured']['count'],64)
                self.assertEqual(c['unique_full_inputs'],128)
                rows=[json.loads(x) for x in (root/c['folder']/'measured.jsonl').read_text().splitlines()]
                self.assertEqual(len({x['key'] for x in rows}),64 if c['profile']=='cold' else c['documents'])
            self.assertEqual(len((root/'calibration/untimed_warmup.jsonl').read_text().splitlines()),3)

    def test_runtime_features_do_not_confuse_autotune_with_backend(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'launch.json').write_text(json.dumps([dict(argv=['python','--enable-prefix-caching','--enforce-eager','--enable-chunked-prefill'])]))
            (root/'worker0.log').write_text('enable_flashinfer_autotune=True\nMUSA platform use FLASH_ATTN with version 3.\n')
            f=suite.runtime_features(types.SimpleNamespace(folder=root,engine='vllm'))
            self.assertTrue(f['requested']['prefix_cache']);self.assertFalse(f['requested']['cuda_graph'])
            self.assertEqual(len(f['attention_backend_log_evidence']),1)
            self.assertIn('FLASH_ATTN',f['attention_backend_log_evidence'][0]['line'])


class NewRouterGatewayTest(unittest.IsolatedAsyncioTestCase):
    async def test_new_router_preserves_sse_payload_usage_and_reservations(self):
        from aiohttp import ClientSession,ClientTimeout
        engines=[await FakeEngine(i).start() for i in range(2)]
        backends=[e.url for e in engines]
        router=Router(backends,phase_model=MODEL,owner_wait_ms=0)
        gateway=Gateway(backends,['model0','model1'],router,0,'nh')
        try:
            await gateway.start()
            url='http://127.0.0.1:'+str(gateway.runner.addresses[0][1])
            async with ClientSession(timeout=ClientTimeout(total=5)) as client:
                records=await asyncio.gather(*(request_one(client,url,make_item('v2-'+str(i)),'alias') for i in range(16)))
            self.assertTrue(all(r['success'] for r in records))
            self.assertTrue(all(r['usage']['completion_tokens']==64 for r in records))
            await asyncio.sleep(.01)
            self.assertEqual(sum(router.reserved),0);self.assertEqual(len(router._waiters),0)
            for e in router.history:
                signature=payload_signature(engines[e['chosen']].payloads[e['request_id']])
                self.assertEqual(signature[0],e['event_payload_sha256'])
                self.assertTrue(e['released'])
                self.assertEqual(e['queue_selected_worker'],e['chosen'])
        finally:
            await gateway.close();router.close()
            for e in engines:await e.close()


if __name__=='__main__':unittest.main()
