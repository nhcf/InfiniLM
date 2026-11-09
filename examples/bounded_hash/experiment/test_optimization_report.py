"""CPU synthetic evidence fixtures; these values are never performance evidence."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import optimization_report as report
import test_length_report as older


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def create_fixture(root):
    # Reuse the proven fixture's workload/metric construction with three profiles
    # and distinct candidate arm names. Then add real-format routing evidence.
    with patch.multiple(older.report, CASES=report.CASES, ARMS={'v3':report.V3,'v2':report.V2,'vllm':report.VLLM},
                        V2=report.V3, MEAN=report.V2):
        older.create_fixture(root)
    plan = json.loads((root/'plan.json').read_text())
    plan.update(cells=9, expected_phase_rows=27)
    dump(root/'plan.json', plan)
    sources = {}
    for module in ('cache_router_v3', 'cache_router', 'mean_router'):
        path = root/'source'/(module+'.py')
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve().parent/(module+'.py'), path)
        sources[module+'.py'] = report.runtime.sha(path)
    dump(root/'fingerprint.json', dict(candidate_sources=sources))
    for cell in root.glob('rounds/round01/cells/*'):
        job=json.loads((cell/'order.json').read_text());arm=job['arm']
        dump(cell/'status.json',dict(status='COMPLETE'))
        module = {report.V3:'cache_router_v3',report.V2:'cache_router',report.VLLM:'mean_router'}[arm]
        dump(cell/'router_source_selection.json',dict(arm=arm,router_class=module+'.Router',
            source_file=module+'.py',source_sha256=sources[module+'.py'],restored_after_cell=True,
            engine='vllm' if arm==report.VLLM else 'nh',
            backend_policy='least_active' if arm==report.VLLM else 'cache_bounded_v2'))
        cumulative=[]
        for phase in ('learn','measured'):
            records=json.loads((cell/(phase+'_requests.json')).read_text())
            for i,r in enumerate(records):
                cached = 0 if i==0 else (None if arm==report.VLLM and i==2 else 7936)
                if cached is not None:r['usage']['prompt_tokens_details']=dict(cached_tokens=cached)
                payload=dict(event_payload_sha256='a'*64,event_payload_count=r['usage']['completion_tokens']+2,event_payload_bytes=1000)
                r.update(payload)
                cumulative.append(dict(request_id=r['id'],chosen=0,released=True,success=True,
                    upstream_http_attempts=1,upstream_force_close=True,
                    generated_tokens=0 if arm==report.VLLM else r['usage']['completion_tokens'],active_before=[0]*8,
                    kv_reserved_before=[0]*8,kv_reserved_tokens=8448,observed_cached_tokens=cached,
                    gateway_wait_ms=.01,admission_observation=dict(status='OBSERVED',chosen_worker=0,
                        workers=[dict(worker=w,pending_request_count=1 if w==0 and i==1 else 0) for w in range(8)]),**payload))
            if arm == report.V3:
                for event in cumulative:
                    event.update(optimization_version='cache_bounded_v3', v3_warm_wait=dict(
                        policy='cache_bounded_v3', checks=1, deferred_checks=0,
                        waited_for_warm_copy=False, final_worker=event['chosen'],
                        prediction_is_not_verified_hit=True, actual_gateway_wait_ms=event['gateway_wait_ms']))
            dump(cell/(phase+'_requests.json'),records)
            dump(cell/(phase+'_routes.json'),dict(policy=report.POLICIES[arm],active=[0]*8,
                kv_reserved_tokens=[0]*8,queue_timeouts=0,queue_cancellations=0,events=list(cumulative)))


class OptimizationReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);create_fixture(self.root)

    def change_json(self, pattern, fn):
        path=next(self.root.glob(pattern));value=json.loads(path.read_text());fn(value);dump(path,value)

    def test_complete_report_recomputes_metrics_and_keeps_unknown_separate(self):
        result=report.analyze(self.root)
        self.assertTrue(result['complete'],result)
        self.assertEqual(result['actual_cells'],9)
        counts=result['phase_summary']['measured']['comparisons']['v3_vs_vllm']
        self.assertEqual(counts['higher'],2);self.assertEqual(counts['at_least_2pct'],1)
        self.assertEqual(counts['both_metrics_better'],0)
        with (self.root/'optimization_cases.csv').open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),6)
        self.assertEqual(rows[0]['v3_zero_cache_requests'],'1')
        self.assertEqual(rows[0]['vllm_unknown_cache_requests'],'1')
        self.assertEqual(rows[0]['vllm_zero_cache_requests'],'1')
        self.assertEqual(rows[3]['v3_zero_cache_requests'],'2')
        self.assertEqual(rows[0]['v3_admissions_with_pending_prefill'],'1')

    def test_candidate_validator_restored_after_success_and_failure(self):
        before = report.summarize_confirmation.CANDIDATE
        self.assertTrue(report.analyze(self.root)['complete'])
        self.assertEqual(report.summarize_confirmation.CANDIDATE, before)
        with (self.root/'confirmation_pairs.csv').open() as stream:
            pairs = list(csv.DictReader(stream))
        self.assertTrue(all(p['candidate'] == report.V3 for p in pairs))
        with patch.object(report.summarize_confirmation, 'analyze', side_effect=ValueError('injected validator failure')):
            self.assertFalse(report.analyze(self.root)['complete'])
        self.assertEqual(report.summarize_confirmation.CANDIDATE, before)

    def test_source_receipt_must_match_saved_fingerprint_and_source(self):
        self.change_json('rounds/round01/cells/*nh_cache_bounded_v3/router_source_selection.json',
                         lambda x:x.update(source_sha256='0'*64))
        self.assertIn('source fingerprint mismatch',report.analyze(self.root)['error'])

    def test_wrong_router_identity_fails(self):
        self.change_json('rounds/round01/cells/*nh_cache_bounded_v3/measured_routes.json',lambda x:x.update(policy='cache_bounded_v2'))
        result=report.analyze(self.root);self.assertFalse(result['complete'])
        self.assertIn('router policy identity',result['error']);self.assertNotIn('phase_summary',result)

    def test_duplicate_route_fails(self):
        self.change_json('rounds/round01/cells/*nh_cache_bounded_v3/measured_routes.json',lambda x:x['events'].append(x['events'][-1]))
        self.assertIn('duplicate request routes',report.analyze(self.root)['error'])

    def test_raw_usage_corruption_fails(self):
        self.change_json('rounds/round01/cells/*/measured_requests.json',lambda x:x[0]['usage'].update(completion_tokens=1))
        self.assertIn('incorrect token usage',report.analyze(self.root)['error'])

    def test_unknown_nh_cache_fails(self):
        self.change_json('rounds/round01/cells/*nh_cache_bounded_v3/measured_requests.json',lambda x:x[0]['usage'].pop('prompt_tokens_details'))
        self.assertIn('NH actual cache usage',report.analyze(self.root)['error'])

    def test_changed_inputs_across_output_length_fails_even_if_hashes_updated(self):
        name='p8192_o64';path=self.root/'workloads'/name/'learn.jsonl'
        rows=[json.loads(s) for s in path.read_text().splitlines()];rows[0]['body']['messages'][0]['content']+='changed'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        shutil.copy2(path,self.root/'rounds/round01/workloads'/name/'learn.jsonl')
        manifest=self.root/'workloads/manifest.json';value=json.loads(manifest.read_text())
        next(c for c in value['cells'] if c['profile']==name)['files']['learn']['sha256']=report.runtime.sha(path)
        dump(manifest,value)
        self.assertIn('output length changed inputs',report.analyze(self.root)['error'])

    def test_merged_timing_corruption_fails(self):
        def edit(rows):
            rows[-1]['wall_s']*=2;rows[-1]['output_tok_s']=rows[-1]['total_output_tokens']/rows[-1]['wall_s']
        self.change_json('rounds/round01/cells/*/cell_statistics.json',edit)
        self.assertIn('combined wall time',report.analyze(self.root)['error'])

    def test_incomplete_run_rejected(self):
        dump(self.root/'status.json',dict(complete=False));self.assertFalse(report.analyze(self.root)['complete'])

    def test_bad_payload_evidence_rejected(self):
        self.change_json('rounds/round01/cells/*nh_cache_bounded_v3/measured_routes.json',lambda x:x['events'][-1].update(event_payload_sha256='b'*64))
        self.assertIn('stream payload evidence differs',report.analyze(self.root)['error'])

    def test_full_reservation_required(self):
        self.change_json('rounds/round01/cells/*nh_cache_bounded_v3/measured_routes.json',lambda x:x['events'][-1].update(kv_reserved_tokens=512))
        self.assertIn('full KV reservation violated',report.analyze(self.root)['error'])

    def test_wait_evidence_is_reported_with_actual_outcome(self):
        cell=next(self.root.glob('rounds/round01/cells/*nh_cache_bounded_v3'))
        records=json.loads((cell/'measured_requests.json').read_text())
        file=cell/'measured_routes.json';state=json.loads(file.read_text())
        event=next(e for e in state['events'] if e['request_id']==records[0]['id'])
        event['owner_wait_requested']=True
        event['v3_warm_wait'].update(checks=2,deferred_checks=1,waited_for_warm_copy=True,
            first_candidates=[dict(worth_waiting=True,pending_requests=0,fits_after_first_release=True,
                evidence='fresh_cache_snapshot',observed_cached_tokens=7936),
                dict(worth_waiting=True,pending_requests=0,fits_after_first_release=True,
                evidence='active_matching_decode_prediction',observed_cached_tokens=0)])
        dump(file,state)
        result=report.audit_routes(cell,'measured',report.V3,records)
        self.assertEqual(result['warm_wait_requests'],1)
        self.assertEqual(result['warm_wait_zero_cache_requests'],1)
        self.assertEqual(result['warm_wait_full_cache_requests'],0)
        self.assertEqual(result['warm_wait_first_fresh_snapshot_candidates'],1)
        self.assertEqual(result['warm_wait_first_active_prediction_candidates'],1)
        self.assertAlmostEqual(result['warm_wait_gateway_total_ms'],.01)
        event['v3_warm_wait']['actual_gateway_wait_ms']=999
        dump(file,state)
        with self.assertRaisesRegex(ValueError,'V3 actual gateway wait'):
            report.audit_routes(cell,'measured',report.V3,records)

    def test_slow_full_cache_pending_classification(self):
        cell=next(self.root.glob('rounds/round01/cells/*nh_cache_bounded_v3'))
        records=json.loads((cell/'measured_requests.json').read_text())
        records[0]['ttft_ms']=4500.;records[1]['ttft_ms']=4500.
        result=report.audit_routes(cell,'measured',report.V3,records)
        self.assertEqual(result['slow_requests_ge4s'],2)
        self.assertEqual(result['slow_zero_cache_requests'],1)
        self.assertEqual(result['slow_full_cache_requests'],1)
        self.assertEqual(result['slow_full_cache_with_pending'],1)


if __name__=='__main__':unittest.main()
