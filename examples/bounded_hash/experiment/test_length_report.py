"""Exercise raw-evidence and throughput-only reporting on a synthetic 18-cell run."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from cache_suite import contract
import length_report as report


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def create_fixture(root, c=16):
    manifest, profiles, jobs = [], {}, []
    cells = root/'rounds/round01/cells'
    for index, (name, (p,o)) in enumerate(report.CASES.items()):
        params = dict(prompt_tokens=p, output_tokens=o, shared_tokens=p-256,
                      documents=8, hot_fraction=.25)
        profiles[name] = params
        case = dict(profile=name, seed=report.SEED, folder=name, files={}, **params)
        cohorts = {}
        for phase, count in [('learn',64),('measured',128)]:
            items = []
            for i in range(count):
                rid = f'{phase}-{i}'
                body = dict(messages=[dict(role='user', content=f'{p}-{rid}')], max_tokens=o)
                items.append(dict(id=rid, key=f'doc-{i%8}', first_block_digest=str(i%8),
                        prefix_digest=f'{p}-{i%8}', prompt_sha256=hashlib.sha256(f'{p}-{rid}'.encode()).hexdigest(),
                        prompt_tokens=p, output_tokens=o, shared_tokens=p-256,body=body))
            path = root/'workloads'/name/(phase+'.jsonl')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(''.join(json.dumps(r)+'\n' for r in items))
            case['files'][phase] = dict(count=count,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            cohorts[phase] = items
        manifest.append(case)
        order = list(report.ARMS.values())
        order = order[index%3:] + order[:index%3]
        for pos, arm in enumerate(order, 1):
            folder = cells/(name+'_'+arm)
            job = dict(serial=len(jobs)+1,round=1,seed=report.SEED,profile=name,concurrency=c,
                       order_position=pos,arm=arm,engine='vllm' if arm==report.VLLM else 'nh')
            jobs.append(job)
            dump(folder/'order.json',dict(job,state='COMPLETE'))
            dump(folder/'gateway_errors.json',[])
            rate = {report.V2:101.0, report.MEAN:100.0,report.VLLM:100.0}[arm] if index==0 else (
                   {report.V2:102.0,report.MEAN:99.,report.VLLM:100.}[arm] if index%2 else 100.)
            latency = 2000.0 if arm==report.V2 else 1000.0
            rows=[]
            for phase in ['learn','measured','learn_plus_measured']:
                if phase=='learn_plus_measured':
                    items = cohorts['learn']+cohorts['measured']
                else:
                    items = cohorts[phase]
                    raw = [dict(id=r['id'],key=r['key'],prompt_sha256=r['prompt_sha256'],
                                ttft_ms=latency,success=True,http_status=200,
                                usage=dict(prompt_tokens=p,completion_tokens=o)) for r in items]
                    dump(folder/(phase+'_requests.json'),raw[::-1])
                total = len(items)*o
                rows.append(dict(profile=name,seed=report.SEED,concurrency=c,arm=arm,phase=phase,
                                 requests=len(items),success=len(items),failures=0,
                                 total_output_tokens=total,wall_s=total/rate,output_tok_s=rate,
                                 ttft_mean_ms=latency,ttft_p95_ms=latency,
                                 ordered_request_contract_sha256=contract(items),valid=True,
                                 route_valid=True,exact_usage=True,cohort_valid=True,
                                 event_payload_transparent=True,cache_policy_valid=True))
            dump(folder/'cell_statistics.json',rows)
    dump(root/'workloads/manifest.json',dict(cells=manifest))
    shutil.copytree(root/'workloads',root/'rounds/round01/workloads')
    dump(root/'plan.json',dict(rounds=1,seeds=[report.SEED],concurrency=[c],arms=list(report.ARMS.values()),
                              profiles=profiles,cells=18,execution_order=jobs,expected_phase_rows=54,
                              learning_requests=64,measured_requests=128,fresh_pool_per_cell=True))
    dump(root/'status.json',dict(complete=True,status='EXPERIMENT_COMPLETE'))
    dump(root/'postprocess_status.json',dict(complete=True))


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        create_fixture(self.root)

    def test_complete_six_case_report_counts_small_wins_two_percent_and_ties(self):
        result = report.analyze(self.root)
        self.assertTrue(result['complete'], result)
        phase = result['phase_summary']['measured']
        self.assertEqual(phase['comparisons']['v2_vs_vllm'],dict(higher=4,at_least_2pct=3))
        self.assertEqual(phase['comparisons']['mean_vs_vllm'],dict(higher=0,at_least_2pct=0))
        self.assertEqual(phase['first_by_throughput'],{'v2':4,'mean|v2|vllm':2})
        with (self.root/'throughput_cases.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows),12)
        self.assertEqual(rows[0]['v2_ttft_mean_ms'],'2000.0')
        self.assertTrue((self.root/'throughput_report.md').is_file())

    def test_bad_raw_usage_suppresses_entire_scoreboard(self):
        path = next(self.root.glob('rounds/round01/cells/*/measured_requests.json'))
        rows = json.loads(path.read_text());rows[0]['usage']['completion_tokens']-=1;dump(path,rows)
        result=report.analyze(self.root)
        self.assertFalse(result['complete'])
        self.assertIn('incorrect token usage',result['error'])
        self.assertNotIn('phase_summary',result)

    def test_mutated_body_between_output_lengths_even_with_matching_new_file_hash_fails(self):
        name='p4096_o128'
        path=self.root/'workloads'/name/'learn.jsonl'
        items=[json.loads(line) for line in path.read_text().splitlines()]
        items[0]['body']['messages'][0]['content'] += 'different input'
        path.write_text(''.join(json.dumps(r)+'\n' for r in items))
        round_path=self.root/'rounds/round01/workloads'/name/'learn.jsonl'
        shutil.copy2(path,round_path)
        mpath=self.root/'workloads/manifest.json'; m=json.loads(mpath.read_text())
        next(c for c in m['cells'] if c['profile']==name)['files']['learn']['sha256']=report.runtime.sha(path)
        dump(mpath,m)
        self.assertIn('changing output length changed input',report.analyze(self.root)['error'])

    def test_edited_learning_inclusive_elapsed_rejected(self):
        path=next(self.root.glob('rounds/round01/cells/*/cell_statistics.json'))
        rows=json.loads(path.read_text()); r=rows[-1]
        r['wall_s']*=2;r['output_tok_s']=r['total_output_tokens']/r['wall_s']
        dump(path,rows)
        self.assertIn('combined wall time',report.analyze(self.root)['error'])

    def test_incomplete_postprocessing_returns_no_wins(self):
        dump(self.root/'postprocess_status.json',dict(complete=False))
        self.assertFalse(report.analyze(self.root)['complete'])


if __name__ == '__main__':
    unittest.main()
