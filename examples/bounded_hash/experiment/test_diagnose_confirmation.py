import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from diagnose_confirmation import inspect_phase, main


class TailDiagnosisTests(unittest.TestCase):
    def fixture(self, root, observation=None, score_units='uncached_tokens_heuristic'):
        cell=root/'blocks'/'r1'/'cells'/'case';cell.mkdir(parents=True)
        req=dict(id='measured',success=True,ttft_ms=4500.0,e2e_ms=5000.0,
                 usage=dict(prompt_tokens=8192,completion_tokens=32,
                            prompt_tokens_details=dict(cached_tokens=7936)))
        route=dict(request_id='measured',chosen=0,preferred=0,gateway_wait_ms=0.1,
                   bounded_candidates=[0],capacity_candidates=[0,1],stable_owners=[0,1],
                   cached_token_hints={'0':7936},scores={'0':256},score_units=score_units)
        if observation is not None:route['admission_observation']=observation
        (cell/'measured_requests.json').write_text(json.dumps([req]))
        (cell/'measured_routes.json').write_text(json.dumps(dict(events=[dict(request_id='learn'),route])))
        (cell/'cell_statistics.json').write_text(json.dumps([dict(phase='measured',shared_tokens=7936,profile='reuse8',seed=1,arm='old_mean',concurrency=16)]))
        return cell/'measured_requests.json'

    def test_missing_outside_hints_remain_unknown_and_token_score_not_ms(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=self.fixture(root)
            rows,audit=inspect_phase(path,root,4000)
            self.assertEqual(audit['noncohort_routes_ignored'],1)
            r=rows[0]
            self.assertIsNone(r['other_observed_warm_excluded_by_count'])
            self.assertIsNone(r['selected_score_ms'])
            self.assertEqual(r['cache_group'],'full')
            self.assertAlmostEqual(r['client_residual_after_gateway_ms'],4499.9)

    def test_readonly_observation_identifies_excluded_warm_and_pending_peers(self):
        observation=dict(status='OBSERVED',all_worker_cache_hints={'0':7936,'1':7936},
            capacity_candidates=[0,1],capture_wall_ms=.04,
            workers=[dict(worker=0,pending_request_count=1,pending_uncached_tokens=8192)],
            chosen_worker_peers=[dict(serial=1,started=False,prompt=8192,digest_match=False)])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=self.fixture(root,observation)
            r=inspect_phase(path,root,4000)[0][0]
            self.assertEqual(r['other_observed_warm_excluded_by_count'],['1'])
            self.assertEqual(r['selected_pending_request_count'],1)
            self.assertIn('observed_warm_alternative_excluded_by_count_bound',r['signals'])

    def test_default_cli_nested_rounds_is_readonly_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';root.mkdir()
            one=self.fixture(root/'rounds'/'round_01')
            two=self.fixture(root/'rounds'/'round_02')
            ignored=root/'previous'/'diagnostics';ignored.mkdir(parents=True)
            (ignored/'measured_requests.json').write_text('not a raw phase')
            original={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(root)]),0)
            status=json.loads((root/'diagnostics'/'diagnosis.json').read_text())
            self.assertEqual(status['request_records'],2)
            self.assertEqual(status['request_phases'],2)
            self.assertEqual(len({r['cell'] for r in status['phase_audits']}),2)
            self.assertEqual(original,{p:p.read_bytes() for p in original})
            output_before=(root/'diagnostics'/'diagnosis.json').read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([str(root)])
            self.assertEqual(output_before,(root/'diagnostics'/'diagnosis.json').read_bytes())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([str(root),'--out',str(root)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(root),'--out',str(Path(temp)/'sibling')]),0)
            sibling=json.loads((Path(temp)/'sibling'/'diagnosis.json').read_text())
            self.assertEqual(sibling['request_records'],2)

if __name__=='__main__':unittest.main()
