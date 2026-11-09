import copy
import json
from pathlib import Path
import tempfile
import unittest
from phase_calibration import overlap_step_ms
from phase_suite import arguments,configuration,make_plan,report,contract


class RunnerTests(unittest.TestCase):
    def test_default_plan_does_not_claim_success(self):
        a=arguments([]);c=configuration(a.config);p=make_plan(a,c)
        self.assertEqual(a.mode,'plan');self.assertEqual(p['cells'],45)
        self.assertEqual(p['total_learning_plus_measured_requests'],45*192)
        self.assertFalse(p['performance_win_claimed']);self.assertEqual(p['gpu_jobs_started'],0)

    def test_selected_short_screen_is_three_real_cells(self):
        a=arguments(['run','--profiles','prefix97','--concurrency','32','--requests','64',
            '--arms','nh_phase_bounded','nh_mean_bounded','vllm_least_active'])
        p=make_plan(a,configuration(a.config));self.assertEqual(p['cells'],3)
        self.assertEqual(p['total_learning_plus_measured_requests'],384)

    def test_invalid_duplicate_selection_refused(self):
        with self.assertRaises(SystemExit):arguments(['plan','--seeds','1','1'])

    def test_full_kv_capacity_equal_and_page_rounded(self):
        a=arguments([]);c=configuration(a.config)
        self.assertEqual(c['max_batch'],3)
        self.assertEqual(c['nh_blocks']*c['nh_block_size'],32768)
        self.assertEqual(c['vllm_gpu_blocks']*c['vllm_block_size'],32768)
        self.assertGreaterEqual(c['max_context'],c['prompt_tokens']+c['output_tokens'])

    def _plan_rows(self):
        a=arguments(['plan','--profiles','prefix97','--concurrency','16','--arms','nh_phase_bounded','vllm_least_active'])
        p=make_plan(a,configuration(a.config));rows=[]
        for arm,t,m in [('nh_phase_bounded',90.,1000.),('vllm_least_active',100.,1100.)]:
            for phase in ('learn','measured','learn_plus_measured'):
                rows.append(dict(profile='prefix97',seed=33959,concurrency=16,arm=arm,phase=phase,
                    valid=True,output_tok_s=t,ttft_mean_ms=m,ttft_p95_ms=1200.,ttft_p99_ms=1250.,e2e_mean_ms=5000.,failures=0,
                    ordered_request_contract_sha256='same'))
        return p,rows

    def test_complete_is_not_performance_pass(self):
        p,rows=self._plan_rows()
        with tempfile.TemporaryDirectory() as d:r=report(Path(d),rows,p)
        self.assertTrue(r['complete']);self.assertFalse(r['performance_win_claimed'])
        self.assertEqual(r['candidate_both_metrics_better_count'],0)

    def test_missing_or_invalid_cells_never_complete(self):
        p,rows=self._plan_rows();rows.pop()
        with tempfile.TemporaryDirectory() as d:r=report(Path(d),rows,p)
        self.assertFalse(r['complete'])
        p,rows=self._plan_rows();rows[0]['valid']=False
        with tempfile.TemporaryDirectory() as d:r=report(Path(d),rows,p)
        self.assertFalse(r['complete'])

    def test_cross_cohort_comparison_refused(self):
        p,rows=self._plan_rows();rows[3]['ordered_request_contract_sha256']='different'
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'cohorts'):report(Path(d),rows,p)

    def test_contract_includes_order_and_token_identity(self):
        a=dict(id='a',prompt_sha256='x',prompt_tokens=2048,shared_tokens=1536,output_tokens=512)
        b=dict(a,id='b',prompt_sha256='y')
        self.assertNotEqual(contract([a,b]),contract([b,a]))
        self.assertNotEqual(contract([a]),contract([dict(a,output_tokens=64)]))

    def test_decode_overlap_excludes_initial_prefill_delay(self):
        # One request starts much earlier. Only the interval after both first tokens is used.
        r1=dict(id='a',token_events=[dict(absolute_s=t,generated_tokens=i+1) for i,t in enumerate([0,.01,.02,.03,.04,.05,.06,.07,.08,.09,.10])])
        r2=dict(id='b',token_events=[dict(absolute_s=t,generated_tokens=i+1) for i,t in enumerate([.03,.04,.05,.06,.07,.08,.09,.10,.11,.12,.13])])
        ms,ev=overlap_step_ms([r1,r2]);self.assertAlmostEqual(ms,10.)
        self.assertEqual(ev['common_start_s'],.03);self.assertEqual(ev['common_end_s'],.10)

    def test_decode_no_overlap_refused(self):
        a=dict(id='a',token_events=[dict(absolute_s=t,generated_tokens=t) for t in (1,2,3)])
        b=dict(id='b',token_events=[dict(absolute_s=t,generated_tokens=t) for t in (4,5,6)])
        with self.assertRaises(ValueError):overlap_step_ms([a,b])

if __name__=='__main__':unittest.main()
