import asyncio
import contextlib
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch
import zipfile

import cache_calibration
import length_runner as runner
from length_workload import PROFILES, SEED


class LengthDriverTests(unittest.TestCase):
    def setUp(self):
        self.cfg = runner.cache_suite.configuration(runner.KIT/'config.musa.json')
        self.ref, self.sources = runner.verify_local(self.cfg)

    def fingerprint(self):
        fp = {name:copy.deepcopy(self.ref[name]) for name in runner.ENVIRONMENT_FIELDS}
        fp.update(protocol=runner.confirm.PROTOCOL, candidate_sources=self.sources)
        return fp

    def test_plans_and_actual_orders_for_both_concurrencies(self):
        for concurrency in (16, 32):
            options = runner.arguments(['plan', '--concurrency', str(concurrency)])
            with runner.adapted_runtime(PROFILES, self.ref, self.sources):
                args = runner.engine_arguments(options, PROFILES, SEED)
                plan = runner.confirm.make_plan(args, self.cfg)
            self.assertEqual(args.seeds, [33962])
            self.assertEqual(args.concurrency, [concurrency])
            self.assertEqual(len(plan['profiles']), 6)
            self.assertEqual(plan['cells'], 18)
            self.assertEqual(plan['total_learning_plus_measured_requests'], 3456)
            for arm in runner.ARMS:
                positions = [j['order_position'] for j in plan['execution_order'] if j['arm'] == arm]
                self.assertEqual(sorted(positions), [1, 1, 2, 2, 3, 3])
            for profile, params in plan['profiles'].items():
                self.assertEqual(params['documents'], 8)
                self.assertEqual(params['shared_tokens'] % 256, 0)
                self.assertLessEqual(params['output_tokens'], 128)
                self.assertLessEqual(params['prompt_tokens'] + params['output_tokens'], self.cfg['max_context'])

    def test_plan_entry_does_not_verify_hardware_or_launch(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), patch.object(runner.confirm, 'execute') as execute:
            rc = runner.main(['plan'])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out.getvalue())['cells'], 18)
        execute.assert_not_called()

    def test_adapter_restored_on_exception(self):
        old_prepare = runner.confirm.prepare_workloads
        old_plan = runner.cache_suite.make_plan
        old_profiles = dict(runner.cache_workload.PROFILES)
        with self.assertRaisesRegex(ValueError, 'fixture'):
            with runner.adapted_runtime(PROFILES, self.ref, self.sources):
                self.assertTrue(set(PROFILES).issubset(runner.cache_workload.PROFILES))
                raise ValueError('fixture')
        self.assertIs(runner.confirm.prepare_workloads, old_prepare)
        self.assertIs(runner.cache_suite.make_plan, old_plan)
        self.assertEqual(runner.cache_workload.PROFILES, old_profiles)

    def test_environment_and_source_drift(self):
        fp = self.fingerprint()
        runner.verify_runtime(fp, self.ref, self.sources)
        for field in ('nh_git','package_versions','native_bindings','hardware','candidate_sources','tokenizer_files'):
            changed = dict(fp, **{field:{}})
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.verify_runtime(changed, self.ref, self.sources)

    def test_enforces_environment_before_workload_build_or_gpu_calibration(self):
        fingerprint = self.fingerprint()
        fingerprint['nh_git'] = {'head':'different'}
        with TemporaryDirectory() as tmp, runner.adapted_runtime(PROFILES, self.ref, self.sources):
            root = Path(tmp)
            a = runner.engine_arguments(runner.arguments(['run']), PROFILES, SEED)
            plan = runner.confirm.make_plan(a, self.cfg)
            with patch.object(runner.harness,'preflight',new=AsyncMock(return_value=fingerprint)), \
                 patch.object(runner.cache_suite,'hardware_fingerprint',new=AsyncMock(return_value=self.ref['hardware'])), \
                 patch.object(runner.harness,'run_logged',new_callable=AsyncMock) as builder, \
                 patch.object(cache_calibration,'calibrate',new_callable=AsyncMock) as calibration, \
                 self.assertRaisesRegex(ValueError,'nh_git'):
                asyncio.run(runner.execute(a,self.cfg,root,plan,self.ref,self.sources))
            builder.assert_not_awaited()
            calibration.assert_not_awaited()

    def test_check_uses_tokenizer_and_no_model_pool(self):
        with TemporaryDirectory() as tmp, runner.adapted_runtime(PROFILES, self.ref, self.sources):
            root = Path(tmp)
            a = runner.engine_arguments(runner.arguments(['check']), PROFILES, SEED)
            plan = runner.confirm.make_plan(a,self.cfg)
            with patch.object(runner.harness,'preflight',new=AsyncMock(return_value=self.fingerprint())), \
                 patch.object(runner.cache_suite,'hardware_fingerprint',new=AsyncMock(return_value=self.ref['hardware'])), \
                 patch.object(runner.confirm,'prepare_workloads',new_callable=AsyncMock) as prepare, \
                 patch.object(cache_calibration,'calibrate',new_callable=AsyncMock) as calibration, \
                 patch.object(runner.confirm.rt,'Pool') as pool:
                result = asyncio.run(runner.execute(a,self.cfg,root,plan,self.ref,self.sources))
            self.assertTrue(result['no_models_loaded'])
            self.assertTrue(result['tokenizer_workload_verified'])
            prepare.assert_awaited_once()
            calibration.assert_not_awaited()
            pool.assert_not_called()

    def test_actual_runtime_executes_18_mock_cells_with_one_calibration(self):
        async def prepare(a,cfg,root,fp):
            (root/'workloads').mkdir()
            return {'cells':[{'profile':p,'seed':SEED} for p in PROFILES]}
        async def job(cfg,root,case,job,model,opts,client):
            return [dict(valid=True,round=job['round']) for _ in ('learn','measured','combined')]
        with TemporaryDirectory() as tmp, runner.adapted_runtime(PROFILES,self.ref,self.sources):
            a = runner.engine_arguments(runner.arguments(['run']),PROFILES,SEED)
            plan = runner.confirm.make_plan(a,self.cfg)
            with patch.object(runner.harness,'preflight',new=AsyncMock(return_value=self.fingerprint())), \
                 patch.object(runner.cache_suite,'hardware_fingerprint',new=AsyncMock(return_value=self.ref['hardware'])), \
                 patch.object(runner.confirm,'prepare_workloads',new=AsyncMock(side_effect=prepare)) as loader, \
                 patch.object(cache_calibration,'calibrate',new=AsyncMock(return_value={})) as calibrate, \
                 patch('phase_router.PhaseModel'), patch('admission_telemetry.install'), \
                 patch.object(runner.confirm,'execute_job',new=AsyncMock(side_effect=job)) as bench, \
                 patch.object(runner.confirm,'validate_manifest'), patch.object(runner.cache_suite,'report'), \
                 patch.object(runner.confirm.rt,'csv_write'), contextlib.redirect_stdout(io.StringIO()):
                result = asyncio.run(runner.execute(a,self.cfg,Path(tmp),plan,self.ref,self.sources))
        self.assertTrue(result['complete'])
        self.assertEqual(result['completed_cells'],18)
        self.assertEqual(result['valid_phase_rows'],54)
        self.assertEqual(bench.await_count,18)
        loader.assert_awaited_once()
        calibrate.assert_awaited_once()

    def check_archival(self, failure):
        async def fake_run(*args):
            if failure == 'benchmark':
                raise RuntimeError('cell failed')
            return {'status':'EXPERIMENT_COMPLETE','complete':True}
        def post(root):
            runner.harness.dump(root/'postprocess_status.json',{'complete':True})
        def metrics(root):
            self.assertTrue(runner.read(root/'postprocess_status.json')['complete'])
            self.assertTrue(runner.read(root/'status.json')['complete'])
            if failure == 'analysis':
                return {'complete':False}
            (root/'throughput_report.md').write_text('Verified fixture report')
            return {'complete':True}
        with TemporaryDirectory() as tmp, runner.adapted_runtime(PROFILES,self.ref,self.sources):
            a=runner.engine_arguments(runner.arguments(['run']),PROFILES,SEED)
            plan=runner.confirm.make_plan(a,self.cfg)
            root=Path(tmp)/'lengths_test'
            with patch.object(runner,'execute',new=AsyncMock(side_effect=fake_run)), \
                 patch.object(runner,'conventional_reports',side_effect=post), \
                 patch('length_report.analyze',side_effect=metrics), contextlib.redirect_stdout(io.StringIO()) as stdout:
                rc = runner.execute_locked(a,self.cfg,root,plan,self.ref,self.sources)
            self.assertEqual(rc, 2 if failure else 0)
            archive=Path(str(root)+'_report.zip')
            self.assertIn('LENGTH_REPORT_ARCHIVE='+str(archive),stdout.getvalue())
            with zipfile.ZipFile(archive) as z:
                result=json.loads(z.read(root.name+'/length_status.json'))
                self.assertEqual(result['complete'],not bool(failure))
                self.assertEqual(root.name+'/throughput_report.md' in z.namelist(),not bool(failure))
                self.assertIn(root.name+'/anchor_source_hashes.json',z.namelist())
                self.assertIn(root.name+'/source/length_runner.py',z.namelist())
            state=runner.read(root/'status.json')
            self.assertEqual(state['complete'],failure != 'benchmark')

    def test_completed_archive_includes_primary_report(self):
        self.check_archival(None)

    def test_failed_benchmark_archives_evidence(self):
        self.check_archival('benchmark')

    def test_incomplete_analysis_fails_and_preserves_engine_result(self):
        self.check_archival('analysis')


if __name__ == '__main__':
    unittest.main()
