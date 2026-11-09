"""CPU-only protocol checks: these tests never start an actual model process."""
import asyncio
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import cache_confirm as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ConfirmationRunnerTests(unittest.TestCase):
    def test_default_latin_rotation_is_balanced_and_has_old_mean(self):
        args = runner.arguments([])
        jobs = runner.schedule(args)
        self.assertEqual(len(jobs), 50)
        positions = defaultdict(list)
        for job in jobs:
            positions[(job['profile'], job['seed'], job['concurrency'], job['arm'])].append(job['order_position'])
        self.assertEqual(len(positions), 10)
        self.assertTrue(all(sorted(p) == [1, 2, 3, 4, 5] for p in positions.values()))
        self.assertIn(('reuse8', 33960, 16, 'nh_mean_bounded'), positions)
        self.assertEqual(jobs, runner.schedule(args))
        self.assertNotEqual(jobs[0]['arm'], jobs[5]['arm'])

    def test_plan_counts_and_screen_label(self):
        args = runner.arguments(['--rounds', '2'])
        plan = runner.make_plan(args, {'model': 'test'})
        self.assertEqual(plan['cells'], 20)
        self.assertFalse(plan['order_balanced'])
        self.assertEqual(plan['evidence_kind'], 'unbalanced_screen_not_confirmation')
        self.assertNotIn('gpu_jobs_started', plan)
        plan = runner.make_plan(runner.arguments([]), {'model': 'test'})
        self.assertEqual(plan['total_learning_plus_measured_requests'], 9600)
        self.assertEqual(plan['expected_phase_rows'], 150)

    def test_subset_rotation(self):
        args = runner.arguments(['--rounds', '2', '--arms', 'nh_cache_bounded_v2', 'vllm_least_active'])
        jobs = runner.schedule(args)
        self.assertEqual(len(jobs), 8)
        for seed in args.seeds:
            for arm in args.arms:
                self.assertEqual(sorted(j['order_position'] for j in jobs if j['seed'] == seed and j['arm'] == arm), [1, 2])

    def frozen_fixture(self, root):
        old = root/'old'
        cfg = {'model': 'same'}
        fp = {'tokenizer_files': {'tokenizer.json': 'same'},
              'nh_git': {'commit': 'same'},
              'nh_files': {'python/infinilm/llm/cache_manager.py': runner.base.EXPECTED_SHA},
              'native_bindings': {'native.so': 'same'},
              'package_versions': {'nh': {'torch': 'same'}, 'vllm': {'torch': 'same'}},
              'hardware': {'available': True, 'devices': [{'name': 'fixture'}]},
              'candidate_sources': {key: 'same' for key in runner.REPLAY_POLICY_SOURCES}}
        write(old/'plan.json', {'protocol': runner.base.PROTOCOL, 'config': cfg})
        write(old/'status.json', {'complete': True})
        write(old/'fingerprint.json', fp)
        case = dict(profile='reuse8', seed=33960, folder='reuse8_seed33960',
                    **runner.PROFILES['reuse8'], files={})
        folder = old/'workloads'/case['folder']
        write(folder/'smoke.json', [{}, {}])
        for phase, count in [('learn', 64), ('measured', 128)]:
            rows = [dict(id=f'{phase}-{i}', prompt_tokens=8192, shared_tokens=7936, output_tokens=32) for i in range(count)]
            file = folder/(phase+'.jsonl')
            file.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            case['files'][phase] = dict(count=count, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        write(old/'workloads/manifest.json', {'cells': [case]})
        args = runner.arguments(['--historical-run', str(old), '--seeds', '33960'])
        return args, cfg, fp, case

    def test_frozen_replay_never_calls_generator_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, cfg, fp, case = self.frozen_fixture(root)
            with patch.object(runner.common, 'run_logged', new_callable=AsyncMock) as generator:
                manifest = asyncio.run(runner.prepare_workloads(args, cfg, root/'new', fp))
                generator.assert_not_called()
            self.assertEqual(manifest['cells'], [case])
            original = args.historical_run/'workloads'/case['folder']/'measured.jsonl'
            copied = root/'new/workloads'/case['folder']/'measured.jsonl'
            self.assertEqual(copied.read_bytes(), original.read_bytes())
            with self.assertRaises(FileExistsError):
                runner.copy_historical_workloads(args, cfg, root/'new', fp)

    def test_historical_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, cfg, fp, case = self.frozen_fixture(root)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                runner.copy_historical_workloads(args, {'model': 'changed'}, root/'bad1', fp)
            with self.assertRaisesRegex(ValueError, 'tokenizer'):
                runner.copy_historical_workloads(args, cfg, root/'bad2', {'tokenizer_files': {}})
            file = args.historical_run/'workloads'/case['folder']/'measured.jsonl'
            file.write_text(file.read_text().replace('measured-0', 'different-0', 1))
            with self.assertRaisesRegex(ValueError, 'count/hash'):
                runner.copy_historical_workloads(args, cfg, root/'bad3', fp)

    def test_phase_model_requires_full_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)/'model.json'
            fp = {'hardware': {'available': True, 'devices': [{'name': 'fixture'}]},
                  'candidate_sources': {'router.py': 'v1'}}
            write(model, {'fingerprint': fp, 'model': 'fixture'})
            self.assertEqual(runner.validated_phase_model(model, fp)['model'], 'fixture')
            changed = dict(fp, candidate_sources={'router.py': 'v2'})
            with self.assertRaisesRegex(ValueError, 'full fingerprint'):
                runner.validated_phase_model(model, changed)
            with self.assertRaisesRegex(ValueError, 'Hardware identity'):
                runner.validated_phase_model(model, dict(fp, hardware={'available': False}))

    def test_replay_rejects_environment_or_original_policy_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, fp, _ = self.frozen_fixture(Path(tmp))
            for field in runner.REPLAY_IDENTITY_FIELDS:
                with self.subTest(field=field):
                    changed = deepcopy(fp)
                    changed[field] = {'changed': True}
                    with self.assertRaisesRegex(ValueError, field):
                        runner.validate_historical_identity(fp, changed)
                    changed = deepcopy(fp)
                    del changed[field]
                    with self.assertRaisesRegex(ValueError, field):
                        runner.validate_historical_identity(fp, changed)
            unknown = deepcopy(fp)
            unknown['hardware'] = {'available': False, 'devices': []}
            with self.assertRaisesRegex(ValueError, 'Hardware identity unavailable'):
                runner.validate_historical_identity(unknown, unknown)
            changed = deepcopy(fp)
            changed['candidate_sources']['mean_router.py'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'mean_router.py'):
                runner.validate_historical_identity(fp, changed)
            changed = deepcopy(fp)
            changed['protocol'] = 'new runner protocol is permitted'
            changed['candidate_sources']['cache_confirm.py'] = 'new runner is permitted'
            runner.validate_historical_identity(fp, changed)

    def test_fresh_pool_per_cell_and_cleanup_on_failure(self):
        events, pools = [], []
        class Pool:
            active = 0
            def __init__(self, cfg, engine, folder):
                self.engine, self.folder = engine, folder
                pools.append(self)
            async def start(self):
                self.folder.mkdir(parents=True)
                assert Pool.active == 0, 'overlapping GPU pools'
                Pool.active += 1
                events.append(('start', self.engine))
            async def close(self):
                Pool.active -= 1
                events.append(('close', self.engine))
        async def run_cell(cfg, root, pool, case, c, arm, *args):
            folder = root/'cells'/f'{case["profile"]}_s{case["seed"]}_c{c}_{arm[0]}'
            folder.mkdir(parents=True)
            if arm[0] == 'nh_mean_bounded':
                raise RuntimeError('fixture failure')
            return [{'valid': True}]*3
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.rt, 'Pool', Pool), patch.object(runner.base, 'runtime_features', return_value={}), patch.object(runner.base, 'run_cell', side_effect=run_cell):
            root = Path(tmp)
            case = {'profile': 'reuse8', 'seed': 33960}
            async def scenario():
                for serial, arm in enumerate(['vllm_least_active', 'nh_cache_bounded_v2', 'nh_mean_bounded'], 1):
                    job = dict(serial=serial, round=1, profile='reuse8', seed=33960,
                               concurrency=16, arm=arm, order_position=serial)
                    if arm == 'nh_mean_bounded':
                        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                            await runner.execute_job({}, root, case, job, {}, {}, None)
                    else:
                        rows = await runner.execute_job({}, root, case, job, {}, {}, None)
                        self.assertEqual(rows[0]['execution_serial'], serial)
            asyncio.run(scenario())
            self.assertEqual(len(pools), 3)
            self.assertEqual(Pool.active, 0)
            self.assertEqual([e[0] for e in events], ['start', 'close']*3)
            failure = json.loads((root/'orders/0003.json').read_text())
            self.assertEqual(failure['state'], 'FAILED')
            self.assertEqual(json.loads((root/'cells/reuse8_s33960_c16_nh_mean_bounded/order.json').read_text())['state'], 'FAILED')

    def test_full_mock_execution_reuses_frozen_input_and_matches_summary(self):
        import admission_telemetry
        import cache_calibration
        import phase_router
        import summarize_confirmation
        starts = []
        class Pool:
            active = 0
            def __init__(self, cfg, engine, folder):
                self.folder, self.engine = folder, engine
            async def start(self):
                self.folder.mkdir(parents=True)
                assert Pool.active == 0
                Pool.active += 1
                starts.append(self.engine)
            async def close(self):
                Pool.active -= 1
        async def run_cell(cfg, root, pool, case, c, arm, *args):
            folder = root/'cells'/f'{case["profile"]}_s{case["seed"]}_c{c}_{arm[0]}'
            folder.mkdir(parents=True)
            rows = []
            for phase, count in [('learn', 64), ('measured', 128), ('learn_plus_measured', 192)]:
                row = dict(profile=case['profile'], seed=case['seed'], concurrency=c,
                           arm=arm[0], phase=phase, valid=True, output_tok_s=100.,
                           ttft_mean_ms=100., ttft_p95_ms=150., ttft_p99_ms=160.,
                           e2e_mean_ms=200., failures=0, requests=count, success=count,
                           wall_s=count*32/100., output_tokens=count*32,
                           total_output_tokens=count*32, ordered_request_contract_sha256='a'*64)
                rows.append(row)
            write(folder/'cell_statistics.json', rows)
            return rows
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, cfg, fp, _ = self.frozen_fixture(root)
            args.mode = 'run'
            fp['nh_files'] = {'python/infinilm/llm/cache_manager.py': runner.base.EXPECTED_SHA}
            out = root/'run'
            out.mkdir()
            plan = runner.make_plan(args, cfg)
            write(out/'plan.json', plan)
            with patch.object(runner.common, 'preflight', new_callable=AsyncMock, return_value=fp), \
                 patch.object(runner.base, 'hardware_fingerprint', new_callable=AsyncMock, return_value=fp['hardware']), \
                 patch.object(cache_calibration, 'calibrate', new_callable=AsyncMock, return_value={}) as calibration, \
                 patch.object(phase_router, 'PhaseModel'), \
                 patch.object(admission_telemetry, 'install') as install, \
                 patch.object(runner.rt, 'Pool', Pool), \
                 patch.object(runner.base, 'runtime_features', return_value={}), \
                 patch.object(runner.base, 'run_cell', side_effect=run_cell), \
                 patch.object(runner.common, 'progress'), \
                 patch.object(runner, 'prepare_workloads', wraps=runner.prepare_workloads) as prepare:
                result = asyncio.run(runner.execute(args, cfg, out, plan))
                prepare.assert_awaited_once()
                calibration.assert_awaited_once()
                install.assert_called_once()
            write(out/'status.json', result)
            self.assertTrue(result['complete'])
            self.assertEqual(len(starts), 25)
            self.assertEqual(Pool.active, 0)
            self.assertEqual(len(runner.partial_rows(out)), 75)
            summary = summarize_confirmation.analyze(out)
            self.assertTrue(summary['complete'], summary)
            self.assertTrue(summary['execution_position_balanced'])
            self.assertEqual(summary['measured_vllm_pairs'], 5)

    def test_failed_startup_closes_pool(self):
        close = AsyncMock()
        pool = SimpleNamespace(start=AsyncMock(side_effect=RuntimeError('startup failed')), close=close)
        job = dict(serial=1, round=1, profile='reuse8', seed=33960,
                   concurrency=16, arm='vllm_least_active', order_position=1)
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.rt, 'Pool', return_value=pool):
            with self.assertRaisesRegex(RuntimeError, 'startup failed'):
                asyncio.run(runner.execute_job({}, Path(tmp), {'profile':'reuse8','seed':33960}, job, {}, {}, None))
            close.assert_awaited_once()
            self.assertEqual(json.loads((Path(tmp)/'orders/0001.json').read_text())['state'], 'FAILED')

    def test_main_preserves_failed_run_and_archive(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runner.base, 'configuration', return_value={'model': 'fixture'}), \
             patch.object(runner, 'snapshot_sources'), \
             patch.object(runner, 'execute', new_callable=AsyncMock, side_effect=RuntimeError('fixture failure')), \
             patch('subprocess.run', return_value=SimpleNamespace(returncode=0)), \
             redirect_stdout(StringIO()):
            code = runner.main(['run', '--output-root', tmp])
            self.assertEqual(code, 2)
            roots = [p for p in Path(tmp).iterdir() if p.is_dir()]
            self.assertEqual(len(roots), 1)
            status = json.loads((roots[0]/'status.json').read_text())
            self.assertEqual(status['status'], 'FAILED')
            self.assertFalse(status['complete'])
            self.assertEqual(len(list(Path(tmp).glob('*_report.zip'))), 1)

    def test_postprocess_errors_return_failure_without_hiding_engine_completion(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(runner.base, 'configuration', return_value={'model': 'fixture'}), \
                 patch.object(runner, 'snapshot_sources'), \
                 patch.object(runner, 'execute', new_callable=AsyncMock,
                              return_value={'status':'EXPERIMENT_COMPLETE','complete':True}), \
                 patch('subprocess.run', side_effect=[SimpleNamespace(returncode=2), SimpleNamespace(returncode=0)]) as postprocess, \
                 patch.object(runner, 'KIT', Path(tmp) if missing else runner.KIT), \
                 redirect_stdout(StringIO()):
                self.assertEqual(runner.main(['run', '--output-root', tmp]), 2)
                root = next(p for p in Path(tmp).iterdir() if p.is_dir())
                self.assertTrue(json.loads((root/'status.json').read_text())['complete'])
                status = json.loads((root/'postprocess_status.json').read_text())
                self.assertFalse(status['complete'])
                self.assertEqual(status['status'], 'FAILED')
                self.assertEqual(status['scripts'][0]['status'], 'MISSING' if missing else 'FAILED')
                self.assertEqual(postprocess.call_count, 0 if missing else 2)
                self.assertEqual(len(list(Path(tmp).glob('*_report.zip'))), 1)


if __name__ == '__main__':
    unittest.main()
