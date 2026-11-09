"""Selection-driver integration with fake child processes; never starts GPUs."""
from __future__ import annotations
from collections import Counter, defaultdict
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import signal
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

import cache_confirm
import select_version as driver
from selection_protocol import digest, validate_spec


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


class FakeChild:
    def __init__(self, code=0):
        self.code = code

    def wait(self):
        return self.code

    def poll(self):
        return self.code

    def send_signal(self, signum):
        raise AssertionError('No child signal expected in this fixture')


class SelectionDriverTests(unittest.TestCase):
    def setUp(self):
        reference = json.loads((driver.KIT/'reference_environment.json').read_text())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.kit = self.root/'kit'
        self.kit.mkdir()
        self.config = self.kit/'config.musa.json'
        self.sources = {'mean_router.py': 'a'*64, 'select_version.py': 'b'*64}
        self.reference = deepcopy(reference)
        self.reference['proven_sources'] = {'mean_router.py': self.sources['mean_router.py']}
        write_json(self.config, self.reference['config'])
        write_json(self.kit/'reference_environment.json', self.reference)
        self.commands = []
        self.children = []
        self.analyzer = Mock()
        self.freezer = Mock()
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous = signal.getsignal(sig)
            self.addCleanup(signal.signal, sig, previous)

    @contextlib.contextmanager
    def patched_identity(self):
        with patch.object(driver, 'KIT', self.kit), \
             patch.object(driver.harness, 'source_hashes', side_effect=lambda: dict(self.sources)):
            yield

    def spec(self):
        with self.patched_identity():
            return driver.build_spec(self.config)

    def fingerprint(self, spec):
        keys = ('config', 'nh_git', 'nh_files', 'native_bindings', 'tokenizer_files',
                'package_versions', 'hardware')
        value = {key: deepcopy(self.reference[key]) for key in keys}
        value.update(protocol=spec['execution_protocol'], candidate_sources=dict(self.sources))
        return value

    def test_actual_command_has_one_observation_per_method_and_per_scenario(self):
        spec = self.spec()
        cmd = driver.command('run', self.root/'out', self.config, spec)
        self.assertNotIn('--historical-run', cmd)
        self.assertNotIn('--phase-model', cmd)
        args = cache_confirm.arguments(cmd[3:])
        jobs = cache_confirm.schedule(args)
        self.assertEqual(len(jobs), 36)
        self.assertEqual(set(args.profiles), {'reuse8', 'reuse16', 'longdecode', 'cold'})
        self.assertEqual(args.concurrency, [16, 32, 64])
        self.assertEqual(args.seeds, [33962])
        self.assertEqual(args.rounds, 1)
        self.assertEqual(args.requests, 128)
        self.assertEqual(set(args.arms), {'nh_mean_bounded', 'nh_cache_bounded_v2', 'vllm_least_active'})
        counts = Counter((j['profile'], j['concurrency']) for j in jobs)
        self.assertEqual(len(counts), 12)
        self.assertEqual(set(counts.values()), {3})
        self.assertEqual(set(j['round'] for j in jobs), {1})
        self.assertEqual(set(j['seed'] for j in jobs), {33962})
        per_case = Counter((j['profile'], j['concurrency'], j['arm']) for j in jobs)
        self.assertEqual(len(per_case), 36)
        self.assertEqual(set(per_case.values()), {1})
        per_profile = defaultdict(list)
        for job in jobs:
            per_profile[job['profile'], job['arm']].append(job['order_position'])
        self.assertEqual(len(per_profile), 12)
        self.assertTrue(all(sorted(p) == [1, 2, 3] for p in per_profile.values()))
        global_positions = Counter((j['arm'], j['order_position']) for j in jobs)
        self.assertEqual(len(global_positions), 9)
        self.assertEqual(set(global_positions.values()), {4})
        self.assertEqual(spec['actual_cells'], 36)
        self.assertEqual(spec['learning_plus_measured_requests'], 6912)

    def test_generic_postprocessor_accepts_all_36_completed_single_round_jobs(self):
        # Exercise the core postprocessor's normal CLI on the complete single-round grid.
        # Per-case repeat confirmation and whole-grid completion are separate outputs.
        import summarize_confirmation as summary
        spec = self.spec()
        cmd = driver.command('run', self.root/'bench', self.config, spec)
        args = cache_confirm.arguments(cmd[3:])
        plan = cache_confirm.make_plan(args, self.reference['config'])
        run = self.root/'single_round'
        write_json(run/'plan.json', plan)
        write_json(run/'status.json', dict(status='EXPERIMENT_COMPLETE', complete=True))
        for job in plan['execution_order']:
            name = f"{job['profile']}_s{job['seed']}_c{job['concurrency']}_{job['arm']}"
            cell = run/'rounds'/'round01'/'cells'/name
            write_json(cell/'order.json', dict(job, state='COMPLETE'))
            records = []
            out_length = plan['profiles'][job['profile']]['output_tokens']
            for phase, count in zip(summary.PHASES, (64, 128, 192)):
                tokens = count * out_length
                records.append(dict(profile=job['profile'], concurrency=job['concurrency'],
                                    arm=job['arm'], seed=job['seed'], phase=phase,
                                    requests=count, success=count, failures=0,
                                    total_output_tokens=tokens, wall_s=tokens / 100,
                                    output_tok_s=100.0, ttft_mean_ms=100, ttft_p95_ms=200,
                                    ordered_request_contract_sha256='a'*64, valid=True,
                                    exact_usage=True, route_valid=True, cohort_valid=True,
                                    cache_policy_valid=True, event_payload_transparent=True))
            write_json(cell/'cell_statistics.json', records)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(summary.main([str(run)]), 0)
        result = json.loads((run/'confirmation_status.json').read_text())
        self.assertTrue(result['complete'])
        self.assertFalse(result['execution_position_balanced'])
        self.assertEqual(result['found_phase_cells'], 108)
        self.assertEqual(result['measured_vllm_pairs'], 12)
        self.assertEqual(result['excluded_pairs_or_metadata'], 0)

    def test_plan_does_not_launch_child_or_create_results(self):
        output = self.root/'unused_results'
        with self.patched_identity(), patch.object(driver.subprocess, 'Popen') as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            code = driver.main(['plan', '--config', str(self.config), '--output-root', str(output)])
        self.assertEqual(code, 0)
        popen.assert_not_called()
        self.assertFalse(output.exists())

    def test_proven_source_drift_rejected_before_child(self):
        self.sources['mean_router.py'] = 'c'*64
        with self.patched_identity(), patch.object(driver.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'verified code'):
                driver.main(['run', '--config', str(self.config), '--output-root', str(self.root/'results')])
        popen.assert_not_called()

    def test_preflight_identity_drift_rejected(self):
        spec = self.spec()
        for key in ('config', 'nh_git', 'nh_files', 'native_bindings', 'tokenizer_files',
                    'package_versions', 'hardware', 'protocol', 'candidate_sources'):
            with self.subTest(field=key):
                fingerprint = self.fingerprint(spec)
                fingerprint[key] = {'changed': True}
                with self.patched_identity(), self.assertRaises(ValueError):
                    driver.check_environment(fingerprint, spec)

    def test_reference_rewrite_after_planning_rejected_even_if_new_fp_matches(self):
        spec = self.spec()
        fingerprint = self.fingerprint(spec)
        changed = deepcopy(self.reference)
        changed['nh_git'] = {'head': 'different-head', 'origin': 'same-origin'}
        fingerprint['nh_git'] = changed['nh_git']
        write_json(self.kit/'reference_environment.json', changed)
        with self.patched_identity(), self.assertRaisesRegex(ValueError, '[Rr]eference'):
            driver.check_environment(fingerprint, spec)

    def run_driver(self, mode='run', fail=None, selected='nh_mean_bounded'):
        original_spec = self.spec()
        fingerprint = self.fingerprint(original_spec)
        decision = dict(complete=True, selected_arm=selected,
                        status='QUALIFIED_WINNER' if selected else 'NO_UNIQUE_MAJORITY_WINNER',
                        scope='MEASURED_AND_COMBINED' if selected else None)
        if fail == 'analysis_incomplete':
            decision.update(complete=False, selected_arm=None, status='INCOMPLETE_EVIDENCE')
        self.analyzer.return_value = decision

        def freeze(run, spec, selected_decision, destination):
            self.assertEqual(selected_decision, decision)
            self.assertTrue(validate_spec(spec))
            self.assertIn('expected_fingerprint', spec)
            destination.mkdir()
            manifest = dict(selected_arm=selected, test_fixture=True)
            write_json(destination/'freeze_manifest.json', manifest)
            with zipfile.ZipFile(Path(str(destination)+'.zip'), 'w') as archive:
                archive.write(destination/'freeze_manifest.json', 'freeze_manifest.json')
            return manifest

        self.freezer.side_effect = freeze

        def child(argv, **kwargs):
            self.commands.append(argv)
            child_mode = argv[3]
            output = Path(argv[argv.index('--output-root')+1])
            run = output/('cache_confirm_'+child_mode+'_fixture')
            self.children.append(run)
            child_failure = fail == child_mode
            write_json(run/'status.json', dict(complete=not child_failure,
                       status='FAILED' if child_failure else 'CHECK_COMPLETE' if child_mode == 'check' else 'EXPERIMENT_COMPLETE'))
            write_json(run/'raw_evidence.json', dict(test_fixture=True, retained=True))
            current_fingerprint = deepcopy(fingerprint)
            if fail == 'environment' and child_mode == 'check':
                current_fingerprint['nh_git']['head'] = 'unexpected-environment'
            if fail == 'runtime_fingerprint' and child_mode == 'run':
                current_fingerprint['package_versions'] = {'unexpected': 'version'}
            write_json(run/'fingerprint.json', current_fingerprint)
            if fail == 'source_after_preflight' and child_mode == 'check':
                self.sources['select_version.py'] = 'd'*64
            if fail == 'config_after_preflight' and child_mode == 'check':
                changed_config = deepcopy(self.reference['config'])
                changed_config['cpu_threads'] += 1
                write_json(self.config, changed_config)
            return FakeChild(2 if child_failure else 0)

        with self.patched_identity(), patch.object(driver.subprocess, 'Popen', side_effect=child), \
             patch.dict('sys.modules', {
                 'selection_analysis': types.SimpleNamespace(analyze=self.analyzer),
                 'freeze_selection': types.SimpleNamespace(freeze=self.freezer)}), \
             contextlib.redirect_stdout(io.StringIO()):
            code = driver.main([mode, '--config', str(self.config), '--output-root', str(self.root/'results')])
        stages = [p for p in (self.root/'results').glob('selection_*') if p.is_dir()]
        self.assertEqual(len(stages), 1)
        return code, stages[0], decision

    def assert_archive(self, stage):
        archive = Path(str(stage)+'_report.zip')
        self.assertTrue(archive.is_file())
        with zipfile.ZipFile(archive) as saved:
            self.assertIn(stage.name+'/stage_status.json', saved.namelist())
            self.assertTrue(any(x.endswith('/raw_evidence.json') for x in saved.namelist()))
            return saved.namelist()

    def test_preflight_failure_stops_before_gpu_and_retains_archive(self):
        code, stage, _ = self.run_driver(fail='check')
        self.assertEqual(code, 2)
        self.assertEqual([cmd[3] for cmd in self.commands], ['check'])
        self.analyzer.assert_not_called()
        self.freezer.assert_not_called()
        self.assertFalse(json.loads((stage/'stage_status.json').read_text())['complete'])
        self.assert_archive(stage)

    def test_changed_preflight_environment_stops_before_gpu(self):
        code, stage, _ = self.run_driver(fail='environment')
        self.assertEqual(code, 2)
        self.assertEqual([cmd[3] for cmd in self.commands], ['check'])
        self.freezer.assert_not_called()
        self.assert_archive(stage)

    def test_source_edit_after_preflight_stops_before_gpu(self):
        code, stage, _ = self.run_driver(fail='source_after_preflight')
        self.assertEqual(code, 2)
        self.assertEqual([cmd[3] for cmd in self.commands], ['check'])
        self.freezer.assert_not_called()
        self.assert_archive(stage)

    def test_config_edit_after_preflight_stops_before_gpu(self):
        code, stage, _ = self.run_driver(fail='config_after_preflight')
        self.assertEqual(code, 2)
        self.assertEqual([cmd[3] for cmd in self.commands], ['check'])
        self.freezer.assert_not_called()
        self.assert_archive(stage)

    def test_check_only_never_launches_run_or_freeze(self):
        code, stage, _ = self.run_driver(mode='check')
        self.assertEqual(code, 0)
        self.assertEqual([cmd[3] for cmd in self.commands], ['check'])
        status = json.loads((stage/'stage_status.json').read_text())
        self.assertTrue(status['complete'])
        self.assertTrue(status['no_models_loaded'])
        self.assertFalse(status['gpu_experiment_requested'])
        self.analyzer.assert_not_called()
        self.freezer.assert_not_called()

    def test_success_calls_symmetric_analyzer_then_freezes_winner(self):
        code, stage, decision = self.run_driver()
        self.assertEqual(code, 0)
        self.assertEqual([cmd[3] for cmd in self.commands], ['check', 'run'])
        self.analyzer.assert_called_once()
        self.freezer.assert_called_once()
        saved = json.loads((stage/'selection_plan.json').read_text())
        self.assertEqual(saved['spec_sha256'], digest({k:v for k,v in saved.items() if k != 'spec_sha256'}))
        self.assertEqual(json.loads((stage/'selection_decision.json').read_text()), decision)
        self.assertEqual(self.freezer.call_args.args[-1], stage/'frozen_release')
        status = json.loads((stage/'stage_status.json').read_text())
        self.assertTrue(status['complete'])
        self.assertEqual(status['selected_arm'], 'nh_mean_bounded')
        self.assert_archive(stage)

    def test_no_winner_archives_both_checkpoint_without_false_release(self):
        code, stage, _ = self.run_driver(selected=None)
        self.assertEqual(code, 0)
        self.assertEqual(self.freezer.call_args.args[-1], stage/'phase_checkpoint')
        status = json.loads((stage/'stage_status.json').read_text())
        self.assertTrue(status['complete'])
        self.assertIsNone(status['selected_arm'])
        self.assertEqual(status['selection_status'], 'NO_UNIQUE_MAJORITY_WINNER')
        self.assertFalse((stage/'frozen_release').exists())
        self.assert_archive(stage)

    def test_failed_benchmark_retains_raw_evidence_and_suppresses_selection(self):
        code, stage, _ = self.run_driver(fail='run')
        self.assertEqual(code, 2)
        self.assertEqual([cmd[3] for cmd in self.commands], ['check', 'run'])
        self.analyzer.assert_not_called()
        self.freezer.assert_not_called()
        status = json.loads((stage/'stage_status.json').read_text())
        self.assertIsNone(status['selected_arm'])
        self.assertFalse(status['complete'])
        names = self.assert_archive(stage)
        self.assertTrue(any('/benchmark/' in x and x.endswith('/raw_evidence.json') for x in names))

    def test_postrun_fingerprint_drift_prevents_selection_and_freeze(self):
        code, stage, _ = self.run_driver(fail='runtime_fingerprint')
        self.assertEqual(code, 2)
        self.analyzer.assert_not_called()
        self.freezer.assert_not_called()
        self.assert_archive(stage)

    def test_incomplete_analysis_never_freezes(self):
        code, stage, _ = self.run_driver(fail='analysis_incomplete')
        self.assertEqual(code, 2)
        self.analyzer.assert_called_once()
        self.freezer.assert_not_called()
        self.assertFalse(json.loads((stage/'stage_status.json').read_text())['complete'])
        self.assert_archive(stage)


if __name__ == '__main__':
    unittest.main()
