"""CPU artifact tests: preserve identity, completeness and no-winner meaning."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

import cache_confirm
import freeze_selection as frozen
from cache_workload import PROFILES
from selection_analysis import validate_single_round_order
from selection_protocol import digest, make_spec


def fixture(parent, selected='nh_mean_bounded', scope='WARM_ONLY'):
    root = parent / 'experiment'
    root.mkdir()
    source = root / 'source'
    source.mkdir()
    (source / 'mean_router.py').write_text('# immutable fixture policy\n')
    (source / 'vendor').mkdir()
    (source / 'vendor' / 'LICENSE').write_text('fixture vendor license\n')
    source_hash = {str(p.relative_to(source)): frozen.sha(p) for p in source.rglob('*') if p.is_file()}
    config = dict(model='/existing/model', device='moore', gpu_ids=list(range(8)))
    fp = dict(config=config, hardware=dict(available=True, devices=[dict(name='fixture')] * 8),
              candidate_sources=source_hash)
    spec = make_spec()
    reference = root / 'reference_environment.json'
    frozen.write(reference, dict(config=config, hardware=fp['hardware']))
    spec.update(config=config, expected_fingerprint=fp, source_sha256=source_hash,
                reference_environment_sha256=frozen.sha(reference))
    spec['spec_sha256'] = digest(spec)
    frozen.write(root / 'selection_plan.json', spec)
    args = cache_confirm.arguments(['plan', '--profiles', *spec['profiles'], '--concurrency', *map(str, spec['concurrency']),
                                    '--seeds', *map(str, spec['seeds']), '--rounds', '1', '--arms', *spec['arms']])
    plan = cache_confirm.make_plan(args, config)
    frozen.write(root / 'plan.json', plan)
    frozen.write(root / 'execution_order.json', plan['execution_order'])
    for job in plan['execution_order']:
        label = f"{job['profile']}_s{job['seed']}_c{job['concurrency']}_{job['arm']}"
        folder = root / 'rounds/round01/cells' / label
        folder.mkdir(parents=True)
        frozen.write(folder / 'order.json', dict(job, state='COMPLETE'))
    frozen.write(root / 'fingerprint.json', fp)
    frozen.write(root / 'phase_model.json', dict(fingerprint=fp, calibration='fixture'))
    frozen.write(root / 'status.json', dict(status='EXPERIMENT_COMPLETE', complete=True,
                 completed_cells=36, valid_phase_rows=108, evidence_kind='unbalanced_screen_not_confirmation'))
    frozen.write(root / 'confirmation_status.json', dict(complete=True, execution_position_balanced=False,
                 expected_phase_cells=108, found_phase_cells=108))
    frozen.write(root / 'postprocess_status.json', dict(complete=True))
    work = root / 'workloads'
    work.mkdir()
    cases = []
    for profile in spec['profiles']:
        for seed in spec['seeds']:
            folder = work / f'{profile}_seed{seed}'
            folder.mkdir()
            (folder / 'smoke.json').write_text('[{},{}]')
            params = PROFILES[profile]
            files = {}
            for phase, count in [('learn', 64), ('measured', 128)]:
                path = folder / (phase + '.jsonl')
                path.write_text(''.join(json.dumps(dict(id=f'{phase}-{i}',
                    **{k:params[k] for k in ('prompt_tokens', 'shared_tokens', 'output_tokens')}))+'\n'
                    for i in range(count)))
                files[phase] = dict(count=count, sha256=frozen.sha(path))
            cases.append(dict(profile=profile, seed=seed, folder=folder.name, files=files, **params))
    frozen.write(work / 'manifest.json', dict(cells=cases))
    (work / 'calibration').mkdir()
    (work / 'calibration' / 'fixture.jsonl').write_text('{}\n')
    decision = dict(status='SELECTED' if selected else 'NO_UNIQUE_MAJORITY_WINNER',
        complete=True, selected_arm=selected, scope=scope if selected else None,
        evidence_level=spec['evidence_level'], cross_case_order=validate_single_round_order(root, plan, spec),
        performance_qualified=bool(selected), spec_sha256=spec['spec_sha256'],
        phase_winners=dict(measured=selected,
                          learn_plus_measured=selected if scope == 'MEASURED_AND_LEARNING_INCLUSIVE' else None),
        per_candidate_coverage=dict(measured={selected: dict(joint_wins=7)} if selected else {}))
    frozen.write(root / 'selection_decision.json', decision)
    (root / 'selection_pairs.csv').write_text('candidate,baseline\n')
    (root / 'selection_cases.csv').write_text('scenario,joint_win\n')
    (root / 'selection_report.md').write_text('# Full result fixture\n')
    return root, spec, decision


class FreezeTests(unittest.TestCase):
    def test_real_analyzer_output_freezes_selected_and_no_winner_panels(self):
        import selection_analysis
        from test_selection_analysis import fixture as full_panel_fixture
        for winning_cases in (7, 6):
            with self.subTest(winning_cases=winning_cases), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp)
                root = parent/'full_panel'
                spec = full_panel_fixture(root, winning_cases=winning_cases)
                auxiliary = parent/'auxiliary'
                auxiliary.mkdir()
                template, _, _ = fixture(auxiliary)
                shutil.copytree(template/'workloads', root/'workloads')
                fp = frozen.read(root/'fingerprint.json')
                frozen.write(root/'reference_environment.json', dict(config=fp['config'], hardware=fp['hardware']))
                spec['reference_environment_sha256'] = frozen.sha(root/'reference_environment.json')
                spec.pop('spec_sha256')
                spec['spec_sha256'] = digest(spec)
                frozen.write(root/'selection_plan.json', spec)
                frozen.write(root/'phase_model.json', dict(fingerprint=fp, calibration='fixture'))
                frozen.write(root/'execution_order.json', frozen.read(root/'plan.json')['execution_order'])
                frozen.write(root/'postprocess_status.json', dict(complete=True))
                frozen.write(root/'status.json', dict(status='EXPERIMENT_COMPLETE', complete=True, completed_cells=36,
                             valid_phase_rows=108, evidence_kind='unbalanced_screen_not_confirmation'))
                decision = selection_analysis.analyze(root, spec)
                self.assertTrue(decision['complete'])
                self.assertFalse(frozen.read(root/'confirmation_status.json')['execution_position_balanced'])
                result = frozen.freeze(root, spec, decision, parent/'frozen')
                self.assertEqual(result['performance_qualified'], winning_cases == 7)
                self.assertEqual(result['selected_arm'], 'nh_mean_bounded' if winning_cases == 7 else None)
                self.assertTrue(Path(result['archive']).is_file())

    def test_selected_bundle_contains_dependency_code_config_inputs_and_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root, spec, decision = fixture(parent)
            before = frozen.tree_manifest(root)
            out = parent / 'release'
            result = frozen.freeze(root, spec, decision, out)
            self.assertEqual(result['status'], 'FROZEN_SELECTED_VERSION')
            self.assertTrue(result['performance_qualified'])
            self.assertEqual(result['scope'], 'WARM_ONLY')
            self.assertEqual(result['evidence_level'], 'SINGLE_SEED_SINGLE_ROUND_SCREEN')
            self.assertFalse(result['within_scenario_order_balanced'])
            marker = frozen.read(out/'selected_version.json')
            self.assertEqual(marker['policy'], 'mean_bounded')
            self.assertEqual(marker['rounds_per_scenario'], 1)
            self.assertEqual(marker['seeds'], [33962])
            self.assertTrue((out/'code/vendor/LICENSE').is_file())
            self.assertEqual(frozen.read(out/'code/config.musa.json'), spec['config'])
            self.assertEqual(frozen.sha(out/'code/reference_environment.json'), spec['reference_environment_sha256'])
            self.assertEqual(frozen.tree_manifest(out/'workloads'), frozen.tree_manifest(root/'workloads'))
            self.assertEqual(frozen.tree_manifest(root), before)
            manifest = frozen.read(out/'FREEZE_MANIFEST.json')
            for name, record in manifest['files'].items():
                self.assertEqual(frozen.sha(out/name), record['sha256'])
            with zipfile.ZipFile(out.with_suffix('.zip')) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn('release/selected_version.json', archive.namelist())

    def test_complete_no_winner_freezes_both_without_selected_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root, spec, decision = fixture(parent, selected=None)
            out = parent/'checkpoint'
            result = frozen.freeze(root, spec, decision, out)
            self.assertEqual(result['status'], 'FROZEN_PHASE_CHECKPOINT')
            self.assertIsNone(result['selected_arm'])
            self.assertFalse(result['performance_qualified'])
            self.assertFalse((out/'selected_version.json').exists())
            checkpoint = frozen.read(out/'checkpoint_versions.json')
            self.assertEqual({r['arm'] for r in checkpoint['candidates']}, set(frozen.CANDIDATES))
            self.assertEqual(checkpoint['evidence_level'], 'SINGLE_SEED_SINGLE_ROUND_SCREEN')

    def test_incomplete_unknown_selection_and_conflicting_scope_rejected(self):
        modifications = [dict(complete=False), dict(selected_arm='vllm_least_active'),
                         dict(scope='UNIVERSAL_WIN'), dict(performance_qualified=False),
                         dict(evidence_level='MULTISEED_CONFIRMATION')]
        for update in modifications:
            with self.subTest(update=update), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp)
                root, spec, decision = fixture(parent)
                decision.update(update)
                frozen.write(root/'selection_decision.json', decision)
                with self.assertRaises(ValueError):
                    frozen.freeze(root, spec, decision, parent/'release')
                self.assertFalse((parent/'release').exists())

    def test_missing_or_misreported_one_pass_order_blocks_freeze(self):
        for kind in ('missing_cell', 'false_confirmation', 'changed_audit', 'incomplete_row_count'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp)
                root, spec, decision = fixture(parent)
                if kind == 'missing_cell':
                    next(root.glob('rounds/round*/cells/*/order.json')).unlink()
                elif kind == 'changed_audit':
                    decision['cross_case_order']['per_arm_position_counts']['nh_mean_bounded']['1'] = 3
                    frozen.write(root/'selection_decision.json', decision)
                else:
                    checks = frozen.read(root/'confirmation_status.json')
                    if kind == 'false_confirmation':
                        checks['execution_position_balanced'] = True
                    else:
                        checks['found_phase_cells'] = 107
                    frozen.write(root/'confirmation_status.json', checks)
                with self.assertRaises(ValueError):
                    frozen.freeze(root, spec, decision, parent/'release')
                self.assertFalse((parent/'release').exists())

    def test_tampered_source_or_workload_or_calibration_rejected(self):
        for kind in ('source', 'workload', 'calibration', 'extra_source'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp)
                root, spec, decision = fixture(parent)
                if kind == 'source':
                    (root/'source/mean_router.py').write_text('# different policy\n')
                elif kind == 'workload':
                    file = next((root/'workloads').glob('*/measured.jsonl'))
                    file.write_text(file.read_text()+'{}\n')
                elif kind == 'calibration':
                    frozen.write(root/'phase_model.json', dict(fingerprint={}))
                else:
                    (root/'source/unrecorded.py').write_text('pass\n')
                with self.assertRaises(ValueError):
                    frozen.freeze(root, spec, decision, parent/'release')
                self.assertFalse((parent/'release').exists())

    def test_missing_evidence_and_different_decision_rejected(self):
        for kind in ('report', 'decision', 'reference'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp)
                root, spec, decision = fixture(parent)
                if kind == 'report':
                    (root/'selection_cases.csv').unlink()
                elif kind == 'reference':
                    (root/'reference_environment.json').unlink()
                else:
                    saved = copy.deepcopy(decision)
                    saved['unexpected'] = 'not the supplied decision'
                    frozen.write(root/'selection_decision.json', saved)
                with self.assertRaises(ValueError):
                    frozen.freeze(root, spec, decision, parent/'release')

    def test_existing_destination_and_in_run_destination_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root, spec, decision = fixture(parent)
            out = parent/'release'
            out.mkdir()
            (out/'keep').write_text('untouched')
            with self.assertRaises(FileExistsError):
                frozen.freeze(root, spec, decision, out)
            self.assertEqual((out/'keep').read_text(), 'untouched')
            with self.assertRaises(ValueError):
                frozen.freeze(root, spec, decision, root/'nested')
            with self.assertRaises(ValueError):
                frozen.freeze(root, spec, decision, parent)

    def test_majority_scope_and_saved_plan_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root, spec, decision = fixture(parent, scope='MEASURED_AND_LEARNING_INCLUSIVE')
            decision['per_candidate_coverage']['measured']['nh_mean_bounded']['joint_wins'] = 6
            frozen.write(root/'selection_decision.json', decision)
            with self.assertRaisesRegex(ValueError, 'majority'):
                frozen.freeze(root, spec, decision, parent/'release')
            decision['per_candidate_coverage']['measured']['nh_mean_bounded']['joint_wins'] = 7
            frozen.write(root/'selection_decision.json', decision)
            saved = copy.deepcopy(spec)
            saved['extra'] = 'different plan'
            frozen.write(root/'selection_plan.json', saved)
            with self.assertRaisesRegex(ValueError, 'saved pre-run plan'):
                frozen.freeze(root, spec, decision, parent/'release')


if __name__ == '__main__':
    unittest.main()
