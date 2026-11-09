"""Synthetic complete-panel regression tests; no GPU or network activity."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cache_confirm
from cache_workload import PROFILES
import selection_analysis as analysis
from selection_protocol import make_spec, digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, winner=analysis.MEAN, winning_cases=12, learning_winner=None, balanced=True):
    """All 36 cell files with coherent learning-inclusive derived metrics."""
    spec = make_spec()
    args = cache_confirm.arguments(['plan', '--profiles', *spec['profiles'], '--concurrency', *map(str, spec['concurrency']),
                                   '--seeds', *map(str, spec['seeds']), '--rounds', '1', '--arms', *spec['arms']])
    config = {'model': '/fixture/model', 'device': 'moore', 'gpu_ids': list(range(8))}
    plan = cache_confirm.make_plan(args, config)
    if not balanced:
        for start in range(0, len(plan['execution_order']), 3):
            trio = sorted(plan['execution_order'][start:start + 3], key=lambda j: j['arm'])
            for pos, job in enumerate(trio, 1):
                job.update(serial=start + pos, order_position=pos)
            plan['execution_order'][start:start + 3] = trio
    source = 'fixture source\n'
    (root / 'source').mkdir(parents=True)
    (root / 'source' / 'fixture.py').write_text(source)
    sources = {'fixture.py': hashlib.sha256(source.encode()).hexdigest()}
    fp = dict(config=config, hardware={'available': True, 'devices': [{'name': 'fixture'}] * 8},
              candidate_sources=sources, protocol=plan['protocol'])
    spec.update(config=config, source_sha256=sources, expected_fingerprint=fp)
    spec['spec_sha256'] = digest(spec)
    cases = [(p, c) for p in spec['profiles'] for c in spec['concurrency']]
    other = analysis.V2 if winner == analysis.MEAN else analysis.MEAN
    for job in plan['execution_order']:
        folder = root / 'rounds' / f"round{job['round']:02d}" / 'cells' / f"{job['profile']}_s{job['seed']}_c{job['concurrency']}_{job['arm']}"
        write(folder / 'order.json', dict(job, state='COMPLETE'))
        params = PROFILES[job['profile']]
        in_winning_set = cases.index((job['profile'], job['concurrency'])) < winning_cases
        effective_winner = winner if in_winning_set else None
        def values(phase):
            if effective_winner is None:
                return 100.0, 100.0
            if phase == 'learn' and learning_winner is not None:
                return {learning_winner: (100.0, 20.0),
                        other if learning_winner == winner else winner: (30.0, 1000.0),
                        analysis.VLLM: (20.0, 2000.0)}[job['arm']]
            return {winner: (120.0, 80.0), other: (110.0, 100.0), analysis.VLLM: (100.0, 120.0)}[job['arm']]
        records = []
        for phase, requested in [('learn', 64), ('measured', 128)]:
            tps, mean = values(phase)
            total = requested * params['output_tokens']
            records.append(dict(profile=job['profile'], seed=job['seed'], concurrency=job['concurrency'], arm=job['arm'],
                                phase=phase, requests=requested, success=requested, failures=0,
                                total_output_tokens=total, output_tokens_per_request=params['output_tokens'],
                                prompt_tokens=params['prompt_tokens'], shared_tokens=params['shared_tokens'],
                                wall_s=total / tps, output_tok_s=tps, ttft_mean_ms=mean,
                                ttft_p95_ms=600.0 if job['arm'] == winner else 400.0,
                                ordered_request_contract_sha256=hashlib.sha256(f"{job['profile']}/{job['seed']}/{phase}".encode()).hexdigest(),
                                valid=True, route_valid=True, exact_usage=True, cohort_valid=True,
                                cache_policy_valid=True, event_payload_transparent=True))
        combined = dict(records[-1], phase='learn_plus_measured', requests=192, success=192,
                        total_output_tokens=192 * params['output_tokens'],
                        wall_s=sum(r['wall_s'] for r in records),
                        ttft_mean_ms=sum(r['ttft_mean_ms'] * r['requests'] for r in records) / 192,
                        ordered_request_contract_sha256=hashlib.sha256(f"{job['profile']}/{job['seed']}/combined".encode()).hexdigest())
        combined['output_tok_s'] = combined['total_output_tokens'] / combined['wall_s']
        records.append(combined)
        write(folder / 'cell_statistics.json', records)
    write(root / 'plan.json', plan)
    write(root / 'fingerprint.json', fp)
    write(root / 'status.json', {'status': 'EXPERIMENT_COMPLETE', 'complete': True})
    return spec


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def path(self, arm=analysis.VLLM, repeat=1):
        return self.root / 'rounds' / f'round{repeat:02d}' / 'cells' / f'reuse8_s33962_c16_{arm}' / 'cell_statistics.json'

    def test_mean_wins_complete_panel_even_with_tail_regressions(self):
        spec = fixture(self.root)
        result = analysis.analyze(self.root, spec)
        self.assertTrue(result['complete'])
        self.assertEqual(result['selected_arm'], analysis.MEAN)
        self.assertEqual(result['scope'], 'MEASURED_AND_LEARNING_INCLUSIVE')
        self.assertEqual(result['evidence_level'], 'SINGLE_SEED_SINGLE_ROUND_SCREEN')
        self.assertEqual(result['paired_trials_per_case'], 1)
        self.assertTrue(result['cross_case_order']['per_profile_rotation_complete'])
        for positions in result['cross_case_order']['per_arm_position_counts'].values():
            self.assertEqual(positions, {'1': 4, '2': 4, '3': 4})
        self.assertFalse(json.loads((self.root/'confirmation_status.json').read_text())['execution_position_balanced'])
        coverage = result['per_candidate_coverage']['measured'][analysis.MEAN]
        self.assertEqual(coverage['joint_wins'], 12)
        self.assertEqual(coverage['vllm_tail_regression_trials'], 12)
        with (self.root / 'selection_pairs.csv').open(encoding='utf-8-sig') as f:
            pairs = list(csv.DictReader(f))
        self.assertEqual(len(pairs), 96)
        self.assertTrue(any(r['candidate'] == analysis.V2 and r['both_metrics_better'] == 'False' for r in pairs))

    def test_v2_wins_symmetrically(self):
        result = analysis.analyze(self.root, fixture(self.root, winner=analysis.V2))
        self.assertEqual(result['selected_arm'], analysis.V2)
        self.assertEqual(result['per_candidate_coverage']['measured'][analysis.V2]['joint_wins'], 12)

    def test_six_cases_do_not_make_a_majority(self):
        result = analysis.analyze(self.root, fixture(self.root, winning_cases=6))
        self.assertTrue(result['complete'])
        self.assertEqual(result['status'], 'NO_UNIQUE_MAJORITY_WINNER')
        self.assertIsNone(result['selected_arm'])
        self.assertEqual(result['per_candidate_coverage']['measured'][analysis.MEAN]['joint_wins'], 6)

    def test_seven_joint_cases_qualify_and_losing_cases_remain(self):
        result = analysis.analyze(self.root, fixture(self.root, winning_cases=7))
        self.assertEqual(result['selected_arm'], analysis.MEAN)
        self.assertEqual(len(result['per_candidate_coverage']['measured'][analysis.MEAN]['not_joint_winning_cases']), 5)

    def test_opposite_phase_winners_prevent_selection(self):
        spec = fixture(self.root, learning_winner=analysis.V2)
        result = analysis.analyze(self.root, spec)
        self.assertTrue(result['complete'])
        self.assertEqual(result['status'], 'PHASE_CONFLICT')
        self.assertIsNone(result['selected_arm'])
        self.assertEqual(result['phase_winners']['measured'], analysis.MEAN)
        self.assertEqual(result['phase_winners']['learn_plus_measured'], analysis.V2)

    def test_measured_only_winner_gets_explicit_warm_scope(self):
        spec = fixture(self.root)
        # Identical expensive learning makes combined throughput differences
        # smaller than the predeclared practical margin, without changing warm results.
        for path in self.root.glob('rounds/round*/cells/*/cell_statistics.json'):
            rows = json.loads(path.read_text())
            rows[0].update(output_tok_s=1., wall_s=float(rows[0]['total_output_tokens']))
            rows[2]['wall_s'] = rows[0]['wall_s'] + rows[1]['wall_s']
            rows[2]['output_tok_s'] = rows[2]['total_output_tokens'] / rows[2]['wall_s']
            write(path, rows)
        result = analysis.analyze(self.root, spec)
        self.assertTrue(result['complete'])
        self.assertEqual(result['selected_arm'], analysis.MEAN)
        self.assertEqual(result['scope'], 'WARM_ONLY')
        self.assertIsNone(result['phase_winners']['learn_plus_measured'])

    def test_missing_cell_prevents_qualified_selection(self):
        spec = fixture(self.root)
        self.path().unlink()
        result = analysis.analyze(self.root, spec)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['selected_arm'])

    def test_unbalanced_positions_prevent_selection(self):
        result = analysis.analyze(self.root, fixture(self.root, balanced=False))
        self.assertFalse(result['complete'])
        self.assertIn('cross_case_arm_position_counts_not_four_each', result['unmet_conditions'])
        self.assertFalse(result['cross_case_order']['complete'])

    def test_mismatched_contract_prevents_selection(self):
        spec = fixture(self.root)
        rows = json.loads(self.path().read_text())
        rows[1]['ordered_request_contract_sha256'] = 'f' * 64
        write(self.path(), rows)
        result = analysis.analyze(self.root, spec)
        self.assertFalse(result['complete'])

    def test_source_or_predeclared_rule_drift_prevents_selection(self):
        spec = fixture(self.root)
        (self.root / 'source/fixture.py').write_text('changed')
        result = analysis.analyze(self.root, spec)
        self.assertFalse(result['complete'])
        self.assertTrue(any('source_snapshot_hash_mismatch' in x for x in result['unmet_conditions']))
        spec['rules']['majority_scenarios'] = 6
        result = analysis.analyze(self.root, spec)
        self.assertTrue(any('invalid_frozen_spec' in x for x in result['unmet_conditions']))

    def test_old_multi_seed_three_round_plan_is_ineligible(self):
        spec = fixture(self.root)
        plan = json.loads((self.root / 'plan.json').read_text())
        plan.update(seeds=[33962, 33963], rounds=3)
        write(self.root / 'plan.json', plan)
        result = analysis.analyze(self.root, spec)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['selected_arm'])
        self.assertIn('run plan:seeds_differs_from_frozen_panel', result['unmet_conditions'])

    def test_stale_completed_order_metadata_is_rejected(self):
        spec = fixture(self.root)
        path = self.path().parent / 'order.json'
        order = json.loads(path.read_text())
        order['order_position'] = order['order_position'] % 3 + 1
        write(path, order)
        result = analysis.analyze(self.root, spec)
        self.assertFalse(result['complete'])
        self.assertFalse(result['cross_case_order']['complete'])
        self.assertIn('confirmation_evidence_incomplete', result['unmet_conditions'])

    def test_equal_global_position_counts_are_insufficient_without_profile_rotation(self):
        spec = fixture(self.root)
        plan = json.loads((self.root / 'plan.json').read_text())
        arms = spec['arms']
        for start in range(0, 36, 3):
            profile_idx = start // 9
            concurrency_idx = (start // 3) % 3
            offset = profile_idx if profile_idx < 3 else concurrency_idx
            rot = arms[offset:] + arms[:offset]
            trio = sorted(plan['execution_order'][start:start + 3], key=lambda j: rot.index(j['arm']))
            for pos, job in enumerate(trio, 1):
                job.update(serial=start + pos, order_position=pos)
                folder = self.root / 'rounds/round01/cells' / f"{job['profile']}_s{job['seed']}_c{job['concurrency']}_{job['arm']}"
                write(folder / 'order.json', dict(job, state='COMPLETE'))
            plan['execution_order'][start:start + 3] = trio
        write(self.root / 'plan.json', plan)
        result = analysis.analyze(self.root, spec)
        self.assertFalse(result['complete'])
        for counts in result['cross_case_order']['per_arm_position_counts'].values():
            self.assertEqual(counts, {'1': 4, '2': 4, '3': 4})
        self.assertIn('profile_does_not_rotate_each_arm_across_concurrency', result['unmet_conditions'])

    def test_failed_phase_forbids_selection(self):
        spec = fixture(self.root)
        rows = json.loads(self.path().read_text())
        rows[1].update(success=127, failures=1, valid=False)
        write(self.path(), rows)
        result = analysis.analyze(self.root, spec)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['selected_arm'])

    def test_margin_boundary_and_smaller_positive_improvements(self):
        fixture(self.root)
        baseline = json.loads(self.path().read_text())[1]
        for throughput, latency, qualifies in ((102., 117.6, True), (101.99, 117.6, False), (102., 117.61, False)):
            candidate = dict(baseline, output_tok_s=throughput,
                             wall_s=baseline['total_output_tokens'] / throughput, ttft_mean_ms=latency)
            comparison = analysis.compare_case(candidate, baseline, {})
            self.assertTrue(comparison['both_metrics_better'])
            self.assertEqual(comparison['margin_win'], qualifies)


if __name__ == '__main__':
    unittest.main()
