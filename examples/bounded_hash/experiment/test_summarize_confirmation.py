"""Offline evidence checks: no GPU, network, or model processes are used."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import summarize_confirmation as summary


ARMS = [summary.CANDIDATE, 'nh_mean_bounded', 'nh_lru_least_active',
        'nh_native_rr', 'vllm_least_active']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read_csv(root, name):
    with (root / name).open() as stream:
        return list(csv.DictReader(stream))


def fixture(root, balanced=True):
    """Match cache_confirm.schedule's actual plan and file schema."""
    jobs = []
    for repeat in range(1, 6):
        offset = ((repeat - 1) + 33960) % 5 if balanced else 0
        order = ARMS[offset:] + ARMS[:offset]
        for position, arm in enumerate(order, 1):
            job = dict(serial=len(jobs) + 1, round=repeat, profile='reuse8',
                       seed=33960, concurrency=16, order_position=position,
                       arm=arm, engine='vllm' if arm.startswith('vllm') else 'nh')
            jobs.append(job)
            folder = root / 'rounds' / f'round{repeat:02d}' / 'cells' / f'reuse8_s33960_c16_{arm}'
            write(folder / 'order.json', dict(job, state='COMPLETE'))
            records = []
            for phase, requested in zip(summary.PHASES, (64, 128, 192)):
                tps = 120.0 if arm == summary.CANDIDATE else 100.0
                records.append(dict(profile='reuse8', seed=33960, concurrency=16,
                                    arm=arm, phase=phase, requests=requested,
                                    success=requested, failures=0,
                                    total_output_tokens=requested * 32,
                                    output_tokens_per_request=32,
                                    wall_s=requested * 32 / tps, output_tok_s=tps,
                                    ttft_mean_ms=100.0 if arm == summary.CANDIDATE else 200.0,
                                    ttft_p95_ms=500.0 if arm == summary.CANDIDATE else 400.0,
                                    ordered_request_contract_sha256={'learn': 'a', 'measured': 'b', 'learn_plus_measured': 'c'}[phase] * 64,
                                    valid=True, route_valid=True, exact_usage=True,
                                    cohort_valid=True, cache_policy_valid=True,
                                    event_payload_transparent=True))
            write(folder / 'cell_statistics.json', records)
    write(root / 'plan.json', dict(arms=[[a, 'vllm' if a.startswith('vllm') else 'nh', 'policy', 'cache'] for a in ARMS],
                                 profiles={'reuse8': {}}, seeds=[33960], concurrency=[16],
                                 rounds=5, expected_phase_rows=75, execution_order=jobs))
    write(root / 'status.json', dict(status='EXPERIMENT_COMPLETE', complete=True))


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture(self.root)

    def path(self, arm='vllm_least_active', round=1):
        return self.root / 'rounds' / f'round{round:02d}' / 'cells' / f'reuse8_s33960_c16_{arm}' / 'cell_statistics.json'

    def mutate(self, update, arm='vllm_least_active', round=1):
        path = self.path(arm, round)
        rows = json.loads(path.read_text())
        update(rows)
        write(path, rows)

    def test_balanced_complete_with_tail_regression_is_retained(self):
        result = summary.analyze(self.root)
        self.assertTrue(result['complete'])
        self.assertTrue(result['execution_position_balanced'])
        self.assertEqual(result['paired_trials'], 60)
        self.assertEqual(result['measured_vllm_pairs'], 5)
        self.assertEqual(result['all_paired_trials_winning_scenarios'], 1)
        self.assertEqual(result['measured_vllm_tail_regressions'], 5)
        records = read_csv(self.root, 'confirmation_summary.csv')
        row = next(r for r in records if r['phase'] == 'measured' and r['baseline'] == 'vllm_least_active')
        self.assertEqual(float(row['candidate_pooled_tok_s']), 120.0)
        self.assertEqual(int(row['candidate_success_count']), 640)
        self.assertEqual(row['all_pairs_both_metrics_better'], 'True')

    def test_losing_result_is_not_excluded(self):
        def change(rows):
            rows[1].update(output_tok_s=200.0, wall_s=4096 / 200.0, ttft_mean_ms=80.0)
        self.mutate(change)
        result = summary.analyze(self.root)
        self.assertTrue(result['complete'])
        self.assertEqual(result['paired_trials'], 60)
        self.assertEqual(result['measured_vllm_both_metrics_wins'], 4)
        self.assertEqual(result['all_paired_trials_winning_scenarios'], 0)
        self.assertEqual(read_csv(self.root, 'confirmation_excluded.csv'), [])
        pair = next(r for r in read_csv(self.root, 'confirmation_pairs.csv')
                    if r['baseline'] == 'vllm_least_active' and r['phase'] == 'measured' and r['round_id'] == '1')
        self.assertEqual(pair['both_metrics_better'], 'False')
        self.assertAlmostEqual(float(pair['throughput_ratio']), 0.6)
        self.assertAlmostEqual(float(pair['mean_ttft_ratio']), 1.25)

    def test_missing_counterpart_never_claims_win(self):
        self.mutate(lambda rows: rows.pop(1))
        result = summary.analyze(self.root)
        self.assertFalse(result['complete'])
        self.assertEqual(result['all_paired_trials_winning_scenarios'], 0)
        exclusions = read_csv(self.root, 'confirmation_excluded.csv')
        self.assertTrue(any(r['reason'] == 'missing_baseline' and r['phase'] == 'measured' for r in exclusions))
        self.assertTrue(all(r['all_pairs_both_metrics_better'] == 'False' for r in read_csv(self.root, 'confirmation_summary.csv')))

    def test_mismatched_request_order_is_excluded(self):
        self.mutate(lambda rows: rows[1].update(ordered_request_contract_sha256='f' * 64))
        result = summary.analyze(self.root)
        self.assertFalse(result['complete'])
        self.assertEqual(result['measured_vllm_pairs'], 4)
        self.assertTrue(any(r['reason'] == 'mismatched_ordered_request_contract' for r in read_csv(self.root, 'confirmation_excluded.csv')))

    def test_invalid_throughput_units_are_excluded(self):
        self.mutate(lambda rows: rows[1].update(output_tok_s=999.0))
        result = summary.analyze(self.root)
        self.assertFalse(result['complete'])
        self.assertTrue(any('throughput_does_not_match' in r['reason'] for r in read_csv(self.root, 'confirmation_excluded.csv')))

    def test_complete_unbalanced_execution_is_not_confirmation_win(self):
        fixture(self.root, balanced=False)
        result = summary.analyze(self.root)
        self.assertTrue(result['complete'])
        self.assertFalse(result['execution_position_balanced'])
        self.assertEqual(result['all_paired_trials_winning_scenarios'], 0)
        self.assertTrue(all(r['eligible_for_case_claim'] == 'False' for r in read_csv(self.root, 'confirmation_summary.csv')))

    def test_final_runner_failure_invalidates_claim_even_with_all_rows(self):
        write(self.root / 'status.json', dict(status='FAILED', complete=False))
        result = summary.analyze(self.root)
        self.assertFalse(result['complete'])
        self.assertEqual(result['paired_trials'], 60)
        self.assertEqual(result['all_paired_trials_winning_scenarios'], 0)
        self.assertTrue(any('runner_did_not_finish' in r['reason'] for r in read_csv(self.root, 'confirmation_excluded.csv')))

    def test_missing_whole_cell_excludes_all_its_phases(self):
        self.path('nh_mean_bounded', 2).unlink()
        result = summary.analyze(self.root)
        self.assertFalse(result['complete'])
        self.assertFalse(result['execution_position_balanced'])
        exclusions = read_csv(self.root, 'confirmation_excluded.csv')
        self.assertEqual(sum(r['baseline'] == 'nh_mean_bounded' and r['reason'] == 'missing_baseline' for r in exclusions), 3)

    def test_actual_order_must_match_plan(self):
        path = self.path().parent / 'order.json'
        value = json.loads(path.read_text())
        value['order_position'] = 3
        write(path, value)
        result = summary.analyze(self.root)
        self.assertFalse(result['complete'])
        self.assertTrue(any('actual_order_differs' in r['reason'] for r in read_csv(self.root, 'confirmation_excluded.csv')))


if __name__ == '__main__':
    unittest.main()
