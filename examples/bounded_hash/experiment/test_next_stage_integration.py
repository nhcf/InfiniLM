"""Exercise the real stage wrapper with fake children; no GPU or subprocesses."""
import contextlib
import csv
import io
import json
import signal
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import next_step


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FakeChild:
    def wait(self):
        return 0

    def poll(self):
        return 0

    def send_signal(self, signum):
        raise AssertionError('No signal should be sent during this test')


class StageIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.commands = []
        self.run_roots = []
        self.selected = next_step.jobs('expand')[:2]
        for sig in (signal.SIGINT, signal.SIGTERM):
            original = signal.getsignal(sig)
            self.addCleanup(signal.signal, sig, original)

    def run_stage(self, fail=None):
        def child(cmd, **kwargs):
            self.commands.append(cmd)
            output = Path(cmd[cmd.index('--output-root') + 1])
            run = output / 'cache_confirm_run_fake'
            self.run_roots.append(run)
            write_json(run / 'status.json', dict(status='EXPERIMENT_COMPLETE', complete=True))
            write_json(run / 'confirmation_status.json', dict(complete=fail != 'incomplete'))
            write_json(run / 'diagnostics/diagnosis.json', dict(status='OFFLINE_DIAGNOSIS_COMPLETE'))
            write_json(run / 'phase_model.json', dict(fingerprint={'test_only': True}))
            if fail != 'missing_csv':
                profile = cmd[cmd.index('--profiles') + 1]
                rows = [dict(profile=profile, concurrency=cmd[cmd.index('--concurrency') + 1],
                             phase=phase, baseline=baseline, paired_trials=2,
                             expected_pairs=2, eligible_for_case_claim=False)
                        for phase in ('learn', 'measured', 'learn_plus_measured')
                        for baseline in ('nh_mean_bounded', 'vllm_least_active')]
                write_rows(run / 'confirmation_summary.csv', rows)
            return FakeChild()
        with mock.patch.object(next_step, 'jobs', return_value=self.selected), \
             mock.patch.object(next_step.subprocess, 'Popen', side_effect=child), \
             contextlib.redirect_stdout(io.StringIO()):
            code = next_step.main(['run', 'expand', '--output-root', str(self.root),
                                   '--config', str(self.root / 'config.json')])
        stages = [p for p in self.root.glob('expand_*') if p.is_dir()]
        self.assertEqual(len(stages), 1)
        return code, stages[0]

    def assert_archived(self, stage):
        archive = Path(str(stage) + '_report.zip')
        self.assertTrue(archive.is_file())
        with zipfile.ZipFile(archive) as stream:
            self.assertIn(stage.name + '/stage_status.json', stream.namelist())
            self.assertTrue(any(name.endswith('/command.json') for name in stream.namelist()))

    def test_two_job_stage_reuses_first_calibration(self):
        code, stage = self.run_stage()
        self.assertEqual(code, 0)
        self.assertEqual(len(self.commands), 2)
        self.assertNotIn('--phase-model', self.commands[0])
        command = self.commands[1]
        self.assertEqual(command[command.index('--phase-model') + 1],
                         str(self.run_roots[0] / 'phase_model.json'))
        status = json.loads((stage / 'stage_status.json').read_text())
        self.assertTrue(status['complete'])
        self.assertEqual(status['completed_jobs'], 2)
        with (stage / 'scenario_scoreboard.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 8)
        self.assertEqual({r['phase'] for r in rows}, {'measured', 'learn_plus_measured'})
        self.assert_archived(stage)

    def test_incomplete_confirmation_stops_later_jobs_and_archives(self):
        code, stage = self.run_stage('incomplete')
        self.assertEqual(code, 2)
        self.assertEqual(len(self.commands), 1)
        status = json.loads((stage / 'stage_status.json').read_text())
        self.assertFalse(status['complete'])
        self.assertEqual(status['status'], 'FAILED')
        self.assertEqual(status['completed_jobs'], 0)
        self.assert_archived(stage)

    def test_missing_summary_stops_later_jobs_and_archives(self):
        code, stage = self.run_stage('missing_csv')
        self.assertEqual(code, 2)
        self.assertEqual(len(self.commands), 1)
        self.assertFalse(json.loads((stage / 'stage_status.json').read_text())['complete'])
        self.assert_archived(stage)

    def test_scoreboard_preserves_mixed_complete_and_partial_columns(self):
        a, b = self.root / 'a', self.root / 'b'
        write_rows(a / 'confirmation_summary.csv', [dict(profile='reuse8', phase='measured', paired_trials=0)])
        write_rows(b / 'confirmation_summary.csv', [dict(profile='cold', phase='measured', paired_trials=2,
                                                        throughput_ratio_geomean=0.8)])
        next_step.scoreboard(self.root, [a, b])
        with (self.root / 'scenario_scoreboard.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['throughput_ratio_geomean'], '')
        self.assertEqual(rows[1]['throughput_ratio_geomean'], '0.8')


if __name__ == '__main__':
    unittest.main()
