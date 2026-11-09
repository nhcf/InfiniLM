"""CPU regression fixtures for the real 127/128 HTTP-reset failure."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from cache_suite import report_invalid_phase


class FailureReportingTests(unittest.TestCase):
    def fixture(self):
        # This failed record is the relevant subset of the uploaded round02 /
        # seed33961 / nh_lru_least_active measured request, copied verbatim.
        failed = {
            'id': 'cache-v2-reuse8-33961-measured-9-7',
            'success': False,
            'http_status': 502,
            'error': 'RuntimeError(\'{"error": "ClientOSError(104, \\\'Connection reset by peer\\\')"}\')',
            'e2e_ms': 15.475819818675518,
        }
        records = [dict(id=f'success-{i}', success=True) for i in range(127)] + [failed]
        row = dict(profile='reuse8', seed=33961, concurrency=16,
                   arm='nh_lru_least_active', phase='measured', requests=128,
                   success=127, failures=1, valid=False, output_tok_s=None,
                   route_valid=False, event_payload_transparent=True,
                   exact_usage=False, cohort_valid=True, cache_policy_valid=False)
        gateway = [dict(id=failed['id'], error="ClientOSError(104, 'Connection reset by peer')")]
        return row, records, gateway

    def test_actual_reset_is_explained_without_changing_invalid_phase(self):
        row, records, gateway = self.fixture()
        before = copy.deepcopy((row, records, gateway))
        with tempfile.TemporaryDirectory() as tmp:
            message = report_invalid_phase(Path(tmp), row, records, gateway, 128)
            details = json.loads((Path(tmp)/'failure_details.json').read_text())
        self.assertIn('Invalid complete cohort', message)
        self.assertIn('success 127/128', message)
        self.assertIn('cache-v2-reuse8-33961-measured-9-7', message)
        self.assertIn('HTTP 502', message)
        self.assertIn('Connection reset by peer', message)
        self.assertEqual((row, records, gateway), before)
        self.assertIsNone(row['output_tok_s'])
        self.assertFalse(details['valid'])
        self.assertEqual((details['requested'], details['received'], details['success'], details['failures']), (128, 128, 127, 1))
        self.assertEqual(details['failed_requests'][0]['error'], records[-1]['error'])
        self.assertEqual(details['audit_flag_failures'], ['valid', 'route_valid', 'exact_usage', 'cache_policy_valid'])
        self.assertEqual(details['phase_gateway_errors'], gateway)
        self.assertEqual(details['requests_retried'], 0)

    def test_all_requests_successful_does_not_hide_audit_failure(self):
        row, records, gateway = self.fixture()
        records[-1] = dict(id=records[-1]['id'], success=True)
        row.update(success=128, failures=0, route_valid=True, cache_policy_valid=True)
        with tempfile.TemporaryDirectory() as tmp:
            message = report_invalid_phase(Path(tmp), row, records, [], 128)
            details = json.loads((Path(tmp)/'failure_details.json').read_text())
        self.assertIn('exact_usage', message)
        self.assertFalse(details['valid'])
        self.assertEqual(details['failures'], 0)
        self.assertEqual(details['failed_requests'], [])
        self.assertIsNone(row['output_tok_s'])

    def test_valid_phase_cannot_be_reported_as_failure(self):
        row, records, gateway = self.fixture()
        row['valid'] = True
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'already invalid'):
                report_invalid_phase(Path(tmp), row, records, gateway, 128)
            self.assertFalse((Path(tmp)/'failure_details.json').exists())


if __name__ == '__main__':
    unittest.main()
