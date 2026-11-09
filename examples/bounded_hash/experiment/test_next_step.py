import unittest
from pathlib import Path
import next_step


class StageTests(unittest.TestCase):
    def test_frozen_confirmation_and_finite_expansion(self):
        j = next_step.jobs('confirm')
        self.assertEqual(sum(len(x['seeds'])*x['rounds']*5 for x in j), 50)
        panel = next_step.jobs('expand')
        self.assertEqual(len(panel), 8)
        self.assertEqual(len({(x['profile'], x['concurrency']) for x in panel}), 8)
        self.assertIn(('cold', 16), next_step.SCREEN_CASES)
        self.assertIn(('longdecode', 16), next_step.SCREEN_CASES)
        self.assertTrue(all(x['arms'] == next_step.SHORT_ARMS for x in panel))
        self.assertTrue(all(x['rounds'] == 3 for x in next_step.jobs('expand-confirm')))

    def test_no_historical_replay_claim_for_expansion(self):
        cmd = next_step.command(next_step.jobs('expand')[0], Path('/config'), Path('/out'))
        self.assertNotIn('--historical-run', cmd)
        cmd = next_step.command(next_step.jobs('confirm')[0], Path('/config'), Path('/out'))
        self.assertIn('--historical-run', cmd)


if __name__ == '__main__':
    unittest.main()
