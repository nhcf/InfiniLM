"""CPU observational transparency, using the shipped router's real test fixture."""
import copy
import unittest
from unittest.mock import patch

import admission_telemetry as telemetry
from affinity_router import Router as BaseRouter
from cache_router import Router as CacheRouter
from mean_router import Router as MeanRouter
from test_phase_router import MODEL, item, place, admit, warm


class AdmissionTelemetryTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        telemetry.uninstall()

    async def execute_fixture(self, factory, enabled):
        telemetry.uninstall()
        if enabled:
            telemetry.install()
        with patch('time.monotonic', return_value=100.0):
            router = factory()
            target = item('target', 'same')
            if hasattr(router, 'owners'):
                place(router, target, 0)
            first = admit(router, item('peer', 'same'), 0, started=True)
            first.produced = 13
            warm(router, 0, target)
            # Worker 1 has a stale warm directory. Worker 2 has pending work,
            # so its directory must remain unusable even when newly observed.
            warm(router, 1, target, age=10)
            admit(router, item('pending', 'other'), 2)
            warm(router, 2, target)
            contract = router._validate_item(target)
            old = copy.deepcopy(router.cache)
            obs, _ = telemetry._capture(router, contract, [0, 1, 2])
            self.assertEqual(old, router.cache)
            self.assertEqual(obs['all_worker_cache_hints'], {'0':7936,'1':0,'2':0})
            lease = await router.acquire(target)
            result = dict(worker=lease.worker, reservation=lease.reserved_tokens,
                owners=copy.deepcopy(getattr(router, 'owners', {})),
                reserved=list(router.reserved), active=[w.active_load for w in router.workers],
                cache=copy.deepcopy(router.cache), route=copy.deepcopy(lease.route))
            if enabled:
                observation = result['route'].pop('admission_observation')
                self.assertEqual(observation['status'], 'OBSERVED')
                self.assertTrue(observation['new_lease_excluded_from_peers'])
                self.assertNotIn(lease.serial, [x['serial'] for x in observation['chosen_worker_peers']])
                if lease.worker == 0:
                    self.assertEqual(observation['chosen_worker_peers'][0]['produced'],13)
                    self.assertTrue(observation['chosen_worker_peers'][0]['digest_match'])
            for x in list(router.active.values()):
                router.release(x, True)
            if hasattr(router, 'close'):
                router.close()
            return result

    async def test_all_five_arm_families_keep_selection_and_state(self):
        factories = [
            lambda: CacheRouter(['a','b','c'], phase_model=MODEL),
            lambda: MeanRouter(['a','b','c'],'mean_bounded'),
            lambda: MeanRouter(['a','b','c'],'least_active'),
            lambda: MeanRouter(['a','b','c'],'round_robin'),
            lambda: MeanRouter(['a','b','c'],'least_active'),  # vLLM uses same router class
        ]
        for factory in factories:
            plain=await self.execute_fixture(factory,False)
            observed=await self.execute_fixture(factory,True)
            self.assertEqual(plain,observed)

    def test_install_is_idempotent_and_uninstalls(self):
        original=BaseRouter._select
        self.assertFalse(telemetry.install()['already_installed'])
        wrapped=BaseRouter._select
        self.assertTrue(telemetry.install()['already_installed'])
        self.assertIs(BaseRouter._select,wrapped)
        self.assertTrue(telemetry.uninstall())
        self.assertIs(BaseRouter._select,original)

if __name__=='__main__':unittest.main()
