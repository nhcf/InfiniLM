#!/usr/bin/env python3
"""Targeted CPU regressions for the prefix_lru RR/V3 ablation."""

import asyncio
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = (
    ROOT / "python/infinilm/routing/bounded_hash"
    / "round_robin_router.py"
)
SPEC = importlib.util.spec_from_file_location("rr_router_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
Router = MODULE.Router


class RoundRobinRouterTest(unittest.IsolatedAsyncioTestCase):
    def router(self, count=3, max_inflight=1, kv_tokens=8448):
        return Router(
            [f"http://replica-{index}.invalid" for index in range(count)],
            max_inflight=max_inflight,
            kv_tokens=kv_tokens,
            page_size=256,
            queue_timeout_s=2,
            expected_prompt_tokens=8192,
            expected_output_tokens=32,
            expected_shared_tokens=7936,
        )

    async def test_fixed_cycle_advances_only_after_atomic_admission(self):
        router = self.router()
        leases = []
        for index in range(3):
            lease = await router.acquire({"id": f"request-{index}"})
            leases.append(lease)
            self.assertEqual(lease.worker, index)
            self.assertEqual(router.cursor, (index + 1) % 3)
        for lease in leases:
            router.release(lease, True)

    async def test_full_cursor_replica_is_skipped(self):
        router = self.router(count=3)
        held = await router.acquire({"id": "held-on-zero"})
        router.cursor = 0
        skipped = await router.acquire({"id": "skip-zero"})
        self.assertEqual((held.worker, skipped.worker), (0, 1))
        router.release(held, True)
        router.release(skipped, True)

    async def test_all_full_waits_and_retries_same_rule(self):
        router = self.router(count=2)
        first = await router.acquire({"id": "first"})
        second = await router.acquire({"id": "second"})
        waiting = asyncio.create_task(router.acquire({"id": "waiting"}))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        router.release(first, True)
        admitted = await asyncio.wait_for(waiting, 1)
        self.assertEqual(admitted.worker, 0)
        router.release(second, True)
        router.release(admitted, True)

    async def test_concurrent_admission_never_exceeds_M_or_K(self):
        router = self.router(count=2, max_inflight=3, kv_tokens=32768)
        peak_active = [0, 0]
        peak_reserved = [0, 0]

        async def one(index):
            lease = await router.acquire({"id": f"concurrent-{index}"})
            snapshot = router.snapshot()
            for worker in range(2):
                peak_active[worker] = max(peak_active[worker], snapshot["active"][worker])
                peak_reserved[worker] = max(
                    peak_reserved[worker], snapshot["kv_reserved_tokens"][worker]
                )
            await asyncio.sleep(0)
            router.release(lease, True)

        await asyncio.gather(*(one(index) for index in range(40)))
        self.assertTrue(all(value <= 3 for value in peak_active))
        self.assertTrue(all(value <= 32768 for value in peak_reserved))
        self.assertEqual(router.snapshot()["active"], [0, 0])
        self.assertEqual(router.snapshot()["kv_reserved_tokens"], [0, 0])

    async def test_cancel_and_exception_release_exactly_once(self):
        router = self.router(count=1)
        held = await router.acquire({"id": "held"})
        waiting = asyncio.create_task(router.acquire({"id": "cancelled"}))
        await asyncio.sleep(0)
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        self.assertEqual(router.snapshot()["queue_waiters"], 0)
        router.release(held, False)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            router.release(held, False)
        self.assertEqual(router.snapshot()["active"], [0])

    async def test_prefix_ownership_cache_hints_and_cost_inputs_cannot_change_rr(self):
        variants = [
            {"prefix_digest": "a", "owner": 0, "cached_tokens": 7936, "cost": 1},
            {"prefix_digest": "b", "owner": 2, "cached_tokens": 0, "cost": 10**9},
        ]
        chosen = []
        for index, variant in enumerate(variants):
            router = self.router()
            lease = await router.acquire({"id": f"variant-{index}", **variant})
            chosen.append(lease.worker)
            self.assertFalse(lease.route["cache_observation_used_for_selection"])
            self.assertFalse(lease.route["cost_input_used_for_selection"])
            router.release(lease, True)
        self.assertEqual(chosen, [0, 0])

    async def test_contract_binding_records_31_pages_and_8448_reservation(self):
        router = self.router()
        lease = await router.acquire({"id": "contract"})
        router.bind_contract(lease, {
            "prompt_tokens": 8192,
            "output_tokens": 32,
            "shared_tokens": 7936,
            "prefix_digest": "verified-prefix",
            "backend_instance_id": "instance-0",
        })
        self.assertEqual(lease.route["prefix_pages"], 31)
        self.assertEqual(lease.reserved_tokens, 8448)
        self.assertEqual(lease.route["contract_rpc_count"], 1)
        router.release(lease, True, {
            "prompt_tokens_details": {"cached_tokens": 7936}
        })
        self.assertEqual(lease.route["observed_cached_tokens"], 7936)


if __name__ == "__main__":
    unittest.main(verbosity=2)
