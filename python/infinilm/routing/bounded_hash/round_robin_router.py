"""Capacity-safe round robin for verified full-model replicas.

This policy deliberately has no hash ring, ownership, cache hint, cost model,
skew bound, or warm-wait state.  One event loop owns all mutable admission
state, so selection, full-sequence reservation, and cursor advancement form one
non-awaiting transaction.  Network contract validation happens after that
transaction and before the single generation POST; it never holds a state lock.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass
import math
import time


@dataclass
class WorkerState:
    healthy: bool = True
    active_load: int = 0
    selections: int = 0
    backend_instance_id: str | None = None


@dataclass
class Lease:
    serial: int
    worker: int
    reserved_tokens: int
    prompt: int
    output: int
    shared: int
    digest: str
    route: dict
    backend_instance_id: str | None = None
    started: bool = False
    produced: int = 0
    released: bool = False


class Router:
    """Fixed-order RR with bounded M/K admission and bounded capacity wait."""

    def __init__(self, backends, *, max_inflight, kv_tokens, page_size,
                 queue_timeout_s, expected_prompt_tokens,
                 expected_output_tokens, expected_shared_tokens):
        if (not backends or len(set(backends)) != len(backends)
                or any(not isinstance(url, str) or not url for url in backends)):
            raise ValueError("Distinct nonempty backends are required.")
        values = (max_inflight, kv_tokens, page_size, expected_prompt_tokens,
                  expected_output_tokens, expected_shared_tokens)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("Positive integer RR capacity parameters are required.")
        if (not math.isfinite(queue_timeout_s) or queue_timeout_s <= 0
                or kv_tokens % page_size or expected_shared_tokens % page_size
                or expected_shared_tokens >= expected_prompt_tokens):
            raise ValueError("Invalid RR capacity, timeout, or prefix alignment.")
        reserve = math.ceil(
            (expected_prompt_tokens + expected_output_tokens) / page_size
        ) * page_size
        if max_inflight * reserve > kv_tokens:
            raise ValueError("M full reservations exceed per-replica K.")
        self.backends = list(backends)
        self.max_inflight = max_inflight
        self.kv_tokens = kv_tokens
        self.page_size = page_size
        self.queue_timeout_s = float(queue_timeout_s)
        self.expected_prompt_tokens = expected_prompt_tokens
        self.expected_output_tokens = expected_output_tokens
        self.expected_shared_tokens = expected_shared_tokens
        self.reserve = reserve
        self.workers = [WorkerState() for _ in backends]
        self.reserved = [0 for _ in backends]
        self.active = {}
        self.history = deque(maxlen=4096)
        self.cursor = 0
        self.serial = 0
        self._changed = asyncio.Event()
        self._waiters = deque()
        self.queue_timeouts = 0
        self.queue_cancellations = 0
        self._closed = False

    def _eligible(self):
        return [
            worker for worker, state in enumerate(self.workers)
            if state.healthy
            and state.active_load < self.max_inflight
            and self.reserved[worker] + self.reserve <= self.kv_tokens
        ]

    def _first_from_cursor(self, eligible):
        allowed = set(eligible)
        count = len(self.workers)
        return next(
            worker for worker in ((self.cursor + offset) % count for offset in range(count))
            if worker in allowed
        )

    async def acquire(self, item):
        """Reserve before contract I/O; only this atomic commit advances RR."""
        if self._closed:
            raise RuntimeError("The router is closed.")
        ident = item.get("id")
        if not isinstance(ident, str) or not ident or any(
                lease.route["request_id"] == ident for lease in self.active.values()):
            raise ValueError("A distinct nonempty request ID is required.")
        started = time.monotonic()
        ticket = object()
        self._waiters.append(ticket)
        try:
            while True:
                healthy = [i for i, state in enumerate(self.workers) if state.healthy]
                if not healthy:
                    raise RuntimeError("No healthy backend.")
                eligible = self._eligible()
                if self._waiters[0] is ticket and eligible:
                    worker = self._first_from_cursor(eligible)
                    wait_ms = 1000 * (time.monotonic() - started)
                    self.serial += 1
                    active_before = [state.active_load for state in self.workers]
                    route = {
                        "serial": self.serial,
                        "request_id": ident,
                        "policy": "round_robin",
                        "chosen": worker,
                        "cursor_before": self.cursor,
                        "eligible": list(eligible),
                        "active_before": active_before,
                        "kv_reserved_before": list(self.reserved),
                        "kv_reserved_tokens": self.reserve,
                        "gateway_wait_ms": wait_ms,
                        "gateway_wait_s": wait_ms / 1000,
                        "contract_rpc_count": 0,
                        "cache_observation_used_for_selection": False,
                        "cost_input_used_for_selection": False,
                        "released": False,
                    }
                    lease = Lease(
                        self.serial, worker, self.reserve,
                        self.expected_prompt_tokens, self.expected_output_tokens,
                        self.expected_shared_tokens, "", route,
                    )
                    state = self.workers[worker]
                    state.active_load += 1
                    state.selections += 1
                    self.reserved[worker] += self.reserve
                    self.active[lease.serial] = lease
                    self.history.append(route)
                    self.cursor = (worker + 1) % len(self.workers)
                    self._waiters.popleft()
                    self._changed.set()
                    return lease
                remaining = self.queue_timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                self._changed.clear()
                await asyncio.wait_for(self._changed.wait(), remaining)
        except asyncio.TimeoutError as exc:
            self.queue_timeouts += 1
            raise TimeoutError(
                f"Gateway capacity wait exceeded {self.queue_timeout_s} seconds."
            ) from exc
        except asyncio.CancelledError:
            self.queue_cancellations += 1
            raise
        finally:
            if ticket in self._waiters:
                self._waiters.remove(ticket)
                self._changed.set()

    def bind_contract(self, lease, contract):
        """Bind authoritative selected-backend metadata before generation."""
        if lease.released or self.active.get(lease.serial) is not lease:
            raise RuntimeError("Cannot bind a released or foreign lease.")
        actual = (
            contract.get("prompt_tokens"), contract.get("output_tokens"),
            contract.get("shared_tokens"),
        )
        expected = (
            self.expected_prompt_tokens, self.expected_output_tokens,
            self.expected_shared_tokens,
        )
        if actual != expected:
            raise ValueError(
                f"Selected backend token contract {actual} differs from frozen {expected}."
            )
        reserve = math.ceil((actual[0] + actual[1]) / self.page_size) * self.page_size
        if reserve != self.reserve:
            raise ValueError("Selected backend changed the full KV reservation.")
        instance = contract.get("backend_instance_id")
        digest = contract.get("prefix_digest")
        if not isinstance(instance, str) or not instance or not isinstance(digest, str) or not digest:
            raise ValueError("Selected backend omitted instance or prefix digest.")
        lease.backend_instance_id = instance
        lease.digest = digest
        self.workers[lease.worker].backend_instance_id = instance
        lease.route.update(
            contract_rpc_count=1,
            backend_instance_id=instance,
            prompt_tokens=actual[0],
            output_tokens=actual[1],
            shared_tokens=actual[2],
            prefix_pages=actual[2] // self.page_size,
            token_contract_verified=True,
        )

    def progress(self, lease, obj):
        if lease.released or self.active.get(lease.serial) is not lease:
            raise RuntimeError("Progress after lease release or foreign lease.")
        metadata = obj.get("x_infinilm") or {}
        instance = metadata.get("backend_instance_id")
        if instance is not None and instance != lease.backend_instance_id:
            raise RuntimeError("The backend restarted during generation.")
        produced = metadata.get("generated_tokens")
        if produced is not None:
            if type(produced) is not int or produced < lease.produced or produced > lease.output:
                raise ValueError("Invalid backend decode progress.")
            lease.produced = produced
        if not lease.started and (
                lease.produced > 0 or any(
                    choice.get("delta", {}).get("content")
                    for choice in obj.get("choices", [])
                )):
            lease.started = True
            lease.route["first_content_observed"] = True

    def release(self, lease, success, usage=None):
        if lease.released or self.active.get(lease.serial) is not lease:
            raise RuntimeError("Duplicate release or foreign lease.")
        lease.released = True
        state = self.workers[lease.worker]
        state.active_load -= 1
        self.reserved[lease.worker] -= lease.reserved_tokens
        del self.active[lease.serial]
        details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if cached is not None and (type(cached) is not int or cached < 0):
            cached = None
        lease.route.update(
            released=True,
            success=bool(success),
            observed_cached_tokens=cached if cached is not None else "NOT_AVAILABLE",
            generated_tokens=lease.produced,
        )
        if state.active_load < 0 or self.reserved[lease.worker] < 0:
            raise RuntimeError("Negative RR admission state.")
        self._changed.set()

    def observe_backend(self, worker, backend_instance_id):
        if not isinstance(backend_instance_id, str) or not backend_instance_id:
            raise ValueError("A nonempty backend instance ID is required.")
        self.workers[worker].backend_instance_id = backend_instance_id
        self.workers[worker].healthy = True
        self._changed.set()

    def mark_unhealthy(self, worker):
        self.workers[worker].healthy = False
        self._changed.set()

    def close(self):
        self._closed = True
        self._changed.set()

    def snapshot(self):
        return {
            "policy": "round_robin",
            "cursor": self.cursor,
            "active": [state.active_load for state in self.workers],
            "kv_reserved_tokens": list(self.reserved),
            "selections": [state.selections for state in self.workers],
            "healthy": [state.healthy for state in self.workers],
            "queue_waiters": len(self._waiters),
            "queue_timeouts": self.queue_timeouts,
            "queue_cancellations": self.queue_cancellations,
            "branches": dict(Counter(route["policy"] for route in self.history)),
            "events": list(self.history),
            "full_reservation_tokens": self.reserve,
            "max_inflight": self.max_inflight,
            "kv_tokens": self.kv_tokens,
            "prefix_pages": self.expected_shared_tokens // self.page_size,
            "uses_hash_affinity": False,
            "uses_cache_hints_for_selection": False,
            "uses_cost_model": False,
            "uses_warm_wait": False,
        }
