"""Opt-in, read-only admission observation for all existing benchmark policies.

install() wraps affinity_router.Router._select exactly once. It does not rescore,
reschedule, invoke GPU APIs, or write cache state. Enable the same observation on
all arms; collecting observations does introduce small controller CPU overhead.
"""
from __future__ import annotations
import time
from functools import wraps

SCHEMA_VERSION = 'admission-observation-v1'


def _capture(router, contract, original_candidates):
    prompt, output, shared, digest, reserve = contract
    now = time.monotonic()
    # Single-loop synchronous read. No await, network, forecasts, or cache writes.
    peers = list(router.active.values())
    workers = []
    for i, worker in enumerate(router.workers):
        local = [x for x in peers if x.worker == i]
        pending = [x for x in local if not x.started]
        state = router.cache[i]
        seen = state.get('observed_at', 0.0)
        age = now - seen if seen > 0 else None
        expired = now - seen > router.cache_ttl_s
        cached = router._cached(i, digest, shared, now) if digest and shared > 0 else 0
        eligible = bool(worker.healthy and worker.active_load < router.max_inflight
                        and router.reserved[i] + reserve <= router.kv_tokens)
        workers.append(dict(worker=i, healthy=bool(worker.healthy), active=worker.active_load,
            reserved_tokens=router.reserved[i], capacity_eligible=eligible,
            observed_cache_hint_tokens=cached,
            cache_hint_is_live_gpu_probe=False,
            cache_snapshot_epoch=state.get('epoch'), cache_snapshot_age_s=age,
            cache_snapshot_expired=expired, cache_hint_blocked_by_pending=bool(pending),
            pending_request_count=len(pending), pending_uncached_tokens=sum(x.pending_tokens for x in pending),
            active_decode_request_count=sum(bool(x.started) for x in local)))
    peer_records = [dict(serial=x.serial, request_id=x.route.get('request_id'), worker=x.worker,
        started=bool(x.started), produced=x.produced, prompt=x.prompt, output=x.output,
        shared=x.shared, digest_match=bool(digest and x.digest == digest),
        pending_tokens=x.pending_tokens, released=bool(x.released)) for x in peers]
    return dict(schema=SCHEMA_VERSION, status='OBSERVED', read_only=True,
        observation_point='immediately_before_original_base_select',
        original_select_candidates=list(original_candidates), workers=workers,
        capacity_candidates=[w['worker'] for w in workers if w['capacity_eligible']],
        all_worker_cache_hints={str(w['worker']):w['observed_cache_hint_tokens'] for w in workers},
        incoming_prompt_tokens=prompt, incoming_output_tokens=output,
        incoming_shared_tokens=shared, incoming_reserved_tokens=reserve), peer_records


def install():
    """Call before router creation/run_cell; returns an idempotent install receipt."""
    from affinity_router import Router
    current = Router._select
    if getattr(current, '_admission_telemetry_schema', None) == SCHEMA_VERSION:
        return dict(installed=True, already_installed=True, schema=SCHEMA_VERSION,
                    target='affinity_router.Router._select', changes_selection=False)

    @wraps(current)
    def observed_select(self, item, contract, eligible, wait_ms):
        began = time.perf_counter()
        try:
            observation, peers = _capture(self, contract, eligible)
        except Exception as exc:
            # Diagnostic collection must not invent state or alter routing fallback.
            observation = dict(schema=SCHEMA_VERSION, status='OBSERVATION_ERROR',
                read_only=True, error_type=type(exc).__name__, error=str(exc))
            peers = []
        capture_ms = 1000 * (time.perf_counter() - began)
        lease = current(self, item, contract, eligible, wait_ms)
        observation['chosen_worker'] = lease.worker
        observation['chosen_worker_peers'] = [p for p in peers if p['worker'] == lease.worker]
        observation['capture_wall_ms'] = capture_ms
        observation['new_lease_excluded_from_peers'] = True
        lease.route['admission_observation'] = observation
        return lease

    observed_select._admission_telemetry_schema = SCHEMA_VERSION
    observed_select._admission_telemetry_original = current
    Router._select = observed_select
    return dict(installed=True, already_installed=False, schema=SCHEMA_VERSION,
                target='affinity_router.Router._select', changes_selection=False)


def uninstall():
    """For CPU transparency tests; not required during normal runner execution."""
    from affinity_router import Router
    current = Router._select
    if getattr(current, '_admission_telemetry_schema', None) == SCHEMA_VERSION:
        Router._select = current._admission_telemetry_original
        return True
    return False
