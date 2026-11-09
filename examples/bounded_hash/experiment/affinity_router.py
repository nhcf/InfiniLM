"""Capacity-safe prefix affinity for the four-way InfiniLM benchmark.

This is a gateway routing candidate, not a change to the engine scheduler.
All methods must run on one asyncio event loop. Cache snapshots are metadata
observations, not reservations of cache blocks. Admission always reserves the
full page-rounded sequence, independently of any estimated prefix reuse.
"""
from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time


MANAGER_SHA = 'ec65247a78c01272bef9c4d51ebdc2bc0b6a1e1383a53d61e1ba5b5c0de56118'


def _manager():
    path = Path(__file__).parent / 'vendor' / 'routing_manager.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != MANAGER_SHA:
        raise ValueError('Audited hash-ring implementation changed')
    name = '_affinity_original_ring'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@dataclass(frozen=True)
class PrefillModel:
    """Measured isolated prefill curve and measured decode step by batch size.

    The result is a ranking estimate, not a guaranteed wall-clock TTFT.
    Unlike the preceding candidate, this model never adds all remaining
    decode tokens of each active request to the queue estimate.
    """
    prefill_coef: tuple
    decode_step_ms: dict

    def __post_init__(self):
        if len(self.prefill_coef) != 3 or not self.decode_step_ms:
            raise ValueError('Three prefill coefficients and decode-step samples are required')
        values = [*self.prefill_coef, *self.decode_step_ms.values()]
        if any(not math.isfinite(float(x)) or float(x) < 0 for x in values):
            raise ValueError('Cost samples must be finite and nonnegative')
        if any(int(k) < 1 for k in self.decode_step_ms):
            raise ValueError('Decode batch sizes must be positive')

    @classmethod
    def from_profile(cls, profile):
        return cls(tuple(float(x) for x in profile['prefill_coef']),
                   {int(k): float(v) for k, v in profile['decode_step_ms'].items()})

    def prefill(self, prompt, cached):
        p = int(prompt)
        c = min(max(0, int(cached)), p - 1)
        q = p - c
        a, b, d = self.prefill_coef
        return max(0.001, a + b*q/1000 + d*(q*q+2*q*c)/1e6)

    def decode_step(self, batch, context=None):
        if batch <= 0:
            return 0.0
        keys = sorted(self.decode_step_ms)
        # Interpolate between measured sizes. Above the largest measured size,
        # scale conservatively; the harness should calibrate every allowed size.
        if batch <= keys[0]:
            return self.decode_step_ms[keys[0]]
        for lo, hi in zip(keys, keys[1:]):
            if lo <= batch <= hi:
                f = (batch-lo)/(hi-lo)
                return self.decode_step_ms[lo]*(1-f) + self.decode_step_ms[hi]*f
        return self.decode_step_ms[keys[-1]] * batch/keys[-1]


@dataclass
class Lease:
    serial: int
    worker: int
    preferred: int
    prompt: int
    output: int
    shared: int
    digest: str
    reserved_tokens: int
    estimated_cached: int
    prefill_ms: float
    pending_tokens: int
    route: dict
    started: bool = False
    produced: int = 0
    released: bool = False


class Router:
    def __init__(self, backends, policy, model=None, max_inflight=3,
                 kv_tokens=32768, page_size=256, skew=1.25,
                 cache_ttl_s=5.0, queue_timeout_s=30.0,
                 prefill_margin_ratio=.05, min_prompt=4096,
                 max_output=128, min_shared=3072, min_shared_ratio=.75):
        if policy not in {'least_active', 'bounded', 'cost_bounded'}:
            raise ValueError('Unknown routing policy: '+str(policy))
        if not backends or len(set(backends)) != len(backends):
            raise ValueError('Distinct nonempty backends required')
        if any(not isinstance(v, int) or v <= 0 for v in (max_inflight, kv_tokens, page_size)):
            raise ValueError('Positive integer capacity limits required')
        if not math.isfinite(skew) or skew < 1:
            raise ValueError('skew must be finite and >=1')
        if any(not math.isfinite(v) or v <= 0 for v in (cache_ttl_s, queue_timeout_s)):
            raise ValueError('Positive finite timeouts required')
        if not 0 <= prefill_margin_ratio < 1 or not 0 <= min_shared_ratio <= 1:
            raise ValueError('Invalid fractional policy thresholds')
        manager = _manager()
        self.workers = [manager.WorkerState('w'+str(i), {'url': b}, 'engine')
                        for i, b in enumerate(backends)]
        self.ring = manager.StableHashRing(self.workers, virtual_nodes=128)
        self.index = {w.member_id: i for i, w in enumerate(self.workers)}
        self.policy, self.model = policy, model
        self.max_inflight, self.kv_tokens, self.page_size = max_inflight, kv_tokens, page_size
        self.skew, self.cache_ttl_s, self.queue_timeout_s = skew, cache_ttl_s, queue_timeout_s
        self.prefill_margin_ratio = prefill_margin_ratio
        self.min_prompt, self.max_output = min_prompt, max_output
        self.min_shared, self.min_shared_ratio = min_shared, min_shared_ratio
        self.active = {}
        self.history = []
        self.turn = self.serial = 0
        self.reserved = [0 for _ in backends]
        self.cache = [dict(epoch=-1, observed_at=0.0, prefixes={}) for _ in backends]
        self._changed = asyncio.Event()
        self._waiters = deque()
        self.queue_timeouts = self.queue_cancellations = 0

    def _validate_item(self, item):
        p, o, shared = (int(item[k]) for k in ('prompt_tokens', 'output_tokens', 'shared_tokens'))
        if p <= 0 or o <= 0 or not 0 <= shared < p:
            raise ValueError('Invalid prompt/output/shared token contract')
        if not isinstance(item.get('id'), str) or not isinstance(item.get('key'), str):
            raise ValueError('Request id and key must be strings')
        digest = item.get('prefix_digest') or ''
        if not isinstance(digest, str):
            raise ValueError('Prefix digest must be a string')
        reserve = math.ceil((p+o)/self.page_size) * self.page_size
        if reserve > self.kv_tokens:
            raise ValueError('One full request exceeds the conservative KV capacity')
        return p, o, shared, digest, reserve

    def affinity_eligible(self, p, o, shared, digest):
        return bool(digest and shared % self.page_size == 0
                    and p >= self.min_prompt and o <= self.max_output
                    and shared >= self.min_shared and shared/p >= self.min_shared_ratio)

    def _pending(self, worker):
        return [lease for lease in self.active.values()
                if lease.worker == worker and not lease.started]

    def _cached(self, worker, digest, shared, now):
        state = self.cache[worker]
        # A newly admitted prefill can evict blocks before it produces its first
        # token. An older request's final snapshot must not mask that uncertainty.
        if self._pending(worker) or now-state['observed_at'] > self.cache_ttl_s:
            return 0
        return min(shared, max(0, int(state['prefixes'].get(digest, 0))))

    def _least(self, eligible, loads):
        low = min(loads[i] for i in eligible)
        n = len(loads)
        return next(i for i in ((self.turn+j) % n for j in range(n))
                    if i in eligible and loads[i] == low)

    def _select(self, item, contract, eligible, wait_ms):
        p, o, shared, digest, reserve = contract
        now = time.monotonic()
        loads = [w.active_load for w in self.workers]
        healthy = [i for i, w in enumerate(self.workers) if w.healthy]
        least = self._least(eligible, loads)
        suitable = self.affinity_eligible(p, o, shared, digest)
        preferred = least
        evidence = {}
        if self.policy == 'least_active' or not suitable or (self.policy == 'cost_bounded' and self.model is None):
            chosen = least
            branch = ('least_active' if self.policy == 'least_active' else
                      'least_active_outside_affinity' if not suitable else
                      'least_active_missing_cost_profile')
        else:
            ring = [self.index[w.member_id] for w in self.ring.walk(item['key']) if w.healthy]
            preferred = ring[0]
            mean = sum(loads[i] for i in healthy)/len(healthy)
            count_limit = self.skew*mean
            bounded = [i for i in ring if i in eligible and loads[i] <= count_limit]
            if not bounded:
                low = min(loads[i] for i in eligible)
                bounded = [i for i in ring if i in eligible and loads[i] == low]
            anchor = bounded[0]
            cached = {i: self._cached(i, digest, shared, now) for i in bounded}
            pending_tokens = {i: sum(x.pending_tokens for x in self._pending(i)) for i in bounded}
            if self.policy == 'bounded':
                scores = {i: pending_tokens[i] + p-cached[i] for i in bounded}
                margin = max(float(self.page_size), p*self.prefill_margin_ratio)
                units = 'uncached_tokens'
                candidate = min(bounded, key=lambda i: scores[i])
                chosen = candidate if scores[anchor]-scores[candidate] > margin else anchor
                branch = ('bounded_hash_hit' if chosen == preferred else
                          'bounded_prefill_spill' if chosen != anchor else 'bounded_capacity_spill')
                pending_cost = None
                candidate_cost = {i: p-cached[i] for i in bounded}
                decode_cost = None
            else:
                pending_cost = {i: sum(x.prefill_ms for x in self._pending(i)) for i in bounded}
                candidate_cost = {i: self.model.prefill(p, cached[i]) for i in bounded}
                decode_cost = {i: self.model.decode_step(
                    sum(x.worker == i and x.started for x in self.active.values()), p) for i in bounded}
                scores = {i: pending_cost[i]+candidate_cost[i]+decode_cost[i] for i in bounded}
                # Margin is tied to PREFILL, never to the full remaining decode.
                margin = max(5.0, self.model.prefill(p, 0)*self.prefill_margin_ratio)
                units = 'estimated_first_token_ms'
                candidate = min(bounded, key=lambda i: scores[i])
                chosen = candidate if scores[anchor]-scores[candidate] > margin else anchor
                branch = ('cost_hash_hit' if chosen == preferred else
                          'cost_prefill_spill' if chosen != anchor else 'cost_capacity_spill')
            evidence = dict(count_limit_before=count_limit, bounded_candidates=bounded,
                            cached_token_hints=cached, pending_uncached_tokens=pending_tokens,
                            pending_prefill_ms=pending_cost, candidate_prefill_cost=candidate_cost,
                            decode_step_ms=decode_cost, scores=scores, score_units=units,
                            comparison_margin=margin, hash_fallback=anchor, best_candidate=candidate)
        estimate = self._cached(chosen, digest, shared, now) if suitable and self.policy != 'least_active' else 0
        prefill = self.model.prefill(p, estimate) if self.model is not None else float(p-estimate)
        observed_at = self.cache[chosen]['observed_at']
        snapshot_age = now-observed_at if observed_at > 0 else None
        self.serial += 1
        route = dict(serial=self.serial, request_id=item['id'], chosen=chosen,
                     preferred=preferred, branch=branch, spill=chosen != preferred,
                     affinity_eligible=suitable, active_before=loads,
                     kv_reserved_before=list(self.reserved), kv_reserved_tokens=reserve,
                     estimated_cached_tokens=estimate, gateway_wait_ms=wait_ms,
                     gateway_wait_s=wait_ms/1000,
                     cache_snapshot_epoch=self.cache[chosen]['epoch'],
                     cache_snapshot_age_s=snapshot_age,
                     cache_snapshot_expired=(snapshot_age is not None and snapshot_age > self.cache_ttl_s),
                     cache_hint_blocked_by_pending=bool(self._pending(chosen)),
                     released=False, **evidence)
        lease = Lease(self.serial, chosen, preferred, p, o, shared, digest,
                      reserve, estimate, prefill, p-estimate, route)
        self.workers[chosen].active_load += 1
        self.workers[chosen].selections += 1
        self.reserved[chosen] += reserve
        self.active[lease.serial] = lease
        self.history.append(route)
        self.turn = (chosen+1) % len(loads)
        # Even a prefix hit needs suffix allocation. Conservatively invalidate
        # the entire directory until a fresh post-prefill snapshot arrives.
        self.cache[chosen]['prefixes'] = {}
        self.cache[chosen]['observed_at'] = 0.0
        return lease

    async def acquire(self, item):
        contract = self._validate_item(item)
        started = time.monotonic()
        ticket = object()
        self._waiters.append(ticket)
        try:
            while True:
                healthy = [i for i, w in enumerate(self.workers) if w.healthy]
                if not healthy:
                    raise RuntimeError('No healthy backend')
                eligible = [i for i in healthy
                            if self.workers[i].active_load < self.max_inflight
                            and self.reserved[i]+contract[-1] <= self.kv_tokens]
                if self._waiters[0] is ticket and eligible:
                    lease = self._select(item, contract, eligible,
                                         1000*(time.monotonic()-started))
                    self._waiters.popleft()
                    self._changed.set()
                    return lease
                remaining = self.queue_timeout_s-(time.monotonic()-started)
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                self._changed.clear()
                await asyncio.wait_for(self._changed.wait(), remaining)
        except asyncio.TimeoutError:
            self.queue_timeouts += 1
            raise TimeoutError('Gateway capacity wait exceeded '+str(self.queue_timeout_s)+' seconds')
        except asyncio.CancelledError:
            self.queue_cancellations += 1
            raise
        finally:
            if ticket in self._waiters:
                self._waiters.remove(ticket)
                self._changed.set()

    def _snapshot_event(self, worker, obj):
        snapshot = (obj.get('x_infinilm') or {}).get('cache_snapshot')
        if snapshot is None:
            return
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get('epoch'), int):
            raise ValueError('Malformed engine cache snapshot')
        prefixes = snapshot.get('prefixes')
        if not isinstance(prefixes, dict) or any(not isinstance(k, str) or not isinstance(v, int) or v < 0
                                                 for k, v in prefixes.items()):
            raise ValueError('Malformed engine prefix directory')
        state = self.cache[worker]
        if snapshot['epoch'] <= state['epoch']:
            return
        state.update(epoch=snapshot['epoch'], prefixes=dict(prefixes), observed_at=time.monotonic())

    def progress(self, lease, obj):
        if lease.released or self.active.get(lease.serial) is not lease:
            raise RuntimeError('Progress after lease release or foreign lease')
        text = any(choice.get('delta', {}).get('content')
                   for choice in obj.get('choices', []))
        if text and not lease.started:
            lease.started = True
            lease.route['first_content_observed'] = True
        produced = (obj.get('x_infinilm') or {}).get('generated_tokens')
        if produced is not None:
            lease.produced = max(lease.produced, min(lease.output, int(produced)))
        self._snapshot_event(lease.worker, obj)

    def release(self, lease, success, usage=None):
        if lease.released or self.active.get(lease.serial) is not lease:
            raise RuntimeError('Duplicate release or foreign lease')
        lease.released = True
        self.workers[lease.worker].active_load -= 1
        self.reserved[lease.worker] -= lease.reserved_tokens
        del self.active[lease.serial]
        observed = ((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens')
        lease.route.update(released=True, success=bool(success),
                           observed_cached_tokens=observed,
                           hint_overestimated=(observed is not None and observed < lease.estimated_cached),
                           generated_tokens=lease.produced)
        if not success:
            self.cache[lease.worker]['prefixes'] = {}
            self.cache[lease.worker]['observed_at'] = 0.0
        if self.reserved[lease.worker] < 0 or self.workers[lease.worker].active_load < 0:
            raise RuntimeError('Negative route reservation')
        self._changed.set()

    def snapshot(self):
        return dict(policy=self.policy, manager_sha256=MANAGER_SHA,
                    active=[w.active_load for w in self.workers],
                    kv_reserved_tokens=list(self.reserved),
                    selections=[w.selections for w in self.workers],
                    queue_waiters=len(self._waiters), queue_timeouts=self.queue_timeouts,
                    queue_cancellations=self.queue_cancellations,
                    cache_hint_overestimates=sum(bool(r.get('hint_overestimated')) for r in self.history),
                    stale_snapshot_selections=sum(bool(r['cache_snapshot_expired']) for r in self.history),
                    branches=dict(Counter(r['branch'] for r in self.history)),
                    events=self.history)
