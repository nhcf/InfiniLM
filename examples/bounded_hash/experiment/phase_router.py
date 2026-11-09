"""Calibrated phase-pressure routing for the existing full-replica gateway.

This is a candidate ranking proxy, not a GPU simulator or a TTFT guarantee.
Decode contributes ONE co-resident decoding round, never all remaining output.
Cache hints retain the original TTL/pending-Prefill invalidation. Admission still
reserves the full page-rounded input plus declared maximum output on every lease.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

from affinity_router import Router as BaseRouter
from mean_router import Router as MeanRouter


def _nonnegative(value, name, positive=False):
    if isinstance(value, bool):
        raise ValueError(name + ' must be a finite numeric value')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + ' must be a finite numeric value') from exc
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError(name + ' must be finite and ' + ('positive' if positive else 'nonnegative'))
    return value


class PhaseModel:
    """Explicit, immutable calibration; never fitted on benchmark outcomes."""

    def __init__(self, source):
        if isinstance(source, (str, Path)):
            source = json.loads(Path(source).read_text())
        if not isinstance(source, dict) or source.get('schema_version') != 1:
            raise ValueError('phase_model requires schema_version=1')
        self.source = json.loads(json.dumps(source, allow_nan=False))
        coefficients = source.get('prefill_coef')
        if not isinstance(coefficients, (list, tuple)) or len(coefficients) != 3:
            raise ValueError('prefill_coef must contain [a,b,d]')
        self.prefill_coef = tuple(_nonnegative(x, 'prefill_coef') for x in coefficients)
        if not any(self.prefill_coef):
            raise ValueError('prefill_coef must describe a nonzero cost')
        raw_steps = source.get('decode_step_ms')
        if not isinstance(raw_steps, dict) or not raw_steps:
            raise ValueError('decode_step_ms must contain calibrated positive batch keys')
        self.decode_steps = {}
        for key, value in raw_steps.items():
            batch = int(key)
            if str(batch) != str(key) or batch < 1:
                raise ValueError('decode_step_ms keys must be positive integer batch sizes')
            self.decode_steps[batch] = _nonnegative(value, 'decode_step_ms', positive=True)
        self.keys = sorted(self.decode_steps)
        if self.keys != list(range(1, self.keys[-1] + 1)):
            raise ValueError('decode calibration must cover every batch from 1 to max batch')
        contexts = source.get('decode_context_tokens')
        if contexts is None:
            per_request = _nonnegative(source.get('decode_reference_context_tokens'),
                                       'decode_reference_context_tokens', positive=True)
            contexts = {str(k): per_request * k for k in self.keys}
        if not isinstance(contexts, dict):
            raise ValueError('decode_context_tokens must be a batch-to-total-context map')
        self.decode_contexts = {k: _nonnegative(contexts.get(str(k), contexts.get(k)),
                                               'decode_context_tokens', positive=True)
                                for k in self.keys}
        # Measurement noise must not make more decoding requests look cheaper.
        high = 0.0
        for k in self.keys:
            high = max(high, self.decode_steps[k])
            self.decode_steps[k] = high
        self.decode_grid = {}
        raw_grid = source.get('decode_context_samples')
        if raw_grid is not None:
            if not isinstance(raw_grid, dict):
                raise ValueError('decode_context_samples must map batch to samples')
            for batch in self.keys:
                points = raw_grid.get(str(batch), raw_grid.get(batch))
                if not isinstance(points, list) or not points:
                    raise ValueError('Context grid must cover every calibrated decode batch')
                clean = sorted((_nonnegative(x['context_tokens'], 'context sample', positive=True),
                                _nonnegative(x['step_ms'], 'decode step sample', positive=True))
                               for x in points)
                if len({x for x,y in clean}) != len(clean):
                    raise ValueError('Duplicate context calibration point')
                monotone, last = [], 0.0
                for context, step in clean:
                    last = max(last, step)
                    monotone.append((context,last))
                self.decode_grid[batch] = monotone
        self.fingerprint = hashlib.sha256(json.dumps(self.source, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    def prefill(self, prompt, cached):
        p = int(prompt)
        if p <= 0:
            raise ValueError('prompt must be positive')
        c = min(max(0, int(cached)), p - 1)
        q = p - c
        a, b, d = self.prefill_coef
        return max(.001, a + b*q/1000 + d*(q*q + 2*q*c)/1e6)

    def decode_step(self, batch, total_context=None):
        if batch == 0:
            return 0.0
        if batch not in self.decode_steps:
            raise ValueError('Uncalibrated decode batch: ' + str(batch))
        context = self.decode_contexts[batch] if total_context is None else _nonnegative(
            total_context, 'active total context')
        if batch in self.decode_grid:
            points = self.decode_grid[batch]
            # Below the smallest measured context retain its fixed FFN cost.
            if context <= points[0][0]:
                return points[0][1]
            for (lo, slo), (hi, shi) in zip(points, points[1:]):
                if context <= hi:
                    return slo + (shi-slo)*(context-lo)/(hi-lo)
            # Only the measured incremental context slope is extrapolated.
            hi, shi = points[-1]
            slope = ((shi-points[-2][1])/(hi-points[-2][0]) if len(points)>1 else 0.0)
            return min(4.0*shi, shi + slope*(context-hi))
        # A simple monotone proxy with bounded extrapolation, explicitly logged.
        ratio = min(4.0, max(.25, context/self.decode_contexts[batch]))
        return self.decode_steps[batch] * ratio


class Router(MeanRouter):
    """Cost-bounded affinity using the existing safe acquire/release lifecycle."""

    def __init__(self, backends, policy='phase_bounded', *, phase_model=None,
                 cost_margin_ms=2.0, **kwargs):
        self.phase_enabled = policy == 'phase_bounded'
        self.phase_model = PhaseModel(phase_model) if self.phase_enabled else None
        self.cost_margin_ms = _nonnegative(cost_margin_ms, 'cost_margin_ms')
        super().__init__(backends, 'mean_bounded' if self.phase_enabled else policy, **kwargs)
        if self.phase_enabled:
            if self.max_inflight > self.phase_model.keys[-1]:
                raise ValueError('Calibrate decode batches through max_inflight')
            self.model = self.phase_model

    def affinity_eligible(self, p, o, shared, digest):
        if not self.phase_enabled:
            return super().affinity_eligible(p, o, shared, digest)
        # Long output and short input are valid phase-aware traffic. Only the
        # prefix telemetry contract (proper, aligned prefix) controls affinity.
        return bool(digest and 0 < shared < p and shared % self.page_size == 0)

    def _preview(self, item, contract, eligible, now):
        if not self.phase_enabled:
            return super()._preview(item, contract, eligible, now)
        p, o, shared, digest, reserve = contract
        ring = self._ring_order(item['key'])
        feasible = [i for i in ring if i in eligible
                    and self.workers[i].active_load < self.max_inflight
                    and self.reserved[i] + reserve <= self.kv_tokens]
        if not feasible:
            raise RuntimeError('No capacity-safe phase candidate')
        affinity = self.affinity_eligible(p, o, shared, digest)
        owner_key = (digest, shared)
        owners = list(self.owners.get(owner_key, ())) if affinity else []
        cached, pending, components, scores = {}, {}, {}, {}
        for i in feasible:
            cached[i] = self._cached(i, digest, shared, now) if affinity else 0
            backlog = self._pending(i)
            pending[i] = sum(x.pending_tokens for x in backlog)
            pending_ms = sum(self.phase_model.prefill(x.prompt, x.estimated_cached)
                             for x in backlog)
            decode = [x for x in self.active.values() if x.worker == i and x.started]
            context = sum(x.prompt + x.produced for x in decode)
            decode_ms = self.phase_model.decode_step(len(decode), context)
            own_ms = self.phase_model.prefill(p, cached[i])
            scores[i] = pending_ms + own_ms + decode_ms
            components[i] = dict(pending_prefill_ms=pending_ms, new_prefill_ms=own_ms,
                                 decode_interference_ms=decode_ms,
                                 active_decode_requests=len(decode),
                                 total_decode_context_tokens=context,
                                 decode_model_kind=('measured_context_grid' if self.phase_model.decode_grid
                                                    else 'bounded_context_ratio'))
        global_best = min(feasible, key=lambda i: (scores[i], self.workers[i].active_load,
                                                  ring.index(i)))
        cost_limit = scores[global_best] * self.skew + self.cost_margin_ms
        bounded = [i for i in feasible if scores[i] <= cost_limit]
        stable = [i for i in bounded if i in owners or (affinity and cached[i] >= shared)]
        stable_best = min(stable, key=lambda i: (scores[i], i not in owners,
                                                ring.index(i))) if stable else None
        chosen = stable_best if stable_best is not None else global_best
        branch = ('phase_owner' if chosen in owners else
                  'phase_resident_copy' if affinity and cached[chosen] >= shared else
                  'phase_cost_spill')
        primary = self.home.get(owner_key, ring[0]) if affinity else ring[0]
        return dict(chosen=chosen, branch=branch, owner_key=owner_key, primary=primary,
                    ring=ring, feasible=feasible, bounded=bounded, cached=cached,
                    pending=pending, scores=scores, components=components,
                    global_best=global_best, stable_best=stable_best,
                    cost_limit_ms=cost_limit, affinity=affinity)

    def _commit(self, item, contract, eligible, wait_ms, plan):
        if not self.phase_enabled:
            return super()._commit(item, contract, eligible, wait_ms, plan)
        if plan is None:
            raise RuntimeError('Phase admission requires a preview')
        chosen = plan['chosen']
        # BaseRouter performs the original full input+output KV reservation,
        # increases in-flight count and invalidates old snapshots on admission.
        lease = BaseRouter._select(self, item, contract, [chosen], wait_ms)
        lease.preferred = plan['primary']
        # BaseRouter's single-target helper logs its old request-count bound.
        # This policy uses a cost bound, so do not leave that stale attribution.
        lease.route.pop('count_limit_before', None)
        lease.route.update(preferred=plan['primary'], spill=chosen != plan['primary'],
            branch=plan['branch'], ring_primary=plan['ring'][0],
            stable_owners=list(self.owners.get(plan['owner_key'], ())),
            bounded_candidates=plan['bounded'], capacity_candidates=plan['feasible'],
            cached_token_hints=plan['cached'], pending_uncached_tokens=plan['pending'],
            scores=plan['scores'], score_units='calibrated_phase_pressure_ms_proxy',
            phase_components=plan['components'], phase_model_sha256=self.phase_model.fingerprint,
            pending_prefill_ms={i:v['pending_prefill_ms'] for i,v in plan['components'].items()},
            candidate_prefill_cost={i:v['new_prefill_ms'] for i,v in plan['components'].items()},
            decode_step_ms={i:v['decode_interference_ms'] for i,v in plan['components'].items()},
            cost_limit_ms=plan['cost_limit_ms'], cost_margin_ms=self.cost_margin_ms,
            comparison_margin=self.cost_margin_ms, hash_fallback=plan['primary'],
            best_candidate=plan['global_best'], stable_candidate=plan['stable_best'],
            selected_score_ms=plan['scores'][chosen],
            selected_cached_hint=plan['cached'][chosen],
            selected_decode_interference_ms=plan['components'][chosen]['decode_interference_ms'],
            decode_rounds_charged=1, decode_remaining_output_tokens_charged=False,
            secondary_committed=False)
        return lease

    def _dispatch(self):
        start = len(self.history)
        super()._dispatch()
        if self.phase_enabled:
            for event in self.history[start:]:
                if 'queue_score_tokens' in event:
                    event['queue_score_ms'] = event.pop('queue_score_tokens')
                event.update(queue_policy='phase_bounded_actual_destination',
                             queue_score_units='calibrated_phase_pressure_ms_proxy')

    def snapshot(self):
        result = super().snapshot()
        if self.phase_enabled:
            for field in ('escape_ratio_threshold', 'escape_prompt_fraction',
                          'retention_credit_scope', 'queue_score_uses_retention_credit'):
                result.pop(field, None)
            result.update(policy='phase_bounded', cost_margin_ms=self.cost_margin_ms,
                          phase_model_sha256=self.phase_model.fingerprint,
                          decode_rounds_charged=1,
                          cost_bound='min_feasible_score * skew + margin_ms',
                          score_units='calibrated_phase_pressure_ms_proxy')
        return result
