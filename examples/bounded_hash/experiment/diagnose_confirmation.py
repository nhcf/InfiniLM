#!/usr/bin/env python3
"""Offline, stdlib-only request/route audit. Never starts models or changes a run.

Works on one cache_suite run, or a directory containing repeated run directories.
Gateway-subtracted TTFT is an observed client residual, NOT measured GPU time.
Missing cache telemetry remains unknown. No counterfactual routing win is claimed.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def integer(value):
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def mapping(value):
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}


def percentile(values, p):
    a = sorted(x for x in values if x is not None)
    if not a:
        return None
    t = (len(a) - 1) * p
    lo, hi = int(math.floor(t)), int(math.ceil(t))
    return a[lo] + (a[hi] - a[lo]) * (t - lo)


def avg(values):
    a = [x for x in values if x is not None]
    return statistics.mean(a) if a else None


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metadata(folder, phase):
    rows = []
    p = folder / 'cell_statistics.json'
    if p.is_file():
        value = read(p)
        rows = value if isinstance(value, list) else []
    if not rows and (folder / 'summary.csv').is_file():
        with (folder / 'summary.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
    return next((r for r in rows if r.get('phase') == phase), {})


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def inspect_phase(path, root, slow_ms):
    phase = path.name.removesuffix('_requests.json')
    folder = path.parent
    meta = metadata(folder, phase)
    records = read(path)
    if not isinstance(records, list):
        raise ValueError(str(path) + ': expected list')
    route_path = folder / (phase + '_routes.json')
    route_value = read(route_path) if route_path.is_file() else {}
    events = route_value.get('events', []) if isinstance(route_value, dict) else route_value
    if not isinstance(events, list):
        raise ValueError(str(route_path) + ': expected events list')
    requested_ids = [r.get('id') for r in records]
    # measured_routes.json is a cumulative snapshot in the original runner.
    event_groups = defaultdict(list)
    for r in events:
        event_groups[r.get('request_id')].append(r)
    shared = meta.get('shared_tokens')
    try:
        shared = int(shared)
    except (TypeError, ValueError):
        shared = None
    warnings = []
    if len(set(requested_ids)) != len(requested_ids):
        warnings.append('duplicate_request_ids')
    duplicates = sum(len(event_groups.get(k, [])) > 1 for k in set(requested_ids))
    if duplicates:
        warnings.append('duplicate_route_ids')
    cell = str(folder.relative_to(root))
    results = []
    for req in records:
        matches = event_groups.get(req.get('id'), [])
        route = matches[0] if len(matches) == 1 else {}
        usage = req.get('usage') or {}
        cached = integer((usage.get('prompt_tokens_details') or {}).get('cached_tokens'))
        category = ('unknown' if cached is None or shared is None or shared <= 0 else
                    'full' if cached >= shared else 'cold' if cached == 0 else 'partial')
        chosen = route.get('chosen')
        components = mapping(route.get('phase_components'))
        component = components.get(str(chosen), {}) or {}
        observed = mapping(route.get('cached_token_hints'))
        observation = route.get('admission_observation') or {}
        observation = observation if isinstance(observation, dict) else {}
        all_observed = mapping(route.get('all_worker_cache_hints') or observation.get('all_worker_cache_hints'))
        all_observed_present = bool(all_observed)
        hints = all_observed if all_observed_present else observed
        bounded = {str(x) for x in route.get('bounded_candidates', [])}
        capacity_list = route.get('capacity_candidates', observation.get('capacity_candidates'))
        capacity = {str(x) for x in capacity_list or []}
        owners = {str(x) for x in route.get('stable_owners', [])}
        warm = ({k for k, v in hints.items() if number(v) is not None and shared is not None and shared > 0 and v >= shared}
                if hints else set())
        other_warm = warm - {str(chosen)}
        # Only assert absent alternatives when the corresponding set was observed.
        warm_bounded = sorted(other_warm & bounded) if hints else None
        warm_excluded = sorted((other_warm & capacity) - bounded) if all_observed_present and 'bounded_candidates' in route and capacity_list is not None else None
        warm_capacity_blocked = sorted(other_warm - capacity) if all_observed_present and capacity_list is not None else None
        excluded_owners = sorted((owners & capacity) - bounded) if capacity_list is not None and 'bounded_candidates' in route else None
        ttft = number(req.get('ttft_ms'))
        wait = number(route.get('gateway_wait_ms'))
        residual = ttft - wait if ttft is not None and wait is not None else None
        units = str(route.get('score_units', ''))
        score = number(route.get('selected_score_ms'))
        if score is None and 'ms' in units:
            score = number(component.get('score_ms'))
        # Old mean scores are counts of tokens, so never compare them to time.
        pred = integer(route.get('predicted_cached_tokens'))
        pending = number(component.get('pending_prefill_ms'))
        scores = {k: number(v) for k, v in mapping(route.get('scores')).items()}
        comparable_scores = {k: v for k, v in scores.items() if v is not None}
        selected_raw = comparable_scores.get(str(chosen))
        cheaper = sorted(k for k, v in comparable_scores.items() if k != str(chosen) and selected_raw is not None and v < selected_raw)
        worker_observation = next((w for w in observation.get('workers', []) if w.get('worker') == chosen), {})
        peers = observation.get('chosen_worker_peers')
        slow = ttft is not None and ttft >= slow_ms
        signals = []
        if slow and category in ('cold', 'partial'):
            signals.append('slow_with_incomplete_cache_reuse')
        if slow and category == 'full':
            signals.append('slow_despite_full_cache_reuse')
            if pending is not None and pending > 0:
                signals.append('selected_worker_has_predicted_pending_prefill')
        if slow and wait is not None and wait >= 1000:
            signals.append('material_gateway_wait')
        if warm_excluded:
            signals.append('observed_warm_alternative_excluded_by_count_bound')
        if warm_bounded and category in ('cold', 'partial'):
            signals.append('observed_warm_alternative_inside_bounded_set')
        if pred is not None and cached is not None and pred > cached:
            signals.append('predicted_reuse_overestimated')
        row = dict(cell=cell, profile=meta.get('profile'), seed=meta.get('seed'),
            concurrency=meta.get('concurrency'), arm=meta.get('arm'), phase=phase,
            request_id=req.get('id'), success=req.get('success'),
            prompt_sha256=req.get('prompt_sha256'), prompt_tokens=usage.get('prompt_tokens'),
            output_tokens=usage.get('completion_tokens'), shared_tokens=shared,
            cache_group=category, cached_tokens=cached,
            ttft_ms=ttft, e2e_ms=number(req.get('e2e_ms')), slow=slow,
            gateway_wait_ms=wait, client_residual_after_gateway_ms=residual,
            route_join_valid=len(matches) == 1, chosen=chosen, preferred=route.get('preferred'),
            branch=route.get('branch'), active_before=route.get('active_before'),
            capacity_candidates=capacity_list, bounded_candidates=route.get('bounded_candidates'),
            selected_score_ms=score, score_units=units,
            residual_minus_predicted_score_ms=(residual-score if residual is not None and score is not None else None),
            predicted_cached_tokens=pred, selected_pending_prefill_ms=pending,
            selected_own_prefill_ms=number(component.get('new_prefill_ms')),
            selected_decode_interference_ms=number(component.get('decode_interference_ms')),
            coalesced_pending_requests=component.get('coalesced_pending_requests'),
            observed_cache_hint=observed.get(str(chosen)),
            owner_wait_requested=route.get('owner_wait_requested'),
            count_bound_relaxed=route.get('count_bound_relaxed'),
            other_observed_warm_in_bounded=warm_bounded,
            other_observed_warm_excluded_by_count=warm_excluded,
            other_observed_warm_capacity_blocked=warm_capacity_blocked,
            owners_excluded_by_count_not_proof_of_warm_cache=excluded_owners,
            all_worker_cache_telemetry_available=all_observed_present,
            admission_observation_status=observation.get('status'),
            admission_observation_capture_ms=number(observation.get('capture_wall_ms')),
            selected_pending_request_count=worker_observation.get('pending_request_count'),
            selected_pending_uncached_tokens=worker_observation.get('pending_uncached_tokens'),
            chosen_worker_peers=peers,
            lower_score_alternatives_not_counterfactual_proof=cheaper,
            signals=signals)
        results.append(row)
    if any(not x['route_join_valid'] for x in results):
        warnings.append('missing_or_ambiguous_route_join')
    return results, dict(cell=cell, phase=phase, request_count=len(records), warnings=warnings,
        noncohort_routes_ignored=len(events)-sum(len(event_groups.get(k, [])) for k in set(requested_ids)))


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['cell'], r['phase'], r['cache_group'])].append(r)
    result = []
    for (cell, phase, cache_group), rr in sorted(groups.items()):
        first = rr[0]
        result.append(dict(cell=cell, profile=first['profile'], seed=first['seed'],
            concurrency=first['concurrency'], arm=first['arm'], phase=phase, cache_group=cache_group,
            requests=len(rr), failed_requests=sum(r['success'] is not True for r in rr),
            slow_requests=sum(r['slow'] for r in rr),
            ttft_mean_ms=avg([r['ttft_ms'] for r in rr]),
            ttft_p95_ms=percentile([r['ttft_ms'] for r in rr], .95),
            gateway_wait_mean_ms=avg([r['gateway_wait_ms'] for r in rr]),
            client_residual_after_gateway_mean_ms=avg([r['client_residual_after_gateway_ms'] for r in rr]),
            pending_prefill_known_requests=sum(r['selected_pending_prefill_ms'] is not None for r in rr),
            predicted_pending_prefill_mean_ms=avg([r['selected_pending_prefill_ms'] for r in rr]),
            warm_excluded_known_requests=sum(r['other_observed_warm_excluded_by_count'] is not None for r in rr),
            observed_warm_excluded_requests=sum(bool(r['other_observed_warm_excluded_by_count']) for r in rr),
            full_worker_cache_telemetry_requests=sum(r['all_worker_cache_telemetry_available'] for r in rr)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--out', type=Path, help='Absent/empty dedicated output directory (default: RUN_ROOT/diagnostics); never overwrites original result files')
    parser.add_argument('--slow-ms', type=float, default=4000)
    args = parser.parse_args(argv)
    root = args.run_root.resolve()
    out = (args.out if args.out is not None else root/'diagnostics').resolve()
    if not root.is_dir() or not math.isfinite(args.slow_ms) or args.slow_ms <= 0:
        parser.error('run_root must exist and slow-ms must be positive and finite')
    if out == root:
        parser.error('--out must be a dedicated directory, not run_root itself')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error('--out must be absent or an empty directory')
    # Repeated runs may be nested under rounds/blocks. Ignore diagnostic trees,
    # including a caller-selected output subtree, so derived files cannot be
    # mistaken for raw request phases. Collect inputs before creating outputs.
    paths = sorted(p for p in (set(root.rglob('learn_requests.json')) | set(root.rglob('measured_requests.json')))
                   if 'diagnostics' not in p.relative_to(root).parts and out not in p.resolve().parents)
    if not paths:
        parser.error('no complete learn_requests.json or measured_requests.json files found')
    rows = []; audits = []
    for path in paths:
        part, audit = inspect_phase(path, root, args.slow_ms)
        rows.extend(part); audits.append(audit)
    grouped = summarize(rows)
    slow = sorted((r for r in rows if r['slow']), key=lambda r:r['ttft_ms'], reverse=True)
    out.mkdir(parents=True, exist_ok=True)
    csv_write(out/'request_diagnostics.csv', rows)
    csv_write(out/'slow_requests.csv', slow)
    csv_write(out/'cache_group_summary.csv', grouped)
    status = dict(status='OFFLINE_DIAGNOSIS_COMPLETE', input_root=str(root), slow_threshold_ms=args.slow_ms,
        request_phases=len(paths), request_records=len(rows), slow_requests=len(slow),
        phase_audits=audits, signals=dict(Counter(s for r in rows for s in r['signals'])),
        all_worker_cache_telemetry_requests=sum(r['all_worker_cache_telemetry_available'] for r in rows),
        causal_claim=False, gpu_jobs_started=0,
        limitations=[
          'Client TTFT minus gateway waiting includes network, backend queue, computation, and stream overhead; it is not GPU kernel time.',
          'Predicted costs and pending-prefill fields are router estimates, not observed backend stage durations.',
          'Existing routes usually omit cache hints outside the bounded set. Unknown alternatives are left blank, never counted as cold.',
          'Owner membership does not prove live cached content; cheaper score alternatives do not prove that rerouting would reduce latency.',
          'No output-quality equivalence or general performance superiority is established by this diagnosis.'])
    (out/'diagnosis.json').write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines=['# Offline tail diagnosis', '',
        f'Inspected {len(rows)} request records across {len(paths)} actual request phases. '
        f'{len(slow)} records have TTFT >= {args.slow_ms:g} ms. No GPU jobs were started.', '',
        'Rows remain separated by exact cell and phase; repetitions are not pooled as independent requests.', '',
        '| Cell | Phase | Cache group | Requests | Slow | Mean TTFT ms | Mean gateway wait ms |',
        '|---|---|---|---:|---:|---:|---:|']
    fmt=lambda x: 'unknown' if x is None else f'{x:.3f}'
    for g in grouped:
        lines.append(f"| {g['cell']} | {g['phase']} | {g['cache_group']} | {g['requests']} | {g['slow_requests']} | {fmt(g['ttft_mean_ms'])} | {fmt(g['gateway_wait_mean_ms'])} |")
    lines += ['', '## Interpretation limits', ''] + ['- '+x for x in status['limitations']]
    lines += ['', '## Next focused checks', '',
        'First separate cold/partial misses from warm requests that wait behind pending prefill. '
        'Inspect observed warm alternatives excluded by capacity/count limits only when all-worker telemetry exists. '
        'Keep the policy frozen for the rotated confirmation. If a pattern persists, test exactly one policy change against this frozen version on identical cohorts, '
        'and retain all throughput, mean TTFT, tail TTFT, and learning-inclusive comparisons.']
    (out/'diagnosis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in status.items() if k not in ('phase_audits','limitations')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
