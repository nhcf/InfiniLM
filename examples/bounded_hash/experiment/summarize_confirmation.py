#!/usr/bin/env python3
"""Summarize matched benchmark repeats without hiding failures or tail regressions.

No request-level confidence intervals are computed. Repeated workload executions
are paired trials, not newly sampled independent requests or new document seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CANDIDATE = 'nh_cache_bounded_v2'
PHASES = ('learn', 'measured', 'learn_plus_measured')
PAIR_FIELDS = ['profile', 'seed', 'concurrency', 'round_id', 'phase', 'baseline',
               'candidate', 'contract_sha256', 'throughput_ratio', 'mean_ttft_ratio',
               'p95_ttft_ratio', 'both_metrics_better', 'tail_regression',
               'candidate_tok_s', 'baseline_tok_s', 'candidate_ttft_ms', 'baseline_ttft_ms',
               'candidate_p95_ms', 'baseline_p95_ms', 'candidate_requests', 'baseline_requests',
               'candidate_success', 'baseline_success', 'candidate_output_tokens',
               'baseline_output_tokens', 'candidate_wall_s', 'baseline_wall_s']
EXCLUDE_FIELDS = ['profile', 'seed', 'concurrency', 'round_id', 'phase', 'baseline', 'reason']


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_write(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def arm_id(arm):
    return arm if isinstance(arm, str) else arm[0]


def truth(value):
    return value is True or value == 'True'


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError('nonfinite_' + key)
    return value


def validate(row):
    """Validate one complete phase summary, including count and throughput units."""
    if not truth(row.get('valid')):
        return 'invalid_phase'
    for field in ('route_valid', 'event_payload_transparent', 'exact_usage', 'cohort_valid', 'cache_policy_valid'):
        if field in row and row[field] not in (None, '') and not truth(row[field]):
            return 'invalid_' + field
    try:
        requested = number(row, 'requests')
        success = number(row, 'success')
        if requested <= 0 or requested != int(requested) or requested != success or number(row, 'failures') != 0:
            return 'incomplete_request_count'
        wall = number(row, 'wall_s')
        total = number(row, 'total_output_tokens' if 'total_output_tokens' in row else 'output_tokens')
        throughput = number(row, 'output_tok_s')
        if wall <= 0 or total <= 0 or throughput <= 0:
            return 'invalid_throughput_input'
        if not math.isclose(throughput, total / wall, rel_tol=1e-7, abs_tol=1e-7):
            return 'throughput_does_not_match_tokens_and_wall'
        if number(row, 'ttft_mean_ms') <= 0 or number(row, 'ttft_p95_ms') <= 0:
            return 'invalid_latency'
        digest = row.get('ordered_request_contract_sha256')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            return 'missing_or_invalid_contract_hash'
    except (KeyError, TypeError, ValueError, OverflowError):
        return 'missing_or_invalid_metric'
    return None


def compare(candidate, baseline, key):
    """Return a paired result or an explicit exclusion; retain poor performance."""
    identity = dict(zip(('profile', 'seed', 'concurrency', 'round_id', 'phase', 'baseline'), key))
    if candidate is None or baseline is None:
        return None, dict(identity, reason='missing_candidate' if candidate is None else 'missing_baseline')
    for name, row in (('candidate', candidate), ('baseline', baseline)):
        error = validate(row)
        if error:
            return None, dict(identity, reason=name + ':' + error)
    if candidate['ordered_request_contract_sha256'] != baseline['ordered_request_contract_sha256']:
        return None, dict(identity, reason='mismatched_ordered_request_contract')
    if int(candidate['requests']) != int(baseline['requests']):
        return None, dict(identity, reason='mismatched_request_counts')
    t = float(candidate['output_tok_s']) / float(baseline['output_tok_s'])
    m = float(candidate['ttft_mean_ms']) / float(baseline['ttft_mean_ms'])
    p = float(candidate['ttft_p95_ms']) / float(baseline['ttft_p95_ms'])
    result = dict(identity, candidate=CANDIDATE,
                  contract_sha256=candidate['ordered_request_contract_sha256'],
                  throughput_ratio=t, mean_ttft_ratio=m, p95_ttft_ratio=p,
                  both_metrics_better=t > 1 and m < 1, tail_regression=p > 1.0)
    for prefix, row in (('candidate', candidate), ('baseline', baseline)):
        result.update({prefix + '_tok_s': float(row['output_tok_s']),
                       prefix + '_ttft_ms': float(row['ttft_mean_ms']),
                       prefix + '_p95_ms': float(row['ttft_p95_ms']),
                       prefix + '_requests': int(row['requests']),
                       prefix + '_success': int(row['success']),
                       prefix + '_output_tokens': int(row['total_output_tokens'] if 'total_output_tokens' in row else row['output_tokens']),
                       prefix + '_wall_s': float(row['wall_s'])})
    return result, None


def geomean(values):
    return math.exp(sum(math.log(x) for x in values) / len(values))


def summarize_pairs(pairs, expected_by_group, eligible_groups):
    groups = defaultdict(list)
    for pair in pairs:
        groups[(pair['profile'], pair['concurrency'], pair['phase'], pair['baseline'])].append(pair)
    output = []
    for key in sorted(set(groups) | set(expected_by_group)):
        rows = groups[key]
        expected = expected_by_group.get(key, 0)
        result = dict(zip(('profile', 'concurrency', 'phase', 'baseline'), key))
        result.update(paired_trials=len(rows), expected_pairs=expected,
                      seeds=len({r['seed'] for r in rows}),
                      both_metrics_better_count=sum(r['both_metrics_better'] for r in rows),
                      tail_regression_count=sum(r['tail_regression'] for r in rows),
                      eligible_for_case_claim=key in eligible_groups and len(rows) == expected and expected > 0,
                      all_pairs_both_metrics_better=key in eligible_groups and bool(rows) and len(rows) == expected and all(r['both_metrics_better'] for r in rows))
        if rows:
            for field in ('throughput_ratio', 'mean_ttft_ratio', 'p95_ttft_ratio'):
                values = [r[field] for r in rows]
                result.update({field + '_geomean': geomean(values), field + '_min': min(values), field + '_max': max(values)})
            for arm in ('candidate', 'baseline'):
                tokens = sum(r[arm + '_output_tokens'] for r in rows)
                wall = sum(r[arm + '_wall_s'] for r in rows)
                requested = sum(r[arm + '_requests'] for r in rows)
                result.update({arm + '_sum_output_tokens': tokens, arm + '_sum_wall_s': wall,
                               arm + '_pooled_tok_s': tokens / wall,
                               arm + '_success_count': sum(r[arm + '_success'] for r in rows),
                               arm + '_request_count': requested,
                               arm + '_weighted_mean_ttft_ms': sum(r[arm + '_ttft_ms'] * r[arm + '_requests'] for r in rows) / requested})
        output.append(result)
    return output


JOB_FIELDS = ('serial', 'round', 'profile', 'seed', 'concurrency', 'order_position', 'arm', 'engine')


def normalize_job(value):
    job = {k: value[k] for k in JOB_FIELDS}
    for k in ('serial', 'round', 'seed', 'concurrency', 'order_position'):
        job[k] = int(job[k])
    return job


def job_key(job):
    return (job['profile'], job['seed'], job['concurrency'], job['round'], job['arm'])


def analyze(root):
    """Read saved runner evidence; never launch or modify a backend.

    Root plan.execution_order is the planned job ledger. Each cell/order.json
    must match its exact job, including position and serial, and be COMPLETE.
    """
    root = Path(root).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    arms = [arm_id(a) for a in plan['arms']]
    if CANDIDATE not in arms or len(arms) != len(set(arms)) or len(arms) < 2:
        raise ValueError('Plan needs one candidate and unique baseline arms')
    baselines = [a for a in arms if a != CANDIDATE]
    profiles = list(plan['profiles'])
    seeds = [int(s) for s in plan['seeds']]
    levels = [int(c) for c in plan['concurrency']]
    repeats = int(plan['rounds'])
    if repeats <= 0 or not profiles or not seeds or not levels:
        raise ValueError('Expected positive rounds and nonempty profile/seed/concurrency')
    if any(len(x) != len(set(x)) for x in (profiles, seeds, levels)):
        raise ValueError('Duplicate profile/seed/concurrency in plan')
    expected_blocks = len(profiles) * len(seeds) * len(levels) * repeats
    expected_cells = expected_blocks * len(arms)
    expected_rows = expected_cells * len(PHASES)
    metadata_errors, exclusions = [], []

    def metadata_error(source, error, job=None):
        metadata_errors.append(dict(file=str(source), error=error))
        job = job or {}
        exclusions.append(dict(profile=job.get('profile', ''), seed=job.get('seed', ''),
                               concurrency=job.get('concurrency', ''), round_id=job.get('round', ''),
                               phase='metadata', baseline=job.get('arm', 'all'),
                               reason=error + ':' + str(source)))

    expected_keys = {(p, s, c, r, a) for p in profiles for s in seeds for c in levels
                     for r in range(1, repeats + 1) for a in arms}
    planned, serials = {}, set()
    for raw_job in plan.get('execution_order', []):
        try:
            job = normalize_job(raw_job)
            key = job_key(job)
            if key not in expected_keys or key in planned or job['serial'] in serials:
                raise ValueError('unexpected_or_duplicate_planned_job')
            if not 1 <= job['order_position'] <= len(arms):
                raise ValueError('invalid_planned_order_position')
            planned[key] = job
            serials.add(job['serial'])
        except (KeyError, TypeError, ValueError) as exc:
            metadata_error('plan.json', str(exc), raw_job)
    if set(planned) != expected_keys or serials != set(range(1, expected_cells + 1)):
        metadata_error('plan.json', 'execution_order_missing_or_invalid_jobs')
    if plan.get('expected_phase_rows', expected_rows) != expected_rows:
        metadata_error('plan.json', 'expected_phase_rows_disagrees_with_grid')

    rows, actual_jobs = {}, {}
    for path in sorted((root / 'rounds').glob('round*/cells/*/cell_statistics.json')):
        job = None
        try:
            order_path = path.parent / 'order.json'
            order = json.loads(order_path.read_text())
            job = normalize_job(order)
            key = job_key(job)
            if key not in planned or job != planned[key]:
                raise ValueError('actual_order_differs_from_planned_job')
            if key in actual_jobs:
                raise ValueError('duplicate_actual_job')
            if path.parents[2].name != f"round{job['round']:02d}":
                raise ValueError('round_directory_differs_from_job')
            if order.get('state') != 'COMPLETE':
                raise ValueError('cell_job_not_complete')
            records = json.loads(path.read_text())
            if not isinstance(records, list):
                raise ValueError('cell_statistics_must_be_list')
            staged = {}
            for row in records:
                actual = (row['profile'], int(row['seed']), int(row['concurrency']), job['round'], row['arm'])
                if actual != key or row['phase'] not in PHASES:
                    raise ValueError('cell_identity_differs_from_order')
                if 'round' in row and int(row['round']) != job['round']:
                    raise ValueError('phase_round_differs_from_order')
                rk = (*key, row['phase'])
                if rk in staged or rk in rows:
                    raise ValueError('duplicate_cell_phase')
                staged[rk] = row
            rows.update(staged)
            actual_jobs[key] = job
        except (KeyError, TypeError, ValueError, OSError) as exc:
            metadata_error(str(path.relative_to(root)), str(exc), job)

    # Position balance is checked from completed cell jobs, not only the plan.
    balanced = {}
    observed_blocks = set()
    for profile in profiles:
        for seed in seeds:
            for c in levels:
                case = (profile, seed, c)
                jobs = [j for k, j in actual_jobs.items() if k[:3] == case]
                for j in jobs:
                    observed_blocks.add((*case, j['round']))
                counts = Counter((j['arm'], j['order_position']) for j in jobs)
                rounds_ok = all(
                    sorted(j['order_position'] for j in jobs if j['round'] == r) == list(range(1, len(arms) + 1))
                    for r in range(1, repeats + 1))
                balanced[case] = (len(jobs) == repeats * len(arms) and rounds_ok and repeats % len(arms) == 0 and
                                  all(counts[(arm, p)] == repeats // len(arms)
                                      for arm in arms for p in range(1, len(arms) + 1)))
    pairs = []
    for profile in profiles:
        for seed in seeds:
            for c in levels:
                for r in range(1, repeats + 1):
                    for phase in PHASES:
                        cand = rows.get((profile, seed, c, r, CANDIDATE, phase))
                        for baseline in baselines:
                            other = rows.get((profile, seed, c, r, baseline, phase))
                            pair, excluded = compare(cand, other, (profile, seed, c, str(r), phase, baseline))
                            if pair:
                                pairs.append(pair)
                            else:
                                exclusions.append(excluded)
    try:
        execution_status = json.loads((root / 'status.json').read_text())
        execution_complete = execution_status.get('complete') is True and execution_status.get('status') == 'EXPERIMENT_COMPLETE'
    except (OSError, TypeError, ValueError):
        execution_complete = False
    if not execution_complete:
        metadata_error('status.json', 'runner_did_not_finish_experiment')
    complete = (execution_complete and len(actual_jobs) == expected_cells and len(rows) == expected_rows
                and not metadata_errors and not exclusions)
    all_balanced = bool(balanced) and all(balanced.values())
    expected_groups = {(profile, c, phase, base): len(seeds) * repeats
                       for profile in profiles for c in levels for phase in PHASES for base in baselines}
    eligible = set(expected_groups) if complete and all_balanced else set()
    summaries = summarize_pairs(pairs, expected_groups, eligible)
    measured_vllm = [r for r in summaries if r['baseline'] == 'vllm_least_active' and r['phase'] == 'measured']
    covered = [r for r in measured_vllm if r['eligible_for_case_claim'] and r['all_pairs_both_metrics_better']]
    result = dict(status='CONFIRMATION_COMPLETE' if complete else 'CONFIRMATION_INCOMPLETE',
                  complete=complete, execution_position_balanced=all_balanced,
                  expected_round_blocks=expected_blocks, found_round_blocks=len(observed_blocks),
                  expected_phase_cells=expected_rows, found_phase_cells=len(rows),
                  paired_trials=len(pairs), excluded_pairs_or_metadata=len(exclusions),
                  metadata_errors=metadata_errors,
                  measured_vllm_scenarios=len(measured_vllm), all_paired_trials_winning_scenarios=len(covered),
                  covered_scenarios=[dict(profile=r['profile'], concurrency=r['concurrency']) for r in covered],
                  measured_vllm_both_metrics_wins=sum(r['both_metrics_better_count'] for r in measured_vllm),
                  measured_vllm_pairs=sum(r['paired_trials'] for r in measured_vllm),
                  measured_vllm_tail_regressions=sum(r['tail_regression_count'] for r in measured_vllm),
                  claim_scope='Only listed frozen scenarios and paired repeats. No broad serving, independent-seed or answer-quality claim.',
                  tail_regression_rule='p95 TTFT ratio > 1.0; reported separately from throughput and mean TTFT wins',
                  statistical_unit='One full paired run; no independent-request confidence intervals',
                  learning_note='learn_plus_measured is a derived aggregation, not extra requests; calibration/startup are excluded')
    csv_write(root / 'confirmation_pairs.csv', pairs, PAIR_FIELDS)
    csv_write(root / 'confirmation_excluded.csv', exclusions, EXCLUDE_FIELDS)
    csv_write(root / 'confirmation_summary.csv', summaries)
    dump(root / 'confirmation_status.json', result)
    write_report(root, result, summaries)
    return result


def write_report(root, status, summaries):
    lines = ['# 有界哈希同轮复验', '',
             f"完成状态：{status['status']}；执行位置平衡：{status['execution_position_balanced']}。",
             f"对 vLLM 的正式阶段配对中，吞吐和平均首响应同时更好：{status['measured_vllm_both_metrics_wins']}/{status['measured_vllm_pairs']}；p95 回退：{status['measured_vllm_tail_regressions']}。",
             f"在全部重复中均获双指标优势且证据完整、顺序平衡的场景：{status['all_paired_trials_winning_scenarios']}/{status['measured_vllm_scenarios']}。",
             '', '比值均为候选 / 对照；吞吐 > 1 更好，平均首响应和 p95 < 1 更好。胜负来自同场景、种子、并发、轮次、阶段的配对。',
             '', '| 场景 | 并发 | 阶段 | 对照 | 有效/应有配对 | 双指标胜出 | 吞吐比几何均值 [最小,最大] | 平均首响应比几何均值 [最小,最大] | p95回退 |',
             '|---|---:|---|---|---:|---:|---|---|---:|']
    for r in summaries:
        if r['phase'] == 'learn':
            continue
        def interval(field):
            if not r['paired_trials']:
                return 'NA'
            return f"{r[field + '_geomean']:.4f} [{r[field + '_min']:.4f}, {r[field + '_max']:.4f}]"
        lines.append(f"| {r['profile']} | {r['concurrency']} | {r['phase']} | {r['baseline']} | {r['paired_trials']}/{r['expected_pairs']} | {r['both_metrics_better_count']} | {interval('throughput_ratio')} | {interval('mean_ttft_ratio')} | {r['tail_regression_count']} |")
    lines += ['', 'learn 为建立缓存阶段；measured 为正式测量；learn_plus_measured 为前两阶段合并，不能当作第三次独立实验。合并吞吐按输出 token 总量除以阶段耗时之和计算，未计入模型启动和离线校准。',
              '', '重复同一请求集只检验固定场景的运行稳定性；不同种子也可能来自开发阶段。没有基于逐请求样本计算虚假的独立样本置信区间。',
              '', '原生轮询对照同时包含缓存替换差异；同 LRU 的 least-active 和旧 mean 对照用于区分路由策略的增益。p95 回退独立保留，不会因平均指标领先而隐藏。',
              '', '每个排除项见 confirmation_excluded.csv；完整逐轮比值见 confirmation_pairs.csv。若存在缺失、无效或请求指纹不一致，不能声称完成版本优势确认。']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.run_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
