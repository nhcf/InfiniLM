"""Audit the frozen nine-cell V3/V2/vLLM experiment from saved evidence only."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import backend_runtime as runtime
import length_report as proven
import summarize_confirmation

V3 = 'nh_cache_bounded_v3'
V2 = 'nh_cache_bounded_v2'
VLLM = 'vllm_least_active'
ARMS = {'v3': V3, 'v2': V2, 'vllm': VLLM}
LABEL = {'v3': 'V3 候选', 'v2': '原 V2', 'vllm': 'vLLM'}
POLICIES = {V3: 'cache_bounded_v3', V2: 'cache_bounded_v2', VLLM: 'least_active'}
CASES = {f'p8192_o{o}': (8192, o) for o in (32, 64, 128)}
PHASES = ('measured', 'learn_plus_measured')
PHASE_LABEL = {'measured': '正式阶段', 'learn_plus_measured': '学习和正式阶段合计'}
COMPARISONS = {'v3_vs_v2': ('v3', 'v2'), 'v3_vs_vllm': ('v3', 'vllm'),
               'v2_vs_vllm': ('v2', 'vllm')}
SEED = 33962
read = proven.read
require = proven.require
matches = proven.matches
verify_phase = proven.verify_phase


def validate_run(root):
    plan = read(root/'plan.json')
    require(read(root/'status.json').get('complete') is True, 'experiment is incomplete')
    require(read(root/'postprocess_status.json').get('complete') is True, 'postprocessing is incomplete')
    require(plan.get('rounds') == 1 and plan.get('seeds') == [SEED], 'round/seed mismatch')
    require(plan.get('concurrency') == [16], 'expected C16 only')
    require(plan.get('learning_requests') == 64 and plan.get('measured_requests') == 128,
            'learning or measured request count differs')
    require(len(plan['arms']) == 3 and {summarize_confirmation.arm_id(a) for a in plan['arms']} == set(ARMS.values()),
            'wrong benchmark methods')
    require(set(plan['profiles']) == set(CASES), 'profile panel differs')
    for name, (p, o) in CASES.items():
        require(plan['profiles'][name] == dict(prompt_tokens=p, shared_tokens=7936, output_tokens=o,
                                             documents=8, hot_fraction=.25), 'profile parameters differ')
    require(plan.get('cells') == 9 and plan.get('expected_phase_rows') == 27
            and plan.get('fresh_pool_per_cell') is True, 'expected nine fresh-pool cells')
    jobs = plan['execution_order']
    require(len(jobs) == 9 and [j['serial'] for j in jobs] == list(range(1, 10)), 'execution serials differ')
    for n in range(0, 9, 3):
        block = jobs[n:n+3]
        require(len({j['profile'] for j in block}) == 1 and
                [j['order_position'] for j in block] == [1, 2, 3] and
                {j['arm'] for j in block} == set(ARMS.values()), 'incorrect per-case order')
    counts = Counter((j['arm'], j['order_position']) for j in jobs)
    require(all(counts[a, p] == 1 for a in ARMS.values() for p in (1, 2, 3)), 'panel positions unbalanced')
    # Keep the proven generic metadata validator, with the candidate explicitly
    # set to V3. Its confirmation_* and report.md are auxiliary dual-metric
    # evidence; optimization_report.md remains the throughput-first report.
    original_candidate = summarize_confirmation.CANDIDATE
    try:
        summarize_confirmation.CANDIDATE = V3
        checked = summarize_confirmation.analyze(root)
    finally:
        summarize_confirmation.CANDIDATE = original_candidate
    require(checked['complete'] and checked['found_phase_cells'] == 27
            and checked['excluded_pairs_or_metadata'] == 0, 'incomplete or invalid matched evidence')
    return plan


def read_workloads(root):
    manifest = read(root/'workloads/manifest.json')
    cases = {}
    signatures = {}
    for case in manifest['cells']:
        profile = case['profile']
        require(profile in CASES and profile not in cases and case['seed'] == SEED, 'invalid manifest case')
        p, o = CASES[profile]
        require(all(case.get(k) == v for k, v in dict(prompt_tokens=p, shared_tokens=7936,
                    output_tokens=o, documents=8, hot_fraction=.25).items()), 'manifest profile mismatch')
        relative = Path(case['folder'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe workload path')
        cohorts = {}
        for phase, count in [('learn', 64), ('measured', 128)]:
            path = root/'workloads'/relative/(phase+'.jsonl')
            other = root/'rounds/round01/workloads'/relative/(phase+'.jsonl')
            declared = case['files'][phase]
            require(runtime.sha(path) == runtime.sha(other) == declared['sha256'], 'workload hash mismatch')
            items = runtime.rows(path)
            require(len(items) == len({x['id'] for x in items}) == count == declared['count'], 'invalid workload count')
            signature = []
            for item in items:
                require((item['prompt_tokens'], item['shared_tokens'], item['output_tokens'], item['body']['max_tokens'])
                        == (p, 7936, o, o), 'workload token contract differs')
                signature.append((item['id'], item['key'], item['prompt_sha256'], item['prefix_digest'],
                                  item['first_block_digest'], proven.normalized_body_digest(item['body'])))
            require(len({i['key'] for i in items}) == 8, 'expected eight documents')
            require(signatures.setdefault(phase, signature) == signature, 'output length changed inputs or arrival order')
            cohorts[phase] = items
        require({x['id'] for x in cohorts['learn']}.isdisjoint(x['id'] for x in cohorts['measured']),
                'learning and measured IDs overlap')
        cases[profile] = cohorts
    require(set(cases) == set(CASES), 'missing workload case')
    return cases


def audit_routes(folder, phase, arm, records, shared=7936):
    snapshot = read(folder/(phase+'_routes.json'))
    require(snapshot.get('policy') == POLICIES[arm], 'router policy identity mismatch: '+arm)
    require(snapshot.get('active') == [0]*8 and snapshot.get('kv_reserved_tokens') == [0]*8,
            'router did not drain')
    require(snapshot.get('queue_timeouts') == 0 and snapshot.get('queue_cancellations') == 0,
            'router timed out or cancelled requests')
    events = snapshot['events']
    expected = {r['id']: r for r in records}
    routes = [e for e in events if e.get('request_id') in expected]
    require(len(routes) == len(expected) and len({e['request_id'] for e in routes}) == len(expected),
            'missing or duplicate request routes')
    result = dict(requests=len(records), zero_cache_requests=0, partial_cache_requests=0,
                  full_cache_requests=0, unknown_cache_requests=0, slow_requests_ge4s=0,
                  slow_zero_cache_requests=0, slow_full_cache_requests=0,
                  admissions_with_pending_prefill=0, slow_full_cache_with_pending=0,
                  gateway_wait_total_ms=0.0, warm_wait_requests=0,
                  warm_wait_gateway_total_ms=0.0, warm_wait_zero_cache_requests=0,
                  warm_wait_partial_cache_requests=0, warm_wait_full_cache_requests=0,
                  warm_wait_unknown_cache_requests=0, warm_wait_slow_requests_ge4s=0,
                  warm_wait_first_fresh_snapshot_candidates=0,
                  warm_wait_first_active_prediction_candidates=0)
    for route in routes:
        r = expected[route['request_id']]
        require(route.get('success') is True and route.get('released') is True, 'unsuccessful or unreleased route')
        require(route.get('upstream_http_attempts') == 1 and route.get('upstream_force_close') is True,
                'HTTP attempts or fresh-connection contract differs')
        for field in ('event_payload_sha256', 'event_payload_count', 'event_payload_bytes'):
            require(field in r and route.get(field) == r[field], 'stream payload evidence differs')
        generated = route.get('generated_tokens')
        # InfiniLM emits engine telemetry; vLLM's absent engine counter is 0/None.
        # The exact client completion count is independently checked for every arm.
        require((arm == VLLM and generated in (None, 0)) or generated == r['usage']['completion_tokens'],
                'route output usage mismatch')
        chosen = route['chosen']
        require(type(chosen) is int and 0 <= chosen < 8, 'invalid chosen worker')
        active, reserved = route['active_before'], route['kv_reserved_before']
        full_reservation = math.ceil((r['usage']['prompt_tokens'] + r['usage']['completion_tokens'])/256)*256
        require(len(active) == len(reserved) == 8 and 0 <= active[chosen] < 3
                and route['kv_reserved_tokens'] == full_reservation
                and 0 <= reserved[chosen] + full_reservation <= 32768,
                'admission count or full KV reservation violated')
        observation = route['admission_observation']
        require(observation.get('status') == 'OBSERVED' and observation.get('chosen_worker') == chosen,
                'admission observation identity mismatch')
        workers = observation['workers']
        require(len(workers) == 8 and {w['worker'] for w in workers} == set(range(8)), 'incomplete worker observations')
        selected = next(w for w in workers if w['worker'] == chosen)
        pending = selected['pending_request_count']
        require(type(pending) is int and pending >= 0, 'invalid pending count')
        result['admissions_with_pending_prefill'] += pending > 0
        wait = float(route['gateway_wait_ms'])
        require(math.isfinite(wait) and wait >= 0, 'invalid gateway wait')
        result['gateway_wait_total_ms'] += wait
        cached = (r['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
        known = type(cached) is int and 0 <= cached <= r['usage']['prompt_tokens']
        require(cached is None or known, 'invalid actual cached-token count')
        require(arm == VLLM or known, 'NH actual cache usage unavailable')
        if route.get('observed_cached_tokens') is not None:
            require(route['observed_cached_tokens'] == cached, 'route actual cache usage differs')
        group = 'unknown' if not known else ('zero' if cached == 0 else ('full' if cached >= shared else 'partial'))
        result[group+'_cache_requests'] += 1
        deferred = route.get('v3_warm_wait')
        if arm != V3:
            require(deferred is None and route.get('optimization_version') is None,
                    'candidate wait evidence found on a baseline')
        else:
            require(route.get('optimization_version') == 'cache_bounded_v3'
                    and isinstance(deferred, dict) and deferred.get('policy') == 'cache_bounded_v3',
                    'V3 wait audit evidence missing')
            checks, held = deferred['checks'], deferred['deferred_checks']
            require(type(checks) is int and type(held) is int and 0 <= held <= checks and checks > 0,
                    'invalid V3 wait counters')
            require(deferred.get('waited_for_warm_copy') is (held > 0)
                    and deferred.get('final_worker') == chosen
                    and deferred.get('prediction_is_not_verified_hit') is True, 'V3 wait evidence inconsistent')
            matches(deferred['actual_gateway_wait_ms'], wait, 'V3 actual gateway wait')
            if held:
                require(route.get('owner_wait_requested') is True, 'V3 deferred marker inconsistent')
                candidates = deferred['first_candidates']
                require(isinstance(candidates, list) and candidates, 'V3 initial wait candidates missing')
                suitable = [x for x in candidates if x.get('worth_waiting') is True]
                require(bool(suitable), 'V3 deferred without suitable evidence')
                for candidate in suitable:
                    kind = candidate['evidence']
                    require(candidate.get('pending_requests') == 0 and candidate.get('fits_after_first_release') is True,
                            'V3 wait candidate violates eligibility')
                    require(kind in ('fresh_cache_snapshot', 'active_matching_decode_prediction'),
                            'unknown V3 cache evidence type')
                    if kind == 'fresh_cache_snapshot':
                        require(candidate['observed_cached_tokens'] >= shared, 'unverified fresh cache wait claim')
                        result['warm_wait_first_fresh_snapshot_candidates'] += 1
                    else:
                        require(candidate['observed_cached_tokens'] < shared, 'prediction mislabeled as observed')
                        result['warm_wait_first_active_prediction_candidates'] += 1
                result['warm_wait_requests'] += 1
                result['warm_wait_gateway_total_ms'] += wait
                result['warm_wait_'+group+'_cache_requests'] += 1
                result['warm_wait_slow_requests_ge4s'] += float(r['ttft_ms']) >= 4000
        slow = float(r['ttft_ms']) >= 4000
        result['slow_requests_ge4s'] += slow
        if slow and group == 'zero': result['slow_zero_cache_requests'] += 1
        if slow and group == 'full':
            result['slow_full_cache_requests'] += 1
            result['slow_full_cache_with_pending'] += pending > 0
    return result


def collect_metrics(root, cohorts):
    metrics = {}
    diagnostics = {}
    folders = list(root.glob('rounds/round01/cells/*/cell_statistics.json'))
    require(len(folders) == 9, 'expected exactly nine cells')
    for path in folders:
        folder = path.parent
        job = read(folder/'order.json')
        arm, profile = job['arm'], job['profile']
        require(arm in ARMS.values() and profile in CASES, 'unexpected cell identity')
        require(read(folder/'gateway_errors.json') == [], 'gateway errors present')
        receipt = read(folder/'router_source_selection.json')
        module = {V3: 'cache_router_v3', V2: 'cache_router', VLLM: 'mean_router'}[arm]
        source_file = module+'.py'
        require(receipt.get('arm') == arm and receipt.get('router_class') == module+'.Router'
                and receipt.get('source_file') == source_file
                and receipt.get('restored_after_cell') is True, 'router source selection receipt differs')
        source_hash = runtime.sha(root/'source'/source_file)
        require(receipt.get('source_sha256') == source_hash
                == read(root/'fingerprint.json')['candidate_sources'].get(source_file),
                'router source fingerprint mismatch')
        expected_policy = 'least_active' if arm == VLLM else 'cache_bounded_v2'
        require(receipt.get('engine') == ('vllm' if arm == VLLM else 'nh')
                and receipt.get('backend_policy') == expected_policy, 'backend policy receipt differs')
        rows = read(path)
        stats = {r['phase']: r for r in rows}
        require(len(rows) == 3 and set(stats) == {'learn', *PHASES}, 'phase count mismatch')
        raw = {p: read(folder/(p+'_requests.json')) for p in ('learn', 'measured')}
        evidence = {}
        for phase in ('learn', 'measured'):
            verify_phase(stats[phase], cohorts[profile][phase], raw[phase])
            evidence[phase] = audit_routes(folder, phase, arm, raw[phase])
        matches(stats['learn_plus_measured']['wall_s'], stats['learn']['wall_s']+stats['measured']['wall_s'], 'combined wall time')
        verify_phase(stats['learn_plus_measured'], cohorts[profile]['learn']+cohorts[profile]['measured'],
                     raw['learn']+raw['measured'])
        evidence['learn_plus_measured'] = {k: evidence['learn'][k] + evidence['measured'][k]
                                           for k in evidence['learn']}
        for phase in PHASES:
            key = profile, phase, arm
            require(key not in metrics, 'duplicate metrics')
            metrics[key] = stats[phase]
            diagnostics[key] = evidence[phase]
    require(len(metrics) == 18, 'missing metrics')
    return metrics, diagnostics


def score(profile, phase, metrics, diagnostics):
    p, o = CASES[profile]
    row = dict(profile=profile, phase=phase, concurrency=16, seed=SEED,
               prompt_tokens=p, shared_tokens=7936, output_tokens_per_request=o)
    for key, arm in ARMS.items():
        stat = metrics[profile, phase, arm]
        for field in ('output_tok_s', 'ttft_mean_ms', 'ttft_p95_ms'):
            row[key+'_'+field] = float(stat[field])
        diagnostic = diagnostics[profile, phase, arm]
        row.update({key+'_'+field: value for field, value in diagnostic.items() if field != 'gateway_wait_total_ms'})
        row[key+'_gateway_wait_mean_ms'] = diagnostic['gateway_wait_total_ms']/diagnostic['requests']
        row[key+'_warm_wait_gateway_mean_ms'] = (diagnostic['warm_wait_gateway_total_ms']/diagnostic['warm_wait_requests']
                                                if diagnostic['warm_wait_requests'] else None)
    rates = {k: row[k+'_output_tok_s'] for k in ARMS}
    row['first_by_throughput'] = '|'.join(k for k in ARMS if rates[k] == max(rates.values()))
    for name, (candidate, baseline) in COMPARISONS.items():
        t = rates[candidate]/rates[baseline]
        m = row[candidate+'_ttft_mean_ms']/row[baseline+'_ttft_mean_ms']
        tail = row[candidate+'_ttft_p95_ms']/row[baseline+'_ttft_p95_ms']
        row.update({name+'_throughput_pct': 100*(t-1), name+'_ttft_pct': 100*(m-1),
                    name+'_p95_pct': 100*(tail-1), name+'_higher': t > 1,
                    name+'_at_least_2pct': t >= 1.02 or math.isclose(t, 1.02, abs_tol=1e-12, rel_tol=0),
                    name+'_both_metrics_better': t > 1 and m < 1,
                    name+'_tail_regression': tail > 1})
    return row


def summarize(rows):
    result = {}
    for phase in PHASES:
        group = [r for r in rows if r['phase'] == phase]
        result[phase] = dict(cases=len(group), first_by_throughput=dict(Counter(r['first_by_throughput'] for r in group)),
            comparisons={key: {field: sum(r[key+'_'+field] for r in group)
                              for field in ('higher', 'at_least_2pct', 'both_metrics_better', 'tail_regression')}
                         for key in COMPARISONS})
    return result


def write_outputs(root, result, rows):
    with (root/'optimization_cases.csv').open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]) if rows else ['profile', 'phase'])
        writer.writeheader(); writer.writerows(rows)
    lines = ['# 长输入缓存与同卡等待优化对比', '',
             '固定 8 文档、输入 8192、共享前缀 7936、hot25、整体并发 16、种子 33962、单轮。',
             '输出长度为 32、64、128。每方法每场景先学习 64 条，再正式测量 128 条。']
    if not result['complete']:
        lines.extend(['', '证据不完整，本报告不生成胜负结论。', '', result['error']])
    else:
        for phase in PHASES:
            group = [r for r in rows if r['phase'] == phase]
            lines.extend(['', '## '+PHASE_LABEL[phase], '',
                '| 输出长度 | V3 tok/s | 原 V2 tok/s | vLLM tok/s | V3 对原 V2 | V3 对 vLLM |',
                '|---|---:|---:|---:|---:|---:|'])
            for r in group:
                lines.append(f"| {r['output_tokens_per_request']} | {r['v3_output_tok_s']:.3f} | {r['v2_output_tok_s']:.3f} | {r['vllm_output_tok_s']:.3f} | {r['v3_vs_v2_throughput_pct']:+.3f}% | {r['v3_vs_vllm_throughput_pct']:+.3f}% |")
            lines.extend(['', '| 比较 | 吞吐领先 | 吞吐至少领先 2% | 吞吐与平均首响应同时改善 | p95 回退 |',
                          '|---|---:|---:|---:|---:|'])
            for name, (a, b) in COMPARISONS.items():
                s = result['phase_summary'][phase]['comparisons'][name]
                lines.append('| '+LABEL[a]+' 对 '+LABEL[b]+' | '+' | '.join(f'{s[k]} / 3' for k in
                    ('higher','at_least_2pct','both_metrics_better','tail_regression'))+' |')
            lines.extend(['', '| 输出长度 | 方法 | 平均首响应 ms | p95 ms | 零命中 / 缓存未知 | ≥4 秒请求 | 完整命中仍慢 / 其中同卡有待预填充请求 |',
                          '|---|---|---:|---:|---:|---:|---:|'])
            for r in group:
                for a in ARMS:
                    lines.append(f"| {r['output_tokens_per_request']} | {LABEL[a]} | {r[a+'_ttft_mean_ms']:.2f} | {r[a+'_ttft_p95_ms']:.2f} | {r[a+'_zero_cache_requests']} / {r[a+'_unknown_cache_requests']} | {r[a+'_slow_requests_ge4s']} | {r[a+'_slow_full_cache_requests']} / {r[a+'_slow_full_cache_with_pending']} |")
            lines.extend(['', '| 输出长度 | V3 实际等待请求 | 等待请求平均网关耗时 ms | 等待后完整 / 零 / 部分 / 未知缓存命中 | 等待后 ≥4 秒 | 首次合格候选数：快照 / 活跃副本预测 |',
                          '|---|---:|---:|---:|---:|---:|'])
            for r in group:
                wait_mean = r['v3_warm_wait_gateway_mean_ms']
                displayed = '—' if wait_mean is None else f'{wait_mean:.3f}'
                cache_counts = ' / '.join(str(r['v3_warm_wait_'+k+'_cache_requests']) for k in ('full','zero','partial','unknown'))
                lines.append(f"| {r['output_tokens_per_request']} | {r['v3_warm_wait_requests']} | {displayed} | {cache_counts} | {r['v3_warm_wait_slow_requests_ge4s']} | {r['v3_warm_wait_first_fresh_snapshot_candidates']} / {r['v3_warm_wait_first_active_prediction_candidates']} |")
        lines.extend(['', '吞吐使用实际输出 token 总数除以阶段总用时，按同一长度分别比较。学习合计按两个阶段耗时之和计算，排除模型启动、探针和校准。',
                      '2% 是事先规定的实用差异门槛，不代表统计显著性。一轮一种子结果只描述本次三组高复用场景。',
                      '三种方法在整个面板的三个执行位置各出现一次。同一场景仅执行一次，未做场景内的重复换位。',
                      '缓存未知单列，不能当作零命中。完整命中慢请求上的待预填充数量是入卡时观测，不是 GPU 阶段计时或因果证明。',
                      'V3 为待验证候选。原 V2 和 vLLM 保留同轮对照；任何退步、慢请求和失败都保留。',
                      '等待统计来自 V3 专属决策记录，并核对实际路由等待时间。首次候选数按候选副本计数，同一请求可有多个候选。预测命中与实际命中分别报告。',
                      '本文件是吞吐主报告。confirmation_* 与 report.md 为以 V3 为候选的辅助双指标配对审计。'])
    (root/'optimization_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    runtime.dump(root/'optimization_report_status.json', result)


def analyze(root):
    root = Path(root).resolve()
    try:
        validate_run(root)
        metrics, diagnostics = collect_metrics(root, read_workloads(root))
        rows = [score(profile, phase, metrics, diagnostics) for phase in PHASES for profile in CASES]
        result = dict(status='OPTIMIZATION_REPORT_COMPLETE', complete=True, concurrency=16, seed=SEED,
                      actual_cells=9, learning_requests=576, measured_requests=1152,
                      total_learning_plus_measured=1728, phase_summary=summarize(rows),
                      evidence_level='single_seed_single_round_favorable_screen',
                      primary='measured output tokens / wall seconds')
    except (KeyError, OSError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        result = dict(status='OPTIMIZATION_REPORT_INCOMPLETE', complete=False, error=str(exc))
        rows = []
    write_outputs(root, result, rows)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    status = analyze(parser.parse_args().run_root)
    print(json.dumps(status, ensure_ascii=False))
    raise SystemExit(0 if status['complete'] else 2)
