#!/usr/bin/env python3
"""Select a phase candidate from one three-way measurement in each of 12 scenarios."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import summarize_confirmation as confirmation
from cache_workload import PROFILES
from selection_protocol import validate_spec

MEAN = 'nh_mean_bounded'
V2 = 'nh_cache_bounded_v2'
VLLM = 'vllm_least_active'
CANDIDATES = (MEAN, V2)
PHASES = ('measured', 'learn_plus_measured')
FIXED_PROFILES = ('reuse8', 'reuse16', 'longdecode', 'cold')
FIXED_LEVELS = (16, 32, 64)
FIXED_SEEDS = (33962,)
ROUNDS = 1
PAIR_COUNT = 1
EVIDENCE_LEVEL = 'SINGLE_SEED_SINGLE_ROUND_SCREEN'
CASE_QUORUM = 7
TPS_MARGIN = 1.02
TTFT_MARGIN = .98


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_write(path, rows, fallback):
    fields = list(dict.fromkeys(k for r in rows for k in r)) or fallback
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(',', ':'))
                             if isinstance(v, (dict, list)) else v for k, v in row.items()})


def compare_case(candidate, baseline, identity):
    """Compare the two phase summaries obtained for the same seed and scenario."""
    for name, row in (('candidate', candidate), ('baseline', baseline)):
        error = confirmation.validate(row)
        if error:
            raise ValueError(name + ':' + error)
    if (candidate['ordered_request_contract_sha256'] != baseline['ordered_request_contract_sha256']
            or candidate['requests'] != baseline['requests']):
        raise ValueError('mismatched paired request contract/count')

    tps_ratio = candidate['output_tok_s'] / baseline['output_tok_s']
    ttft_ratio = candidate['ttft_mean_ms'] / baseline['ttft_mean_ms']
    p95_ratio = candidate['ttft_p95_ms'] / baseline['ttft_p95_ms']
    fits = {
        'throughput_margin': tps_ratio >= TPS_MARGIN or math.isclose(tps_ratio, TPS_MARGIN, rel_tol=0, abs_tol=1e-12),
        'mean_ttft_margin': ttft_ratio <= TTFT_MARGIN or math.isclose(ttft_ratio, TTFT_MARGIN, rel_tol=0, abs_tol=1e-12),
    }
    pair = dict(identity, seed=FIXED_SEEDS[0], round=1,
                request_contract_sha256=candidate['ordered_request_contract_sha256'],
                candidate_tok_s=candidate['output_tok_s'], baseline_tok_s=baseline['output_tok_s'],
                candidate_mean_ttft_ms=candidate['ttft_mean_ms'], baseline_mean_ttft_ms=baseline['ttft_mean_ms'],
                candidate_p95_ttft_ms=candidate['ttft_p95_ms'], baseline_p95_ttft_ms=baseline['ttft_p95_ms'],
                requests=candidate['requests'], throughput_ratio=tps_ratio, mean_ttft_ratio=ttft_ratio,
                p95_ttft_ratio=p95_ratio, both_metrics_better=tps_ratio > 1 and ttft_ratio < 1,
                margin_win=all(fits.values()), tail_regression=p95_ratio > 1,
                unmet_conditions=[name for name, passed in fits.items() if not passed])
    return pair


def validate_single_round_order(root, plan, spec):
    """Audit saved serials as well as position labels across the 12 actual job triples."""
    errors = []
    arms = (MEAN, V2, VLLM)
    counts = {arm: {str(position): 0 for position in range(1, 4)} for arm in arms}
    profile_counts = {profile: {arm: [] for arm in arms} for profile in FIXED_PROFILES}
    rotation_complete = False
    try:
        planned = [confirmation.normalize_job(job) for job in plan['execution_order']]
        if [job['serial'] for job in planned] != list(range(1, 37)):
            raise ValueError('expected_36_sequential_planned_jobs')
        observed = {}
        for path in Path(root).glob('rounds/round*/cells/*/order.json'):
            raw = read(path)
            job = confirmation.normalize_job(raw)
            if raw.get('state') != 'COMPLETE' or job['serial'] in observed:
                raise ValueError('incomplete_or_duplicate_observed_job')
            observed[job['serial']] = job
        if len(observed) != 36 or any(observed.get(job['serial']) != job for job in planned):
            raise ValueError('observed_jobs_differ_from_predeclared_execution_order')

        for index, scenario in enumerate(spec['scenarios']):
            profile, concurrency = scenario['profile'], scenario['concurrency']
            trio = [observed[3 * index + position] for position in (1, 2, 3)]
            if {j['arm'] for j in trio} != set(arms):
                raise ValueError('scenario_missing_or_duplicate_arm')
            for position, job in enumerate(trio, 1):
                if (job['round'], job['seed'], job['profile'], job['concurrency'], job['order_position']) != (
                        1, FIXED_SEEDS[0], profile, concurrency, position):
                    raise ValueError('observed_order_position_or_scenario_differs_from_execution_serial')
                arm = job['arm']
                counts[arm][str(position)] += 1
                profile_counts[profile][arm].append(position)
        rotation_complete = all(sorted(positions) == [1, 2, 3]
                                for by_arm in profile_counts.values() for positions in by_arm.values())
        if not rotation_complete:
            errors.append('profile_does_not_rotate_each_arm_across_concurrency')
        if any(n != 4 for by_position in counts.values() for n in by_position.values()):
            errors.append('cross_case_arm_position_counts_not_four_each')
    except (KeyError, TypeError, ValueError, OSError) as exc:
        errors.append('single_round_order_error:' + str(exc))
    return dict(complete=not errors, errors=errors, per_arm_position_counts=counts,
                per_profile_rotation_complete=rotation_complete)


def validate_panel(plan, spec):
    """Do not reinterpret a smaller or edited panel after seeing its results."""
    expected = dict(profiles=list(FIXED_PROFILES), concurrency=list(FIXED_LEVELS),
                    seeds=list(FIXED_SEEDS), rounds=ROUNDS,
                    arms=[MEAN, V2, VLLM], learning_requests=64, measured_requests=128)
    errors = []
    try:
        validate_spec(spec)
    except (TypeError, ValueError) as exc:
        errors.append('invalid_frozen_spec:' + str(exc))
    if plan.get('protocol') != spec.get('execution_protocol'):
        errors.append('execution_protocol_differs_from_spec')
    if 'config' in spec and plan.get('config') != spec['config']:
        errors.append('runtime_config_differs_from_frozen_spec')
    for label, value in (('spec', spec), ('run plan', plan)):
        for key, want in expected.items():
            actual = value.get(key)
            if key == 'profiles' and isinstance(actual, dict):
                actual = list(actual)
            elif key == 'arms' and isinstance(actual, list):
                actual = [confirmation.arm_id(a) for a in actual]
            if isinstance(want, list):
                if not isinstance(actual, list) or len(actual) != len(set(actual)) or set(actual) != set(want):
                    errors.append(label + ':' + key + '_differs_from_frozen_panel')
            elif actual != want:
                errors.append(label + ':' + key + '_differs_from_frozen_panel')
    if isinstance(plan.get('profiles'), dict):
        for name in FIXED_PROFILES:
            if plan['profiles'].get(name) != PROFILES[name]:
                errors.append('run plan:profile_parameters_differ:' + name)
    else:
        errors.append('run plan:missing_profile_parameters')
    return errors


def validate_fingerprint(root, plan, spec):
    errors = []
    try:
        fp = read(root / 'fingerprint.json')
        if 'expected_fingerprint' in spec and fp != spec['expected_fingerprint']:
            errors.append('runtime_fingerprint_differs_from_predeclared_spec')
        if not fp.get('config') or fp['config'] != plan.get('config'):
            errors.append('runtime_config_differs_from_plan')
        hw = fp.get('hardware', {})
        if hw.get('available') is not True or len(hw.get('devices', [])) != 8:
            errors.append('missing_eight_gpu_hardware_identity')
        sources = fp.get('candidate_sources')
        if not isinstance(sources, dict) or not sources:
            errors.append('missing_source_fingerprint')
        else:
            for name, digest in sources.items():
                rel = Path(name)
                if rel.is_absolute() or '..' in rel.parts:
                    errors.append('unsafe_source_path')
                    continue
                path = root / 'source' / rel
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    errors.append('source_snapshot_hash_mismatch:' + name)
            expected_sources = spec.get('source_sha256')
            if expected_sources is not None and expected_sources != sources:
                errors.append('source_fingerprint_differs_from_predeclared_spec')
    except (OSError, TypeError, ValueError) as exc:
        errors.append('missing_or_invalid_fingerprint:' + str(exc))
    return errors


def decision_from_cases(cases, complete, errors):
    coverage = {phase: {} for phase in PHASES}
    for phase in PHASES:
        for arm in CANDIDATES:
            own = [r for r in cases if r['phase'] == phase and r['candidate'] == arm]
            coverage[phase][arm] = dict(
                vllm_wins=sum(r['vllm_margin_win'] for r in own),
                head_to_head_wins=sum(r['other_margin_win'] for r in own),
                joint_wins=sum(r['joint_win'] for r in own),
                joint_winning_cases=[r['scenario'] for r in own if r['joint_win']],
                not_joint_winning_cases=[r['scenario'] for r in own if not r['joint_win']],
                vllm_tail_regression_trials=sum(r['vllm_tail_regression'] for r in own),
                vllm_tail_regression_cases=sum(r['vllm_tail_regression'] > 0 for r in own))
    winners = {}
    for phase in PHASES:
        eligible = [arm for arm in CANDIDATES if coverage[phase][arm]['joint_wins'] >= CASE_QUORUM] if complete else []
        winners[phase] = eligible[0] if len(eligible) == 1 else None
    selected, scope = None, None
    if not complete:
        status = 'SELECTION_INCOMPLETE'
    elif winners['measured'] is None:
        status = 'NO_UNIQUE_MAJORITY_WINNER'
    elif winners['learn_plus_measured'] not in (None, winners['measured']):
        status = 'PHASE_CONFLICT'
    else:
        selected = winners['measured']
        scope = ('MEASURED_AND_LEARNING_INCLUSIVE' if winners['learn_plus_measured'] == selected else 'WARM_ONLY')
        status = 'SELECTED'
    unmet = list(errors)
    if complete and selected is None:
        unmet.append('fewer_than_7_joint_winning_cases_in_primary_phase' if winners['measured'] is None else 'primary_and_learning_inclusive_winners_conflict')
    return dict(status=status, complete=complete, selected_arm=selected, scope=scope,
                performance_qualified=selected is not None, scenario_count=12,
                majority_required=CASE_QUORUM, paired_trials_per_case=PAIR_COUNT,
                evidence_level=EVIDENCE_LEVEL,
                performance_qualification_basis='single-pass observed margins',
                practical_margins=dict(throughput_ratio_at_least=TPS_MARGIN, mean_ttft_ratio_at_most=TTFT_MARGIN,
                                       statistical_significance_claim=False),
                per_candidate_coverage=coverage, phase_winners=winners,
                unmet_conditions=unmet,
                interpretation='Each profile/concurrency has one vote. A joint win requires beating both vLLM and the other version under the same fixed rule.',
                scope_limit='Only the declared hardware/runtime configuration and 12 synthetic scenarios; no untested-hardware or general-serving claim.',
                tail_rule='Every paired p95 ratio >1 is reported separately; p95 is not a hidden winner veto.',
                request_quality_claim=False)


def write_report(root, decision, cases):
    names = {MEAN: '旧版 mean', V2: 'V2', VLLM: 'vLLM'}
    lines = ['# 固定 12 场景单轮版本筛查', '',
             f"状态为 {decision['status']}。证据完整为 {decision['complete']}。",
             f"筛查选定版本为 {names.get(decision['selected_arm'], '未选定')}，适用范围为 {decision['scope'] or '无'}。", '',
             '本轮使用种子 33962，每个场景的三种方法各测一次。每次测量含 64 条学习和 128 条正式请求。执行顺序在同一负载的三个并发间轮换，完整面板中每个方法在第一、第二、第三位各出现四次。',
             '每个场景等权一票。候选相对另一版本及 vLLM 均达到吞吐高至少 2%、平均首响应低至少 2% 时得票，正式阶段至少 7/12 票可选定冻结候选。这里的得票反映这个固定种子的一次测量。', '',
             '| 阶段 | 版本 | 胜过 vLLM 的场景 | 胜过另一版本的场景 | 同时胜过两者 | 对 vLLM 的 p95 回退场景 |',
             '|---|---|---:|---:|---:|---:|']
    for phase in PHASES:
        for arm in CANDIDATES:
            c = decision['per_candidate_coverage'][phase][arm]
            lines.append(f"| {phase} | {names[arm]} | {c['vllm_wins']}/12 | {c['head_to_head_wins']}/12 | {c['joint_wins']}/12 | {c['vllm_tail_regression_trials']}/12 |")
    lines += ['', 'WARM_ONLY 表示该候选在建立缓存后的正式阶段达到多数场景门槛。学习与正式合计按两段实际请求和耗时计算。若两个阶段有相反胜者，或正式阶段没有候选达到门槛，则保留双方代码与本次证据。', '',
              '| 场景 | 阶段 | 版本 | 对 vLLM 吞吐比 | 对 vLLM 平均首响应比 | 对另一版本吞吐比 | 对另一版本平均首响应比 | 获得场景票 |',
              '|---|---|---|---:|---:|---:|---:|---|']
    for r in cases:
        lines.append(f"| {r['scenario']} | {r['phase']} | {names[r['candidate']]} | {r['vllm_throughput_ratio']:.4f} | {r['vllm_mean_ttft_ratio']:.4f} | {r['other_throughput_ratio']:.4f} | {r['other_mean_ttft_ratio']:.4f} | {r['joint_win']} |")
    if decision['unmet_conditions']:
        lines += ['', '未满足的条件'] + ['- ' + str(x) for x in decision['unmet_conditions']]
    lines += ['', '每个比值均为该次候选测量值除以同场景对照测量值。吞吐越高、平均首响应越低越好。p95 回退和未获票场景另行保留。冻结范围限定为本次硬件、配置、种子与场景。']
    (root / 'selection_report.md').write_text('\n'.join(lines) + '\n')


def analyze(run_root, spec):
    root = Path(run_root).resolve()
    errors, pairs, cases = [], [], []
    cross_case_order = {'complete': False, 'errors': ['evidence_unavailable']}
    try:
        plan = read(root / 'plan.json')
        errors += validate_panel(plan, spec)
        errors += validate_fingerprint(root, plan, spec)
        if not errors:
            checked = confirmation.analyze(root)
            if not checked['complete']:
                errors.append('confirmation_evidence_incomplete')
            cross_case_order = validate_single_round_order(root, plan, spec)
            errors.extend(cross_case_order['errors'])
        if not errors:
            rows = {}
            contracts = {}
            for path in sorted((root / 'rounds').glob('round*/cells/*/cell_statistics.json')):
                order = read(path.parent / 'order.json')
                cell_rows = read(path)
                by_phase = {row['phase']: row for row in cell_rows}
                learn, measured, combined = (by_phase[p] for p in ('learn', 'measured', 'learn_plus_measured'))
                if not math.isclose(float(combined['wall_s']), float(learn['wall_s']) + float(measured['wall_s']), rel_tol=1e-9):
                    raise ValueError('combined_elapsed_is_not_sum_of_phases')
                expected_mean = (float(learn['ttft_mean_ms']) * 64 + float(measured['ttft_mean_ms']) * 128) / 192
                if not math.isclose(float(combined['ttft_mean_ms']), expected_mean, rel_tol=1e-9):
                    raise ValueError('combined_mean_is_not_request_weighted')
                for row in cell_rows:
                    parameters = PROFILES[row['profile']]
                    expected_requests = {'learn': 64, 'measured': 128, 'learn_plus_measured': 192}[row['phase']]
                    if int(row['requests']) != expected_requests:
                        raise ValueError('actual_phase_request_count_differs_from_spec')
                    expected_total = expected_requests * parameters['output_tokens']
                    total = row['total_output_tokens'] if 'total_output_tokens' in row else row['output_tokens']
                    if total != expected_total or row.get('output_tokens_per_request') != parameters['output_tokens']:
                        raise ValueError('actual_output_token_contract_differs_from_profile')
                    if row.get('prompt_tokens') != parameters['prompt_tokens'] or row.get('shared_tokens') != parameters['shared_tokens']:
                        raise ValueError('actual_input_token_contract_differs_from_profile')
                    contract_key = (row['profile'], int(row['seed']), row['phase'])
                    digest = row['ordered_request_contract_sha256']
                    if contract_key in contracts and contracts[contract_key] != digest:
                        raise ValueError('frozen_request_contract_changed_between_rounds_or_concurrency')
                    contracts[contract_key] = digest
                    if row['phase'] in PHASES:
                        rows[(row['profile'], int(row['concurrency']), row['phase'], row['arm'], int(row['seed']), int(order['round']))] = row
            for profile in FIXED_PROFILES:
                for c in FIXED_LEVELS:
                    for phase in PHASES:
                        for arm in CANDIDATES:
                            other = V2 if arm == MEAN else MEAN
                            outputs = {}
                            for key, baseline in (('vllm', VLLM), ('other', other)):
                                a = rows[(profile, c, phase, arm, FIXED_SEEDS[0], 1)]
                                b = rows[(profile, c, phase, baseline, FIXED_SEEDS[0], 1)]
                                result = compare_case(a, b, dict(profile=profile, concurrency=c, phase=phase, candidate=arm, baseline=baseline))
                                outputs[key] = result
                                pairs.append(result)
                            item = dict(scenario=f'{profile}_c{c}', profile=profile, concurrency=c, phase=phase,
                                        candidate=arm, other_candidate=other,
                                        joint_win=outputs['vllm']['margin_win'] and outputs['other']['margin_win'])
                            for key, result in outputs.items():
                                item.update({key + '_' + name: value for name, value in result.items()
                                             if name not in ('profile', 'concurrency', 'phase', 'candidate', 'baseline')})
                            cases.append(item)
    except (OSError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        errors.append('evidence_error:' + repr(exc))
    complete = not errors and len(cases) == 48 and len(pairs) == 96
    decision = decision_from_cases(cases, complete, errors)
    decision.update(spec_sha256=spec.get('spec_sha256'), execution_protocol=spec.get('execution_protocol'),
                    cross_case_order=cross_case_order)
    csv_write(root / 'selection_pairs.csv', pairs, ['profile', 'concurrency', 'phase', 'candidate', 'baseline'])
    csv_write(root / 'selection_cases.csv', cases, ['scenario', 'phase', 'candidate', 'joint_win'])
    dump(root / 'selection_decision.json', decision)
    write_report(root, decision, cases)
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.run_root, read(args.spec))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
