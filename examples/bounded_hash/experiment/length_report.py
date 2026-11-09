"""Throughput-first report for the eight-document, single-seed length panel."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import backend_runtime as runtime
from cache_suite import contract
import summarize_confirmation


MEAN = 'nh_mean_bounded'
V2 = 'nh_cache_bounded_v2'
VLLM = 'vllm_least_active'
ARMS = {'mean': MEAN, 'v2': V2, 'vllm': VLLM}
LABEL = {'mean': '旧 mean', 'v2': 'V2', 'vllm': 'vLLM'}
COMPARISONS = {'v2_vs_vllm': ('v2', 'vllm'), 'mean_vs_vllm': ('mean', 'vllm'),
               'v2_vs_mean': ('v2', 'mean')}
CASES = {f'p{p}_o{o}': (p, o) for p in (4096, 6144, 8192) for o in (32, 128)}
PHASES = ('measured', 'learn_plus_measured')
PHASE_LABEL = {'measured': '正式阶段', 'learn_plus_measured': '学习和正式阶段合计'}
SEED = 33962


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def matches(actual, expected, name):
    require(math.isclose(float(actual), float(expected), abs_tol=1e-8, rel_tol=1e-8),
            f'{name}: observed {actual} differs from recomputed {expected}')


def validate_run(root):
    plan = read(root/'plan.json')
    require(plan.get('rounds') == 1 and plan.get('seeds') == [SEED], 'round/seed mismatch')
    require(plan.get('concurrency') in ([16], [32]), 'expected one concurrency panel at C16 or C32')
    require(plan.get('learning_requests') == 64 and plan.get('measured_requests') == 128,
            'learning or measured request count changed')
    require({summarize_confirmation.arm_id(v) for v in plan['arms']} == set(ARMS.values())
            and len(plan['arms']) == 3, 'wrong methods')
    require(set(plan['profiles']) == set(CASES), 'length panel differs from specification')
    for name, (p, o) in CASES.items():
        require(plan['profiles'][name] == dict(prompt_tokens=p, output_tokens=o,
                shared_tokens=p-256, documents=8, hot_fraction=.25), 'wrong configuration: '+name)
    require(plan.get('cells') == 18 and plan.get('fresh_pool_per_cell') is True,
            'expected 18 fresh-pool benchmark cells')
    require(read(root/'postprocess_status.json').get('complete') is True, 'postprocessing incomplete')
    checked = summarize_confirmation.analyze(root)
    require(checked['complete'] and checked['found_phase_cells'] == 54
            and checked['excluded_pairs_or_metadata'] == 0, 'incomplete or invalid matched evidence')
    jobs = plan['execution_order']
    require([j['serial'] for j in jobs] == list(range(1, 19)), 'invalid execution serials')
    for i in range(0, 18, 3):
        block = jobs[i:i+3]
        require(len({j['profile'] for j in block}) == 1
                and [j['order_position'] for j in block] == [1, 2, 3]
                and {j['arm'] for j in block} == set(ARMS.values()), 'invalid scenario order')
    counts = Counter((job['arm'], job['order_position']) for job in jobs)
    require(all(counts[a, n] == 2 for a in ARMS.values() for n in (1, 2, 3)),
            'panel execution positions are unbalanced')
    return plan


def normalized_body_digest(body):
    # Same input/output-length variants may differ only in their generation budget.
    value = {k: v for k, v in body.items() if k != 'max_tokens'}
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def read_workloads(root):
    work = root/'workloads'
    manifest = read(work/'manifest.json')
    by_case = {}
    same_prompt = {}
    across_lengths = {}
    for case in manifest['cells']:
        name = case['profile']
        require(name in CASES and name not in by_case and case['seed'] == SEED, 'invalid manifest case')
        p, o = CASES[name]
        require(all(case.get(k) == value for k, value in dict(prompt_tokens=p, output_tokens=o,
                shared_tokens=p-256, documents=8, hot_fraction=.25).items()), 'manifest profile mismatch')
        relative = Path(case['folder'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe workload path')
        cohorts = {}
        for phase, expected_count in (('learn', 64), ('measured', 128)):
            path = work/relative/(phase+'.jsonl')
            round_path = root/'rounds/round01/workloads'/relative/(phase+'.jsonl')
            declared = case['files'][phase]
            require(runtime.sha(path) == declared['sha256'] == runtime.sha(round_path),
                    'frozen workload digest mismatch')
            items = runtime.rows(path)
            require(len(items) == len({item['id'] for item in items}) == expected_count == declared['count'],
                    'invalid workload size or duplicate IDs')
            signature, sequence, small = [], [], []
            for item in items:
                require((item['prompt_tokens'], item['shared_tokens'], item['output_tokens'], item['body']['max_tokens'])
                        == (p, p-256, o, o), 'item token contract mismatch')
                signature.append((item['id'], item['key'], item['prompt_sha256'], item['prefix_digest'],
                                  item['first_block_digest'], normalized_body_digest(item['body'])))
                sequence.append((item['id'], item['key'], item['first_block_digest']))
                small.append({k:item[k] for k in ('id','key','prompt_sha256','prompt_tokens',
                                                  'shared_tokens','output_tokens')})
            require(len({i['key'] for i in items}) == 8, 'expected eight document keys')
            require(same_prompt.setdefault((p, phase), signature) == signature,
                    'changing output length changed input/cohort')
            require(across_lengths.setdefault(phase, sequence) == sequence,
                    'document identity, leading block or arrival order changed across lengths')
            cohorts[phase] = small
        require({x['id'] for x in cohorts['learn']}.isdisjoint(x['id'] for x in cohorts['measured']),
                'learning/measured IDs overlap')
        by_case[name] = cohorts
    require(set(by_case) == set(CASES), 'missing frozen workload case')
    return by_case


def verify_phase(row, inputs, records):
    expected = {i['id']: i for i in inputs}
    require(len(inputs) == len(records) == len(expected) == int(row['requests']), 'bad phase count')
    require({r['id'] for r in records} == expected.keys(), 'actual requests differ from frozen inputs')
    require(row['ordered_request_contract_sha256'] == contract(inputs), 'ordered request digest incorrect')
    require(summarize_confirmation.validate(row) is None, 'invalid phase statistics')
    ttft = []
    tokens = 0
    for record in records:
        item = expected[record['id']]
        require(record.get('success') is True and record.get('http_status') == 200, 'failed request')
        require((record['key'], record['prompt_sha256']) == (item['key'], item['prompt_sha256']),
                'request identity differs')
        usage = record['usage']
        require((usage['prompt_tokens'], usage['completion_tokens']) ==
                (item['prompt_tokens'], item['output_tokens']), 'incorrect token usage')
        value = float(record['ttft_ms'])
        require(math.isfinite(value) and value > 0, 'invalid raw TTFT')
        ttft.append(value)
        tokens += usage['completion_tokens']
    require(tokens == row['total_output_tokens'], 'token total mismatch')
    matches(row['output_tok_s'], tokens / float(row['wall_s']), 'throughput')
    matches(row['ttft_mean_ms'], statistics.mean(ttft), 'mean TTFT')
    matches(row['ttft_p95_ms'], runtime.quantile(ttft, .95), 'p95 TTFT')


def collect_metrics(root, cohorts):
    data = {}
    for path in sorted(root.glob('rounds/round01/cells/*/cell_statistics.json')):
        folder = path.parent
        job = read(folder/'order.json')
        arm, profile = job['arm'], job['profile']
        require(read(folder/'gateway_errors.json') == [], 'gateway errors present')
        stats = {r['phase']:r for r in read(path)}
        require(set(stats) == {'learn', *PHASES}, 'missing phase')
        raw = {phase: read(folder/(phase+'_requests.json')) for phase in ('learn','measured')}
        for phase in ('learn','measured'):
            verify_phase(stats[phase], cohorts[profile][phase], raw[phase])
        matches(stats['learn_plus_measured']['wall_s'], stats['learn']['wall_s']+stats['measured']['wall_s'],
                'combined wall time')
        verify_phase(stats['learn_plus_measured'], cohorts[profile]['learn']+cohorts[profile]['measured'],
                     raw['learn']+raw['measured'])
        for phase in PHASES:
            data[profile,phase,arm] = stats[phase]
    require(len(data) == 36, 'missing or duplicate metrics')
    return data


def score(profile, phase, concurrency, data):
    p, o = CASES[profile]
    out = dict(profile=profile, phase=phase, concurrency=concurrency, seed=SEED,
               prompt_tokens=p, shared_tokens=p-256, output_tokens_per_request=o)
    for name, arm in ARMS.items():
        row = data[profile,phase,arm]
        for column in ('output_tok_s','ttft_mean_ms','ttft_p95_ms'):
            out[name+'_'+column] = float(row[column])
    rates = {name: out[name+'_output_tok_s'] for name in ARMS}
    out['first_by_throughput'] = '|'.join(name for name in ARMS if rates[name] == max(rates.values()))
    for comparison, (cand, base) in COMPARISONS.items():
        ratio = rates[cand] / rates[base]
        out[comparison+'_pct'] = 100 * (ratio - 1)
        out[comparison+'_higher'] = rates[cand] > rates[base]
        out[comparison+'_at_least_2pct'] = ratio >= 1.02 or math.isclose(ratio, 1.02, rel_tol=0, abs_tol=1e-12)
    return out


def summarize(rows):
    result = {}
    for phase in PHASES:
        group = [r for r in rows if r['phase'] == phase]
        wins = {key: {'higher':sum(r[key+'_higher'] for r in group),
                      'at_least_2pct':sum(r[key+'_at_least_2pct'] for r in group)}
                for key in COMPARISONS}
        winners = Counter(r['first_by_throughput'] for r in group)
        result[phase] = dict(cases=len(group), comparisons=wins, first_by_throughput=dict(winners))
    return result


def write_outputs(root, result, rows):
    with (root/'throughput_cases.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['phase','profile'])
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 八文档复用长度实验', '', '固定种子 33962，每种方法在每组长度下运行一次。']
    if not result['complete']:
        lines.extend(['', '本次未取得完整的六组可比成绩。', '', result['error']])
    else:
        lines.extend(['',f"全局并发为 {result['concurrency']}。每种方法先运行 64 条学习请求，再测量 128 条正式请求。吞吐为输出 token 总量除以阶段总用时。不同长度各占一个场景。"])
        for phase in PHASES:
            summary = result['phase_summary'][phase]
            lines.extend(['', '## '+PHASE_LABEL[phase], '',
                          '| 输入 / 输出 | 旧 mean tok/s | V2 tok/s | vLLM tok/s | V2 对 vLLM | 旧 mean 对 vLLM | V2 对旧 mean | 吞吐最高 |',
                          '|---|---:|---:|---:|---:|---:|---:|---|'])
            group = [r for r in rows if r['phase'] == phase]
            for r in group:
                lines.append(f"| {r['prompt_tokens']} / {r['output_tokens_per_request']} | {r['mean_output_tok_s']:.3f} | {r['v2_output_tok_s']:.3f} | {r['vllm_output_tok_s']:.3f} | {r['v2_vs_vllm_pct']:+.3f}% | {r['mean_vs_vllm_pct']:+.3f}% | {r['v2_vs_mean_pct']:+.3f}% | "+'、'.join(LABEL[k] for k in r['first_by_throughput'].split('|'))+' |')
            lines.extend(['', '| 比较 | 吞吐数值领先 | 吞吐至少领先 2% |','|---|---:|---:|'])
            for key, (cand, base) in COMPARISONS.items():
                w = summary['comparisons'][key]
                lines.append(f"| {LABEL[cand]} 对 {LABEL[base]} | {w['higher']} / 6 | {w['at_least_2pct']} / 6 |")
            top_counts = '，'.join('、'.join(LABEL[n] for n in k.split('|')) + f' {v} 组' for k,v in summary['first_by_throughput'].items())
            lines.extend(['', '按三方实际吞吐排名第一的场景数为 '+top_counts+'。相等时列为并列第一。', '',
                          '| 输入 / 输出 | 旧 mean 平均 TTFT (ms) | 旧 mean p95 (ms) | V2 平均 TTFT (ms) | V2 p95 (ms) | vLLM 平均 TTFT (ms) | vLLM p95 (ms) |',
                          '|---|---:|---:|---:|---:|---:|---:|'])
            for r in group:
                lines.append('| '+f"{r['prompt_tokens']} / {r['output_tokens_per_request']}"+' | '+' | '.join(f"{r[a+'_'+m]:.2f}" for a in ARMS for m in ('ttft_mean_ms','ttft_p95_ms'))+' |')
        lines.extend(['', '吞吐胜负按原始精度计算。2% 是本次预设的实用差异门槛。TTFT 与 p95 单列展示，反映这一批请求的首响应体验。'])
    (root/'throughput_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    runtime.dump(root/'throughput_status.json',result)


def analyze(root):
    root = Path(root).resolve()
    try:
        plan = validate_run(root)
        data = collect_metrics(root, read_workloads(root))
        c = plan['concurrency'][0]
        rows = [score(profile, phase, c, data) for phase in PHASES for profile in CASES]
        result = dict(status='THROUGHPUT_COMPLETE', complete=True, concurrency=c, seed=SEED,
                      actual_cells=18, learning_requests=1152, measured_requests=2304,
                      total_learning_plus_measured=3456, phase_summary=summarize(rows),
                      evidence_level='single_seed_single_round', primary='measured output tokens / wall seconds')
    except (KeyError, OSError, ValueError, TypeError, ZeroDivisionError) as exc:
        result = dict(status='THROUGHPUT_INCOMPLETE', complete=False, error=str(exc))
        rows = []
    write_outputs(root, result, rows)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    status = analyze(parser.parse_args().run_root)
    print(json.dumps(status, ensure_ascii=False))
    raise SystemExit(0 if status['complete'] else 2)
