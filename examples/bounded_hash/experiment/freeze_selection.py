#!/usr/bin/env python3
"""Freeze a validated selection or an explicit no-winner phase checkpoint.

No deployment, model loading, mutation of benchmark evidence, or overwrite is
performed. The complete raw experiment is referenced by a byte-level manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

from selection_protocol import digest, validate_spec

CANDIDATES = ('nh_mean_bounded', 'nh_cache_bounded_v2')
EVIDENCE_LEVEL = 'SINGLE_SEED_SINGLE_ROUND_SCREEN'
ARM_POLICY = {'nh_mean_bounded': 'mean_bounded', 'nh_cache_bounded_v2': 'cache_bounded_v2'}
REQUIRED_REPORTS = ('plan.json', 'selection_plan.json', 'execution_order.json', 'status.json',
                    'confirmation_status.json', 'postprocess_status.json',
                    'selection_decision.json', 'selection_pairs.csv',
                    'selection_cases.csv', 'selection_report.md')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            result.update(block)
    return result.hexdigest()


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe relative artifact path: ' + str(name))
    return path


def tree_manifest(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink is not a frozen local artifact: ' + str(path))
        if path.is_file():
            result[str(path.relative_to(root))] = dict(sha256=sha(path), bytes=path.stat().st_size)
    return result


def validate_decision(decision, spec):
    if decision.get('complete') is not True:
        raise ValueError('Incomplete evidence cannot be frozen as a completed phase')
    if decision.get('spec_sha256') != spec['spec_sha256']:
        raise ValueError('Decision does not reference the frozen selection plan')
    if decision.get('evidence_level') != EVIDENCE_LEVEL:
        raise ValueError('Decision must identify the single-seed single-round screen')
    selected = decision.get('selected_arm')
    if selected is not None and selected not in CANDIDATES:
        raise ValueError('Unknown selected candidate')
    status, scope = decision.get('status'), decision.get('scope')
    winners = decision.get('phase_winners', {})
    if selected is not None:
        if status != 'SELECTED' or decision.get('performance_qualified') is not True:
            raise ValueError('Selected candidate lacks a qualified selection status')
        if winners.get('measured') != selected:
            raise ValueError('Selected candidate differs from primary-phase winner')
        combined = winners.get('learn_plus_measured')
        expected_scope = 'MEASURED_AND_LEARNING_INCLUSIVE' if combined == selected else 'WARM_ONLY'
        if combined not in (None, selected) or scope != expected_scope:
            raise ValueError('Conflicting or incorrect selection scope')
        coverage = decision.get('per_candidate_coverage', {}).get('measured', {}).get(selected, {})
        if coverage.get('joint_wins', 0) < spec['rules']['majority_scenarios']:
            raise ValueError('Selected candidate lacks the majority of joint-winning scenarios')
    else:
        if status not in ('NO_UNIQUE_MAJORITY_WINNER', 'PHASE_CONFLICT'):
            raise ValueError('No-selection checkpoint needs an explicit completed decision')
        if decision.get('performance_qualified') is not False or scope is not None:
            raise ValueError('No-winner checkpoint must not claim qualified performance')
    return selected


def validate_inputs(root, spec, decision):
    validate_spec(spec)
    if not spec.get('spec_sha256'):
        raise ValueError('A predeclared hashed selection plan is required')
    selected = validate_decision(decision, spec)
    for name in REQUIRED_REPORTS:
        if not (root / name).is_file():
            raise ValueError('Required completed evidence missing: ' + name)
    if read(root / 'selection_decision.json') != decision:
        raise ValueError('Supplied decision differs from saved analysis')
    if read(root / 'selection_plan.json') != spec:
        raise ValueError('Supplied specification differs from saved pre-run plan')
    state = read(root / 'status.json')
    checks = read(root / 'confirmation_status.json')
    post = read(root / 'postprocess_status.json')
    plan = read(root / 'plan.json')
    cells = spec['actual_cells']
    rows = cells * 3
    expected_fields = [
        ('runtime', state, dict(status='EXPERIMENT_COMPLETE', complete=True, completed_cells=cells,
                               valid_phase_rows=rows, evidence_kind='unbalanced_screen_not_confirmation')),
        ('confirmation', checks, dict(complete=True, execution_position_balanced=False,
                                     expected_phase_cells=rows, found_phase_cells=rows)),
        ('postprocessing', post, dict(complete=True)),
        ('plan', plan, dict(rounds=1, seeds=spec['seeds'], cells=cells,
                           evidence_kind='unbalanced_screen_not_confirmation', order_balanced=False)),
    ]
    for label, record, required in expected_fields:
        for field, expected in required.items():
            if record.get(field) != expected:
                raise ValueError(f'Unexpected single-pass {label} field {field}')
    if read(root / 'execution_order.json') != plan['execution_order']:
        raise ValueError('Archived execution order differs from the run plan')
    from selection_analysis import validate_single_round_order
    order = validate_single_round_order(root, plan, spec)
    if order.get('complete') is not True or order != decision.get('cross_case_order'):
        raise ValueError('Actual cross-scenario order is incomplete or differs from the analyzed screen')
    fp = read(root / 'fingerprint.json')
    if fp != spec.get('expected_fingerprint'):
        raise ValueError('Fingerprint differs from the pre-run frozen identity')
    if fp.get('config') != spec.get('config') or plan.get('config') != spec.get('config'):
        raise ValueError('Configuration differs from the frozen plan')
    sources = fp.get('candidate_sources')
    if not sources or sources != spec.get('source_sha256'):
        raise ValueError('Source fingerprint differs from the frozen plan')
    actual_sources = tree_manifest(root / 'source')
    if set(actual_sources) != set(sources):
        raise ValueError('Source snapshot has missing or unexpected files')
    for name, expected in sources.items():
        safe_relative(name)
        if actual_sources[name]['sha256'] != expected:
            raise ValueError('Source snapshot hash mismatch: ' + name)
    phase_model = read(root / 'phase_model.json')
    if phase_model.get('fingerprint') != fp:
        raise ValueError('Calibration model differs from frozen runtime identity')
    from cache_confirm import validate_manifest
    workloads = root / 'workloads'
    # Reject symlinks before following workload paths. The reused validator also
    # checks exact profile parameters, IDs, cohort sizes and declared file hashes.
    workload_files = tree_manifest(workloads)
    workload_manifest = read(workloads / 'manifest.json')
    cases = validate_manifest(workloads, workload_manifest,
                              SimpleNamespace(profiles=spec['profiles'], seeds=spec['seeds'], requests=spec['measured_requests']))
    if len(workload_manifest['cells']) != len(cases):
        raise ValueError('Unexpected workload cases outside the frozen panel')
    if len({c['folder'] for c in cases}) != len(cases):
        raise ValueError('Workload cases share a folder')
    if spec.get('reference_environment_sha256'):
        reference = root / 'reference_environment.json'
        if not reference.is_file() or sha(reference) != spec['reference_environment_sha256']:
            raise ValueError('Reference environment artifact missing or differs')
    return selected, fp, actual_sources, workload_files


def freeze(run_root, spec, decision, destination):
    root, out = Path(run_root).resolve(), Path(destination).resolve()
    archive = out.with_suffix('.zip')
    if out == root or root in out.parents or out in root.parents:
        raise ValueError('Freeze destination must be outside and separate from the source run')
    if out.exists() or archive.exists():
        raise FileExistsError('Never overwrite a frozen bundle or archive: ' + str(out))
    # Do not follow an untrusted artifact link while reading JSON or copying
    # source files; external paths in model configuration remain metadata only.
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlink is not a frozen local artifact: ' + str(path))
    selected, fp, sources, workloads = validate_inputs(root, spec, decision)
    # Enumerate and hash evidence before producing new files. No external model
    # weights are read; file paths inside configuration are metadata only.
    evidence = tree_manifest(root)
    identity = dict(selection_spec_sha256=spec['spec_sha256'],
                    source_sha256=spec['source_sha256'], config=spec['config'],
                    fingerprint_sha256=digest(fp), workload_files=workloads,
                    decision=decision)
    version = ('selected-' if selected else 'checkpoint-') + digest(identity)[:24]
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root / 'source', out / 'code')
    for config_name in ('config.musa.json', 'config.musa.example.json', 'config32b.json'):
        write(out / 'code' / config_name, spec['config'])
    if (root / 'reference_environment.json').is_file():
        shutil.copy2(root / 'reference_environment.json', out / 'code' / 'reference_environment.json')
    shutil.copytree(root / 'workloads', out / 'workloads')
    for name in ('fingerprint.json', 'phase_model.json'):
        shutil.copy2(root / name, out / name)
    write(out / 'selection_plan.json', spec)
    reports = out / 'reports'
    reports.mkdir()
    names = set(REQUIRED_REPORTS)
    names.update(p.name for pattern in ('selection_*', 'confirmation_*') for p in root.glob(pattern) if p.is_file())
    names.update(name for name in ('report.md', 'summary.csv') if (root / name).is_file())
    for name in sorted(names):
        shutil.copy2(root / name, reports / name)
    write(out / 'evidence_manifest.json', dict(source_run=str(root), files=evidence,
          note='The raw result directory is preserved separately in the full experiment report archive. This manifest verifies its bytes.'))
    if selected:
        write(out / 'selected_version.json', dict(version_id=version, arm=selected,
              policy=ARM_POLICY[selected], cache_policy='prefix_lru', scope=decision['scope'],
              selection_status=decision['status'], performance_qualified=True,
              evidence_level=EVIDENCE_LEVEL, seeds=spec['seeds'], rounds_per_scenario=spec['rounds']))
    else:
        write(out / 'checkpoint_versions.json', dict(version_id=version,
              candidates=[dict(arm=arm, policy=ARM_POLICY[arm]) for arm in CANDIDATES],
              selection_status=decision['status'], selected_arm=None, performance_qualified=False,
              evidence_level=EVIDENCE_LEVEL, seeds=spec['seeds'], rounds_per_scenario=spec['rounds']))
    text = ('# 单轮实验阶段归档\n\n'
            f'归档编号 `{version}`，选中版本 `{selected or "NONE"}`，适用范围 `{decision.get("scope") or "未分出多数赢家"}`。\n\n'
            f'本次在种子 {spec["seeds"][0]} 下，每个场景运行一轮。实际执行记录显示，三种方法在全部 12 个场景中各占第一、第二和第三位置各 4 次。'
            '这份归档记录本次单轮筛查的结果。场景内的运行波动和不同种子间的稳定性仍待独立复测。\n\n'
            '`reports/selection_report.md` 列出各场景胜负、计入学习阶段后的结果及 p95 首响应。'
            '`WARM_ONLY` 表示所选版本仅在学习后的正式阶段达到多数门槛。\n\n'
            '`code/` 保存依赖源码和配置，`workloads/` 保存固定请求，`phase_model.json` 和 `fingerprint.json` 保存校准及运行环境。'
            '完整原始实验另存于阶段报告包，`evidence_manifest.json` 记录原始文件校验值。模型身份核对覆盖配置和分词器。\n')
    (out / 'README_FREEZE.md').write_text(text, encoding='utf-8')
    bundle_files = tree_manifest(out)
    manifest = dict(status='FROZEN_SELECTED_VERSION' if selected else 'FROZEN_PHASE_CHECKPOINT',
                    complete=True, version_id=version, selected_arm=selected,
                    performance_qualified=selected is not None, scope=decision.get('scope'),
                    evidence_level=EVIDENCE_LEVEL, seeds=spec['seeds'], rounds_per_scenario=spec['rounds'],
                    within_scenario_order_balanced=False, cross_case_order=decision['cross_case_order'],
                    selection_status=decision['status'], spec_sha256=spec['spec_sha256'],
                    fingerprint_sha256=digest(fp), source_files=sources,
                    workload_files=workloads, source_run=str(root),
                    evidence_manifest_sha256=sha(out / 'evidence_manifest.json'),
                    files=bundle_files, bundle_directory=str(out), archive=str(archive),
                    model_weights_bundled=False, model_weight_hash_claim=False,
                    no_untested_hardware_or_general_serving_claim=True)
    write(out / 'FREEZE_MANIFEST.json', manifest)
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                zf.write(path, Path(out.name) / path.relative_to(out))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = freeze(args.run_root, read(args.spec), read(args.run_root / 'selection_decision.json'), args.out)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
