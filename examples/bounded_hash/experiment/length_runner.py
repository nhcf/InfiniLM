#!/usr/bin/env python3
"""Run one paired eight-document throughput panel at a fixed global concurrency."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager, redirect_stdout
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import cache_confirm as confirm
import cache_suite
import cache_workload
import diagnose_confirmation
import harness
import summarize_confirmation

KIT = Path(__file__).resolve().parent
STUDY = 'reuse8-matched-lengths-v1'
ARMS = ['nh_cache_bounded_v2', 'nh_mean_bounded', 'vllm_least_active']
REFERENCE_SHA256 = '7344691f0e0d4f44f780936279ee14aaf6e2f5e733dbd90796d4914596827ddc'
ANCHOR_MANIFEST_SHA256 = '1290accd099c6ae866305cd22413a6620542bd44f58d9ce92b2a34030d6399b6'
ENVIRONMENT_FIELDS = ('config', 'nh_git', 'nh_files', 'native_bindings',
                      'tokenizer_files', 'package_versions', 'hardware')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_local(cfg):
    reference_path = KIT/'reference_environment.json'
    if confirm._sha(reference_path) != REFERENCE_SHA256:
        raise ValueError('Reference environment identity has changed')
    ref = read(reference_path)
    if cfg != ref['config']:
        raise ValueError('Configuration differs from the verified MUSA environment')
    sources = harness.source_hashes()
    changed = [name for name, digest in ref['proven_sources'].items() if sources.get(name) != digest]
    if changed:
        raise ValueError('Previously verified runtime source differs: '+', '.join(changed))
    anchors = KIT/'anchor_data'
    if confirm._sha(anchors/'source_hashes.json') != ANCHOR_MANIFEST_SHA256:
        raise ValueError('Frozen source workload hash manifest differs')
    for name, digest in read(anchors/'source_hashes.json').items():
        if confirm._sha(anchors/name) != digest:
            raise ValueError('Frozen source workload differs: '+name)
    return ref, sources


def verify_runtime(fp, reference, sources):
    for name in ENVIRONMENT_FIELDS:
        if fp.get(name) != reference[name]:
            raise ValueError('Runtime differs from the verified MUSA configuration: '+name)
    if fp.get('protocol') != confirm.PROTOCOL:
        raise ValueError('Serving transport protocol differs from the verified run')
    if fp.get('candidate_sources') != sources or harness.source_hashes() != sources:
        raise ValueError('Package source changed between planning and execution')
    if fp['hardware'].get('available') is not True or len(fp['hardware'].get('devices', [])) != 8:
        raise ValueError('Expected eight visible verified MUSA devices')


async def prepare_lengths(a, cfg, root, fingerprint, reference, sources):
    """The proven execution engine calls this after preflight and before calibration."""
    verify_runtime(fingerprint, reference, sources)
    output = root/'workloads'
    cmd = [cfg['nh_python'], '-B', str(KIT/'length_workload.py'),
           '--model', cfg['model'], '--out', str(output)]
    await harness.run_logged(cmd, confirm.rt.backend_env(cfg, 'nh', cfg['gpu_ids'][0], '/tmp'),
                             cfg['repo'], root/'workload_build.log', timeout=1800)
    manifest = read(output/'manifest.json')
    cases = confirm.validate_manifest(output, manifest, a)
    if len(cases) != 6 or len(manifest['cells']) != 6:
        raise ValueError('Incomplete or unexpected matched-length workload')
    harness.dump(root/'workload_origin.json', dict(
        mode='matched_length_family_from_frozen_reuse8',
        source_checkpoint='selection_20261103T023718_124257Z/phase_checkpoint',
        source_anchor_manifest_sha256=ANCHOR_MANIFEST_SHA256,
        files={str(p.relative_to(output)): confirm._sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return manifest


def study_metadata():
    return dict(study=STUDY, primary='measured output tokens / measured wall seconds',
                comparison='V2, old mean and vLLM paired at the same input, output and concurrency',
                historical_relation='Reuse8 seed33962 frozen document identities. Each panel measures new scores including the 8192/32 anchor.',
                source_qualification='Eight shared documents, hot25, fixed seed, one pass per length combination.',
                caveat='Descriptive throughput screen. Mean/p95 TTFT and learning-inclusive scores are secondary.')


@contextmanager
def adapted_runtime(profiles, reference, sources):
    """Limit the registry/loader adapters to this invocation and restore them on exit."""
    if set(cache_workload.PROFILES).intersection(profiles):
        raise ValueError('Length profile collides with an existing runtime profile')
    original_prepare = confirm.prepare_workloads
    original_plan = cache_suite.make_plan
    original_protocol = harness.PROTOCOL

    def base_plan(a, cfg):
        result = original_plan(a, cfg)
        result.update(study_metadata())
        return result

    async def prepare(a, cfg, root, fingerprint):
        return await prepare_lengths(a, cfg, root, fingerprint, reference, sources)

    cache_workload.PROFILES.update(profiles)
    confirm.prepare_workloads = prepare
    cache_suite.make_plan = base_plan
    try:
        yield
    finally:
        confirm.prepare_workloads = original_prepare
        cache_suite.make_plan = original_plan
        harness.PROTOCOL = original_protocol
        for name in profiles:
            del cache_workload.PROFILES[name]


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('plan', 'check', 'run'), nargs='?', default='plan')
    p.add_argument('--concurrency', type=int, choices=(16, 32), default=16)
    p.add_argument('--output-root', type=Path, default=KIT/'results_lengths')
    return p.parse_args(argv)


def engine_arguments(args, profiles, seed):
    return confirm.arguments([args.mode, '--profiles', *profiles, '--concurrency', str(args.concurrency),
                              '--seeds', str(seed), '--rounds', '1', '--requests', '128', '--arms', *ARMS,
                              '--config', str(KIT/'config.musa.json'), '--output-root', str(args.output_root)])


async def execute(a, cfg, root, plan, reference, sources):
    result = await confirm.execute(a, cfg, root, plan)
    fp = read(root/'fingerprint.json')
    verify_runtime(fp, reference, sources)
    if a.mode == 'check':
        await confirm.prepare_workloads(a, cfg, root, fp)
        result['tokenizer_workload_verified'] = True
    return result


def conventional_reports(root):
    records = []
    for name, entry in [('diagnose_confirmation', diagnose_confirmation.main),
                        ('summarize_confirmation', summarize_confirmation.main)]:
        result = dict(script=name+'.py', complete=False)
        try:
            with (root/(name+'.log')).open('w', encoding='utf-8') as log, redirect_stdout(log):
                rc = entry([str(root)])
            result.update(complete=rc == 0, returncode=rc)
        except (Exception, SystemExit) as exc:
            result['error'] = repr(exc)
        records.append(result)
    passed = all(record['complete'] for record in records)
    harness.dump(root/'postprocess_status.json', dict(status='COMPLETE' if passed else 'FAILED',
                                                       complete=passed, scripts=records))
    if not passed:
        raise RuntimeError('Request/confirmation postprocessing failed; inspect individual stage logs')


def execute_locked(a, cfg, root, plan, reference, sources):
    """Save execution evidence first, then postprocess, then make one authoritative archive."""
    root.mkdir(parents=True, exist_ok=False)
    print('RUN_DIR='+str(root), flush=True)
    harness.dump(root/'plan.json', plan)
    harness.dump(root/'execution_order.json', plan['execution_order'])
    harness.dump(root/'status.json', dict(status='RUNNING', complete=False))
    stage = dict(status='RUNNING', complete=False, study=STUDY, concurrency=a.concurrency[0])
    harness.dump(root/'length_status.json', stage)
    code = 2
    try:
        confirm.snapshot_sources(root)
        shutil.copy2(KIT/'reference_environment.json', root/'reference_environment.json')
        shutil.copy2(KIT/'anchor_data/source_hashes.json', root/'anchor_source_hashes.json')
        result = asyncio.run(execute(a, cfg, root, plan, reference, sources))
        harness.dump(root/'status.json', result)
        if result.get('complete') is not True:
            raise RuntimeError('Benchmark engine produced an incomplete result')
        if a.mode == 'run':
            conventional_reports(root)
            from length_report import analyze
            report = analyze(root)
            if report.get('complete') is not True:
                raise RuntimeError('Throughput analysis incomplete; inspect throughput_status.json')
            print('THROUGHPUT_REPORT='+str(root/'throughput_report.md'), flush=True)
        stage.update(complete=True, status='LENGTH_EXPERIMENT_COMPLETE' if a.mode == 'run' else 'CHECK_COMPLETE')
        code = 0
    except BaseException as exc:
        status = 'INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else 'FAILED'
        failure = dict(status=status, complete=False, error=repr(exc), traceback=traceback.format_exc())
        stage.update(failure)
        if read(root/'status.json').get('status') == 'RUNNING':
            harness.dump(root/'status.json', failure)
        print(status+' '+repr(exc), flush=True)
    finally:
        harness.dump(root/'length_status.json', stage)
        archive = shutil.make_archive(str(root)+'_report', 'zip', root.parent, root.name)
        print('LENGTH_REPORT_ARCHIVE='+archive, flush=True)
    return code


def main(argv=None):
    from length_workload import PROFILES, SEED
    args = arguments(argv)
    cfg = cache_suite.configuration(KIT/'config.musa.json')
    reference, sources = verify_local(cfg)
    with adapted_runtime(PROFILES, reference, sources):
        a = engine_arguments(args, PROFILES, SEED)
        plan = confirm.make_plan(a, cfg)
        plan.update(study_metadata())
        if args.mode == 'plan':
            fields = ('study', 'model', 'profiles', 'concurrency', 'seeds', 'rounds', 'arms', 'cells',
                      'learning_requests', 'measured_requests', 'total_learning_plus_measured_requests', 'primary')
            print(json.dumps({k:plan[k] for k in fields}, ensure_ascii=False, indent=2))
            return 0
        with open('/tmp/infinilm-fourway-'+str(os.getuid())+'.lock', 'a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print('Another benchmark holds the eight-GPU lock.', file=sys.stderr)
                return 2
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            root = args.output_root.resolve()/f'lengths_{args.mode}_c{args.concurrency}_{stamp}'
            return execute_locked(a, cfg, root, plan, reference, sources)


if __name__ == '__main__':
    raise SystemExit(main())
