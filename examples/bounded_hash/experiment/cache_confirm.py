#!/usr/bin/env python3
"""Frozen paired repeats with all engines interleaved and fresh pools per cell.

Only run starts model services. Plan is local/read-only; check does source/runtime
preflight without loading models. The underlying routers and measured timer are
reused from the supplied cache-v2 implementation without modifications.
"""
from __future__ import annotations
import argparse
import asyncio
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import traceback

import cache_suite as base
import backend_runtime as rt
import harness as common
from cache_workload import PROFILES

KIT = Path(__file__).resolve().parent
PROTOCOL = 'infinilm-cache-paired-confirm-v2-fresh-http'
ARM_MAP = {a[0]: tuple(a) for a in base.ARMS}
REPLAY_IDENTITY_FIELDS = ('tokenizer_files', 'nh_git', 'nh_files', 'native_bindings',
                          'package_versions', 'hardware')
REPLAY_POLICY_SOURCES = ('affinity_router.py', 'stable_router.py', 'mean_router.py',
                         'cost_router.py', 'phase_router.py', 'cache_router.py',
                         'cache_policy.py', 'vendor/routing_manager.py',
                         'vendor/nhcf_cache/cache_manager.py', 'vendor/nhcf_cache/prefix_cache.py')


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['plan', 'check', 'run'], nargs='?', default='plan')
    p.add_argument('--config', type=Path, default=KIT/'config.musa.json')
    p.add_argument('--historical-run', type=Path)
    p.add_argument('--phase-model', type=Path)
    p.add_argument('--profiles', nargs='+', choices=list(PROFILES), default=['reuse8'])
    p.add_argument('--seeds', nargs='+', type=int, default=[33960, 33961])
    p.add_argument('--concurrency', nargs='+', type=int, choices=[16, 32, 64], default=[16])
    p.add_argument('--requests', type=int, choices=[64, 128, 256], default=128)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--arms', nargs='+', choices=list(ARM_MAP), default=list(ARM_MAP))
    p.add_argument('--output-root', type=Path, default=KIT/'results_confirm')
    a = p.parse_args(argv)
    for seq in (a.profiles, a.seeds, a.concurrency, a.arms):
        if len(set(seq)) != len(seq) or not seq:
            p.error('Selections must be nonempty and unique')
    if a.rounds < 1 or a.rounds > 20 or any(s < 0 for s in a.seeds):
        p.error('Use 1..20 rounds and nonnegative seeds')
    if not {'nh_cache_bounded_v2', 'vllm_least_active'}.issubset(a.arms):
        p.error('Include both nh_cache_bounded_v2 and vllm_least_active')
    # Frozen candidate settings. Performance exploration must be a new version.
    a.no_coalescing = False
    a.owner_wait_ms = 500.0
    return a


def schedule(a):
    """Each arm occupies every position once in each full N-round cycle."""
    arms = [x for x in base.ARMS if x[0] in a.arms]
    jobs = []
    serial = 0
    for r in range(a.rounds):
        for pi, profile in enumerate(a.profiles):
            for seed in a.seeds:
                for ci, c in enumerate(a.concurrency):
                    shift = (r + seed + pi + ci) % len(arms)
                    order = arms[shift:] + arms[:shift]
                    for position, arm in enumerate(order, 1):
                        serial += 1
                        jobs.append(dict(serial=serial, round=r+1, profile=profile,
                                         seed=seed, concurrency=c, order_position=position,
                                         arm=arm[0], engine=arm[1]))
    return jobs


def make_plan(a, cfg):
    p = base.make_plan(a, cfg)
    p.pop('gpu_jobs_started', None)
    per_round_cells = p['cells']
    p.update(protocol=PROTOCOL, mode=a.mode, rounds=a.rounds,
             per_round_cells=per_round_cells, cells=per_round_cells*a.rounds,
             expected_phase_rows=per_round_cells*a.rounds*3,
             total_learning_plus_measured_requests=per_round_cells*a.rounds*(64+a.requests),
             execution_order=schedule(a),
             order_balanced=(a.rounds % len(p['arms']) == 0),
             evidence_kind=('balanced_repeated_comparison' if a.rounds % len(p['arms']) == 0
                            else 'unbalanced_screen_not_confirmation'),
             fresh_pool_per_cell=True,
             transport=dict(connection_reuse=False, force_close=True,
                            automatic_inference_retries=0, applies_to='all arms, client-to-gateway, gateway-to-backend, calibration',
                            timing='connection establishment is inside request timing',
                            prior_result_pooling=False),
             fairness='Every arm gets a fresh eight-replica pool per cell; all arms are interleaved in a fixed cyclic Latin order. Cache probes, learning and measured requests use the original timing/audit code.',
             startup_cost='One new model pool per cell, deliberately excluded from serving throughput; wall-clock execution can be lengthy. No two eight-GPU pools coexist.',
             historical_workloads=str(a.historical_run.resolve()) if a.historical_run else None,
             historical_relation=('Reuse saved workload bytes after strict configuration, hardware, runtime, engine, original routing-policy source, tokenizer, profile, count and hash checks; historical performance rows are not counted as new controls.'
                                  if a.historical_run else 'Generate each profile/seed workload once before any repeated cells, then freeze the same bytes across all methods and rounds.'),
             source_qualification='Seeds are repeated workload strata, not independent per-request samples. Development seeds do not become unseen holdouts by repetition.',
             comparison='candidate vs same-round same-profile seed C baselines, including old mean when selected; warm and learning-inclusive phases remain separate',
             calibration='Fresh calibration by default. Saved model accepted only with exact current full fingerprint; changed package sources require recalibration.',
             admission_observation='Read-only pre-admission worker/peer snapshots applied equally to all methods; no router selection code is modified.')
    return p


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe_relative(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe workload path')
    return p


def validate_manifest(work, manifest, a):
    want = {(p, s) for p in a.profiles for s in a.seeds}
    cases = [c for c in manifest['cells'] if (c['profile'], c['seed']) in want]
    if {(c['profile'], c['seed']) for c in cases} != want or len(cases) != len(want):
        raise ValueError('Required frozen profile/seed cohorts missing or duplicated')
    for c in cases:
        expected = PROFILES[c['profile']]
        if any(c.get(k) != v for k, v in expected.items()):
            raise ValueError('Historical profile parameters differ from requested profile')
        folder = work/_safe_relative(c['folder'])
        probes = json.loads((folder/'smoke.json').read_text())
        if len(probes) != 2:
            raise ValueError('Expected cold/warm protocol probes')
        for phase, count in [('learn', 64), ('measured', a.requests)]:
            file = folder/(phase+'.jsonl')
            declaration = c['files'][phase]
            rows = rt.rows(file)
            if declaration['count'] != count or len(rows) != count or _sha(file) != declaration['sha256']:
                raise ValueError('Frozen workload count/hash mismatch; refusing changed requests')
            if len({r['id'] for r in rows}) != count:
                raise ValueError('Duplicate request IDs in frozen cohort')
            if any((r['prompt_tokens'], r['shared_tokens'], r['output_tokens']) !=
                   (c['prompt_tokens'], c['shared_tokens'], c['output_tokens']) for r in rows):
                raise ValueError('Frozen workload token contract differs')
    return cases


def validate_historical_identity(prior, current):
    """Allow new runner/protocol files, but reject environment or policy drift."""
    for key in REPLAY_IDENTITY_FIELDS:
        if key not in prior or key not in current or prior[key] is None or prior[key] != current[key]:
            raise ValueError(f'Historical {key} identity missing or differs; refusing exact replay')
    hardware = current['hardware']
    if not hardware.get('available') or not hardware.get('devices'):
        raise ValueError('Hardware identity unavailable; cannot validate exact historical replay')
    for source in REPLAY_POLICY_SOURCES:
        old_hash = prior.get('candidate_sources', {}).get(source)
        new_hash = current.get('candidate_sources', {}).get(source)
        if not old_hash or old_hash != new_hash:
            raise ValueError(f'Historical routing-policy source differs: {source}')


def copy_historical_workloads(a, cfg, root, fingerprint):
    old = a.historical_run.resolve()
    plan = json.loads((old/'plan.json').read_text())
    status = json.loads((old/'status.json').read_text())
    prior_fp = json.loads((old/'fingerprint.json').read_text())
    if plan.get('protocol') != base.PROTOCOL or not status.get('complete'):
        raise ValueError('Historical workload source must be a complete cache-v2 run')
    if plan.get('config') != cfg:
        raise ValueError('Historical runtime configuration differs; do not label this an exact replay')
    validate_historical_identity(prior_fp, fingerprint)
    manifest = json.loads((old/'workloads/manifest.json').read_text())
    cases = validate_manifest(old/'workloads', manifest, a)
    target = root/'workloads'
    if target.exists():
        raise FileExistsError('Never overwrite a frozen workload tree')
    shutil.copytree(old/'workloads', target)
    manifest = dict(manifest, cells=cases, profiles=list(a.profiles), seeds=list(a.seeds))
    common.dump(target/'manifest.json', manifest)
    validate_manifest(target, manifest, a)
    common.dump(root/'workload_origin.json', dict(mode='exact_copy', source=str(old),
               source_manifest_sha256=_sha(old/'workloads/manifest.json'),
               verified_identity_fields=list(REPLAY_IDENTITY_FIELDS),
               verified_original_policy_sources=list(REPLAY_POLICY_SOURCES),
               copied_files={str(p.relative_to(target)): _sha(p) for p in sorted(target.rglob('*')) if p.is_file()}))
    return manifest


async def prepare_workloads(a, cfg, root, fingerprint):
    if a.historical_run:
        return copy_historical_workloads(a, cfg, root, fingerprint)
    cmd = [cfg['nh_python'], str(base.KIT/'cache_workload.py'), '--model', cfg['model'],
           '--out', str(root/'workloads'), '--profiles', *a.profiles,
           '--seeds', *map(str, a.seeds), '--requests', str(a.requests)]
    await common.run_logged(cmd, rt.backend_env(cfg, 'nh', cfg['gpu_ids'][0], '/tmp'),
                            cfg['repo'], root/'workload_build.log', timeout=1800)
    manifest = json.loads((root/'workloads/manifest.json').read_text())
    validate_manifest(root/'workloads', manifest, a)
    common.dump(root/'workload_origin.json', dict(mode='generated_once_before_all_rounds',
               files={str(p.relative_to(root/'workloads')): _sha(p) for p in sorted((root/'workloads').rglob('*')) if p.is_file()}))
    return manifest


async def execute_job(cfg, round_root, case, job, model, router_options, client):
    """Pool lifetime is exactly one cell, with cleanup even after startup failure."""
    arm = ARM_MAP[job['arm']]
    label = f'{case["profile"]}_s{case["seed"]}_c{job["concurrency"]}_{arm[0]}'
    pool = rt.Pool(cfg, arm[1], round_root/'pools'/f'{job["serial"]:04d}_{label}')
    common.dump(round_root/'orders'/f'{job["serial"]:04d}.json', dict(job, state='STARTING'))
    try:
        await pool.start()
        common.dump(pool.folder/'runtime_features.json', base.runtime_features(pool))
        rows = await base.run_cell(cfg, round_root, pool, case, job['concurrency'], arm,
                                   client, model, router_options)
        common.dump(round_root/'cells'/label/'order.json', dict(job, state='COMPLETE'))
        common.dump(round_root/'orders'/f'{job["serial"]:04d}.json', dict(job, state='COMPLETE'))
        return [dict(r, round=job['round'], order_position=job['order_position'],
                     execution_serial=job['serial']) for r in rows]
    except BaseException as exc:
        if (round_root/'cells'/label).exists():
            common.dump(round_root/'cells'/label/'order.json',
                        dict(job, state='FAILED', error=repr(exc)))
        common.dump(round_root/'orders'/f'{job["serial"]:04d}.json',
                    dict(job, state='FAILED', error=repr(exc)))
        raise
    finally:
        await pool.close()


def partial_rows(root):
    rows = []
    for folder in sorted((root/'rounds').glob('round*')):
        r = int(folder.name[5:])
        for file in sorted((folder/'cells').glob('*/cell_statistics.json')):
            order_file = file.parent/'order.json'
            order = json.loads(order_file.read_text()) if order_file.exists() else {}
            for row in json.loads(file.read_text()):
                rows.append(dict(row, round=r, order_position=order.get('order_position'),
                                 execution_serial=order.get('serial')))
    return rows


def validated_phase_model(path, fingerprint):
    model = json.loads(path.read_text())
    if not fingerprint['hardware'].get('available') or not fingerprint['hardware'].get('devices'):
        raise ValueError('Hardware identity unavailable; cannot reuse calibration')
    if model.get('fingerprint') != fingerprint:
        raise ValueError('Saved phase model differs from current full fingerprint; omit --phase-model to recalibrate')
    return model


async def execute(a, cfg, root, plan):
    from aiohttp import ClientSession, ClientTimeout, TCPConnector
    from cache_calibration import calibrate
    from phase_router import PhaseModel
    from admission_telemetry import install
    common.PROTOCOL = PROTOCOL
    task = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
    fp = await common.preflight(cfg, root)
    fp['hardware'] = await base.hardware_fingerprint(cfg, root)
    common.dump(root/'fingerprint.json', fp)
    if fp['nh_files']['python/infinilm/llm/cache_manager.py'] != base.EXPECTED_SHA:
        raise RuntimeError('BlockManager changed; use the verified code identity')
    saved_model = validated_phase_model(a.phase_model, fp) if a.phase_model else None
    if saved_model is not None:
        PhaseModel(saved_model)
    if a.mode == 'check':
        if a.historical_run:
            copy_historical_workloads(a, cfg, root, fp)
        return dict(status='CHECK_COMPLETE', complete=True, model_pool_start_attempts=0,
                    gpu_benchmark_started=False, no_models_loaded=True)
    manifest = await prepare_workloads(a, cfg, root, fp)
    if saved_model is not None:
        model = saved_model
        common.dump(root/'phase_model.json', model)
    else:
        model = await calibrate(cfg, root, root/'workloads', fp)
    PhaseModel(model)
    install()
    cases = {(c['profile'], c['seed']): c for c in manifest['cells']}
    per_round_plan = base.make_plan(a, cfg)
    per_round_plan.pop('gpu_jobs_started', None)
    per_round_plan.update(protocol=PROTOCOL, fresh_pool_per_cell=True,
                          historical_relation=plan['historical_relation'])
    all_rows = []
    attempts = 0
    async with ClientSession(timeout=ClientTimeout(total=1800), connector=TCPConnector(limit=0, force_close=True), read_bufsize=2**20) as client:
        for job in plan['execution_order']:
            rr = root/'rounds'/f'round{job["round"]:02d}'
            if not rr.exists():
                rr.mkdir(parents=True)
                shutil.copytree(root/'workloads', rr/'workloads')
                common.dump(rr/'plan.json', dict(per_round_plan, round=job['round']))
            # No regeneration is permitted between arms or repeats.
            validate_manifest(rr/'workloads', manifest, a)
            attempts += 1
            common.dump(root/'execution_state.json', dict(model_pool_start_attempts=attempts,
                        completed_cells=len(all_rows)//3, current=job))
            common.progress(root, 'ORDERED_CELL_BEGIN', **job)
            rows = await execute_job(cfg, rr, cases[(job['profile'], job['seed'])],
                                     job, model, plan['router_options'], client)
            all_rows.extend(rows)
            round_rows = [r for r in all_rows if r['round'] == job['round']]
            base.report(rr, round_rows, per_round_plan)
            rt.csv_write(root/'summary.csv', all_rows)
            common.progress(root, 'ORDERED_CELL_COMPLETE', **job)
    expected = plan['expected_phase_rows']
    if len(all_rows) != expected or not all(r.get('valid') for r in all_rows):
        raise RuntimeError('Incomplete or invalid repeated comparison')
    return dict(status='EXPERIMENT_COMPLETE', complete=True,
                model_pool_start_attempts=attempts, completed_cells=attempts,
                valid_phase_rows=len(all_rows), expected_phase_rows=expected,
                order_balanced=plan['order_balanced'], evidence_kind=plan['evidence_kind'],
                performance_win_claimed=False,
                interpretation='Execution complete. Evaluate same-cell paired ratios and repeated outcomes; completion is not a performance win.')


def snapshot_sources(root):
    for relative in common.source_hashes():
        source = common.KIT/relative
        dest = root/'source'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def main(argv=None):
    a = arguments(argv)
    cfg = base.configuration(a.config)
    plan = make_plan(a, cfg)
    if a.mode == 'plan':
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    lock = open('/tmp/infinilm-fourway-'+str(os.getuid())+'.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('A benchmark already owns this lock. Do not delete it.', file=sys.stderr)
        lock.close()
        return 2
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root = a.output_root.resolve()/f'cache_confirm_{a.mode}_{stamp}'
    root.mkdir(parents=True)
    common.dump(root/'plan.json', plan)
    common.dump(root/'status.json', dict(status='RUNNING', complete=False))
    common.dump(root/'execution_order.json', plan['execution_order'])
    print('RUN_DIR='+str(root), flush=True)
    code = 0
    try:
        snapshot_sources(root)
        result = asyncio.run(execute(a, cfg, root, plan))
        common.dump(root/'status.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException as exc:
        common.dump(root/'status.json', dict(status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else 'FAILED',
                    complete=False, error=repr(exc), traceback=traceback.format_exc()))
        print('FAILED '+repr(exc)+'; evidence='+str(root), flush=True)
        code = 2
    finally:
        processing = dict(status='NOT_REQUIRED' if a.mode != 'run' else 'RUNNING',
                          complete=a.mode != 'run', scripts=[])
        try:
            rows = partial_rows(root)
            if rows:
                rt.csv_write(root/'summary.csv', rows)
            # Both programs are offline: they cannot start engines or change routing.
            for script in ['diagnose_confirmation.py', 'summarize_confirmation.py']:
                target = KIT/script
                if a.mode != 'run':
                    continue
                item = dict(script=script)
                processing['scripts'].append(item)
                if not target.is_file():
                    item.update(status='MISSING', complete=False)
                else:
                    try:
                        import subprocess
                        with (root/(target.stem+'.log')).open('w') as log:
                            completed = subprocess.run([sys.executable, '-B', str(target), str(root)],
                                                       stdout=log, stderr=subprocess.STDOUT)
                        item.update(status='COMPLETE' if completed.returncode == 0 else 'FAILED',
                                    complete=completed.returncode == 0, returncode=completed.returncode)
                    except Exception as exc:
                        item.update(status='ERROR', complete=False, error=repr(exc))
                if not item['complete']:
                    code = 2
                    print('POSTPROCESS_FAILED '+script+'; inspect '+str(root), flush=True)
            if a.mode == 'run':
                processing['complete'] = all(item['complete'] for item in processing['scripts'])
                processing['status'] = 'COMPLETE' if processing['complete'] else 'FAILED'
        except Exception as exc:
            common.dump(root/'postprocess_error.json', dict(error=repr(exc), traceback=traceback.format_exc()))
            processing.update(status='ERROR', complete=False, error=repr(exc))
            code = 2
            print('POSTPROCESS_FAILED '+repr(exc), flush=True)
        finally:
            common.dump(root/'postprocess_status.json', processing)
            lock.close()
            archive = shutil.make_archive(str(root)+'_report', 'zip', root.parent, root.name)
            print('REPORT_ARCHIVE='+archive, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
