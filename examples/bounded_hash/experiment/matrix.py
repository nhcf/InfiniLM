#!/usr/bin/env python3
"""Small Qwen3-32B screen: five methods x C16/C32/C64, one frozen workload.

Four model-pool launches suffice. NH resets only while idle; vLLM starts fresh at
each concurrency, without assuming that its older reset HTTP API checks success.
Every cell then receives identical independent cold/warm probes and a new router.
"""
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
import statistics
import sys
import traceback

import backend_runtime as rt
import harness as common
from cache_policy import EXPECTED_SHA
from gateway import Gateway
from stable_router import Router

KIT = Path(__file__).resolve().parent
ROOT = Path('/mnt/zj-data/data/chengcp/infinilm')
MODEL = '/mnt/zj-data/models/Qwen3-32B/'
PROTOCOL = 'qwen32-cache-affinity-c16-c32-c64-v1'
# name, engine, gateway policy, cache replacement policy
ARMS = [('nh_round_robin', 'nh', 'round_robin', 'native'),
        ('nh_least_active', 'nh', 'least_active', 'native'),
        ('nh_lru_least_active', 'nh', 'least_active', 'prefix_lru'),
        ('nh_lru_stable_bounded', 'nh', 'stable_bounded', 'prefix_lru'),
        ('vllm_least_active', 'vllm', 'least_active', 'vllm_native')]
CONCURRENCY = [16, 32, 64]
dump, progress = common.dump, common.progress


def configuration():
    old = ROOT/'qwen3_32b_fourway_20261021/config32b.json'
    cfg = json.loads(old.read_text()) if old.is_file() else json.loads((KIT/'config32b.json').read_text())
    if Path(cfg['model']).resolve() != Path(MODEL).resolve() or Path(cfg['repo']).resolve() != (ROOT/'InfiniLM').resolve():
        raise ValueError('Use the existing nhcf/InfiniLM environment and exact Qwen3-32B model path')
    cfg.update(model=MODEL, repo=str(ROOT/'InfiniLM'), prompt_tokens=8192, output_tokens=64,
               max_context=9216, max_batch=3, token_budget=8192, nh_blocks=128, nh_block_size=256,
               vllm_gpu_blocks=2048, vllm_block_size=16, admission_cap=3, skew=1.25,
               cache_ttl_s=5.0, queue_timeout_s=600.0)
    cfg['vllm_env'] = dict(cfg.get('vllm_env', {}))
    cfg['vllm_env'].pop('VLLM_SERVER_DEV_MODE', None)
    return cfg


def make_plan(cfg):
    return dict(protocol=PROTOCOL, config=cfg, arms=ARMS, concurrency=CONCURRENCY,
                model=MODEL, input_tokens=8192, shared_tokens=7680, output_tokens=64,
                keys=16, hot_fraction=.25, seed=33959, learn_requests=32, measured_requests=128,
                cells=15, total_learning_and_measured_requests=2400,
                primary_metric_phase='measured_128_including_ramp_and_drain',
                replicas=8, tp=1, max_worker_inflight=3, max_backend_inflight=24,
                client_concurrency_includes_gateway_queue=True,
                optimized_queue=dict(window=16,max_bypasses=8),
                ttft='client send to first nonempty content; includes gateway and backend wait',
                throughput='all exact successful output tokens / complete phase wall time; invalid phases have no throughput',
                baseline='Round-robin and least-active rules at a common full-replica gateway; same capacity guard',
                cache_reset='NH acknowledged idle reset; vLLM fresh processes per C; same independent probes after reset/start',
                model_pool_starts=4, probe_requests_per_cell=16,
                comparison_scope='Qwen3-32B MUSA BF16 TP1 eager, equal 32768 KV tokens per worker',
                qualification='screen only; paired throughput >=0.95 and TTFT p95 <=1.05 of vLLM at SAME concurrency',
                calibration=False, installs=False, rebuilds=False, performance_win_claimed=False)


async def reset_pool(client, pool, mode, folder):
    if pool.engine != 'nh':
        raise ValueError('Older vLLM reset HTTP 200 does not confirm reset; use fresh processes')
    async def worker(url):
        endpoint = '/_matrix/reset_cache'
        kwargs = {'json': {'mode': mode}}
        for attempt in range(60):
            async with client.post(url+endpoint, **kwargs) as response:
                raw = await response.text()
                try:
                    result = json.loads(raw) if raw.strip() else None
                except json.JSONDecodeError:
                    result = None
                success = response.status == 200 and isinstance(result, dict) and result.get('success') is True
                if success:
                    if result.get('cache_mode') != mode or result.get('resident_blocks') != 0:
                        raise RuntimeError('NH cache reset returned inconsistent state')
                    return dict(url=url, status=response.status, result=result)
                if response.status in (404, 405):
                    raise RuntimeError('Installed engine does not expose '+endpoint+'; no measurement was started')
                if response.status not in (200, 409):
                    raise RuntimeError('Cache reset failed: '+str(response.status)+' '+raw[:500])
            await asyncio.sleep(.5)
        raise RuntimeError('Engine did not acknowledge an idle prefix-cache reset')
    evidence = await asyncio.gather(*(worker(url) for url in pool.urls))
    dump(folder/'cache_reset.json', evidence)


def audit(router, gateway, records, routes):
    by_id = {r['id']:r for r in records}
    transparent = len({e.get('request_id') for e in routes}) == len(records) and all(
        e.get('request_id') in by_id and all(e.get(k) == by_id[e['request_id']].get(k)
        for k in ('event_payload_sha256', 'event_payload_count', 'event_payload_bytes')) for e in routes)
    state = router.snapshot()
    valid = (transparent and not gateway.errors and len(routes) == len(records)
             and not any(state['active']) and not any(state['kv_reserved_tokens'])
             and not state['queue_waiters']
             and all(e.get('success') and e.get('released') for e in routes))
    return valid, transparent, state


def statistics_extra(records, routes):
    known = [r for r in records if isinstance((r.get('usage', {}).get('prompt_tokens_details') or {}).get('cached_tokens'), int)]
    full = [r for r in known if r['usage']['prompt_tokens_details']['cached_tokens'] >= 7680]
    zero = [r for r in known if r['usage']['prompt_tokens_details']['cached_tokens'] == 0]
    return dict(cache_counter_requests=len(known), full_prefix_hit_requests=len(full), zero_cache_requests=len(zero),
                full_prefix_hit_fraction=len(full)/len(records) if len(known) == len(records) else None,
                full_hit_ttft_mean_ms=statistics.mean(r['ttft_ms'] for r in full) if full else None,
                zero_hit_ttft_mean_ms=statistics.mean(r['ttft_ms'] for r in zero) if zero else None,
                client_postfirst_mean_ms=statistics.mean(r['e2e_ms']-r['ttft_ms'] for r in records if r.get('success')) if any(r.get('success') for r in records) else None,
                gateway_wait_p95_ms=rt.quantile([e['gateway_wait_ms'] for e in routes], .95),
                cache_hint_overestimates=sum(bool(e.get('hint_overestimated')) for e in routes),
                spills=sum(bool(e.get('spill')) for e in routes))


async def measure_with_heartbeat(pool, router, root, label, phase, *args):
    task=asyncio.create_task(common.measure(*args))
    try:
        while True:
            done,_=await asyncio.wait({task},timeout=20)
            if done:
                return task.result()
            pool.assert_alive()
            progress(root,'PHASE_HEARTBEAT',cell=label,phase=phase,
                     active=sum(w.active_load for w in router.workers),queued=len(router._waiters),
                     released=sum(bool(e.get('released')) for e in router.history))
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)


async def run_cell(cfg, root, pool, case, concurrency, arm, client):
    name, engine, policy, mode = arm
    label = f'c{concurrency}_{name}'
    folder = root/'cells'/label
    folder.mkdir(parents=True)
    dump(folder/'status.json', dict(status='RUNNING'))
    progress(root, 'CELL_BEGIN', cell=label, cache_mode=mode)
    gateway = None
    phases, all_records, durations = [], [], []
    try:
        if engine == 'nh':
            await reset_pool(client, pool, mode, folder)
        else:
            dump(folder/'cache_reset.json', dict(method='fresh_processes', pool=str(pool.folder)))
        probes = json.loads((root/'workloads'/case['folder']/'smoke.json').read_text())
        await common.smoke_pool(client, pool, probes, folder/'smoke.json')
        if engine == 'nh':
            probe_records = json.loads((folder/'smoke.json').read_text())
            if any(pair[0]['usage']['prompt_tokens_details'].get('cached_tokens') != 0 for pair in probe_records):
                raise RuntimeError('NH reset failed the independent cold-probe check')
        router = Router(pool.urls, policy, max_inflight=3, kv_tokens=32768, page_size=256,
                        skew=1.25, cache_ttl_s=5.0, queue_timeout_s=cfg['queue_timeout_s'])
        gateway = Gateway(pool.urls, pool.models, router, cfg['port'], engine)
        await gateway.start()
        for phase, filename in [('learn', 'learn.jsonl'), ('measured', 'steady.jsonl')]:
            items = rt.rows(root/'workloads'/case['folder']/filename)
            begin = len(router.history)
            progress(root, 'PHASE_BEGIN', cell=label, phase=phase, requests=len(items))
            records, elapsed = await measure_with_heartbeat(pool,router,root,label,phase,
                client,f'http://127.0.0.1:{cfg["port"]}','gateway',items,concurrency,root,label,phase)
            routes = router.history[begin:]
            route_valid, transparent, state = audit(router, gateway, records, routes)
            mode_valid = engine != 'nh' or all(r.get('engine_telemetry',{}).get('cache_policy') == mode for r in records)
            meta = dict(arm=name, seed=case['seed'], concurrency=concurrency, phase=phase,
                        cache_policy=mode, gateway_policy=policy, hot_fraction=.25)
            row = common.phase_stats(records, elapsed, meta)
            row.update(statistics_extra(records, routes), route_valid=route_valid,
                       event_payload_transparent=transparent, pool=str(pool.folder))
            row['cache_policy_valid'] = mode_valid
            row['valid'] = row['valid'] and route_valid and mode_valid
            if not row['valid']:
                row['output_tok_s'] = row['request_s'] = None
            dump(folder/(phase+'_requests.json'), records)
            dump(folder/(phase+'_routes.json'), state)
            dump(folder/'gateway_errors.json', gateway.errors)
            phases.append(row)
            dump(folder/'cell_statistics.json', phases)
            rt.csv_write(folder/'summary.csv', phases)
            pool.assert_alive()
            if not row['valid']:
                raise RuntimeError('Invalid request/stream/accounting cohort: '+label)
            all_records.extend(records); durations.append(elapsed)
            progress(root, 'PHASE_END', cell=label, phase=phase, output_tok_s=row['output_tok_s'],
                     ttft_p95_ms=row['ttft_p95_ms'], full_hit_fraction=row['full_prefix_hit_fraction'],
                     queue_p95_ms=row['gateway_wait_p95_ms'])
        combined = common.phase_stats(all_records, sum(durations), dict(meta, phase='learn_plus_measured'))
        combined.update(route_valid=True, event_payload_transparent=True)
        phases.append(combined); rt.csv_write(folder/'summary.csv', phases)
        dump(folder/'cell_statistics.json', phases)
        dump(folder/'status.json', dict(status='PASS'))
        return phases
    except BaseException as exc:
        dump(folder/'status.json', dict(status='FAILED', error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if gateway:
            await gateway.close()


def paired_rows(rows):
    # Concurrency MUST be part of the pairing key; never mix C16/32/64.
    lookup = {(r['arm'],r['seed'],r['concurrency'],r['phase']):r for r in rows if r.get('valid')}
    pairs = []
    for (arm, seed, concurrency, phase), row in lookup.items():
        refs = ['vllm_least_active'] if arm != 'vllm_least_active' else []
        if arm == 'nh_lru_stable_bounded':
            refs += ['nh_lru_least_active', 'nh_least_active', 'nh_round_robin']
        if arm == 'nh_lru_least_active':
            refs += ['nh_least_active']
        for baseline in refs:
            other = lookup.get((baseline,seed,concurrency,phase))
            if other:
                tr = row['output_tok_s']/other['output_tok_s']
                pr = row['ttft_p95_ms']/other['ttft_p95_ms']
                pairs.append(dict(candidate=arm, baseline=baseline, concurrency=concurrency,
                                  seed=seed, phase=phase, throughput_ratio=tr, ttft_p95_ratio=pr,
                                  within_5_percent=tr>=.95 and pr<=1.05))
    return pairs


def report(root, rows):
    rt.csv_write(root/'summary.csv', rows)
    pairs = paired_rows(rows); rt.csv_write(root/'paired.csv', pairs)
    complete = {(r['arm'],r['concurrency']) for r in rows if r.get('valid') and r['phase']=='measured'} == {
        (a[0], c) for a in ARMS for c in CONCURRENCY}
    relevant = [p for p in pairs if p['candidate']=='nh_lru_stable_bounded' and p['baseline']=='vllm_least_active' and p['phase']=='measured']
    result = dict(complete=complete, per_concurrency=relevant, screen_only=True,
                  all_three_within_5_percent=complete and len(relevant)==3 and all(p['within_5_percent'] for p in relevant),
                  performance_win_claimed=False)
    dump(root/'qualification.json', result)
    lines = ['# Qwen3-32B C16/C32/C64 small screen', '',
             'Model: '+MODEL, '', '128 measured requests per cell; 32 separately reported learning requests. One seed; screening evidence, not a confidence interval.', '',
             'Round-robin is reproduced at the common capacity-guarded gateway. Native cache mode delegates the existing NH allocator. LRU arms use a process-local replacement policy.', '',
             'C32/C64 include gateway queueing above 24 backend slots; all queue time stays in TTFT.', '',
             '| Method | C | tok/s | TTFT mean ms | TTFT p95 ms | full-prefix hit fraction | queue p95 ms |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['phase']=='measured':
            def fmt(k):
                v=r.get(k); return f'{v:.3f}' if isinstance(v,(int,float)) else 'NA'
            lines.append(f'| {r["arm"]} | {r["concurrency"]} | {fmt("output_tok_s")} | {fmt("ttft_mean_ms")} | {fmt("ttft_p95_ms")} | {fmt("full_prefix_hit_fraction")} | {fmt("gateway_wait_p95_ms")} |')
    lines += ['', 'See paired.csv for ratios at matched concurrency. Missing vLLM cache counters remain unknown. Invalid cells have no comparable throughput.', '',
              'Separate attribution: NH LRU least-active / NH native least-active measures cache-policy changes; NH LRU stable bounded / NH LRU least-active measures routing changes.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return result


async def run(cfg, root):
    from aiohttp import ClientSession, ClientTimeout, TCPConnector
    common.PROTOCOL = PROTOCOL
    task = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
    fp = await common.preflight(cfg, root)
    if fp['nh_files']['python/infinilm/llm/cache_manager.py'] != EXPECTED_SHA:
        raise RuntimeError('Cache manager changed since the uploaded run; no GPU job started')
    opts = dict(seeds=[33959], requests=128, keys=16, prompt_tokens=8192,
                output_tokens=64, shared_tokens=7680, learn_repeats=2, calibration=False)
    await common.run_logged([cfg['nh_python'], str(KIT/'workload.py'), '--model', MODEL,
                            '--out', str(root/'workloads'), '--options', json.dumps(opts)],
                           rt.backend_env(cfg,'nh',cfg['gpu_ids'][0],'/tmp'), cfg['repo'],
                           root/'workload_build.log', timeout=900)
    case = json.loads((root/'workloads/manifest.json').read_text())['cells'][0]
    rows = []
    try:
        pool_specs = [('nh', CONCURRENCY)] + [('vllm', [c]) for c in CONCURRENCY]
        for engine, levels in pool_specs:
            pool = rt.Pool(cfg, engine, root/'pools'/(engine+'_'+str(levels[0])))
            try:
                await pool.start()
                async with ClientSession(timeout=ClientTimeout(total=1800), connector=TCPConnector(limit=0), read_bufsize=2**20) as client:
                    arms = [arm for arm in ARMS if arm[1]==engine]
                    for index, concurrency in enumerate(levels):
                        shift=index%len(arms)
                        for arm in arms[shift:]+arms[:shift]:
                            rows += await run_cell(cfg,root,pool,case,concurrency,arm,client)
                            report(root,rows)
            finally:
                await pool.close()
    finally:
        # Include invalid/partial cell statistics rather than discarding them.
        rows = [r for path in sorted((root/'cells').glob('*/cell_statistics.json'))
                for r in json.loads(path.read_text())]
        report(root,rows)
    result=report(root,rows)
    if not result['complete']:
        raise RuntimeError('Some matrix cells are missing')
    return result


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['plan','run'],nargs='?',default='plan')
    a=ap.parse_args(argv);cfg=configuration();plan=make_plan(cfg)
    if a.mode=='plan':
        print(json.dumps(plan,ensure_ascii=False,indent=2));print('GPU_JOBS_STARTED=0');return 0
    if not KIT.is_relative_to(ROOT):
        raise SystemExit('Extract this kit under '+str(ROOT))
    lock=open(f'/tmp/infinilm-fourway-{os.getuid()}.lock','a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('An existing benchmark owns the lock. Do not delete the lock or start duplicate GPU jobs.',file=sys.stderr)
        return 2
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root=KIT/'results32b'/('matrix_'+stamp);root.mkdir(parents=True)
    dump(root/'plan.json',plan);dump(root/'status.json',dict(status='RUNNING'))
    source=root/'source';source.mkdir()
    for relative in common.source_hashes():
        dest=source/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(KIT/relative,dest)
    print('RUN_DIR='+str(root),flush=True)
    try:
        result=asyncio.run(run(cfg,root))
        dump(root/'status.json',dict(status='COMPLETE',**result))
        print('MATRIX_COMPLETE '+json.dumps(result),flush=True)
        code=0
    except BaseException as exc:
        dump(root/'status.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc()))
        print('FAILED: '+repr(exc)+'; evidence='+str(root),file=sys.stderr,flush=True)
        code=2
    finally:
        lock.close()
        archive=shutil.make_archive(str(root)+'_report','zip',root.parent,root.name)
        print('REPORT_ARCHIVE='+archive,flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
