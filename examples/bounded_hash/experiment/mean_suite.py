#!/usr/bin/env python3
"""Qwen3-32B declared long-prefix mean-TTFT screen; no GPU result is assumed.

Every engine sees identical complete requests. Fixed independent probes verify
cache availability; equal learning is separately reported and also included in
combined results. C32/C64 include gateway queuing above 24 backend slots.
"""
import argparse
import asyncio
from collections import defaultdict
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
import matrix as previous
from cache_policy import EXPECTED_SHA
from gateway import Gateway

KIT = Path(__file__).resolve().parent
ROOT = Path('/mnt/zj-data/data/chengcp/infinilm')
MODEL = '/mnt/zj-data/models/Qwen3-32B/'
PROTOCOL = 'qwen32-mean-affinity-v1'
ARMS = [('nh_round_robin', 'nh', 'round_robin', 'native'),
        ('nh_lru_least_active', 'nh', 'least_active', 'prefix_lru'),
        ('nh_lru_stable_bounded', 'nh', 'stable_bounded', 'prefix_lru'),
        ('nh_lru_mean_bounded', 'nh', 'mean_bounded', 'prefix_lru'),
        ('vllm_least_active', 'vllm', 'least_active', 'vllm_native')]
CANDIDATE = 'nh_lru_mean_bounded'
PROFILES = {'prefix97': 7936, 'replay94': 7680}
MEASURED = {16: 256, 32: 128, 64: 128}
PHASES = ('learn', 'measured', 'learn_plus_measured')
dump, progress = common.dump, common.progress


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['plan', 'run'], nargs='?', default='plan')
    ap.add_argument('--profile', choices=list(PROFILES), default='prefix97')
    ap.add_argument('--concurrency', type=int, choices=[16,32,64], nargs='+', default=[16,32,64])
    ap.add_argument('--seeds', type=int, nargs='+', default=[33959])
    args = ap.parse_args(argv)
    if len(set(args.concurrency)) != len(args.concurrency) or len(set(args.seeds)) != len(args.seeds):
        ap.error('concurrency and seeds must not contain duplicates')
    if any(seed < 0 for seed in args.seeds):
        ap.error('seeds must be nonnegative')
    return args


def configuration():
    cfg = previous.configuration()
    # Keep the proven Qwen3-32B model, eager kernels and full-sequence capacity.
    if cfg['model'] != MODEL or cfg['repo'] != str(ROOT/'InfiniLM'):
        raise ValueError('Use the existing nhcf/InfiniLM checkout and exact Qwen3-32B model path')
    if len(cfg.get('gpu_ids', [])) != 8 or len(set(cfg['gpu_ids'])) != 8:
        raise ValueError('This comparison requires eight distinct full-replica GPUs')
    return cfg


def make_plan(args, cfg):
    counts = {str(c):MEASURED[c] for c in args.concurrency}
    return dict(protocol=PROTOCOL, config=cfg, profile=args.profile, arms=ARMS,
                candidate=CANDIDATE, concurrency=args.concurrency, seeds=args.seeds,
                model=MODEL, input_tokens=8192, shared_tokens=PROFILES[args.profile], output_tokens=64,
                keys=16, hot_fraction=.25, learn_requests=64, measured_requests_by_concurrency=counts,
                cells=len(ARMS)*len(args.concurrency)*len(args.seeds),
                total_learning_and_measured_requests=len(ARMS)*len(args.seeds)*sum(64+MEASURED[c] for c in args.concurrency),
                primary_metric_phase='measured_after_equal_64_request_learning_including_ramp_and_drain',
                replicas=8, tp=1, max_worker_inflight=3, max_backend_inflight=24,
                client_concurrency_includes_gateway_queue=True,
                optimized_queue=dict(window=16,max_bypasses=8),
                ttft='client send to first nonempty content; includes gateway and backend wait; smaller is better',
                throughput='all exact successful output tokens / complete phase wall time; invalid phases have no throughput',
                baseline='Round-robin and least-active rules at a common full-replica gateway with the same capacity guard',
                cache_reset='NH acknowledged idle reset; vLLM fresh processes per C/seed; same independent probes after reset/start',
                model_pool_starts=1+len(args.concurrency)*len(args.seeds), probe_requests_per_cell=16,
                comparison_scope='Qwen3-32B MUSA BF16 TP1 eager, equal 32768 KV tokens per worker; cache on, NH flash-attn',
                workload_scope='declared synthetic long shared prefix / short output / hot25 scenario; not a general serving benchmark',
                primary_criterion='mean TTFT ratio < 1.0 and throughput ratio >= 0.95 of vLLM at SAME profile, seed and concurrency',
                secondary_metrics=['TTFT p95','TTFT p99','requests/s','learn_plus_measured','cache hit fraction'],
                calibration=False, installs=False, rebuilds=False, screen_only=True,
                seed_scope='33959 is a development seed already used previously; new seeds are confirmation only if not used for tuning')


def stats_extra(records, routes, shared_tokens):
    known = [r for r in records if isinstance((r.get('usage',{}).get('prompt_tokens_details') or {}).get('cached_tokens'),int)]
    full = [r for r in known if r['usage']['prompt_tokens_details']['cached_tokens'] >= shared_tokens]
    zero = [r for r in known if r['usage']['prompt_tokens_details']['cached_tokens'] == 0]
    succeeded = [r for r in records if r.get('success')]
    return dict(cache_counter_requests=len(known), full_prefix_hit_requests=len(full),zero_cache_requests=len(zero),
                full_prefix_hit_fraction=len(full)/len(records) if records and len(known)==len(records) else None,
                full_hit_ttft_mean_ms=statistics.mean(r['ttft_ms'] for r in full) if full else None,
                zero_hit_ttft_mean_ms=statistics.mean(r['ttft_ms'] for r in zero) if zero else None,
                client_postfirst_mean_ms=statistics.mean(r['e2e_ms']-r['ttft_ms'] for r in succeeded) if succeeded else None,
                gateway_wait_mean_ms=statistics.mean(e['gateway_wait_ms'] for e in routes) if routes else None,
                gateway_wait_p95_ms=rt.quantile([e['gateway_wait_ms'] for e in routes],.95),
                cache_hint_overestimates=sum(bool(e.get('hint_overestimated')) for e in routes),
                spills=sum(bool(e.get('spill')) for e in routes))


def workload_selection(items):
    """Fingerprint exact ordered subset, including prompt and output contracts."""
    values=[{k:item[k] for k in ('id','prompt_sha256','prompt_tokens','shared_tokens','output_tokens')} for item in items]
    payload=json.dumps(values,sort_keys=True,separators=(',',':')).encode()
    return dict(requests=len(items),ordered_request_contract_sha256=hashlib.sha256(payload).hexdigest(),
                first_request_id=items[0]['id'] if items else None,last_request_id=items[-1]['id'] if items else None)


async def run_cell(cfg,root,pool,case,concurrency,arm,client,plan):
    from mean_router import Router
    name,engine,policy,mode=arm
    label=f'{plan["profile"]}_s{case["seed"]}_c{concurrency}_{name}'
    folder=root/'cells'/label
    folder.mkdir(parents=True)
    dump(folder/'status.json',dict(status='RUNNING'))
    progress(root,'CELL_BEGIN',cell=label,cache_mode=mode,primary_metric='mean_TTFT')
    gateway=None
    phases,all_records,all_routes,all_items,durations=[],[],[],[],[]
    try:
        if engine=='nh':
            await previous.reset_pool(client,pool,mode,folder)
        else:
            dump(folder/'cache_reset.json',dict(method='fresh_processes',pool=str(pool.folder)))
        work=root/'workloads'/case['folder']
        probes=json.loads((work/'smoke.json').read_text())
        await common.smoke_pool(client,pool,probes,folder/'smoke.json')
        if engine=='nh':
            probe_records=json.loads((folder/'smoke.json').read_text())
            if any(pair[0]['usage']['prompt_tokens_details'].get('cached_tokens')!=0 for pair in probe_records):
                raise RuntimeError('NH reset failed the independent cold-probe check')
        router=Router(pool.urls,policy,max_inflight=3,kv_tokens=32768,page_size=256,
                      skew=1.25,cache_ttl_s=5.0,queue_timeout_s=cfg['queue_timeout_s'])
        gateway=Gateway(pool.urls,pool.models,router,cfg['port'],engine)
        await gateway.start()
        selection={}
        for phase,filename,count in [('learn','learn.jsonl',64),('measured','steady.jsonl',MEASURED[concurrency])]:
            source=work/filename
            items=rt.rows(source)[:count]
            if len(items)!=count:
                raise RuntimeError('Workload is shorter than the declared cohort')
            if any((i['prompt_tokens'],i['shared_tokens'],i['output_tokens'])!=(8192,plan['shared_tokens'],64) for i in items):
                raise RuntimeError('Workload does not match declared model input/output profile')
            selection[phase]=dict(source=str(source),source_sha256=rt.sha(source),**workload_selection(items))
            dump(folder/'workload_selection.json',selection)
            begin=len(router.history)
            progress(root,'PHASE_BEGIN',cell=label,phase=phase,requests=len(items))
            records,elapsed=await previous.measure_with_heartbeat(pool,router,root,label,phase,
                client,f'http://127.0.0.1:{cfg["port"]}','gateway',items,concurrency,root,label,phase)
            routes=router.history[begin:]
            valid_routes,transparent,state=previous.audit(router,gateway,records,routes)
            mode_valid=engine!='nh' or all(r.get('engine_telemetry',{}).get('cache_policy')==mode for r in records)
            meta=dict(arm=name,seed=case['seed'],concurrency=concurrency,phase=phase,profile=plan['profile'],
                      cache_policy=mode,gateway_policy=policy,hot_fraction=.25,shared_tokens=plan['shared_tokens'])
            expected_ids=[i['id'] for i in items]
            actual_ids=[r.get('id') for r in records]
            cohort_count_valid=(len(records)==len(items) and len(set(expected_ids))==len(items)
                                and len(set(actual_ids))==len(records) and set(actual_ids)==set(expected_ids))
            cohort_contract_valid=all(r.get('usage',{}).get('prompt_tokens')==8192
                                      and r.get('usage',{}).get('completion_tokens')==64 for r in records)
            row=common.phase_stats(records,elapsed,meta)
            row.update(stats_extra(records,routes,plan['shared_tokens']),route_valid=valid_routes,
                       event_payload_transparent=transparent,pool=str(pool.folder),cache_policy_valid=mode_valid,
                       ordered_request_contract_sha256=selection[phase]['ordered_request_contract_sha256'],
                       cohort_count_valid=cohort_count_valid,cohort_contract_valid=cohort_contract_valid)
            row['valid']=row['valid'] and valid_routes and mode_valid and cohort_count_valid and cohort_contract_valid
            if not row['valid']:row['output_tok_s']=row['request_s']=None
            dump(folder/(phase+'_requests.json'),records)
            dump(folder/(phase+'_routes.json'),state)
            dump(folder/'gateway_errors.json',gateway.errors)
            phases.append(row)
            dump(folder/'cell_statistics.json',phases)
            rt.csv_write(folder/'summary.csv',phases)
            pool.assert_alive()
            if not row['valid']:raise RuntimeError('Invalid request/stream/accounting cohort: '+label)
            all_records.extend(records);all_routes.extend(routes);all_items.extend(items);durations.append(elapsed)
            progress(root,'PHASE_END',cell=label,phase=phase,output_tok_s=row['output_tok_s'],
                     ttft_mean_ms=row['ttft_mean_ms'],ttft_p95_ms=row['ttft_p95_ms'],full_hit_fraction=row['full_prefix_hit_fraction'])
        combined=common.phase_stats(all_records,sum(durations),dict(meta,phase='learn_plus_measured'))
        combined.update(stats_extra(all_records,all_routes,plan['shared_tokens']),route_valid=True,
                        event_payload_transparent=True,cache_policy_valid=True,pool=str(pool.folder),
                        cohort_count_valid=True,cohort_contract_valid=True,
                        ordered_request_contract_sha256=workload_selection(all_items)['ordered_request_contract_sha256'],
                        wall_definition='sum of timed learn and measured phases; excludes intervening artifact writes')
        phases.append(combined)
        rt.csv_write(folder/'summary.csv',phases);dump(folder/'cell_statistics.json',phases)
        dump(folder/'status.json',dict(status='PASS'))
        return phases
    except BaseException as exc:
        dump(folder/'status.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        if gateway:await gateway.close()


def paired_rows(rows):
    lookup={}
    for row in rows:
        if not row.get('valid'):continue
        key=tuple(row[k] for k in ('arm','profile','seed','concurrency','phase'))
        if key in lookup:raise ValueError('Duplicate cell/phase statistics: '+str(key))
        lookup[key]=row
    pairs=[]
    for (arm,profile,seed,concurrency,phase),row in lookup.items():
        refs=['vllm_least_active'] if arm!='vllm_least_active' else []
        if arm in ('nh_lru_stable_bounded',CANDIDATE):refs+=['nh_lru_least_active','nh_round_robin']
        if arm==CANDIDATE:refs+=['nh_lru_stable_bounded']
        for baseline in refs:
            other=lookup.get((baseline,profile,seed,concurrency,phase))
            if other is None:continue
            # Exact workload-subset hashes must match, including the frozen order.
            ah,bh=row.get('ordered_request_contract_sha256'),other.get('ordered_request_contract_sha256')
            if ah is not None and bh is not None and ah!=bh:
                raise ValueError('Paired cells used different ordered workload contracts')
            tr=row['output_tok_s']/other['output_tok_s']
            mr=row['ttft_mean_ms']/other['ttft_mean_ms']
            pr=row['ttft_p95_ms']/other['ttft_p95_ms']
            pairs.append(dict(candidate=arm,baseline=baseline,profile=profile,concurrency=concurrency,seed=seed,phase=phase,
                              throughput_ratio=tr,mean_ttft_ratio=mr,ttft_p95_ratio=pr,
                              throughput_gain_pct=100*(tr-1),mean_ttft_reduction_pct=100*(1-mr),
                              mean_ttft_faster=mr<1.0,throughput_within_5_percent=tr>=.95,
                              throughput_faster=tr>1.0,mean_goal_met=mr<1.0 and tr>=.95,
                              historical_p95_goal_met=tr>=.95 and pr<=1.05))
    return pairs


def report(root,rows,plan):
    rt.csv_write(root/'summary.csv',rows)
    pairs=paired_rows(rows);rt.csv_write(root/'paired.csv',pairs)
    expected={(a[0],plan['profile'],s,c,p) for a in plan['arms'] for s in plan['seeds'] for c in plan['concurrency'] for p in PHASES}
    valid={tuple(r[k] for k in ('arm','profile','seed','concurrency','phase')) for r in rows if r.get('valid')}
    complete=valid==expected
    candidates=[p for p in pairs if p['candidate']==CANDIDATE and p['baseline']=='vllm_least_active' and p['phase']=='measured']
    expected_comparisons=len(plan['seeds'])*len(plan['concurrency'])
    candidate_results={}
    for name in ('nh_lru_stable_bounded',CANDIDATE):
        measured=[p for p in pairs if p['candidate']==name and p['baseline']=='vllm_least_active' and p['phase']=='measured']
        candidate_results[name]=dict(per_seed_concurrency=measured,
            all_requested_mean_goals_met=complete and len(measured)==expected_comparisons and all(p['mean_goal_met'] for p in measured))
    result=dict(complete=complete,primary_metric='mean_TTFT_lower_is_better',predeclared_candidate=CANDIDATE,
                per_seed_concurrency=candidates,candidate_results=candidate_results,screen_only=True,
                all_requested_mean_goals_met=complete and len(candidates)==expected_comparisons and all(p['mean_goal_met'] for p in candidates),
                statistical_significance_claimed=False,tail_latency_is_diagnostic=True)
    dump(root/'qualification.json',result)
    grouped=defaultdict(list)
    for p in pairs:grouped[(p['candidate'],p['baseline'],p['profile'],p['concurrency'],p['phase'])].append(p)
    aggregate=[]
    for key,values in sorted(grouped.items()):
        r=dict(zip(('candidate','baseline','profile','concurrency','phase'),key));r['rounds']=len(values)
        for metric in ('throughput_ratio','mean_ttft_ratio','ttft_p95_ratio'):
            vals=[p[metric] for p in values]
            r.update({metric+'_median':statistics.median(vals),metric+'_min':min(vals),metric+'_max':max(vals)})
        r['all_rounds_mean_goal_met']=all(p['mean_goal_met'] for p in values);aggregate.append(r)
    rt.csv_write(root/'paired_aggregate.csv',aggregate)
    lines=['# Qwen3-32B mean-TTFT affinity screen','',
           'Model: '+MODEL,'',
           f'Profile {plan["profile"]}: input8192/shared{plan["shared_tokens"]}/output64; 16 documents; hot25. Primary: LOWER mean TTFT and output throughput >=95% of same-cell vLLM.',
           'Each cell has 64 separately timed learning requests. C16 has 256 measured requests; C32/C64 have 128. All phase times include ramp/drain. Learning plus measured totals are also reported.',
           'This is a selected synthetic workload, not a broad benchmark. Each seed is retained; a single seed is screening evidence, not a confidence interval.',
           'RR is reproduced at a common capacity-guarded gateway. All engines have 24 backend slots; C32/C64 include gateway queueing in TTFT.',
           '','| Method | Seed | C | tok/s | requests/s | Mean TTFT ms | p95 ms | p99 ms | Full hit fraction |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in sorted(rows,key=lambda r:(r['seed'],r['concurrency'],r['arm'],r['phase'])):
        if row['phase']!='measured':continue
        def fmt(k):
            v=row.get(k);return f'{v:.3f}' if isinstance(v,(int,float)) else 'NA'
        lines.append(f'| {row["arm"]} | {row["seed"]} | {row["concurrency"]} | {fmt("output_tok_s")} | {fmt("request_s")} | {fmt("ttft_mean_ms")} | {fmt("ttft_p95_ms")} | {fmt("ttft_p99_ms")} | {fmt("full_prefix_hit_fraction")} |')
    lines+=['','Paired comparisons: paired.csv; across-seed median/min/max: paired_aggregate.csv. Primary comparisons never mix profiles, seeds, concurrency, or phases.',
            'Use LRU hash / LRU least-active to isolate the routing plus queue-policy contribution. Hash / native RR combines cache replacement and routing changes. Old stable / new mean hash isolates the new policy.',
            'Missing vLLM cached-token counters stay unknown. Unsuccessful cohorts have no comparable throughput. Tail metrics are retained even when the mean criterion is met.',
            '','## Learning-inclusive comparison','', '| Method | Seed | C | tok/s | Mean TTFT ms | p95 ms |','|---|---:|---:|---:|---:|---:|']
    for row in sorted(rows,key=lambda r:(r['seed'],r['concurrency'],r['arm'],r['phase'])):
        if row['phase']=='learn_plus_measured':
            vals=[row.get(k) for k in ('output_tok_s','ttft_mean_ms','ttft_p95_ms')]
            text=['NA' if not isinstance(v,(int,float)) else f'{v:.3f}' for v in vals]
            lines.append(f'| {row["arm"]} | {row["seed"]} | {row["concurrency"]} | '+ ' | '.join(text)+' |')
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return result


async def run(cfg,root,plan):
    from aiohttp import ClientSession,ClientTimeout,TCPConnector
    common.PROTOCOL=PROTOCOL
    task=asyncio.current_task()
    for sig in (signal.SIGINT,signal.SIGTERM):asyncio.get_running_loop().add_signal_handler(sig,task.cancel)
    fp=await common.preflight(cfg,root)
    if fp['nh_files']['python/infinilm/llm/cache_manager.py']!=EXPECTED_SHA:
        raise RuntimeError('Cache manager changed since the verified run; no GPU job started')
    opts=dict(seeds=plan['seeds'],requests=max(MEASURED[c] for c in plan['concurrency']),keys=16,
              prompt_tokens=8192,output_tokens=64,shared_tokens=plan['shared_tokens'],learn_repeats=4,calibration=False)
    progress(root,'WORKLOAD_BUILD_BEGIN',options=opts)
    await common.run_logged([cfg['nh_python'],str(KIT/'mean_workload.py'),'--model',MODEL,
                            '--out',str(root/'workloads'),'--options',json.dumps(opts)],
                           rt.backend_env(cfg,'nh',cfg['gpu_ids'][0],'/tmp'),cfg['repo'],root/'workload_build.log',timeout=900)
    manifest=json.loads((root/'workloads/manifest.json').read_text())
    cases=manifest['cells']
    if [c['seed'] for c in cases]!=plan['seeds']:raise RuntimeError('Workload seeds differ from plan')
    progress(root,'WORKLOAD_BUILD_PASS',manifest_sha256=rt.sha(root/'workloads/manifest.json'))
    rows=[]
    try:
        # Reuse NH weights, switching cache mode only after idle acknowledgement.
        # Fresh vLLM processes avoid assuming an unchecked reset HTTP 200 means success.
        specs=[('nh',cases,plan['concurrency'])]
        specs += [('vllm',[case],[c]) for case in cases for c in plan['concurrency']]
        for pool_index,(engine,pool_cases,levels) in enumerate(specs):
            pool=rt.Pool(cfg,engine,root/'pools'/f'{pool_index:02d}_{engine}')
            try:
                await pool.start()
                async with ClientSession(timeout=ClientTimeout(total=1800),connector=TCPConnector(limit=0),read_bufsize=2**20) as client:
                    arms=[a for a in plan['arms'] if a[1]==engine]
                    for seed_index,case in enumerate(pool_cases):
                        for level_index,c in enumerate(levels):
                            shift=(seed_index+level_index)%len(arms)
                            for arm in arms[shift:]+arms[:shift]:
                                rows += await run_cell(cfg,root,pool,case,c,arm,client,plan)
                                report(root,rows,plan)
            finally:await pool.close()
    finally:
        rows=[r for path in sorted((root/'cells').glob('*/cell_statistics.json')) for r in json.loads(path.read_text())]
        report(root,rows,plan)
    result=report(root,rows,plan)
    if not result['complete']:raise RuntimeError('Some requested cells or phases are missing')
    return result


def main(argv=None):
    args=parse_args(argv);cfg=configuration();plan=make_plan(args,cfg)
    if args.mode=='plan':
        print(json.dumps(plan,ensure_ascii=False,indent=2));print('GPU_JOBS_STARTED=0');return 0
    if not KIT.is_relative_to(ROOT):raise SystemExit('Extract this kit under '+str(ROOT))
    lock=open(f'/tmp/infinilm-fourway-{os.getuid()}.lock','a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();print('An existing benchmark owns the lock. Do not delete the lock or start duplicate GPU jobs.',file=sys.stderr);return 2
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root=KIT/'results32b'/('mean_'+args.profile+'_'+stamp);root.mkdir(parents=True)
    dump(root/'plan.json',plan);dump(root/'status.json',dict(status='RUNNING'))
    source=root/'source';source.mkdir()
    for relative in common.source_hashes():
        dest=source/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(KIT/relative,dest)
    print('RUN_DIR='+str(root),flush=True)
    try:
        result=asyncio.run(run(cfg,root,plan))
        dump(root/'status.json',dict(status='COMPLETE',**result));code=0
        print('MEAN_SUITE_COMPLETE '+json.dumps(result,ensure_ascii=False),flush=True)
        print((root/'report.md').read_text(),flush=True)
    except BaseException as exc:
        dump(root/'status.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc()));code=2
        print('FAILED: '+repr(exc)+'; evidence='+str(root),file=sys.stderr,flush=True)
    finally:
        lock.close()
        archive=shutil.make_archive(str(root)+'_report','zip',root.parent,root.name)
        print('REPORT_ARCHIVE='+archive,flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
