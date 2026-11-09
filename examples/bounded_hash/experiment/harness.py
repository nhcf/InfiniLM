#!/usr/bin/env python3
"""Frozen long-prefix screen, followed by two-seed holdout confirmation.

GPU execution occurs only in screen/screen16/confirm, on the user's host.
No measured winner is assumed. Calibration and probe requests are excluded
from performance timing; learning and steady results are reported separately.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import defaultdict
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import sys
import time
import traceback
import backend_runtime as rt

KIT = Path(__file__).resolve().parent
ARMS = [('nh_least_active','nh','least_active'), ('nh_bounded','nh','bounded'),
        ('nh_cost_bounded','nh','cost_bounded'), ('vllm_least_active','vllm','least_active')]
HOLDOUT_SEEDS = [133962,233965]
PROTOCOL = 'qwen32-long-prefix-final-v1'


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def progress(root, event, **fields):
    value = dict(event=event,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields)
    dump(root/'progress.json',value)
    print(event+' '+json.dumps(fields,ensure_ascii=False,separators=(',',':')),flush=True)


def source_hashes():
    paths = [p for p in KIT.iterdir() if p.is_file() and p.suffix in {'.py','.sh'}]
    paths += [p for p in (KIT/'vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    return {str(p.relative_to(KIT)):rt.sha(p) for p in sorted(paths)}


def validate_fixed(cfg):
    expected = dict(model='/mnt/zj-data/models/Qwen3-32B/',
                    repo='/mnt/zj-data/data/chengcp/infinilm/InfiniLM',
                    prompt_tokens=8192,output_tokens=64,max_context=9216,max_batch=3,
                    token_budget=8192,nh_blocks=128,nh_block_size=256,
                    vllm_gpu_blocks=2048,vllm_block_size=16,admission_cap=3,
                    skew=1.25,cache_ttl_s=5.0,queue_timeout_s=30.0)
    for key,value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'Frozen configuration requires {key}={value!r}, got {cfg.get(key)!r}')
    if len(cfg.get('gpu_ids',[])) != 8 or len(set(cfg['gpu_ids'])) != 8:
        raise ValueError('Eight unique GPUs are required')
    reservation = math.ceil((8192+64)/256)*256
    if reservation*cfg['admission_cap'] > 32768:
        raise ValueError('Page-rounded full-sequence admission exceeds worker KV capacity')
    return reservation


def selected_screen(path):
    root = Path(path).resolve()
    plan = json.loads((root/'plan.json').read_text())
    status = json.loads((root/'status.json').read_text())
    qualification = json.loads((root/'qualification.json').read_text())
    if plan.get('protocol') != PROTOCOL or plan.get('mode') not in ('screen','screen16'):
        raise ValueError('Confirmation requires a screen from this frozen protocol')
    if status.get('status') != 'SCREEN_COMPLETE' or not qualification.get('winner'):
        raise ValueError('Screen incomplete or neither hash candidate met the declared threshold; confirmation refused')
    winner = qualification['winner']
    if winner not in ('nh_bounded','nh_cost_bounded') or qualification['concurrency'] != plan['concurrency']:
        raise ValueError('Malformed screen selection')
    return root,plan,qualification


def make_plan(a,cfg):
    reserve = validate_fixed(cfg)
    seeds,requests,concurrency,arms = [33959],128,(16 if a.mode=='screen16' else 8),list(ARMS)
    screen = None
    if a.mode=='confirm':
        if not a.screen:
            raise ValueError('confirm requires --screen RUN_DIR')
        screen,screen_plan,selection = selected_screen(a.screen)
        concurrency = screen_plan['concurrency']
        arms = [arm for arm in ARMS if arm[0] in ('nh_least_active',selection['winner'],'vllm_least_active')]
        seeds,requests = HOLDOUT_SEEDS,256
    elif a.screen:
        raise ValueError('--screen is only accepted by confirm')
    return dict(protocol=PROTOCOL,mode=a.mode,config=cfg,concurrency=concurrency,seeds=seeds,
                steady_requests=requests,learn_requests=64,keys=16,hot_fraction=.25,
                prompt_tokens=8192,output_tokens=64,shared_tokens=7680,arms=arms,
                cells=len(seeds)*len(arms),formal_requests=len(seeds)*len(arms)*(64+requests),
                cache='on both engines; NH flash-attn; both eager',
                tp=1,replicas=8,pd_split=False,kv_tokens_per_worker=32768,
                kv_GiB_per_worker=8,page_rounded_reservation_per_request=reserve,
                max_inflight_per_worker=3,graphs=False,
                calibration='12 cold/warm prefill samples + 3 decode warmups + 6 measured requests in C1/C2/C3; one independent shared prefix per level; separate single NH replica',
                probe_requests_per_cell=16,
                arrival_order='hot25 shuffled within each 64-request cohort',
                ttft='client send through gateway to first nonempty content; includes capacity wait',
                throughput='all exact output tokens / full phase wall clock including ramp and drain',
                native_baseline='InfiniLM engine with least-active policy reproduced at common full-replica gateway',
                selection='steady-only: throughput/vLLM>=0.95 and p95TTFT/vLLM<=1.05, zero errors; maximize min(tput_ratio,1/p95_ratio), tie bounded',
                confirmation_screen=str(screen) if screen else None,
                performance_win_claimed=False)


def nh_metadata(item):
    return dict(item,body=dict(item['body'],_fourway_prefix={
        'digest':item['prefix_digest'],'tokens':item['shared_tokens']}))


async def smoke_pool(client,pool,probes,destination=None):
    from streaming import request_one
    async def worker(i):
        results=[]
        for item in probes:
            payload=nh_metadata(item) if pool.engine=='nh' else item
            results.append(await request_one(client,pool.urls[i],payload,pool.models[i],False))
        return results
    results=await asyncio.gather(*(worker(i) for i in range(pool.count)))
    dump(destination or pool.folder/'smoke.json',results)
    if not all(r['success'] for pair in results for r in pair):
        raise RuntimeError('Direct streaming/usage probe failed; inspect workers/smoke.json')
    for pair in results:
        cached=(pair[1]['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0)
        if not isinstance(cached,int) or cached < probes[1]['shared_tokens']:
            raise RuntimeError('Repeated probe did not demonstrate the full shared-prefix cache hit')
        if pool.engine=='nh':
            snapshot=pair[1].get('engine_telemetry',{}).get('cache_snapshot') or {}
            if snapshot.get('prefixes',{}).get(probes[1]['prefix_digest'],0) < probes[1]['shared_tokens']:
                raise RuntimeError('NH actual-resident cache metadata missing for the repeated probe')
    pool.assert_alive()


async def measure(client,url,model,items,concurrency,root,label,phase):
    from streaming import request_one
    iterator=iter(items)
    results=[]
    async def worker():
        while True:
            try:
                item=next(iterator)
            except StopIteration:
                return
            results.append(await request_one(client,url,item,model,True))
            if len(results)%16==0 or len(results)==len(items):
                dump(root/'cells'/label/(phase+'_partial_requests.json'),results)
                progress(root,'REQUEST_PROGRESS',cell=label,phase=phase,completed=len(results),total=len(items),
                         failures=sum(not r['success'] for r in results))
    start=time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(min(len(items),concurrency))))
    elapsed=time.perf_counter()-start
    by_id={r['id']:r for r in results}
    if len(by_id)!=len(items):
        raise RuntimeError('Missing or duplicate request identity')
    return [by_id[item['id']] for item in items],elapsed


async def calibrate(cfg,root,work,fp):
    from aiohttp import ClientSession,ClientTimeout,TCPConnector
    from streaming import request_one
    folder=root/'calibration'
    pool=rt.Pool(cfg,'nh',folder,count=1)
    prefill,decode,warmup=[],[],[]
    try:
        progress(root,'CALIBRATION_BEGIN',prefill_samples=12,decode_warmup_requests=3,decode_requests=6,replicas=1)
        await pool.start()
        async with ClientSession(timeout=ClientTimeout(total=1800),connector=TCPConnector(limit=0),read_bufsize=2**20) as client:
            for index,item in enumerate(rt.rows(work/'calibration.jsonl')):
                result=await request_one(client,pool.urls[0],nh_metadata(item),pool.models[0],False)
                result['calibration_stage']=item['calibration_stage']
                prefill.append(result)
                dump(folder/'prefill_requests.json',prefill)
                if not result['success']:
                    raise RuntimeError('Calibration stream failed: '+str(result))
                cached=(result['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
                if not isinstance(cached,int) or not 0<=cached<item['prompt_tokens']:
                    raise RuntimeError('Calibration needs an exact initial cached-token counter')
                if item['calibration_stage']=='cold' and cached!=0:
                    raise RuntimeError('Independent cold calibration document unexpectedly reused a full cache block')
                if item['calibration_stage']=='warm' and cached<item['shared_tokens']:
                    raise RuntimeError('Warm calibration did not reuse its declared prefix')
                progress(root,'CALIBRATION_SAMPLE',completed=index+1,total=12,prompt_tokens=item['prompt_tokens'],cached_tokens=cached)
            samples=rt.rows(work/'decode_calibration.jsonl')
            warmups=rt.rows(work/'decode_warmup.jsonl')
            steps={}
            for batch in (1,2,3):
                for item in (r for r in warmups if r['batch']==batch):
                    result=await request_one(client,pool.urls[0],nh_metadata(item),pool.models[0],False)
                    result['concurrency']=batch
                    warmup.append(result)
                    dump(folder/'decode_warmup_requests.json',warmup)
                    if not result['success']:
                        raise RuntimeError('Decode calibration warmup failed')
                group=[r for r in samples if r['batch']==batch]
                values=await asyncio.gather(*(request_one(client,pool.urls[0],nh_metadata(item),pool.models[0],False) for item in group))
                for result in values:
                    result['concurrency']=batch
                decode.extend(values)
                dump(folder/'decode_requests.json',decode)
                if not all(r['success'] for r in values):
                    raise RuntimeError('Decode-step calibration stream failed')
                progress(root,'DECODE_CACHE_CHECK',concurrency=batch,
                         cached_tokens=[(r['usage'].get('prompt_tokens_details') or {}).get('cached_tokens') for r in values],
                         required_shared_tokens=7680,layout='shared-prefix-distinct-suffixes')
                if not all((r['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0)>=7680 for r in values):
                    raise RuntimeError('Decode probes did not reuse their pre-warmed shared prefixes')
                steps[str(batch)]=statistics.mean(max(0,r['e2e_ms']-r['ttft_ms'])/(r['usage']['completion_tokens']-1) for r in values)
                progress(root,'DECODE_CALIBRATION',concurrency=batch,observed_step_ms=steps[str(batch)])
        x,y=[],[]
        for r in prefill:
            p=r['usage']['prompt_tokens']; c=r['usage']['prompt_tokens_details']['cached_tokens']; q=p-c
            x.append([1,q/1000,(q*q+2*q*c)/1e6]); y.append(r['ttft_ms'])
        coef=rt.nnls_small(x,y)
        predicted=[sum(a*b for a,b in zip(coef,row)) for row in x]
        profile=dict(fingerprint=fp,prefill_coef=coef,decode_step_ms=steps,prefill_samples=len(prefill),
                     prefill_in_sample_mae_ms=statistics.mean(abs(a-b) for a,b in zip(y,predicted)),
                     meaning='Heuristic ranking model. Direct client TTFT and pre-warmed concurrent client-effective decode-step estimates (one independent shared prefix per concurrency level, distinct suffixes). Residual prefill/interleaving and stream finalization remain; not GPU timing or guaranteed queue delay.',
                     calibration_is_performance_data=False)
        dump(root/'cost_profile.json',profile)
        progress(root,'CALIBRATION_END',coefficients=coef,decode_step_ms=steps)
        return profile
    finally:
        await pool.close()


def phase_stats(records,elapsed,meta):
    result=rt.statistics_row(records,elapsed,meta)
    result['valid']=result['failures']==0 and result['success']==result['requests']
    # Failed cohorts must not produce a partial-success throughput comparison.
    if not result['valid']:
        result['output_tok_s']=None
        result['request_s']=None
    return result


async def run_cell(cfg,root,case,concurrency,arm,profile):
    from aiohttp import ClientSession,ClientTimeout,TCPConnector
    from affinity_router import Router,PrefillModel
    from gateway import Gateway
    name,engine,policy=arm
    label=f'{case["folder"]}_c{concurrency}_{name}'
    folder=root/'cells'/label
    folder.mkdir(parents=True)
    pool=rt.Pool(cfg,engine,folder/'workers')
    gateway=None
    status=dict(status='RUNNING',cell=label,arm=name)
    dump(folder/'status.json',status)
    phases,all_records,durations=[],[],[]
    try:
        progress(root,'CELL_BEGIN',cell=label,engine=engine,policy=policy)
        await pool.start()
        router=Router(pool.urls,policy,model=PrefillModel.from_profile(profile) if policy=='cost_bounded' else None,
                      max_inflight=3,kv_tokens=32768,page_size=256,skew=1.25,
                      cache_ttl_s=5.0,queue_timeout_s=30.0)
        gateway=Gateway(pool.urls,pool.models,router,cfg['port'],engine)
        await gateway.start()
        case_dir=root/'workloads'/case['folder']
        async with ClientSession(timeout=ClientTimeout(total=1800),connector=TCPConnector(limit=0),read_bufsize=2**20) as client:
            progress(root,'CACHE_PROBE_BEGIN',cell=label)
            await smoke_pool(client,pool,json.loads((case_dir/'smoke.json').read_text()))
            progress(root,'CACHE_PROBE_PASS',cell=label)
            for phase in ('learn','steady'):
                items=rt.rows(case_dir/(phase+'.jsonl'))
                begin=len(router.history)
                progress(root,'PHASE_BEGIN',cell=label,phase=phase,requests=len(items))
                records,elapsed=await measure(client,f'http://127.0.0.1:{cfg["port"]}','gateway',items,concurrency,root,label,phase)
                all_records.extend(records); durations.append(elapsed)
                routes=router.history[begin:]
                state=router.snapshot()
                by_id={record['id']:record for record in records}
                transparent=(len({route.get('request_id') for route in routes})==len(items) and
                             all(route.get('request_id') in by_id and all(
                                 route.get(key)==by_id[route['request_id']].get(key)
                                 for key in ('event_payload_sha256','event_payload_count','event_payload_bytes'))
                                 for route in routes))
                valid_routes=(transparent and not gateway.errors and not any(state['active']) and not any(state['kv_reserved_tokens'])
                              and len(routes)==len(items) and all(r.get('released') and r.get('success') for r in routes))
                summary=phase_stats(records,elapsed,dict(arm=name,seed=case['seed'],hot_fraction=.25,concurrency=concurrency,phase=phase))
                summary.update(route_requests=len(routes),route_valid=valid_routes,event_payload_transparent=transparent,
                               spills=sum(bool(r.get('spill')) for r in routes),
                               cache_hint_overestimates=sum(bool(r.get('hint_overestimated')) for r in routes),
                               queue_wait_p95_ms=rt.quantile([r.get('gateway_wait_ms',0) for r in routes],.95))
                summary['valid']=summary['valid'] and valid_routes
                if not summary['valid']:
                    summary['output_tok_s']=None; summary['request_s']=None
                dump(folder/(phase+'_requests.json'),records)
                dump(folder/(phase+'_routes.json'),state)
                dump(folder/'gateway_errors.json',gateway.errors)
                phases.append(summary); rt.csv_write(folder/'summary.csv',phases)
                pool.assert_alive()
                if not summary['valid']:
                    raise RuntimeError('Invalid stream/usage/route lease cohort; inspect raw requests, routes and gateway_errors')
                progress(root,'PHASE_END',cell=label,phase=phase,output_tok_s=summary['output_tok_s'],ttft_p95_ms=summary['ttft_p95_ms'],cache_fraction=summary['cache_fraction'])
        combined=phase_stats(all_records,sum(durations),dict(arm=name,seed=case['seed'],hot_fraction=.25,concurrency=concurrency,phase='learn_plus_steady'))
        combined['wall_definition']='sum of measured learning and steady phases; excludes intervening artifact writes'
        phases.append(combined); rt.csv_write(folder/'summary.csv',phases)
        status['status']='PASS'
        return phases
    except BaseException as exc:
        status.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        dump(folder/'status.json',status)
        try:
            if gateway:
                await gateway.close()
        finally:
            await pool.close()


def summarize(root,results):
    rt.csv_write(root/'summary.csv',results)
    groups=defaultdict(list)
    for r in results:
        if r.get('valid'):
            groups[(r['arm'],r['hot_fraction'],r['concurrency'],r['phase'])].append(r)
    aggregate=[]
    for (arm,hot,concurrency,phase),values in groups.items():
        row=dict(arm=arm,hot_fraction=hot,concurrency=concurrency,phase=phase,rounds=len(values))
        for metric in ('output_tok_s','ttft_mean_ms','ttft_p50_ms','ttft_p95_ms','ttft_p99_ms','cache_fraction'):
            data=[r[metric] for r in values if r.get(metric) is not None]
            if data:
                row.update({metric+'_mean':statistics.mean(data),metric+'_min':min(data),metric+'_max':max(data)})
        aggregate.append(row)
    rt.csv_write(root/'aggregate.csv',aggregate)
    pairs=paired_rows(results)
    rt.csv_write(root/'paired.csv',pairs)
    return pairs


def paired_rows(results):
    lookup={(r['arm'],r['seed'],r['phase']):r for r in results if r.get('valid')}
    pairs=[]
    for (arm,seed,phase),r in lookup.items():
        if arm not in ('nh_bounded','nh_cost_bounded'):
            continue
        for baseline in ('nh_least_active','vllm_least_active'):
            b=lookup.get((baseline,seed,phase))
            if b:
                tr=r['output_tok_s']/b['output_tok_s']; pr=r['ttft_p95_ms']/b['ttft_p95_ms']
                pairs.append(dict(candidate=arm,baseline=baseline,seed=seed,hot_fraction=.25,concurrency=r['concurrency'],phase=phase,
                                  throughput_ratio=tr,ttft_p95_ratio=pr,throughput_gain_pct=100*(tr-1),
                                  ttft_p95_reduction_pct=100*(1-pr),
                                  ttft_mean_reduction_pct=100*(1-r['ttft_mean_ms']/b['ttft_mean_ms']),
                                  within_5_percent=tr>=.95 and pr<=1.05))
    return pairs


def qualification(plan,results,pairs):
    expected={(arm[0],seed) for arm in plan['arms'] for seed in plan['seeds']}
    got={(r['arm'],r['seed']) for r in results if r['phase']=='steady' and r.get('valid')}
    complete=got==expected
    comparisons=[r for r in pairs if r['phase']=='steady' and r['baseline']=='vllm_least_active']
    candidates={}
    for name in ('nh_bounded','nh_cost_bounded'):
        records=[r for r in comparisons if r['candidate']==name]
        if not records:
            continue
        qualifies=complete and len(records)==len(plan['seeds']) and all(r['within_5_percent'] for r in records)
        candidates[name]=dict(qualifies=qualifies,score=min(min(r['throughput_ratio'],1/r['ttft_p95_ratio']) for r in records),per_seed=records)
    valid=[name for name,r in candidates.items() if r['qualifies']]
    winner=max(valid,key=lambda name:(candidates[name]['score'],name=='nh_bounded')) if valid else None
    return dict(protocol=PROTOCOL,complete=complete,concurrency=plan['concurrency'],candidates=candidates,winner=winner,
                steady_only=True,criteria=dict(throughput_ratio_min=.95,ttft_p95_ratio_max=1.05,errors=0),
                result='WITHIN_5_PERCENT' if winner else 'NO_QUALIFIED_CANDIDATE',
                interpretation='Engineering threshold, not confidence interval. Screening is not a holdout result.')


async def run_logged(argv,env,cwd,path,timeout=180):
    started=time.monotonic()
    process=await asyncio.create_subprocess_exec(*argv,env=env,cwd=cwd,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:
        communication=asyncio.create_task(process.communicate())
        while True:
            remaining=timeout-(time.monotonic()-started)
            if remaining<=0:
                raise TimeoutError('Subprocess timeout: '+str(path))
            done,_=await asyncio.wait({communication},timeout=min(20,remaining))
            if done:
                stdout,_=communication.result()
                break
            print('SUBPROCESS_RUNNING '+json.dumps(dict(log=str(path),elapsed_s=round(time.monotonic()-started))),flush=True)
    except BaseException:
        if process.returncode is None:
            process.kill(); await process.wait()
        if 'communication' in locals() and not communication.done():
            communication.cancel()
        raise
    path.write_bytes(stdout)
    if process.returncode:
        raise RuntimeError('Preflight subprocess failed; see '+str(path))
    return stdout


async def preflight(cfg,root):
    progress(root,'PREFLIGHT_BEGIN')
    rt.check_config(cfg)
    fp=rt.fingerprint(cfg)
    fp.update(protocol=PROTOCOL,candidate_sources=source_hashes())
    code="import json,importlib.metadata as m; names=['torch','torch-musa','transformers','vllm','vllm-musa','aiohttp','infinilm']; out={};\nfor n in names:\n try: out[n]=m.version(n)\n except m.PackageNotFoundError: pass\nprint(json.dumps(out,sort_keys=True))"
    versions={}
    for engine in ('nh','vllm'):
        env=rt.backend_env(cfg,engine,cfg['gpu_ids'][0],'/tmp')
        text=await run_logged([cfg[engine+'_python'],'-c',code],env,cfg['repo'],root/(engine+'_versions.log'))
        versions[engine]=json.loads(text.decode().splitlines()[-1])
        argv=rt.command(cfg,engine,cfg['port']+1)
        help_argv=argv[:2 if engine=='nh' else 3]+['--help' if engine=='nh' else '--help=all']
        text=await run_logged(help_argv,env,cfg['repo'],root/(engine+'_help.log'))
        required=['--enable-paged-attn','--disable-prefix-caching','--ignore-eos','--attn'] if engine=='nh' else ['--enable-prefix-caching','--enable-prompt-tokens-details','--num-gpu-blocks-override','--enable-chunked-prefill','--enforce-eager']
        if any(flag.encode() not in text for flag in required):
            raise RuntimeError(engine+' installed CLI does not expose the required verified flags')
    fp['package_versions']=versions
    receipt=Path(cfg['repo']).parent/'workspace_receipt.json'
    if receipt.is_file():
        fp['workspace_receipt_sha256']=rt.sha(receipt)
        shutil.copy2(receipt,root/'workspace_receipt.json')
    dump(root/'versions.json',versions); dump(root/'fingerprint.json',fp)
    progress(root,'PREFLIGHT_PASS')
    return fp


async def main_async(a,cfg,root,plan):
    task=asyncio.current_task()
    loop=asyncio.get_running_loop()
    for sig in (signal.SIGTERM,signal.SIGINT):
        loop.add_signal_handler(sig,task.cancel)
    fp=await preflight(cfg,root)
    profile_source=Path(a.profile).resolve() if a.profile else None
    if a.mode=='confirm':
        screen=Path(plan['confirmation_screen'])
        if json.loads((screen/'fingerprint.json').read_text()) != fp:
            raise ValueError('Screen fingerprint differs from current engines/configuration/sources; refusing a changed holdout experiment')
        screen_results=[]
        # Recompute the frozen decision from the original cell statistics instead
        # of trusting a manually edited winner field alone.
        for path in sorted((screen/'cells').glob('*/summary.csv')):
            import csv
            for row in csv.DictReader(path.open()):
                converted={k:v for k,v in row.items()}
                for k in ('seed','concurrency','requests','success','failures'):
                    if converted.get(k): converted[k]=int(converted[k])
                for k in ('output_tok_s','ttft_p95_ms','ttft_mean_ms'):
                    if converted.get(k): converted[k]=float(converted[k])
                converted['valid']=converted.get('valid')=='True'
                screen_results.append(converted)
        original=json.loads((screen/'qualification.json').read_text())
        screen_plan=json.loads((screen/'plan.json').read_text())
        recomputed=qualification(screen_plan,screen_results,paired_rows(screen_results))
        if not recomputed['complete'] or not recomputed['winner'] or recomputed['winner'] != original['winner']:
            raise ValueError('Screen statistics do not support the declared frozen selection')
        if any(json.loads(path.read_text()).get('status')!='PASS' for path in (screen/'cells').glob('*/status.json')):
            raise ValueError('Screen contains a failed cell')
        profile_source=screen/'cost_profile.json'
    profile=None
    if profile_source:
        profile=json.loads(profile_source.read_text())
        if profile.get('fingerprint') != fp:
            raise ValueError('Cost profile fingerprint does not exactly match this engine/configuration/source version')
        dump(root/'cost_profile.json',profile)
        progress(root,'CALIBRATION_REUSED',source=str(profile_source))
    opts=dict(seeds=plan['seeds'],requests=plan['steady_requests'],keys=16,shared_tokens=7680,prompt_tokens=8192,output_tokens=64)
    work=root/'workloads'
    progress(root,'WORKLOAD_BUILD_BEGIN',seeds=plan['seeds'],steady_requests=plan['steady_requests'])
    await run_logged([cfg['nh_python'],str(KIT/'workload.py'),'--model',cfg['model'],'--out',str(work),'--options',json.dumps(opts)],
                     rt.backend_env(cfg,'nh',cfg['gpu_ids'][0],'/tmp'),cfg['repo'],root/'workload_build.log',timeout=900)
    progress(root,'WORKLOAD_BUILD_PASS')
    if profile is None:
        profile=await calibrate(cfg,root,work,fp)
    results=[]
    manifest=json.loads((work/'manifest.json').read_text())
    try:
        for index,case in enumerate(manifest['cells']):
            arms=plan['arms']; offset=index%len(arms)
            for arm in arms[offset:]+arms[:offset]:
                results+=await run_cell(cfg,root,case,plan['concurrency'],arm,profile)
                pairs=summarize(root,results)
                dump(root/'qualification.json',qualification(plan,results,pairs))
    finally:
        pairs=summarize(root,results)
        dump(root/'qualification.json',qualification(plan,results,pairs))
    final=qualification(plan,results,pairs)
    if not final['complete']:
        raise RuntimeError('Expected experiment cells are missing')
    return final


def render_report(root):
    try:
        from report import render
        render(root)
    except Exception as exc:
        dump(root/'report_error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        print('REPORT_RENDER_FAILED: '+repr(exc)+'; raw evidence retained in '+str(root),file=sys.stderr,flush=True)


def arguments(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(KIT/'config32b_final.json'))
    parser.add_argument('--mode',choices=('plan','screen','screen16','confirm'),default='plan')
    parser.add_argument('--out',default=str(KIT/'results32b_final'))
    parser.add_argument('--screen')
    parser.add_argument('--profile',help='Screen only: exact-fingerprint calibration reuse, e.g. screen16 after screen')
    args=parser.parse_args(argv)
    if args.mode=='confirm' and args.profile:
        parser.error('Confirm automatically reuses its frozen screen profile; do not pass --profile')
    return args


def main(argv=None):
    a=arguments(argv)
    cfg=json.loads(Path(a.config).read_text())
    plan=make_plan(a,cfg)
    if a.mode=='plan':
        print(json.dumps(plan,ensure_ascii=False,indent=2)); print('GPU_JOBS_STARTED=0',flush=True)
        return 0
    lock=open(f'/tmp/infinilm-fourway-{os.getuid()}.lock','a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print('Another fourway job still owns the host lock. Do not delete the lock or start duplicate GPU jobs. Inspect the existing controller process.',file=sys.stderr)
        return 2
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root=Path(a.out).resolve()/(a.mode+'_'+stamp)
    root.mkdir(parents=True)
    dump(root/'plan.json',plan); dump(root/'status.json',dict(status='RUNNING',performance_win_claimed=False))
    source=root/'source'; source.mkdir()
    for relative in source_hashes():
        destination=source/relative; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(KIT/relative,destination)
    print('RUN_DIR='+str(root),flush=True)
    try:
        result=asyncio.run(main_async(a,cfg,root,plan))
        status='CONFIRM_COMPLETE' if a.mode=='confirm' else 'SCREEN_COMPLETE'
        claim=a.mode=='confirm' and bool(result['winner'])
        dump(root/'status.json',dict(status=status,result=result['result'],winner=result['winner'],
                                     performance_win_claimed=False,within_5_percent_on_each_holdout_seed=claim))
        progress(root,status,result=result['result'],winner=result['winner'],summary=str(root/'summary.csv'))
        print('REPORT_RUN_DIR='+str(root),flush=True)
        render_report(root)
        return 0
    except BaseException as exc:
        dump(root/'status.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),performance_win_claimed=False))
        print('FAILED: '+repr(exc)+'; evidence='+str(root),file=sys.stderr,flush=True)
        render_report(root)
        return 2
    finally:
        lock.close()


if __name__=='__main__':
    raise SystemExit(main())
