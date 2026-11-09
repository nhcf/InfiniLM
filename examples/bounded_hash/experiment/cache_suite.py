#!/usr/bin/env python3
"""Cache-preserving bounded-hash screen; run defaults to three methods at C16."""
import argparse
import asyncio
from collections import defaultdict
import datetime
import fcntl
import json
import math
from pathlib import Path
import shutil
import signal
import statistics
import sys
import traceback
import backend_runtime as rt
import harness as common
from cache_policy import EXPECTED_SHA
from cache_workload import PROFILES

KIT=Path(__file__).resolve().parent
PROTOCOL='infinilm-cache-bounded-v2'
ARMS=[('nh_cache_bounded_v2','nh','cache_bounded_v2','prefix_lru'),
      ('nh_mean_bounded','nh','mean_bounded','prefix_lru'),
      ('nh_lru_least_active','nh','least_active','prefix_lru'),
      ('nh_native_rr','nh','round_robin','native'),
      ('vllm_least_active','vllm','least_active','vllm_native')]


def arguments(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',nargs='?',choices=['plan','check','calibrate','run'],default='plan')
    ap.add_argument('--config',type=Path,default=KIT/'config.musa.example.json')
    ap.add_argument('--profiles',nargs='+',choices=list(PROFILES),default=['reuse8'])
    ap.add_argument('--concurrency',nargs='+',type=int,choices=[16,32,64],default=[16])
    ap.add_argument('--seeds',nargs='+',type=int,default=[33959])
    ap.add_argument('--requests',type=int,choices=[64,128,256],default=128)
    ap.add_argument('--arms',nargs='+',choices=[a[0] for a in ARMS],default=['nh_cache_bounded_v2','nh_native_rr','vllm_least_active'])
    ap.add_argument('--phase-model',type=Path)
    ap.add_argument('--no-coalescing',action='store_true',help='Diagnostic ablation; forecast no same-prefix reuse')
    ap.add_argument('--owner-wait-ms',type=float,default=500.0)
    ap.add_argument('--output-root',type=Path,default=KIT/'results')
    a=ap.parse_args(argv)
    for values in (a.profiles,a.concurrency,a.seeds,a.arms):
        if not values or len(set(values))!=len(values):ap.error('Selections must be nonempty and unique')
    if any(s<0 for s in a.seeds):ap.error('Seeds must be nonnegative')
    if not math.isfinite(a.owner_wait_ms) or not 0 <= a.owner_wait_ms <= 2000:ap.error('owner-wait-ms must be in [0,2000]')
    if a.mode=='run' and 'nh_cache_bounded_v2' not in a.arms:ap.error('A run must include the candidate')
    return a


def configuration(path):
    cfg=json.loads(path.read_text())
    if 'config' in cfg:cfg=cfg['config']
    cfg=dict(cfg)
    # Runtime locations are copied from this explicit configuration, never guessed.
    cfg.update(prompt_tokens=8192,output_tokens=512,max_context=9216,max_batch=3,
        token_budget=8192,nh_blocks=128,nh_block_size=256,vllm_gpu_blocks=2048,
        vllm_block_size=16,admission_cap=3,skew=1.25,cache_ttl_s=5.0,queue_timeout_s=600.0)
    if cfg.get('device') not in ('moore','nvidia'):raise ValueError('Use a verified moore or nvidia runtime')
    if len(cfg.get('gpu_ids',[]))!=8 or len(set(cfg['gpu_ids']))!=8:raise ValueError('Eight unique TP1 replica GPUs required')
    if cfg['nh_blocks']*cfg['nh_block_size']!=cfg['vllm_gpu_blocks']*cfg['vllm_block_size']:
        raise ValueError('Equal KV token capacity required')
    reserve=math.ceil((cfg['prompt_tokens']+cfg['output_tokens'])/256)*256
    if 3*reserve>32768:raise ValueError('Full page-rounded reservations exceed physical per-worker KV capacity')
    return cfg


def make_plan(a,cfg):
    arms=[x for x in ARMS if x[0] in a.arms]
    cells=len(arms)*len(a.profiles)*len(a.concurrency)*len(a.seeds)
    return dict(protocol=PROTOCOL,config=cfg,profiles={n:PROFILES[n] for n in a.profiles},
        arms=arms,concurrency=a.concurrency,seeds=a.seeds,learning_requests=64,measured_requests=a.requests,
        cells=cells,total_learning_plus_measured_requests=cells*(64+a.requests),
        model=cfg['model'],replicas=8,tp=1,pd_split=False,backend_slots=24,
        calibration='Independent untimed 2K/4K/8K warmup, reset, then C1 cold/warm and overlap-only B1/B2/B3 decode; raw events retained',
        router_options=dict(coalescing=not a.no_coalescing,owner_wait_ms=a.owner_wait_ms),
        primary='same profile/seed/C: lower mean TTFT AND higher output tok/s than vLLM; also report old >=95% throughput screen',
        cohort_definition='learning64 separately timed, then measured; both include ramp/drain; combined is sum of the two timed phases',
        source_qualification='Declared favorable long-prefix/short-output workloads, not a broad serving benchmark. Seeds may have been used in prior development.',
        historical_relation='reuse8/reuse16 are newly declared favorable profiles. prefix97 has historical parameters but new paired documents/IDs; no replay or reproduction of historical absolute scores is claimed.',
        fairness='All arms same model/dtype/8TP1/24slots/32768KVtokens, cache on, eager. NH cache-mode idle reset, vLLM fresh process pool for every profile/seed/C.',
        caveat='RR/native comparison includes cache replacement changes; same-LRU least-active and old-mean isolate candidate changes.',
        gpu_jobs_started=0,performance_win_claimed=False)


def contract(items):
    import hashlib
    value=[{k:i[k] for k in ('id','prompt_sha256','prompt_tokens','shared_tokens','output_tokens')} for i in items]
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def phase_statistics(records,elapsed,meta):
    result=common.phase_stats(records,elapsed,meta)
    result['total_output_tokens']=result['output_tokens']
    result['output_tokens_per_request']=meta['output_tokens']
    return result


def report_invalid_phase(folder,row,records,gateway_errors,requested_count):
    """Explain an already-invalid phase without relaxing or changing its audit.

    Failed requests are evidence, not retry instructions. Preserve the complete
    errors in JSON and add the first observed failure to the raised message.
    """
    if row.get('valid'):
        raise ValueError('Failure reporting requires an already invalid phase')
    failed=[{k:r.get(k) for k in ('id','http_status','error','e2e_ms')}
            for r in records if r.get('success') is not True]
    flags=('valid','route_valid','event_payload_transparent','exact_usage',
           'cohort_valid','cache_policy_valid')
    bad_flags=[name for name in flags if name in row and not row[name]]
    phase_ids={r.get('id') for r in records}
    phase_errors=[e for e in gateway_errors if e.get('id') in phase_ids]
    success=sum(r.get('success') is True for r in records)
    details={k:row.get(k) for k in ('profile','seed','concurrency','arm','phase')}
    details.update(status='INVALID_PHASE',valid=False,
        requested=requested_count,received=len(records),success=success,
        failures=len(failed),missing_requests=max(0,requested_count-len(records)),
        failed_requests=failed,audit_flag_failures=bad_flags,
        gateway_errors=list(gateway_errors),phase_gateway_errors=phase_errors,
        throughput_comparison_suppressed=True,requests_retried=0)
    common.dump(Path(folder)/'failure_details.json',details)
    message=(f'Invalid complete cohort; throughput comparison suppressed; '
             f'{details["arm"]}/{details["phase"]}: success {success}/{requested_count}, '
             f'received {len(records)}, failures {len(failed)}; '
             f'failed checks: {",".join(bad_flags) or "see phase statistics"}')
    if failed:
        first=failed[0]
        error=first.get('error')
        if not error:
            error=next((e.get('error') for e in phase_errors if e.get('id')==first['id']),None)
        message+=f'; first failed request {first["id"]}, HTTP {first["http_status"]}: {str(error or "no error text recorded")[:600]}'
    message+='; see failure_details.json'
    return message


def runtime_features(pool):
    """Capture requested flags plus actually logged backend selection separately."""
    import re
    launch=json.loads((pool.folder/'launch.json').read_text())
    args=[w['argv'] for w in launch]
    def enabled(flag):return all(flag in argv for argv in args)
    lines=[]
    for path in sorted(pool.folder.glob('worker*.log')):
        for line in path.read_text(errors='replace').splitlines():
            if re.search(r'platform use .*ATTN|Using .*attention backend|Using .*Attention backend|Using .*Attention',line):
                lines.append(dict(worker=path.stem,line=line))
    nh=pool.engine=='nh'
    requested=dict(prefix_cache=(enabled('--enable-paged-attn') and not any('--disable-prefix-caching' in a for a in args)) if nh else enabled('--enable-prefix-caching'),
        attention='flash-attn' if nh and all('--attn' in a and a[a.index('--attn')+1]=='flash-attn' for a in args) else 'runtime_auto',
        cuda_graph=False if nh and not any('--enable-graph' in a for a in args) else not enabled('--enforce-eager'),
        chunked_prefill=enabled('--enable-chunked-prefill') if not nh else 'no_engine_scheduler_change')
    return dict(engine=pool.engine,workers=len(args),requested=requested,
        attention_backend_log_evidence=lines,
        note='Requested attention flags are not a GPU kernel trace. Cache activation and actual prefix hits are also verified in smoke.json and per-request usage. enable_flashinfer_autotune alone does not select FlashInfer attention.')


def extra_stats(records,routes,shared):
    known=[r for r in records if isinstance((r.get('usage',{}).get('prompt_tokens_details') or {}).get('cached_tokens'),int)]
    hit=[r for r in known if r['usage']['prompt_tokens_details']['cached_tokens']>=shared] if shared else []
    good=[r for r in records if r.get('success')]
    forecasts=[r for r in routes if 'predicted_cached_tokens' in r]
    comparable=[r for r in forecasts if isinstance(r.get('observed_cached_tokens'),int)]
    tpot=[max(0,r['e2e_ms']-r['ttft_ms'])/(r['usage']['completion_tokens']-1) for r in good if r['usage']['completion_tokens']>1]
    return dict(full_prefix_hit_fraction=len(hit)/len(records) if shared and len(known)==len(records) else None,
        client_tpot_mean_ms=statistics.mean(tpot) if tpot else None,client_tpot_p95_ms=rt.quantile(tpot,.95),
        client_tpot_definition='Per-request (client E2E minus client TTFT)/(exact completion_tokens-1); includes stream finalization, not token ITL or SSE chunk interval',
        e2e_p95_ms=rt.quantile([r['e2e_ms'] for r in good],.95),
        gateway_wait_mean_ms=statistics.mean(r['gateway_wait_ms'] for r in routes) if routes else None,
        gateway_wait_p95_ms=rt.quantile([r['gateway_wait_ms'] for r in routes],.95),
        cache_hint_overestimates=sum(bool(r.get('hint_overestimated')) for r in routes),
        spills=sum(bool(r.get('spill')) for r in routes),
        predicted_reuse_admissions=sum(r['predicted_cached_tokens']>r.get('estimated_cached_tokens',0) for r in forecasts) if forecasts else None,
        predicted_reuse_overestimates=sum(r['predicted_cached_tokens']>r['observed_cached_tokens'] for r in comparable) if len(comparable)==len(forecasts) and forecasts else None,
        owner_wait_admissions=sum(bool(r.get('owner_wait_requested')) for r in forecasts) if forecasts else None,
        count_bound_relaxations=sum(bool(r.get('count_bound_relaxed')) for r in forecasts) if forecasts else None)


async def smoke_pool(client,pool,probes,destination):
    from streaming import request_one
    async def worker(i):
        results=[]
        for item in probes:
            payload=item
            if pool.engine=='nh' and item['shared_tokens']>0:
                payload=common.nh_metadata(item)
            results.append(await request_one(client,pool.urls[i],payload,pool.models[i],False))
        return results
    results=await asyncio.gather(*(worker(i) for i in range(pool.count)))
    common.dump(destination,results)
    for pair in results:
        if not all(r['success'] for r in pair):raise RuntimeError('Direct cache/stream probe failed')
        shared=probes[1]['shared_tokens']
        if shared:
            cached=(pair[1]['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
            if not isinstance(cached,int) or cached<shared:raise RuntimeError('Warm probe did not confirm prefix cache')
            if pool.engine=='nh' and pair[1].get('engine_telemetry',{}).get('cache_snapshot',{}).get('prefixes',{}).get(probes[1]['prefix_digest'],0)<shared:
                raise RuntimeError('NH cache snapshot did not confirm actual resident prefix')
    pool.assert_alive()


async def hardware_fingerprint(cfg,root):
    """Read-only device identity probe. Unknown identity prevents model reuse only."""
    code='''import json, torch
try:
 import torch_musa
except ImportError:
 pass
d=getattr(torch,'musa',None) if DEVICE=='moore' else torch.cuda
out={'device':DEVICE,'available':False,'devices':[],'torch_version':torch.__version__,'cuda_runtime':getattr(torch.version,'cuda',None)}
try:
 out['available']=bool(d and d.is_available())
 if out['available']:
  for i in range(d.device_count()):
   p=d.get_device_properties(i)
   out['devices'].append({k:getattr(p,k,None) for k in ('name','total_memory','major','minor','multi_processor_count')})
except Exception as e:
 out['error']=repr(e)
print(json.dumps(out,sort_keys=True))
'''.replace('DEVICE',repr(cfg['device']))
    env=rt.backend_env(cfg,'nh',cfg['gpu_ids'][0],'/tmp')
    env['CUDA_VISIBLE_DEVICES']=','.join(map(str,cfg['gpu_ids']))
    if cfg['device']=='moore':env['MUSA_VISIBLE_DEVICES']=env['CUDA_VISIBLE_DEVICES']
    try:
        raw=await common.run_logged([cfg['nh_python'],'-c',code],env,cfg['repo'],root/'hardware_probe.log')
        value=json.loads(raw.decode().splitlines()[-1])
    except Exception as exc:value=dict(device=cfg['device'],available=False,error=repr(exc),devices=[])
    # NVIDIA provides a stable driver identifier without allocating model memory.
    import shutil as _shutil
    if cfg['device']=='nvidia' and _shutil.which('nvidia-smi'):
        try:
            proc=await asyncio.create_subprocess_exec('nvidia-smi','--query-gpu=driver_version','--format=csv,noheader',stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL)
            stdout,_=await asyncio.wait_for(proc.communicate(),timeout=10)
            value['driver_versions']=sorted(set(stdout.decode().splitlines())) if proc.returncode==0 else None
        except Exception:value['driver_versions']=None
    common.dump(root/'hardware.json',value)
    return value


async def run_cell(cfg,root,pool,case,c,arm,client,model,router_options=None):
    import matrix as old_matrix
    from gateway import Gateway
    from mean_router import Router as MeanRouter
    from cache_router import Router as CacheRouter
    name,engine,policy,cache_mode=arm
    label=f'{case["profile"]}_s{case["seed"]}_c{c}_{name}'
    folder=root/'cells'/label;folder.mkdir(parents=True)
    common.dump(folder/'status.json',dict(status='RUNNING'))
    gateway=None;router=None;stats=[];all_records=[];all_routes=[];durations=[]
    try:
        if engine=='nh':await old_matrix.reset_pool(client,pool,cache_mode,folder)
        else:common.dump(folder/'cache_reset.json',dict(method='fresh_processes',pool=str(pool.folder)))
        work=root/'workloads'/case['folder']
        probes=json.loads((work/'smoke.json').read_text())
        await smoke_pool(client,pool,probes,folder/'smoke.json')
        if engine=='nh' and any(pair[0]['usage']['prompt_tokens_details'].get('cached_tokens')!=0 for pair in json.loads((folder/'smoke.json').read_text())):
            raise RuntimeError('NH idle reset cold probe did not prove zero initial cache reuse')
        kwargs=dict(max_inflight=3,kv_tokens=32768,page_size=256,skew=1.25,cache_ttl_s=5.,queue_timeout_s=600.)
        router=CacheRouter(pool.urls,policy,phase_model=model,**(router_options or {}),**kwargs) if policy=='cache_bounded_v2' else MeanRouter(pool.urls,policy,**kwargs)
        gateway=Gateway(pool.urls,pool.models,router,cfg['port'],engine);await gateway.start()
        common.progress(root,'CELL_BEGIN',cell=label)
        for phase in ('learn','measured'):
            file=work/(phase+'.jsonl')
            items=rt.rows(file)
            declared=case['files'][phase]
            if len(items)!=declared['count'] or rt.sha(file)!=declared['sha256']:
                raise ValueError('Frozen workload file count/hash differs from its generated manifest')
            if phase=='learn' and len(items)!=64:raise ValueError('Learning cohort must contain exactly 64 requests')
            if any((i['prompt_tokens'],i['shared_tokens'],i['output_tokens'])!=(case['prompt_tokens'],case['shared_tokens'],case['output_tokens']) for i in items):
                raise ValueError('Workload does not match profile')
            begin=len(router.history)
            records,elapsed=await old_matrix.measure_with_heartbeat(pool,router,root,label,phase,
                client,f'http://127.0.0.1:{cfg["port"]}','gateway',items,c,root,label,phase)
            routes=router.history[begin:]
            valid,transparent,state=old_matrix.audit(router,gateway,records,routes)
            exact=all((r.get('usage',{}).get('prompt_tokens'),r.get('usage',{}).get('completion_tokens'))==(case['prompt_tokens'],case['output_tokens']) for r in records)
            cohort=len(items)==len(records) and len({r['id'] for r in records})==len(items) and {r['id'] for r in records}=={i['id'] for i in items}
            modes=engine!='nh' or all(r.get('engine_telemetry',{}).get('cache_policy')==cache_mode
                and r.get('engine_telemetry',{}).get('prefix_caching') is True for r in records)
            meta=dict(profile=case['profile'],seed=case['seed'],concurrency=c,arm=name,phase=phase,
                cache_policy=cache_mode,prompt_tokens=case['prompt_tokens'],shared_tokens=case['shared_tokens'],output_tokens=case['output_tokens'],
                ordered_request_contract_sha256=contract(items))
            row=phase_statistics(records,elapsed,meta)
            row.update(extra_stats(records,routes,case['shared_tokens']),route_valid=valid,event_payload_transparent=transparent,
                exact_usage=exact,cohort_valid=cohort,cache_policy_valid=modes)
            row['valid']=bool(row['valid'] and valid and exact and cohort and modes)
            if not row['valid']:row['output_tok_s']=row['request_s']=None
            common.dump(folder/(phase+'_requests.json'),records);common.dump(folder/(phase+'_routes.json'),state)
            common.dump(folder/'gateway_errors.json',gateway.errors)
            stats.append(row);common.dump(folder/'cell_statistics.json',stats)
            if not row['valid']:
                raise RuntimeError(report_invalid_phase(folder,row,records,gateway.errors,len(items)))
            all_records.extend(records);all_routes.extend(routes);durations.append(elapsed);pool.assert_alive()
            common.progress(root,'PHASE_END',cell=label,phase=phase,output_tok_s=row['output_tok_s'],mean_ttft_ms=row['ttft_mean_ms'])
        combined=phase_statistics(all_records,sum(durations),dict(meta,phase='learn_plus_measured',
            ordered_request_contract_sha256=contract(rt.rows(work/'learn.jsonl')+rt.rows(work/'measured.jsonl'))))
        combined.update(extra_stats(all_records,all_routes,case['shared_tokens']),route_valid=True,event_payload_transparent=True,
            wall_definition='sum of timed learning and measured phases; excludes artifact writes and calibration')
        stats.append(combined);common.dump(folder/'cell_statistics.json',stats);rt.csv_write(folder/'summary.csv',stats)
        common.dump(folder/'status.json',dict(status='COMPLETE'))
        common.dump(folder/'runtime_features.json',runtime_features(pool))
        return stats
    except BaseException as exc:
        common.dump(folder/'status.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        if gateway:await gateway.close()
        if router is not None and hasattr(router,'close'):router.close()


def report(root,rows,plan):
    rt.csv_write(root/'summary.csv',rows)
    lookup={}
    for r in rows:
        key=tuple(r[k] for k in ('profile','seed','concurrency','arm','phase'))
        if key in lookup:raise ValueError('Duplicate cell/phase')
        lookup[key]=r
    pairs=[]
    for key,r in lookup.items():
        if not r.get('valid') or r['arm']!='nh_cache_bounded_v2':continue
        for baseline in [a[0] for a in plan['arms'] if a[0]!='nh_cache_bounded_v2']:
            other=lookup.get((r['profile'],r['seed'],r['concurrency'],baseline,r['phase']))
            if not other or not other.get('valid'):continue
            if r['ordered_request_contract_sha256']!=other['ordered_request_contract_sha256']:
                raise ValueError('Compared ordered token cohorts differ')
            t=r['output_tok_s']/other['output_tok_s'];m=r['ttft_mean_ms']/other['ttft_mean_ms'];p=r['ttft_p95_ms']/other['ttft_p95_ms']
            pairs.append(dict(profile=r['profile'],seed=r['seed'],concurrency=r['concurrency'],phase=r['phase'],baseline=baseline,
                throughput_ratio=t,mean_ttft_ratio=m,p95_ttft_ratio=p,
                both_metrics_better=t>1 and m<1,old_mean_screen_met=t>=.95 and m<1,
                tail_regression=p>1.05))
    rt.csv_write(root/'paired.csv',pairs)
    expected={(pr,se,c,a[0],ph) for pr in plan['profiles'] for se in plan['seeds'] for c in plan['concurrency'] for a in plan['arms'] for ph in ('learn','measured','learn_plus_measured')}
    actual={key for key,row in lookup.items() if row.get('valid')}
    v=[r for r in pairs if r['baseline']=='vllm_least_active' and r['phase']=='measured']
    result=dict(complete=actual==expected,expected_phase_cells=len(expected),valid_phase_cells=len(actual),
        measured_vllm_comparisons=len(v),candidate_both_metrics_better_count=sum(r['both_metrics_better'] for r in v),
        candidate_tail_regression_count=sum(r['tail_regression'] for r in v),performance_win_claimed=False,
        interpretation='Per-cell ratios only; a single seed provides no confidence interval. Completeness is not a performance PASS.')
    common.dump(root/'qualification.json',result)
    lines=['# InfiniLM cache-preserving bounded-hash benchmark','',json.dumps(result,ensure_ascii=False),'',
        'Only same profile / seed / concurrency / phase pairs are compared. All wall times include ramp/drain.',
        'Calibration overhead is separately reported in phase_model.json; learn_plus_measured includes learning, not calibration.',
        'Client TPOT is per-request (E2E-TTFT)/(exact output tokens-1), including stream finalization; it is not token ITL or SSE chunk count.',
        '','| Profile | Seed | C | Method | Phase | tok/s | Mean TTFT ms | p95 ms | p99 ms | E2E mean ms | client TPOT mean ms | failures |',
        '|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def val(x):return 'NA' if x is None else f'{x:.3f}'
    for r in rows:
        if r['phase']!='measured':continue
        lines.append(f'| {r["profile"]} | {r["seed"]} | {r["concurrency"]} | {r["arm"]} | {r["phase"]} | {val(r["output_tok_s"])} | {val(r["ttft_mean_ms"])} | {val(r["ttft_p95_ms"])} | {val(r["ttft_p99_ms"])} | {val(r["e2e_mean_ms"])} | {val(r.get("client_tpot_mean_ms"))} | {r["failures"]} |')
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return result


async def execute(a,cfg,root,plan):
    from aiohttp import ClientSession,ClientTimeout,TCPConnector
    from cache_calibration import calibrate
    common.PROTOCOL=PROTOCOL
    task=asyncio.current_task()
    for sig in (signal.SIGINT,signal.SIGTERM):asyncio.get_running_loop().add_signal_handler(sig,task.cancel)
    fp=await common.preflight(cfg,root)
    fp['hardware']=await hardware_fingerprint(cfg,root)
    common.dump(root/'fingerprint.json',fp)
    if fp['nh_files']['python/infinilm/llm/cache_manager.py']!=EXPECTED_SHA:
        raise RuntimeError('BlockManager changed from verified source; inspect exact source before adapting the wrapper')
    if a.mode=='check':return dict(status='CHECK_COMPLETE',gpu_jobs_started=0,model_loads=0)
    argv=[cfg['nh_python'],str(KIT/'cache_workload.py'),'--model',cfg['model'],'--out',str(root/'workloads'),
        '--profiles',*a.profiles,'--seeds',*map(str,a.seeds),'--requests',str(a.requests)]
    await common.run_logged(argv,rt.backend_env(cfg,'nh',cfg['gpu_ids'][0],'/tmp'),cfg['repo'],root/'workload_build.log',timeout=1800)
    manifest=json.loads((root/'workloads'/'manifest.json').read_text())
    expected_cases={(p,se) for p in a.profiles for se in a.seeds}
    if {(c['profile'],c['seed']) for c in manifest['cells']}!=expected_cases or len(manifest['cells'])!=len(expected_cases):
        raise ValueError('Generated profile/seed cells differ from plan')
    if any(c['files']['learn']['count']!=64 or c['files']['measured']['count']!=a.requests for c in manifest['cells']):
        raise ValueError('Generated cohort counts differ from plan')
    if a.phase_model:
        model=json.loads(a.phase_model.read_text())
        if not fp['hardware'].get('available') or not fp['hardware'].get('devices'):
            raise ValueError('Hardware identity unavailable: do not reuse a saved model; calibrate within this run instead')
        if model.get('fingerprint')!=fp:raise ValueError('Calibration model fingerprint differs from current runtime/config/source')
        common.dump(root/'phase_model.json',model)
    else:model=await calibrate(cfg,root,root/'workloads',fp)
    from phase_router import PhaseModel
    PhaseModel(model)  # Validate calibrated units and all measured-grid values before any 8-replica pool starts.
    if a.mode=='calibrate':return dict(status='CALIBRATION_COMPLETE',model=str(root/'phase_model.json'),performance_win_claimed=False)
    # NH weights persist across drained/reset cells. vLLM is fresh per case/C.
    specifications=[('nh',manifest['cells'],a.concurrency)]
    specifications += [('vllm',[case],[c]) for case in manifest['cells'] for c in a.concurrency if any(x[1]=='vllm' for x in plan['arms'])]
    rows=[]
    try:
        for pool_no,(engine,cases,levels) in enumerate(specifications):
            arms=[x for x in plan['arms'] if x[1]==engine]
            if not arms:continue
            pool=rt.Pool(cfg,engine,root/'pools'/f'{pool_no:03d}_{engine}')
            try:
                await pool.start()
                common.dump(pool.folder/'runtime_features.json',runtime_features(pool))
                async with ClientSession(timeout=ClientTimeout(total=1800),connector=TCPConnector(limit=0),read_bufsize=2**20) as client:
                    for case_no,case in enumerate(cases):
                        for level_no,c in enumerate(levels):
                            shift=(case_no+level_no)%len(arms)
                            for arm in arms[shift:]+arms[:shift]:
                                rows += await run_cell(cfg,root,pool,case,c,arm,client,model,plan['router_options'])
                                report(root,rows,plan)
            finally:await pool.close()
    finally:
        rows=[r for path in sorted((root/'cells').glob('*/cell_statistics.json')) for r in json.loads(path.read_text())]
        final=report(root,rows,plan)
    if not final['complete']:raise RuntimeError('Some required cells or phases are missing/invalid')
    return dict(status='EXPERIMENT_COMPLETE',**final)


def main(argv=None):
    a=arguments(argv);cfg=configuration(a.config);plan=make_plan(a,cfg)
    if a.mode=='plan':print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    import importlib.util
    missing=[name for name in ('aiohttp','numpy') if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit('Controller interpreter '+sys.executable+' lacks '+','.join(missing)+'. Set CONTROLLER_PYTHON to the previously verified controller; no packages were installed.')
    lock=open('/tmp/infinilm-fourway-'+str(__import__('os').getuid())+'.lock','a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('An existing benchmark owns this lock; do not delete it.',file=sys.stderr);return 2
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root=a.output_root.resolve()/('cache_v2_'+a.mode+'_'+stamp);root.mkdir(parents=True)
    common.dump(root/'plan.json',plan);common.dump(root/'status.json',dict(status='RUNNING'))
    for rel in common.source_hashes():
        to=root/'source'/rel;to.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(KIT/rel,to)
    print('RUN_DIR='+str(root),flush=True)
    try:
        result=asyncio.run(execute(a,cfg,root,plan));common.dump(root/'status.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True);code=0
    except BaseException as exc:
        common.dump(root/'status.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc()))
        print('FAILED '+repr(exc)+'; evidence='+str(root),file=sys.stderr,flush=True);code=2
    finally:
        lock.close();archive=shutil.make_archive(str(root)+'_report','zip',root.parent,root.name)
        print('REPORT_ARCHIVE='+archive,flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
