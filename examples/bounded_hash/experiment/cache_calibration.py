"""V2 calibration: independent untimed warmup, reset, then retained samples."""
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
import backend_runtime as rt
import harness as common
from streaming import Observation, events, request_one


async def timed_decode(client,url,item,model):
    payload=common.nh_metadata(item)
    body=dict(payload['body'],model=model,stream=True,stream_options={'include_usage':True})
    obs=Observation();begin=time.perf_counter();samples=[]
    async with client.post(url+'/v1/chat/completions',json=body) as response:
        if response.status!=200:raise RuntimeError('Calibration HTTP '+str(response.status))
        async for data in events(response.content):
            now=time.perf_counter();obj=obs.accept(data,now)
            if obj is not None and any(c.get('delta',{}).get('content') for c in obj.get('choices',[])):
                count=(obj.get('x_infinilm') or {}).get('generated_tokens')
                if not isinstance(count,int):raise RuntimeError('Decode calibration needs exact generated_tokens telemetry')
                if samples and count<samples[-1]['generated_tokens']:raise RuntimeError('Generated-token counter decreased')
                if not samples or count>samples[-1]['generated_tokens']:
                    samples.append(dict(absolute_s=now,generated_tokens=count))
    obs.validate(item)
    return dict(id=item['id'],success=True,group=item['group'],batch=item['batch'],
        context_tokens=item['context_tokens'],repeat=item['repeat'],usage=obs.usage,
        ttft_ms=1000*(obs.first-begin),e2e_ms=1000*(time.perf_counter()-begin),token_events=samples)


def overlap_step_ms(records):
    """Count per-request token advances only while every request is decoding."""
    if not records or any(len(r['token_events'])<3 for r in records):
        raise ValueError('Insufficient decode event samples')
    start=max(r['token_events'][0]['absolute_s'] for r in records)
    end=min(r['token_events'][-1]['absolute_s'] for r in records)
    if end<=start:raise ValueError('No shared all-decoding interval')
    rates=[];evidence=[]
    for r in records:
        selected=[e for e in r['token_events'] if start<=e['absolute_s']<=end]
        if len(selected)<3:raise ValueError('Too few events inside shared decode interval')
        first,last=selected[0],selected[-1]
        n=last['generated_tokens']-first['generated_tokens']
        elapsed=last['absolute_s']-first['absolute_s']
        if n<4 or elapsed<=0:raise ValueError('Insufficient exact token advances for decode calibration')
        rates.append(elapsed*1000/n)
        evidence.append(dict(id=r['id'],first=first,last=last,token_advances=n))
    return statistics.mean(rates),dict(common_start_s=start,common_end_s=end,per_request=evidence)


async def calibrate(cfg,root,work,fp):
    from aiohttp import ClientSession,ClientTimeout,TCPConnector
    from matrix import reset_pool
    folder=root/'calibration';folder.mkdir(parents=True,exist_ok=True)
    pool=rt.Pool(cfg,'nh',folder/'workers',count=1)
    prefill,decode,warmups=[],[],[]
    began=time.perf_counter()
    try:
        common.progress(root,'CALIBRATION_BEGIN',replicas=1,other_pool_not_started=True)
        await pool.start()
        async with ClientSession(timeout=ClientTimeout(total=1800),connector=TCPConnector(limit=0, force_close=True),read_bufsize=2**20) as client:
            await reset_pool(client,pool,'prefix_lru',folder)
            warmup_records = []
            for item in rt.rows(work/'calibration'/'untimed_warmup.jsonl'):
                result = await request_one(client,pool.urls[0],common.nh_metadata(item),pool.models[0],False)
                warmup_records.append(result)
                common.dump(folder/'untimed_warmup_requests.json',warmup_records)
                if not result['success']:
                    raise RuntimeError('Untimed calibration warmup failed')
            await reset_pool(client,pool,'prefix_lru',folder)
            common.progress(root,'UNTIMED_WARMUP_COMPLETE',requests=len(warmup_records),included_in_fit=False)
            for item in rt.rows(work/'calibration'/'prefill.jsonl'):
                result=await request_one(client,pool.urls[0],common.nh_metadata(item),pool.models[0],False)
                result.update(calibration_stage=item['calibration_stage'],repeat=item['repeat'])
                prefill.append(result);common.dump(folder/'prefill_requests.json',prefill)
                if not result['success']:raise RuntimeError('Isolated prefill calibration failed')
                cached=(result['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
                if not isinstance(cached,int) or cached<0 or cached>=item['prompt_tokens']:
                    raise RuntimeError('Exact initial cache counter missing')
                if item['calibration_stage']=='cold' and cached!=0:raise RuntimeError('Cold calibration reused cached block')
                if item['calibration_stage']=='warm' and cached<item['shared_tokens']:raise RuntimeError('Warm calibration missed declared prefix')
                common.progress(root,'PREFILL_CALIBRATION_SAMPLE',completed=len(prefill),input_tokens=item['prompt_tokens'],cached_tokens=cached)
            samples=rt.rows(work/'calibration'/'decode.jsonl')
            groups=[]
            for warm in rt.rows(work/'calibration'/'decode_warmup.jsonl'):
                # A reset and explicit warmup isolate every group and both contexts.
                await reset_pool(client,pool,'prefix_lru',folder)
                result=await request_one(client,pool.urls[0],common.nh_metadata(warm),pool.models[0],False)
                result.update(group=warm['group']);warmups.append(result)
                common.dump(folder/'decode_warmup_requests.json',warmups)
                if not result['success']:raise RuntimeError('Decode warmup failed')
                items=[i for i in samples if i['group']==warm['group']]
                records=await asyncio.gather(*(timed_decode(client,pool.urls[0],i,pool.models[0]) for i in items))
                if any((r['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',-1)<warm['shared_tokens'] for r in records):
                    raise RuntimeError('Decode calibration did not hit warm prefixes')
                step,evidence=overlap_step_ms(records)
                decode.extend(records);common.dump(folder/'decode_requests.json',decode)
                groups.append(dict(group=warm['group'],batch=warm['batch'],context_tokens=warm['context_tokens']*warm['batch'],
                    step_ms=step,overlap=evidence));common.dump(folder/'decode_groups.json',groups)
                common.progress(root,'DECODE_CALIBRATION_GROUP',group=warm['group'],step_ms=step)
        x=[];y=[]
        for r in prefill:
            p=r['usage']['prompt_tokens'];c=r['usage']['prompt_tokens_details']['cached_tokens'];q=p-c
            x.append([1,q/1000,(q*q+2*q*c)/1e6]);y.append(r['ttft_ms'])
        coef=rt.nnls_small(x,y)
        if any(not math.isfinite(v) or v<0 for v in coef):raise RuntimeError('Invalid fitted prefill model')
        grid={};steps={};contexts={}
        for b in (1,2,3):
            grid[str(b)]=[]
            for p in (2048,8192):
                observed=[g['step_ms'] for g in groups if g['batch']==b and g['context_tokens']==p*b]
                grid[str(b)].append(dict(context_tokens=p*b,step_ms=statistics.median(observed)))
            steps[str(b)]=grid[str(b)][-1]['step_ms'];contexts[str(b)]=8192*b
        predicted=[sum(a*b for a,b in zip(coef,r)) for r in x]
        profile=dict(schema_version=1,calibration_id=root.name,prefill_coef=coef,decode_step_ms=steps,
            decode_context_tokens=contexts,decode_context_samples=grid,fingerprint=fp,
            calibration_wall_s=time.perf_counter()-began,untimed_warmup_samples=len(warmup_records),
            untimed_warmup_included_in_fit=False,prefill_samples=len(prefill),decode_groups=len(groups),
            prefill_fit_mae_ms=statistics.mean(abs(a-b) for a,b in zip(y,predicted)),
            provenance='Independent 2K/4K/8K warmup, then cache reset before retained samples. Isolated NH replica only. Sequential C1 cold/warm TTFT fit; concurrent B1/B2/B3 warm decode at context2048/8192, each repeated twice. Token advances measured only inside the common all-decoding interval. Client effective proxies, not GPU event times or guaranteed TTFT.',
            calibration_is_performance_data=False,performance_win_claimed=False)
        common.dump(root/'phase_model.json',profile)
        common.progress(root,'CALIBRATION_COMPLETE',path=str(root/'phase_model.json'))
        return profile
    finally:await pool.close()
