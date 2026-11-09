"""Deterministic token-exact profiles and independent calibration requests."""
import argparse
import json
from pathlib import Path
import random
from workload import document, row, order, write_rows

PROFILES = {
    'prefix97': dict(prompt_tokens=8192, shared_tokens=7936, output_tokens=64),
    'prefix50': dict(prompt_tokens=8192, shared_tokens=4096, output_tokens=64),
    'longdecode': dict(prompt_tokens=2048, shared_tokens=1536, output_tokens=512),
    'cold': dict(prompt_tokens=8192, shared_tokens=0, output_tokens=64),
}


def build(tokenizer, out, profiles, seeds, requests):
    if requests < 64 or requests % 64:
        raise ValueError('requests must be a positive whole number of 64-request cohorts')
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(cells=[], profiles=profiles, seeds=seeds, measured_requests=requests,
        purpose='Synthetic systems workloads; no task-quality/general-serving claim',
        ordering='Identical stored token-exact ordered cohorts for every method in the same cell',
        hot_fraction=.25, learning_requests=64, key_selection='document hash, independent of hash ring and results')
    first_blocks = set()
    for profile in profiles:
        params = PROFILES[profile]
        p,s,o = params['prompt_tokens'],params['shared_tokens'],params['output_tokens']
        for seed in seeds:
            folder = out/f'{profile}_seed{seed}'
            folder.mkdir()
            docs = [document(tokenizer,seed,f'{profile}-{i}',s or 512) for i in range(16)]
            probe_doc = document(tokenizer,seed,f'{profile}-independent-probe',s or 512)
            probes = [row(tokenizer,probe_doc,f'{profile}-{seed}-probe-{j}',p,o,s) for j in range(2)]
            rng=random.Random(seed+17)
            learn=[]
            for _ in range(4):
                visit=list(range(16));rng.shuffle(visit);learn.extend(visit)
            sequences={'learn':learn,'measured':order(16,requests,.25,seed+31)}
            seen,prefixes,blocks,files=set(),{},set(),{}
            for phase,groups in sequences.items():
                items=[]
                for i,g in enumerate(groups):
                    ident=f'{profile}-{seed}-{phase}-{i}-{g}'
                    doc=docs[g] if s else document(tokenizer,seed,ident,512)
                    item=row(tokenizer,doc,ident,p,o,s)
                    if item['prompt_sha256'] in seen:raise ValueError('Duplicate complete input')
                    seen.add(item['prompt_sha256'])
                    if item['key'] in prefixes and prefixes[item['key']]!=item['prefix_digest']:
                        raise ValueError('Same document key has different declared prefix')
                    prefixes[item['key']]=item['prefix_digest'];blocks.add(item['first_block_digest'])
                    items.append(item)
                files[phase]=write_rows(folder/(phase+'.jsonl'),items)
            if s and len(blocks)!=16:raise ValueError('Documents must have disjoint first blocks')
            if blocks & first_blocks:raise ValueError('Profiles/seeds must not reuse first cache blocks')
            if probes[0]['first_block_digest'] in blocks:raise ValueError('Independent probe overlaps measured inputs')
            first_blocks.update(blocks);first_blocks.add(probes[0]['first_block_digest'])
            (folder/'smoke.json').write_text(json.dumps(probes,ensure_ascii=False))
            manifest['cells'].append(dict(profile=profile,seed=seed,folder=folder.name,**params,
                files=files,unique_full_inputs=len(seen),documents=len(prefixes),cold_distinct_documents=(s==0)))
    calibration(tokenizer,out/'calibration',first_blocks)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


def calibration(tokenizer,out,used):
    out.mkdir(parents=True,exist_ok=True)
    prefill=[]
    for p,shared in [(2048,1536),(4096,3584),(8192,7936),(8192,4096)]:
        for repeat in range(2):
            doc=document(tokenizer,91827461,f'phase-prefill-{p}-{shared}-{repeat}',shared)
            for stage in ('cold','warm'):
                item=row(tokenizer,doc,f'cal-{p}-{shared}-{repeat}-{stage}',p,32,shared)
                item.update(calibration_stage=stage,repeat=repeat)
                if stage=='cold':
                    if item['first_block_digest'] in used:raise ValueError('Calibration overlap')
                    used.add(item['first_block_digest'])
                prefill.append(item)
    write_rows(out/'prefill.jsonl',prefill)
    decode,warmup=[],[]
    for p,shared in [(2048,1536),(8192,7936)]:
        for batch in (1,2,3):
            for repeat in range(2):
                group=f'decode-{p}-b{batch}-r{repeat}'
                doc=document(tokenizer,91827461,'phase-'+group,shared)
                warm=row(tokenizer,doc,group+'-warm',p,16,shared)
                warm.update(group=group,batch=batch,context_tokens=p,repeat=repeat)
                if warm['first_block_digest'] in used:raise ValueError('Decode calibration overlap')
                used.add(warm['first_block_digest']);warmup.append(warm)
                for i in range(batch):
                    item=row(tokenizer,doc,group+f'-{i}',p,64,shared)
                    if item['prefix_digest']!=warm['prefix_digest']:raise ValueError('Decode prefix mismatch')
                    item.update(group=group,batch=batch,context_tokens=p,repeat=repeat)
                    decode.append(item)
    write_rows(out/'decode.jsonl',decode);write_rows(out/'decode_warmup.jsonl',warmup)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--profiles',nargs='+',choices=list(PROFILES),default=list(PROFILES)[:3])
    ap.add_argument('--seeds',nargs='+',type=int,default=[33959]);ap.add_argument('--requests',type=int,default=128)
    a=ap.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
    m=build(tok,Path(a.out),a.profiles,a.seeds,a.requests)
    print('TOKEN_VERIFICATION_PASS cells='+str(len(m['cells'])),flush=True)

if __name__=='__main__':main()
