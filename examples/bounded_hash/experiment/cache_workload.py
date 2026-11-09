"""Declared favorable prefix-reuse workloads, paired identically across arms."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from workload import document, row, write_rows
from phase_workload import calibration as build_calibration

PROFILES = {
    'reuse8': dict(prompt_tokens=8192, shared_tokens=7936, output_tokens=32,
                   documents=8, hot_fraction=.25),
    'reuse16': dict(prompt_tokens=8192, shared_tokens=7936, output_tokens=32,
                    documents=16, hot_fraction=.25),
    'prefix97': dict(prompt_tokens=8192, shared_tokens=7936, output_tokens=64,
                     documents=16, hot_fraction=.25),
    'longdecode': dict(prompt_tokens=2048, shared_tokens=1536, output_tokens=512,
                       documents=16, hot_fraction=.25),
    'cold': dict(prompt_tokens=8192, shared_tokens=0, output_tokens=32,
                 documents=0, hot_fraction=0.),
}


def arrivals(documents, requests, hot_fraction, seed):
    """No topology/ring-dependent input selection or worker-directed warmup."""
    if documents not in (8, 16) or requests % 64 or hot_fraction != .25:
        raise ValueError('Use 8/16 documents and complete hot25 cohorts')
    rng = random.Random(seed)
    hot_keys = list(range(documents))
    rng.shuffle(hot_keys)
    result = []
    for start in range(0, requests, 64):
        hot = hot_keys[(start // 256) % documents]
        background = [i for i in range(documents) if i != hot]
        rng.shuffle(background)
        cohort = [hot] * 16 + (background * 8)[:48]
        rng.shuffle(cohort)
        result.extend(cohort)
    return result


def build(tokenizer, out, profiles, seeds, requests):
    if requests not in (64, 128, 256):
        raise ValueError('Use 64, 128 or 256 measured requests')
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(cells=[], profiles=profiles, seeds=seeds, measured_requests=requests,
        learning_requests=64, arrival_model='closed-loop; shuffled hot25 arrivals',
        purpose='Declared favorable long-prefix/short-output screen, not general-serving superiority',
        key_selection='document hash, independent of ring and measured performance',
        warmup='Identical 64 learning requests through each actual gateway; no worker-specific priming')
    used_blocks = set()
    for profile in profiles:
        params = PROFILES[profile]
        p, s, o, n = (params[k] for k in ('prompt_tokens','shared_tokens','output_tokens','documents'))
        for seed in seeds:
            folder = out / f'{profile}_seed{seed}'
            folder.mkdir()
            docs = [document(tokenizer,seed,f'cache-v2-{profile}-{i}',s or 512) for i in range(n)]
            probe = document(tokenizer,seed,f'cache-v2-{profile}-probe',s or 512)
            probes = [row(tokenizer,probe,f'cache-v2-{profile}-{seed}-probe-{i}',p,o,s) for i in range(2)]
            rng = random.Random(seed+17)
            learn = []
            if n:
                for _ in range(64//n):
                    cycle = list(range(n)); rng.shuffle(cycle); learn.extend(cycle)
                measured = arrivals(n, requests, params['hot_fraction'], seed+31)
            else:
                learn, measured = list(range(64)), list(range(requests))
            sequences = dict(learn=learn, measured=measured)
            seen, prefixes, blocks, files = set(), {}, set(), {}
            for phase, sequence in sequences.items():
                items = []
                for i, g in enumerate(sequence):
                    ident = f'cache-v2-{profile}-{seed}-{phase}-{i}-{g}'
                    doc = docs[g] if n else document(tokenizer,seed,ident,512)
                    item = row(tokenizer,doc,ident,p,o,s)
                    if item['prompt_sha256'] in seen:
                        raise ValueError('Duplicate full input')
                    seen.add(item['prompt_sha256'])
                    if item['key'] in prefixes and prefixes[item['key']] != item['prefix_digest']:
                        raise ValueError('Document prefix is not stable')
                    prefixes[item['key']] = item['prefix_digest']
                    blocks.add(item['first_block_digest'])
                    items.append(item)
                files[phase] = write_rows(folder/(phase+'.jsonl'), items)
            if s and len(blocks) != n:
                raise ValueError('Documents share a first cache block')
            if blocks & used_blocks or probes[0]['first_block_digest'] in blocks | used_blocks:
                raise ValueError('Workload/probe overlap')
            used_blocks.update(blocks); used_blocks.add(probes[0]['first_block_digest'])
            (folder/'smoke.json').write_text(json.dumps(probes,ensure_ascii=False))
            manifest['cells'].append(dict(profile=profile,seed=seed,folder=folder.name,**params,
                files=files,unique_full_inputs=len(seen),cold_distinct_documents=(s==0)))
    calibration_dir = out/'calibration'
    build_calibration(tokenizer,calibration_dir,used_blocks)
    # These requests run before, and are excluded from, all calibration fits.
    # Cache is then reset so the first retained cold sample is genuinely cold.
    untimed = []
    for p in (2048,4096,8192):
        doc = document(tokenizer,991827361,f'cache-v2-untimed-{p}',p-256)
        item = row(tokenizer,doc,f'cache-v2-untimed-{p}',p,8,p-256)
        if item['first_block_digest'] in used_blocks:
            raise ValueError('Untimed warmup overlaps measured/calibration prefixes')
        used_blocks.add(item['first_block_digest']); untimed.append(item)
    write_rows(calibration_dir/'untimed_warmup.jsonl',untimed)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--profiles',nargs='+',choices=list(PROFILES),default=['reuse8'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[33959])
    ap.add_argument('--requests',type=int,choices=[64,128,256],default=128)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
    result = build(tok,Path(a.out),a.profiles,a.seeds,a.requests)
    print('TOKEN_VERIFICATION_PASS cells='+str(len(result['cells'])),flush=True)


if __name__ == '__main__':
    main()
