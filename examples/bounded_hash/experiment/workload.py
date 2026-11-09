"""Build identical, token-verified long-prefix requests before GPU timing."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def prompt_ids(tokenizer, text):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True,
                                         add_generation_prompt=True, enable_thinking=False)


def exact(tokenizer, text, target):
    count = target - len(prompt_ids(tokenizer, text))
    for _ in range(20):
        if count < 0:
            break
        content = text + ' detail' * count
        ids = prompt_ids(tokenizer, content)
        if len(ids) == target:
            return content, ids
        count += target - len(ids)
    raise ValueError('Cannot construct exact prompt length; inspect tokenizer/template')


def document(tokenizer, seed, group, shared):
    identity = hashlib.sha256(f'fourway:{seed}:{group}'.encode()).hexdigest()
    unit = ('The service records capacity, completed operations, retry counts, '
            'and maintenance intervals. Compare observations using only this document. ')
    text = f'Document {identity}. Tenant operational notes. /no_think\n' + unit*1400
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[:shared], skip_special_tokens=False)


def row(tokenizer, doc, identifier, p=2048, o=512, shared=1536):
    tail = f'\nQuestion {identifier}: Summarize capacity and reliability, then state one limitation. /no_think\n'
    content, ids = exact(tokenizer, doc + tail, p)
    # Count actual full input tokens, including the chat template. Every same-key
    # row is checked for equality of these entire shared prefix blocks below.
    return dict(id=str(identifier), key='doc-'+hashlib.sha256(doc.encode()).hexdigest(),
                prompt_tokens=p, output_tokens=o, shared_tokens=shared,
                prefix_digest=digest(ids[:shared]), first_block_digest=digest(ids[:256]),
                prompt_sha256=digest(ids),
                body=dict(messages=[{'role': 'user', 'content': content}], temperature=0.0,
                          top_p=1.0, top_k=1, max_tokens=o, ignore_eos=True, n=1,
                          chat_template_kwargs={'enable_thinking': False}))


def order(keys, requests, hot_fraction, seed):
    """One 25% hot document per 256 arrivals; shuffled within each 64 cohort."""
    if requests % 64 or keys != 16 or hot_fraction != .25:
        raise ValueError('Frozen scenario requires 16 documents, hot25, whole 64-request cohorts')
    rng = random.Random(seed)
    hot_keys = list(range(keys))
    rng.shuffle(hot_keys)
    result = []
    for start in range(0, requests, 64):
        hot = hot_keys[(start//256) % keys]
        background = [i for i in range(keys) if i != hot]
        rng.shuffle(background)
        cohort = [hot]*16 + (background*4)[:48]
        rng.shuffle(cohort)
        result.extend(cohort)
    return result


def write_rows(path, values):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in values))
    return dict(count=len(values), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def build(tokenizer, out, opts):
    out.mkdir(parents=True, exist_ok=True)
    p, o, shared = 8192, 64, 7680
    if (opts['prompt_tokens'], opts['output_tokens'], opts['shared_tokens'], opts['keys']) != (p,o,shared,16):
        raise ValueError('This final workload is frozen at 8192/64/7680 and 16 documents')
    manifest = dict(options=opts, arrival_model='closed-loop; hot25 shuffled within each 64-arrival cohort',
                    key_selection='sha256(document); no ring search', cells=[])
    all_blocks = set()
    for seed in opts['seeds']:
        docs = [document(tokenizer, seed, i, shared) for i in range(16)]
        probe_doc = document(tokenizer, seed, 'independent-smoke-final', shared)
        probes = [row(tokenizer, probe_doc, f'smoke-{seed}-{stage}', p,o,shared) for stage in ('cold','warm')]
        folder = out/f'seed{seed}_hot25'
        folder.mkdir()
        rng = random.Random(seed+17)
        learn = []
        for _ in range(opts.get('learn_repeats', 4)):
            visit = list(range(16)); rng.shuffle(visit); learn.extend(visit)
        plans = dict(learn=learn, steady=order(16,opts['requests'],.25,seed+31))
        seen, prefixes, blocks, files = set(), {}, {}, {}
        for phase, groups in plans.items():
            items = [row(tokenizer,docs[g],f'{seed}-{phase}-{i}-{g}',p,o,shared) for i,g in enumerate(groups)]
            for item in items:
                if item['prompt_sha256'] in seen:
                    raise ValueError('Duplicate full prompt')
                seen.add(item['prompt_sha256'])
                key = item['key']
                if key in prefixes and prefixes[key] != item['prefix_digest']:
                    raise ValueError('Same document must have exactly the declared shared prefix')
                prefixes[key], blocks[key] = item['prefix_digest'], item['first_block_digest']
            files[phase] = write_rows(folder/(phase+'.jsonl'), items)
        if len(set(blocks.values())) != 16 or probes[0]['first_block_digest'] in blocks.values():
            raise ValueError('Formal and probe documents must have disjoint first cache blocks')
        if probes[0]['prefix_digest'] != probes[1]['prefix_digest']:
            raise ValueError('Probe prefix differs')
        if all_blocks & set(blocks.values()):
            raise ValueError('Seeds must use disjoint documents')
        all_blocks.update(blocks.values()); all_blocks.add(probes[0]['first_block_digest'])
        (folder/'smoke.json').write_text(json.dumps(probes,ensure_ascii=False))
        manifest['cells'].append(dict(seed=seed,hot_fraction=.25,folder=folder.name,
                                      unique_full_inputs=len(seen),files=files))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if opts.get('calibration', True):
        calibration(tokenizer,out,all_blocks)


def calibration(tokenizer,out,formal_blocks):
    items, used = [], set(formal_blocks)
    for p in (2048,4096,8192):
        shared = p-512
        for repeat in range(2):
            doc = document(tokenizer,91827461,f'final-cal-{p}-{repeat}',shared)
            pair = []
            for stage in ('cold','warm'):
                item = row(tokenizer,doc,f'cal-{p}-{repeat}-{stage}',p,16,shared)
                item['calibration_stage'] = stage
                pair.append(item); items.append(item)
            if pair[0]['prefix_digest'] != pair[1]['prefix_digest']:
                raise ValueError('Calibration prefix mismatch')
            if pair[0]['first_block_digest'] in used:
                raise ValueError('Calibration prefix overlaps another pair or test')
            used.add(pair[0]['first_block_digest'])
    write_rows(out/'calibration.jsonl',items)
    decode, warmup = [], []
    # One independent long document per concurrency level. Concurrent probes
    # share its prefix and have distinct suffixes. This avoids requiring several
    # unpinned documents to survive eviction in the already-used calibration pool.
    # These documents remain disjoint from prefill calibration and formal tests.
    for batch in (1,2,3):
        doc = document(tokenizer,91827461,f'final-decode-shared-{batch}',7680)
        warm = row(tokenizer,doc,f'decode-warm-{batch}',8192,16,7680)
        warm['batch'] = batch
        if warm['first_block_digest'] in used:
            raise ValueError('Decode calibration document overlaps')
        used.add(warm['first_block_digest'])
        warmup.append(warm)
        full_inputs = {warm['prompt_sha256']}
        for i in range(batch):
            item = row(tokenizer,doc,f'decode-{batch}-{i}',8192,16,7680)
            item['batch'] = batch
            if item['prefix_digest'] != warm['prefix_digest']:
                raise ValueError('Decode probe does not share its warmup prefix')
            if item['prompt_sha256'] in full_inputs:
                raise ValueError('Decode calibration requires distinct suffixes')
            full_inputs.add(item['prompt_sha256'])
            decode.append(item)
    write_rows(out/'decode_calibration.jsonl',decode)
    write_rows(out/'decode_warmup.jsonl',warmup)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True); ap.add_argument('--out',required=True); ap.add_argument('--options',required=True)
    a=ap.parse_args()
    if Path(a.model).resolve() != Path('/mnt/zj-data/models/Qwen3-32B/').resolve():
        raise ValueError('Use exactly /mnt/zj-data/models/Qwen3-32B/')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
    build(tokenizer,Path(a.out),json.loads(a.options))
    print('TOKEN_VERIFICATION_PASS: every formal input8192/output64/shared7680; calibration disjoint',flush=True)


if __name__=='__main__':
    main()
