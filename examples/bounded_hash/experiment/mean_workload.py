"""Frozen synthetic prefix-reuse workload for Qwen3-32B mean-TTFT experiments.

This is a favorable long-prefix stress test, not an application-quality dataset.
No router, hash ring, backend identity or benchmark result enters construction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from workload import document, row, order, write_rows

MODEL = '/mnt/zj-data/models/Qwen3-32B/'
KIT = Path(__file__).resolve().parent
PROFILES = {'prefix97': 7936, 'replay94': 7680}


def build(tokenizer, out, opts):
    p, o, shared = opts['prompt_tokens'], opts['output_tokens'], opts['shared_tokens']
    if (p, o, opts['keys']) != (8192, 64, 16) or shared not in PROFILES.values():
        raise ValueError('Frozen workload requires 8192 input, 64 output, 16 keys, shared 7936 or 7680')
    if opts.get('learn_repeats', 4) != 4 or opts.get('calibration', False):
        raise ValueError('Exactly 64 learning requests, no calibration')
    if opts['requests'] not in (128, 256):
        raise ValueError('Use 128 or 256 measured requests; cohorts are 64 arrivals')
    if not opts['seeds'] or len(set(opts['seeds'])) != len(opts['seeds']):
        raise ValueError('Seeds must be nonempty and unique')
    profile = next(k for k,v in PROFILES.items() if v == shared)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(options=opts, profile=profile, cells=[],
                    purpose='favorable synthetic long shared-prefix serving workload; not application-quality evaluation',
                    arrival_model='closed-loop; one hot document receives 25% in each shuffled 64-request cohort; hot identity changes after 256 measured requests',
                    key_selection='sha256(document); independent of ring, backend identities and benchmark outcomes',
                    learning='four independently shuffled passes over all 16 documents; same for every method',
                    primary_phase='measured after learning; include ramp/drain; separately report learning plus measured',
                    declared_shared_fraction=shared/p,
                    subset_rule='C16 uses first 256 measured records, C32/C64 use first 128; pair only at same C/seed/profile')
    all_blocks = set()
    for seed in opts['seeds']:
        docs = [document(tokenizer, seed, i, shared) for i in range(16)]
        probe_doc = document(tokenizer, seed, 'independent-smoke-final', shared)
        probes = [row(tokenizer, probe_doc, f'smoke-{seed}-{stage}', p,o,shared) for stage in ('cold','warm')]
        folder = out/f'seed{seed}_hot25'; folder.mkdir()
        rng = random.Random(seed+17)
        learn = []
        for _ in range(4):
            visit = list(range(16)); rng.shuffle(visit); learn.extend(visit)
        plans = dict(learn=learn, steady=order(16, opts['requests'], .25, seed+31))
        seen, prefixes, blocks, files = set(), {}, {}, {}
        for phase, groups in plans.items():
            items = [row(tokenizer, docs[g], f'{seed}-{phase}-{i}-{g}', p,o,shared) for i,g in enumerate(groups)]
            for item in items:
                if item['prompt_sha256'] in seen:
                    raise ValueError('Duplicate full prompt')
                seen.add(item['prompt_sha256'])
                key = item['key']
                if key in prefixes and prefixes[key] != item['prefix_digest']:
                    raise ValueError('Same document does not have identical declared shared tokens')
                prefixes[key], blocks[key] = item['prefix_digest'], item['first_block_digest']
            files[phase] = write_rows(folder/(phase+'.jsonl'), items)
            # The downloadable test set is a frozen reference, not a separate
            # source of prompts. Regeneration must reproduce every record.
            reference = KIT/'datasets'/profile/folder.name/(phase+'.jsonl')
            if reference.is_file():
                expected = reference.read_text().splitlines()[:len(items)]
                if len(expected) != len(items) or expected != (folder/(phase+'.jsonl')).read_text().splitlines():
                    raise ValueError('Local model tokenizer/template differs from frozen test set: '+str(reference))
        if len(set(blocks.values())) != 16 or probes[0]['first_block_digest'] in blocks.values():
            raise ValueError('Each document and the independent probes require disjoint first cache blocks')
        if probes[0]['prefix_digest'] != probes[1]['prefix_digest']:
            raise ValueError('Probe shared prefixes differ')
        if all_blocks & (set(blocks.values()) | {probes[0]['first_block_digest']}):
            raise ValueError('Seeds must have disjoint first cache blocks')
        all_blocks.update(blocks.values()); all_blocks.add(probes[0]['first_block_digest'])
        (folder/'smoke.json').write_text(json.dumps(probes, ensure_ascii=False))
        manifest['cells'].append(dict(seed=seed, profile=profile, hot_fraction=.25, folder=folder.name,
                                      unique_full_inputs=len(seen), shared_tokens=shared, files=files,
                                      distinct_documents=len(prefixes)))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--options', required=True)
    args = ap.parse_args()
    if Path(args.model).resolve() != Path(MODEL).resolve():
        raise ValueError('Use exactly '+MODEL)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    manifest = build(tokenizer, Path(args.out), json.loads(args.options))
    hashes = {name: hashlib.sha256((Path(args.model)/name).read_bytes()).hexdigest()
              for name in ('tokenizer.json', 'tokenizer_config.json')}
    manifest['local_tokenizer_sha256'] = hashes
    (Path(args.out)/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('TOKEN_VERIFICATION_PASS '+json.dumps(dict(profile=manifest['profile'], tokenizer_sha256=hashes)), flush=True)


if __name__ == '__main__':
    main()
