"""Derive matched eight-document requests from the verified seed-33962 archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from workload import digest, prompt_ids, row, write_rows

SEED = 33962
INPUTS = (4096, 6144, 8192)
OUTPUTS = (32, 128)
PROFILES = {
    f'p{size}_o{answer}': dict(prompt_tokens=size, shared_tokens=size - 256,
                               output_tokens=answer, documents=8, hot_fraction=.25)
    for size in INPUTS for answer in OUTPUTS
}
ANCHOR = Path(__file__).resolve().parent / 'anchor_data'
CASES = {'learn': 64, 'measured': 128, 'smoke': 2}
FILENAMES = dict(learn='learn.jsonl', measured='measured.jsonl', smoke='smoke.json')
CALIBRATION = ('prefill.jsonl', 'decode.jsonl', 'decode_warmup.jsonl', 'untimed_warmup.jsonl')


def _load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def verify_anchor_files(anchor_dir=ANCHOR):
    anchor_dir = Path(anchor_dir)
    recorded = _load(anchor_dir / 'source_hashes.json')
    expected_paths = {'original_case.json'}
    expected_paths.update('reuse8_seed33962/' + x for x in FILENAMES.values())
    expected_paths.update('calibration/' + x for x in CALIBRATION)
    if set(recorded) != expected_paths:
        raise ValueError('Anchor file manifest does not match the archived case layout')
    for name in sorted(expected_paths):
        path = anchor_dir / name
        if not path.is_file() or _sha(path) != recorded[name]:
            raise ValueError('Anchor file is missing or changed: ' + name)
    case = _load(anchor_dir / 'original_case.json')
    if (case['profile'], case['seed'], case['folder'], case['prompt_tokens'],
            case['shared_tokens'], case['output_tokens'], case['documents'], case['hot_fraction']) != (
            'reuse8', SEED, 'reuse8_seed33962', 8192, 7936, 32, 8, .25):
        raise ValueError('Frozen reference case differs from reuse8 C16/C32 input contract')
    for phase in ('learn', 'measured'):
        if case['files'][phase] != {'count': CASES[phase], 'sha256': recorded['reuse8_seed33962/' + FILENAMES[phase]]}:
            raise ValueError('Original case declaration disagrees with anchor: ' + phase)
    return recorded


def _actual_ids(tokenizer, item):
    body = item['body']
    messages = body['messages']
    if (len(messages) != 1 or messages[0].get('role') != 'user'
            or body.get('max_tokens') != item['output_tokens']
            or body.get('ignore_eos') is not True
            or body.get('chat_template_kwargs') != {'enable_thinking': False}):
        raise ValueError('Unsupported request/template contract: ' + item['id'])
    ids = prompt_ids(tokenizer, messages[0]['content'])
    if (len(ids) != item['prompt_tokens'] or digest(ids) != item['prompt_sha256']
            or digest(ids[:item['shared_tokens']]) != item['prefix_digest']
            or digest(ids[:256]) != item['first_block_digest']):
        raise ValueError('Local tokenizer does not reproduce archived tokens: ' + item['id'])
    return ids


def _anchor_contract(tokenizer, anchor_dir):
    base = anchor_dir / 'reuse8_seed33962'
    cohorts = {phase: (_load(base / filename) if phase == 'smoke' else _lines(base / filename))
               for phase, filename in FILENAMES.items()}
    documents, prefixes, blocks = {}, {}, {}
    unique_ids, unique_inputs = set(), set()
    group_keys = {}
    for phase, records in cohorts.items():
        if len(records) != CASES[phase]:
            raise ValueError('Incomplete archived cohort: ' + phase)
        for index, item in enumerate(records):
            key, ident = item['key'], item['id']
            if (item['prompt_tokens'], item['shared_tokens'], item['output_tokens']) != (8192, 7936, 32):
                raise ValueError('Unexpected archived profile at ' + ident)
            ids = _actual_ids(tokenizer, item)
            content = item['body']['messages'][0]['content']
            document_text, separator, _ = content.partition('\nQuestion ' + ident + ': ')
            if not separator or key != 'doc-' + hashlib.sha256(document_text.encode()).hexdigest():
                raise ValueError('Request/document key mismatch: ' + ident)
            if ident in unique_ids or item['prompt_sha256'] in unique_inputs:
                raise ValueError('Duplicate archived request: ' + ident)
            unique_ids.add(ident)
            unique_inputs.add(item['prompt_sha256'])
            if key not in documents:
                documents[key] = document_text
                blocks[key] = item['first_block_digest']
                prefixes[key] = {size: digest(ids[:size - 256]) for size in INPUTS}
            if documents[key] != document_text or prefixes[key][8192] != item['prefix_digest']:
                raise ValueError('Unstable document prefix: ' + ident)
            if phase != 'smoke':
                group = int(ident.rsplit('-', 1)[-1])
                if (ident != f'cache-v2-reuse8-{SEED}-{phase}-{index}-{group}'
                        or group not in range(8) or group_keys.setdefault(group, key) != key):
                    raise ValueError('Invalid archived request ordering/key: ' + ident)
    learn = [r['key'] for r in cohorts['learn']]
    measured = [r['key'] for r in cohorts['measured']]
    formal_keys = set(group_keys.values())
    if len(formal_keys) != 8 or set(learn) != formal_keys or set(measured) != formal_keys:
        raise ValueError('Reference must have exactly eight formal documents')
    if any(learn.count(key) != 8 for key in formal_keys):
        raise ValueError('Learning cohort must visit each of the eight documents eight times')
    for batch in (measured[:64], measured[64:]):
        if max(batch.count(key) for key in formal_keys) != 16:
            raise ValueError('Reference hot25 cohort changed')
    probe_keys = {r['key'] for r in cohorts['smoke']}
    if len(probe_keys) != 1 or formal_keys & probe_keys or len(set(blocks.values())) != 9:
        raise ValueError('Formal and smoke documents must have distinct first cache blocks')
    calibration_blocks = set()
    for filename in CALIBRATION:
        for r in _lines(anchor_dir / 'calibration' / filename):
            _actual_ids(tokenizer, r)
            calibration_blocks.add(r['first_block_digest'])
    if set(blocks.values()) & calibration_blocks:
        raise ValueError('Smoke or formal first block overlaps calibration')
    return cohorts, documents, prefixes


def _derive_inputs(tokenizer, size, cohorts, documents, prefixes):
    if size == 8192:
        return cohorts
    shared = size - 256
    shortened = {key: tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:shared],
                                       skip_special_tokens=False) for key, text in documents.items()}
    matched = {}
    seen = set()
    for phase, records in cohorts.items():
        new_rows = []
        for original in records:
            key = original['key']
            item = row(tokenizer, shortened[key], original['id'], size, 32, shared)
            # Keep the archived logical-document key on the hash ring. The new
            # prefix digest and prompt hash still represent actual shortened tokens.
            item['key'] = key
            if (item['prefix_digest'] != prefixes[key][size]
                    or item['first_block_digest'] != original['first_block_digest']):
                raise ValueError('Shortened prompt lost the archived nested prefix: ' + original['id'])
            if item['prompt_sha256'] in seen:
                raise ValueError('Two requests became identical after shortening')
            seen.add(item['prompt_sha256'])
            new_rows.append(item)
        matched[phase] = new_rows
    return matched


def _with_output(item, target):
    return dict(item, output_tokens=target, body=dict(item['body'], max_tokens=target))


def _write_case(work, anchor_dir, size, answer, inputs):
    name = f'p{size}_o{answer}'
    dest = work / f'{name}_seed{SEED}'
    dest.mkdir()
    files = {}
    for phase, rows in inputs.items():
        filename = FILENAMES[phase]
        if size == 8192 and answer == 32:
            shutil.copyfile(anchor_dir / 'reuse8_seed33962' / filename, dest / filename)
        elif phase == 'smoke':
            _dump(dest / filename, [_with_output(r, answer) for r in rows])
        else:
            write_rows(dest / filename, [_with_output(r, answer) for r in rows])
        if phase != 'smoke':
            files[phase] = dict(count=len(rows), sha256=_sha(dest / filename))
    return dict(profile=name, seed=SEED, folder=dest.name, **PROFILES[name], files=files,
                unique_full_inputs=192, cold_distinct_documents=False)


def build(tokenizer, out, anchor_dir=None):
    out = Path(out).resolve()
    anchor_dir = Path(anchor_dir or ANCHOR)
    if out.exists():
        raise FileExistsError('Will not overwrite an existing workload: ' + str(out))
    recorded = verify_anchor_files(anchor_dir)
    cohorts, docs, prefixes = _anchor_contract(tokenizer, anchor_dir)
    manifest = dict(cells=[], profiles=list(PROFILES), seeds=[SEED], measured_requests=128,
                    learning_requests=64, arrival_model='frozen closed-loop hot25 cohort order',
                    key_selection='same archived logical document key at every input/output length',
                    warmup='64 identical-schedule learning requests on a new pool for each case')
    receipt = dict(seed=SEED, original_case='reuse8', archived_files=recorded, records_per_arm=192,
                   anchor_byte_identical=True, logical_keys_and_order_matched=True,
                   nested_prefix_tokens_checked=True, tokenizer_verified=True,
                   same_input_across_output_lengths=True, input_cohort_hashes={})
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix='.matched-lengths-') as temp:
        work = Path(temp) / 'workloads'
        work.mkdir()
        for size in INPUTS:
            inputs = _derive_inputs(tokenizer, size, cohorts, docs, prefixes)
            receipt['input_cohort_hashes'][str(size)] = {
                phase: digest([[r['id'], r['key'], r['prefix_digest'], r['prompt_sha256']] for r in rows])
                for phase, rows in inputs.items()}
            for answer in OUTPUTS:
                manifest['cells'].append(_write_case(work, anchor_dir, size, answer, inputs))
            # Audit the emitted files as the engine will read them, including the
            # archived 8192/32 byte copies. This catches drift in sampling/body.
            folders = [work / f'p{size}_o{o}_seed{SEED}' for o in OUTPUTS]
            for phase, filename in FILENAMES.items():
                loader = _load if phase == 'smoke' else _lines
                a, b = (loader(folder / filename) for folder in folders)
                if len(a) != CASES[phase] or len(b) != len(a) or any(
                        _with_output(first, OUTPUTS[1]) != second for first, second in zip(a, b)):
                    raise ValueError(f'Output-length variants changed prompt, order or sampling: {size}/{phase}')
        shutil.copytree(anchor_dir / 'calibration', work / 'calibration')
        _dump(work / 'length_match_receipt.json', receipt)
        _dump(work / 'manifest.json', manifest)
        work.rename(out)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    manifest = build(tokenizer, a.out)
    print('TOKEN_VERIFICATION_PASS profiles=' + str(len(manifest['cells'])), flush=True)


if __name__ == '__main__':
    main()
