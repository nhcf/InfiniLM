"""Use verified 8192-token inputs with three output budgets; never regenerate text."""
import argparse
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import length_workload as proven

SEED = 33962
PROFILES = {f'p8192_o{o}': dict(prompt_tokens=8192, shared_tokens=7936,
             output_tokens=o, documents=8, hot_fraction=.25) for o in (32, 64, 128)}


@contextmanager
def original_builder_scope():
    old = proven.INPUTS, proven.OUTPUTS, proven.PROFILES
    proven.INPUTS = (8192,)
    proven.OUTPUTS = (32, 128)
    proven.PROFILES = PROFILES
    try:
        yield
    finally:
        proven.INPUTS, proven.OUTPUTS, proven.PROFILES = old


def build(tokenizer, out, anchor_dir=None):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Will not overwrite an existing workload: '+str(out))
    anchor = Path(anchor_dir or proven.ANCHOR)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix='.optimization-workload-') as temporary:
        work = Path(temporary)/'workloads'
        with original_builder_scope():
            manifest = proven.build(tokenizer, work, anchor)
            inputs = {phase:(proven._load if phase == 'smoke' else proven._lines)(
                      work/'p8192_o32_seed33962'/name)
                      for phase,name in proven.FILENAMES.items()}
            middle = proven._write_case(work, anchor, 8192, 64, inputs)
        lookup = {c['profile']:c for c in manifest['cells']+[middle]}
        manifest['cells'] = [lookup[name] for name in PROFILES]
        manifest['profiles'] = list(PROFILES)
        for phase, filename in proven.FILENAMES.items():
            loader = proven._load if phase == 'smoke' else proven._lines
            original = inputs[phase]
            for output in (32,64,128):
                items = loader(work/f'p8192_o{output}_seed33962'/filename)
                if items != [proven._with_output(r,output) for r in original]:
                    raise ValueError('Output budget changed input/body/order')
                for item in items:
                    proven._actual_ids(tokenizer,item)
        receipt = proven._load(work/'length_match_receipt.json')
        receipt.update(output_lengths=[32,64,128],
            derivation='Verified archived 8192/32 input; only output_tokens and body.max_tokens change')
        proven._dump(work/'length_match_receipt.json',receipt)
        proven._dump(work/'manifest.json',manifest)
        work.rename(out)
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    result=build(tokenizer,args.out)
    print('TOKEN_VERIFICATION_PASS profiles='+str(len(result['cells'])),flush=True)


if __name__=='__main__':
    main()
