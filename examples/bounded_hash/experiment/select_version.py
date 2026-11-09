#!/usr/bin/env python3
"""One finite predeclared comparison, symmetric selection, then a local freeze."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import traceback
import zipfile

import cache_suite
import harness
from selection_protocol import make_spec, validate_spec, digest

KIT = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_spec(config):
    cfg = cache_suite.configuration(config)
    ref_path = KIT/'reference_environment.json'
    reference = json.loads(ref_path.read_text())
    if cfg != reference['config']:
        raise ValueError('Configuration differs from the proven MUSA run. This final panel uses the verified configuration unchanged.')
    sources = harness.source_hashes()
    for name, expected in reference['proven_sources'].items():
        if sources.get(name) != expected:
            raise ValueError('Previously verified code was changed or omitted: '+name)
    spec = make_spec()
    spec.update(config=cfg, source_sha256=sources,
                reference_environment_sha256=sha(ref_path),
                reference_run=reference['verified_source_run'])
    spec['spec_sha256'] = digest(spec)
    validate_spec(spec)
    return spec


def check_environment(fingerprint, spec):
    reference_path = KIT/'reference_environment.json'
    if sha(reference_path) != spec['reference_environment_sha256']:
        raise ValueError('Reference environment file changed after the plan was prepared')
    reference = json.loads(reference_path.read_text())
    for key in ('config', 'nh_git', 'nh_files', 'native_bindings', 'tokenizer_files', 'package_versions', 'hardware'):
        if fingerprint.get(key) != reference[key]:
            raise ValueError('Preflight differs from proven run: '+key)
    if fingerprint.get('protocol') != spec['execution_protocol']:
        raise ValueError('Execution protocol changed')
    if fingerprint.get('candidate_sources') != spec['source_sha256']:
        raise ValueError('Code changed after the selection plan was prepared')
    hardware = fingerprint.get('hardware') or {}
    if hardware.get('available') is not True or len(hardware.get('devices', [])) != 8:
        raise ValueError('Eight verified devices are required')


def command(mode, output, config, spec):
    return [sys.executable, '-B', str(KIT/'cache_confirm.py'), mode,
            '--config', str(config.resolve()), '--profiles', *spec['profiles'],
            '--concurrency', *map(str, spec['concurrency']),
            '--seeds', *map(str, spec['seeds']), '--rounds', str(spec['rounds']),
            '--requests', str(spec['measured_requests']), '--arms', *spec['arms'],
            '--output-root', str(output)]


def find_run(output, mode):
    found = [p for p in output.glob('cache_confirm_'+mode+'_*') if p.is_dir()]
    if len(found) != 1:
        raise RuntimeError('Expected exactly one child run at '+str(output))
    return found[0]


def archive_stage(root):
    path = Path(str(root)+'_report.zip')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(root.rglob('*')):
            if f.is_file() and f.suffix != '.zip':
                z.write(f, Path(root.name)/f.relative_to(root))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'check', 'run'])
    parser.add_argument('--config', type=Path, default=KIT/'config.musa.json')
    parser.add_argument('--output-root', type=Path, default=KIT/'results_selection')
    args = parser.parse_args(argv)
    spec = build_spec(args.config)
    if args.mode == 'plan':
        print(json.dumps({k:v for k,v in spec.items() if k not in ('source_sha256', 'config')}, ensure_ascii=False, indent=2))
        return 0
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root = args.output_root.resolve()/('selection_'+stamp)
    root.mkdir(parents=True, exist_ok=False)
    dump(root/'selection_proposal.json', spec)
    result = dict(status='RUNNING', complete=False, selected_arm=None, gpu_experiment_requested=args.mode == 'run')
    dump(root/'stage_status.json', result)
    print('SELECTION_DIR='+str(root), flush=True)
    child = None
    interrupted = False
    old_handlers = {}

    def handle(signum, frame):
        nonlocal interrupted
        interrupted = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, handle)

    def run_child(mode, output):
        nonlocal child
        if interrupted:
            raise InterruptedError('User interrupted selection')
        output.mkdir(parents=True)
        argv = command(mode, output, args.config, spec)
        dump(output/'command.json', argv)
        child = subprocess.Popen(argv, cwd=KIT, start_new_session=True)
        code = child.wait()
        child = None
        if interrupted:
            raise InterruptedError('User interrupted selection')
        if code:
            raise RuntimeError(f'Child {mode} failed with exit {code}; raw evidence is at {output}')
        run = find_run(output, mode)
        status = json.loads((run/'status.json').read_text())
        if status.get('complete') is not True:
            raise RuntimeError('Incomplete child status: '+str(run))
        return run

    code = 2
    try:
        checked = run_child('check', root/'preflight')
        fingerprint = json.loads((checked/'fingerprint.json').read_text())
        check_environment(fingerprint, spec)
        spec.pop('spec_sha256')
        spec['expected_fingerprint'] = fingerprint
        spec['spec_sha256'] = digest(spec)
        validate_spec(spec)
        dump(root/'selection_plan.json', spec)
        if args.mode == 'check':
            result.update(status='CHECK_COMPLETE', complete=True, no_models_loaded=True)
            code = 0
        else:
            check_environment(fingerprint, spec)
            if harness.source_hashes() != spec['source_sha256']:
                raise RuntimeError('Source files changed after preflight; no models will be started')
            if cache_suite.configuration(args.config) != spec['config']:
                raise RuntimeError('Configuration changed after preflight; no models will be started')
            print('PLAN_FROZEN='+spec['spec_sha256'], flush=True)
            print(f"BENCHMARK_BEGIN cells={spec['actual_cells']} requests={spec['learning_plus_measured_requests']} "
                  f"profiles={','.join(spec['profiles'])} concurrency={spec['concurrency']} "
                  f"arms={len(spec['arms'])} seeds={spec['seeds']} rounds={spec['rounds']}", flush=True)
            run = run_child('run', root/'benchmark')
            runtime_fingerprint = json.loads((run/'fingerprint.json').read_text())
            if runtime_fingerprint != fingerprint:
                raise RuntimeError('Environment or code changed between preflight and benchmark')
            if harness.source_hashes() != spec['source_sha256']:
                raise RuntimeError('Local source files changed during the run; selection is suppressed')
            if interrupted:
                raise InterruptedError('Interrupted before selection')
            dump(run/'selection_plan.json', spec)
            shutil.copy2(KIT/'reference_environment.json', run/'reference_environment.json')
            from selection_analysis import analyze
            decision = analyze(run, spec)
            dump(root/'selection_decision.json', decision)
            if decision.get('complete') is not True:
                raise RuntimeError('Incomplete evidence; no version will be frozen')
            if interrupted:
                raise InterruptedError('Interrupted before freezing')
            from freeze_selection import freeze
            destination = root/('frozen_release' if decision.get('selected_arm') else 'phase_checkpoint')
            frozen = freeze(run, spec, decision, destination)
            result.update(status='SELECTION_COMPLETE', complete=True,
                          selected_arm=decision.get('selected_arm'), scope=decision.get('scope'),
                          selection_status=decision.get('status'), freeze_directory=str(destination),
                          freeze_manifest=frozen)
            print('SELECTED_ARM='+str(decision.get('selected_arm')), flush=True)
            print('SELECTION_STATUS='+str(decision.get('status')), flush=True)
            print('FREEZE_BUNDLE='+str(destination)+'.zip', flush=True)
            code = 0
    except BaseException as exc:
        result.update(status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
                      complete=False, selected_arm=None, error=repr(exc), traceback=traceback.format_exc())
        print('SELECTION_FAILED '+repr(exc), flush=True)
    finally:
        dump(root/'stage_status.json', result)
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)
        path = archive_stage(root)
        print('SELECTION_REPORT='+str(path), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
