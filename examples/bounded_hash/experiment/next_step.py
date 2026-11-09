#!/usr/bin/env python3
"""Small explicit stages; never silently expand a Cartesian benchmark matrix."""
from __future__ import annotations
import argparse
import csv
import datetime
import json
from pathlib import Path
import signal
import subprocess
import sys

KIT = Path(__file__).resolve().parent
SCREEN_CASES = [('reuse8', 32), ('reuse8', 64), ('reuse16', 16),
                ('reuse16', 32), ('prefix97', 16), ('prefix97', 32),
                ('longdecode', 16), ('cold', 16)]
SHORT_ARMS = ['nh_cache_bounded_v2', 'nh_mean_bounded', 'vllm_least_active']


def jobs(stage):
    if stage == 'confirm':
        return [dict(profile='reuse8', concurrency=16, seeds=[33960, 33961],
                     rounds=5, historical=True, arms=None)]
    return [dict(profile=p, concurrency=c, seeds=[33962, 33963],
                 rounds=3 if stage == 'expand-confirm' else 1,
                 historical=False, arms=SHORT_ARMS) for p, c in SCREEN_CASES]


def command(job, config, output, phase_model=None, mode='run'):
    cmd = [sys.executable, '-B', str(KIT/'cache_confirm.py'), mode,
           '--config', str(config), '--profiles', job['profile'],
           '--concurrency', str(job['concurrency']), '--seeds', *map(str, job['seeds']),
           '--requests', '128', '--rounds', str(job['rounds']), '--output-root', str(output)]
    if job['historical']:
        cmd += ['--historical-run', str(KIT/'frozen_report')]
    if job['arms']:
        cmd += ['--arms', *job['arms']]
    if phase_model:
        cmd += ['--phase-model', str(phase_model)]
    return cmd


def scoreboard(root, run_roots):
    rows = []
    for run in run_roots:
        file = run/'confirmation_summary.csv'
        if file.exists():
            with file.open(newline='') as stream:
                rows += [dict(r, run_directory=str(run.relative_to(root)))
                         for r in csv.DictReader(stream) if r['phase'] != 'learn']
    if rows:
        with (root/'scenario_scoreboard.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['plan', 'run'])
    ap.add_argument('stage', choices=['confirm', 'expand', 'expand-confirm'])
    ap.add_argument('--config', type=Path, default=KIT/'config.musa.json')
    ap.add_argument('--output-root', type=Path, default=KIT/'results_next')
    a = ap.parse_args(argv)
    selected = jobs(a.stage)
    cells = sum(len(j['seeds'])*j['rounds']*len(j['arms'] or range(5)) for j in selected)
    plan = dict(stage=a.stage, jobs=selected, actual_cells=cells,
                protocol='infinilm-cache-paired-confirm-v2-fresh-http',
                http_connection_reuse=False, automatic_inference_retries=0,
                learning_plus_measured_requests=cells*192,
                balanced=a.stage != 'expand',
                purpose='Frozen same-round V2 / old mean / vLLM comparison; all declared outcomes retained.',
                startup_note='Each cell restarts all eight replicas. Calibration, startup, and probes are additional and excluded from serving throughput.',
                expansion_note='expand is an exploratory one-round screen, not balanced confirmation. expand-confirm repeats the entire same eight-case panel in three balanced rounds.',
                config=str(a.config.resolve()))
    if a.action == 'plan':
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root = a.output_root.resolve()/f'{a.stage}_{stamp}'
    root.mkdir(parents=True, exist_ok=False)
    (root/'stage_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    print('STAGE_DIR='+str(root), flush=True)
    child = None
    interrupted = False
    runs = []
    result = dict(complete=False, status='RUNNING', completed_jobs=0, expected_jobs=len(selected))
    (root/'stage_status.json').write_text(json.dumps(result, indent=2)+'\n')
    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)
    exit_code = 2
    try:
        phase_model = None
        for i, job in enumerate(selected, 1):
            if interrupted:
                raise RuntimeError('Stage interrupted; no additional jobs started')
            out = root/f'job{i:02d}_{job["profile"]}_c{job["concurrency"]}'
            out.mkdir()
            cmd = command(job, a.config.resolve(), out, phase_model)
            (out/'command.json').write_text(json.dumps(cmd, ensure_ascii=False, indent=2)+'\n')
            print(f'JOB {i}/{len(selected)} {job["profile"]} C{job["concurrency"]}', flush=True)
            # Separate group keeps terminal SIGINT from being delivered twice;
            # only this exact child receives the forwarded signal.
            child = subprocess.Popen(cmd, cwd=KIT, start_new_session=True)
            code = child.wait()
            child = None
            found = sorted(p for p in out.glob('cache_confirm_run_*') if p.is_dir())
            runs.extend(found)
            scoreboard(root, runs)
            if code or len(found) != 1:
                raise RuntimeError(f'Job {i} failed/incomplete, return code {code}; evidence retained')
            run = found[0]
            for status_name in ('status.json', 'confirmation_status.json', 'diagnostics/diagnosis.json'):
                status = json.loads((run/status_name).read_text())
                if status_name != 'diagnostics/diagnosis.json' and not status.get('complete'):
                    raise RuntimeError(f'Incomplete {status_name}: {run}')
                if status_name == 'diagnostics/diagnosis.json' and status.get('status') != 'OFFLINE_DIAGNOSIS_COMPLETE':
                    raise RuntimeError(f'Incomplete diagnosis: {run}')
            with (run/'confirmation_summary.csv').open(newline='') as stream:
                summary_rows = list(csv.DictReader(stream))
            required = [r for r in summary_rows if r.get('phase') == 'measured'
                        and r.get('baseline') == 'vllm_least_active'
                        and r.get('profile') == job['profile']
                        and int(r.get('concurrency', 0)) == job['concurrency']]
            if len(required) != 1 or int(required[0].get('paired_trials', 0)) != len(job['seeds'])*job['rounds']:
                raise RuntimeError(f'Missing or incomplete measured vLLM summary: {run}')
            if not phase_model:
                # Runner verifies exact hardware, runtime, config, and source identity.
                phase_model = run/'phase_model.json'
            result['completed_jobs'] = i
            (root/'stage_status.json').write_text(json.dumps(result, indent=2)+'\n')
        result.update(complete=True, status='STAGE_COMPLETE', performance_win_claimed=False)
        exit_code = 0
    except Exception as exc:
        result.update(status='INTERRUPTED' if interrupted else 'FAILED', error=repr(exc))
        print('STAGE_FAILED '+repr(exc), flush=True)
    finally:
        (root/'stage_status.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        scoreboard(root, runs)
        # Inner per-job report zips duplicate directory evidence; omit only those
        # copies from the stage archive, retaining all raw directories and logs.
        import zipfile
        archive = Path(str(root)+'_report.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(root.rglob('*')):
                if p.is_file() and not p.name.endswith('_report.zip'):
                    zf.write(p, Path(root.name)/p.relative_to(root))
        print('STAGE_REPORT='+str(archive), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
