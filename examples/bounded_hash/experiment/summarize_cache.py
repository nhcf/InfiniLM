#!/usr/bin/env python3
"""Print every measured cell and paired result; never select only winners."""
import argparse
import csv
import json
from pathlib import Path


def latest_run(parent):
    runs=[p for p in parent.glob('cache_v2_run_*') if p.is_dir()]
    if not runs:
        raise ValueError('No cache_v2_run directory found under '+str(parent))
    return max(runs,key=lambda p:p.name)


def summarize(root):
    def number(value, scale=1.):
        return 'NA' if value in (None,'','None') else f'{float(value)/scale:.3f}'
    status=json.loads((root/'status.json').read_text())
    lines=['RUN_DIR='+str(root),'STATUS='+status.get('status','UNKNOWN')]
    if not (root/'summary.csv').is_file():
        lines.append('尚无已完成的测量结果；查看日志和 progress.json。')
        return '\n'.join(lines)
    rows=list(csv.DictReader((root/'summary.csv').open()))
    for phase in ('measured','learn_plus_measured'):
        lines.append('\nPHASE='+phase)
        for r in rows:
            if r['phase']!=phase:
                continue
            lines.append(f"{r['profile']} seed={r['seed']} C={r['concurrency']} {r['arm']}: "
                f"tok/s={number(r['output_tok_s'])} TTFT_mean={number(r['ttft_mean_ms'],1000)}s "
                f"TTFT_p95={number(r['ttft_p95_ms'],1000)}s failures={r['failures']} valid={r['valid']}")
    if (root/'paired.csv').is_file():
        lines.append('\nPAIRED_MEASURED_RESULTS')
        for r in csv.DictReader((root/'paired.csv').open()):
            if r['phase']!='measured':
                continue
            lines.append(f"{r['profile']} seed={r['seed']} C={r['concurrency']} vs {r['baseline']}: "
                f"throughput_ratio={float(r['throughput_ratio']):.4f} "
                f"mean_TTFT_ratio={float(r['mean_ttft_ratio']):.4f} "
                f"p95_TTFT_ratio={float(r['p95_ttft_ratio']):.4f} "
                f"both_metrics_better={r['both_metrics_better']} tail_regression={r['tail_regression']}")
    lines.append('\n吞吐比>1 且平均TTFT比<1，才算该次测量双指标胜出；单种子仅为筛查结果。')
    lines.append('REPORT_ARCHIVE='+str(root)+'_report.zip')
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',nargs='?',type=Path)
    ap.add_argument('--latest',action='store_true')
    a=ap.parse_args()
    if a.run is not None and a.latest:
        ap.error('Use either a run path or --latest')
    root=a.run or latest_run(Path(__file__).resolve().parent/'results')
    print(summarize(root))


if __name__=='__main__':main()
