#!/usr/bin/env python3
"""Fixed C16, one-seed comparison of V3, frozen V2 and vLLM using proven serving code."""
from __future__ import annotations
import argparse
import asyncio
from contextlib import contextmanager, redirect_stdout
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import cache_confirm as confirm
import cache_router
import cache_suite
import cache_workload
import diagnose_confirmation
import harness
import length_runner as proven
from optimization_workload import PROFILES, SEED

KIT=Path(__file__).resolve().parent
STUDY='reuse8-v3-fixed-c16-20261104'
BASELINE_MANIFEST_SHA256='79fd4d14f2db3238edc760f5f3c9677306e6274818c23d32c600518f41878ac9'
V3_ARM=('nh_cache_bounded_v3','nh','cache_bounded_v2','prefix_lru')
ARMS=['nh_cache_bounded_v3','nh_cache_bounded_v2','vllm_least_active']
read=proven.read
verify_runtime=proven.verify_runtime


def verify_local(cfg):
    reference,sources=proven.verify_local(cfg)
    if confirm._sha(KIT/'baseline_sources.json')!=BASELINE_MANIFEST_SHA256:
        raise ValueError('Successful-run source manifest identity differs')
    baseline=read(KIT/'baseline_sources.json')
    if not baseline or any(sources.get(name)!=digest for name,digest in baseline.items()):
        raise ValueError('Files from the successful length run have changed')
    for name in ('cache_router.py','length_runner.py','length_workload.py','cache_confirm.py',
                 'cache_suite.py','gateway.py','streaming.py'):
        if name not in baseline:
            raise ValueError('Baseline source declaration incomplete: '+name)
    return reference,sources


def study_metadata():
    return dict(study=STUDY, primary='measured output tokens / measured wall seconds',
        comparison='New V3 versus unchanged V2 and vLLM, paired on identical input cohorts',
        historical_relation='Archived successful reuse8 seed33962 inputs; every performance value is measured again.',
        source_qualification='Selected favorable 8192 input/7936 shared, 8-document hot25 workloads, C16, one seed and one round.',
        caveat='Three output lengths are not independent seeds. Performance superiority requires measured evidence.',
        panel_position_balance=True, per_scenario_repeated_rotation=False,
        arm_router_options={'nh_cache_bounded_v2':dict(owner_wait_ms=500.0,coalescing=True),
                            'nh_cache_bounded_v3':dict(owner_wait_ms=500.0,coalescing=True,warm_wait_ms=1000.0)},
        v3_wait_rule='min(1000 ms, 25% of calibrated cold-to-warm saving), single fixed deadline; 5% saving margin',
        intended_router_classes={'nh_cache_bounded_v3':'cache_router_v3.Router',
                                 'nh_cache_bounded_v2':'cache_router.Router',
                                 'vllm_least_active':'mean_router.Router'})


async def prepare_workloads(a,cfg,root,fingerprint,reference,sources):
    verify_runtime(fingerprint,reference,sources)
    output=root/'workloads'
    await harness.run_logged([cfg['nh_python'],'-B',str(KIT/'optimization_workload.py'),
        '--model',cfg['model'],'--out',str(output)],
        confirm.rt.backend_env(cfg,'nh',cfg['gpu_ids'][0],'/tmp'),cfg['repo'],
        root/'workload_build.log',timeout=1800)
    manifest=read(output/'manifest.json')
    cases=confirm.validate_manifest(output,manifest,a)
    if len(cases)!=3 or len(manifest['cells'])!=3:
        raise ValueError('Expected exactly three matched long-input profiles')
    harness.dump(root/'workload_origin.json',dict(mode='archived_8192_inputs_output_budget_only',
        source_anchor_manifest_sha256=proven.ANCHOR_MANIFEST_SHA256,
        files={str(p.relative_to(output)):confirm._sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return manifest


@contextmanager
def adapted_runtime(reference,sources):
    """Select the new router only inside the V3 cell; the engine and timers stay original."""
    if set(PROFILES).intersection(cache_workload.PROFILES):
        raise ValueError('Optimization profile collides with runtime profiles')
    from cache_router_v3 import Router as NewRouter
    old_prepare,old_plan,old_run_cell=confirm.prepare_workloads,cache_suite.make_plan,cache_suite.run_cell
    old_arms,old_map=cache_suite.ARMS,dict(confirm.ARM_MAP)
    original_router,original_protocol=cache_router.Router,harness.PROTOCOL

    def base_plan(a,cfg):
        result=old_plan(a,cfg)
        result.update(study_metadata())
        return result

    async def prepare(a,cfg,root,fingerprint):
        return await prepare_workloads(a,cfg,root,fingerprint,reference,sources)

    async def run_cell(cfg,root,pool,case,c,arm,client,model,router_options=None):
        # confirm.execute runs exactly one cell at a time and closes its pool
        # before starting the next. Always restore the router, including errors.
        if cache_router.Router is not original_router:
            raise RuntimeError('Unexpected concurrent router replacement')
        selected=NewRouter if arm[0]==V3_ARM[0] else original_router
        cache_router.Router=selected
        try:
            options=dict(router_options or {})
            if arm[0]==V3_ARM[0]:
                options['warm_wait_ms']=1000.0
            return await old_run_cell(cfg,root,pool,case,c,arm,client,model,options)
        finally:
            cache_router.Router=original_router
            label=f'{case["profile"]}_s{case["seed"]}_c{c}_{arm[0]}'
            folder=root/'cells'/label
            if folder.exists():
                module=('cache_router_v3' if arm[0]==V3_ARM[0] else
                        'cache_router' if arm[0]=='nh_cache_bounded_v2' else 'mean_router')
                harness.dump(folder/'router_source_selection.json',dict(arm=arm[0],
                    router_class=module+'.Router', source_file=module+'.py',
                    source_sha256=confirm._sha(KIT/(module+'.py')),
                    engine=arm[1], backend_policy=arm[2], restored_after_cell=True))

    cache_workload.PROFILES.update(PROFILES)
    cache_suite.ARMS=old_arms+[V3_ARM]
    confirm.ARM_MAP[V3_ARM[0]]=V3_ARM
    confirm.prepare_workloads=prepare
    cache_suite.make_plan=base_plan
    cache_suite.run_cell=run_cell
    try:
        yield
    finally:
        cache_router.Router=original_router
        confirm.prepare_workloads=old_prepare
        cache_suite.make_plan=old_plan
        cache_suite.run_cell=old_run_cell
        cache_suite.ARMS=old_arms
        confirm.ARM_MAP.clear();confirm.ARM_MAP.update(old_map)
        harness.PROTOCOL=original_protocol
        for name in PROFILES:
            del cache_workload.PROFILES[name]


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('plan','check','run'),nargs='?',default='plan')
    p.add_argument('--concurrency',type=int,choices=(16,),default=16)
    p.add_argument('--output-root',type=Path,default=KIT/'results_optimized')
    return p.parse_args(argv)


def engine_arguments(args):
    return confirm.arguments([args.mode,'--profiles',*PROFILES,'--concurrency','16',
        '--seeds',str(SEED),'--rounds','1','--requests','128','--arms',*ARMS,
        '--config',str(KIT/'config.musa.json'),'--output-root',str(args.output_root)])


async def execute(a,cfg,root,plan,reference,sources):
    result=await confirm.execute(a,cfg,root,plan)
    fingerprint=read(root/'fingerprint.json')
    verify_runtime(fingerprint,reference,sources)
    if a.mode=='check':
        await confirm.prepare_workloads(a,cfg,root,fingerprint)
        result['tokenizer_workload_verified']=True
    return result


def conventional_reports(root):
    with (root/'diagnose_confirmation.log').open('w',encoding='utf-8') as log,redirect_stdout(log):
        result=diagnose_confirmation.main([str(root)])
    harness.dump(root/'postprocess_status.json',dict(complete=result==0,
        status='COMPLETE' if result==0 else 'FAILED',script='diagnose_confirmation.py',returncode=result))
    if result!=0:
        raise RuntimeError('Offline request diagnosis failed')


def execute_locked(a,cfg,root,plan,reference,sources):
    root.mkdir(parents=True,exist_ok=False)
    print('RUN_DIR='+str(root),flush=True)
    harness.dump(root/'plan.json',plan)
    harness.dump(root/'execution_order.json',plan['execution_order'])
    harness.dump(root/'status.json',dict(status='RUNNING',complete=False))
    stage=dict(status='RUNNING',complete=False,study=STUDY,concurrency=16)
    harness.dump(root/'optimization_status.json',stage)
    code=2
    try:
        confirm.snapshot_sources(root)
        for name in ('reference_environment.json','baseline_sources.json'):
            shutil.copy2(KIT/name,root/name)
        shutil.copy2(KIT/'anchor_data/source_hashes.json',root/'anchor_source_hashes.json')
        result=asyncio.run(execute(a,cfg,root,plan,reference,sources))
        harness.dump(root/'status.json',result)
        if result.get('complete') is not True:
            raise RuntimeError('Benchmark execution is incomplete')
        if a.mode=='run':
            conventional_reports(root)
            from optimization_report import analyze
            report=analyze(root)
            if report.get('complete') is not True:
                raise RuntimeError('Optimization report incomplete; inspect optimization_report_status.json')
            print('OPTIMIZATION_REPORT='+str(root/'optimization_report.md'),flush=True)
        stage.update(complete=True,status='OPTIMIZATION_EXPERIMENT_COMPLETE' if a.mode=='run' else 'CHECK_COMPLETE')
        code=0
    except BaseException as exc:
        failure=dict(status='INTERRUPTED' if isinstance(exc,(KeyboardInterrupt,asyncio.CancelledError)) else 'FAILED',
                     complete=False,error=repr(exc),traceback=traceback.format_exc())
        stage.update(failure)
        if read(root/'status.json').get('status')=='RUNNING':
            harness.dump(root/'status.json',failure)
        print(stage['status']+' '+repr(exc),flush=True)
    finally:
        harness.dump(root/'optimization_status.json',stage)
        archive=shutil.make_archive(str(root)+'_report','zip',root.parent,root.name)
        print('OPTIMIZATION_REPORT_ARCHIVE='+archive,flush=True)
    return code


def main(argv=None):
    args=arguments(argv)
    cfg=cache_suite.configuration(KIT/'config.musa.json')
    reference,sources=verify_local(cfg)
    with adapted_runtime(reference,sources):
        a=engine_arguments(args)
        plan=confirm.make_plan(a,cfg)
        plan.update(study_metadata())
        plan['admission_observation']='Original read-only observer is unchanged for all arms. V3 alone changes admission deferral through cache_router_v3.Router.'
        if args.mode=='plan':
            print(json.dumps(plan,ensure_ascii=False,indent=2))
            return 0
        with open('/tmp/infinilm-fourway-'+str(os.getuid())+'.lock','a') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print('Another benchmark holds the eight-GPU lock.',file=sys.stderr)
                return 2
            stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            root=args.output_root.resolve()/f'optimized_{args.mode}_c16_{stamp}'
            return execute_locked(a,cfg,root,plan,reference,sources)


if __name__=='__main__':
    raise SystemExit(main())
