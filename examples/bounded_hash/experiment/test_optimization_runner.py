"""CPU integration checks for the new entry and unchanged benchmark lifecycle."""
import asyncio
import contextlib
import copy
import io
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch
import zipfile

import cache_calibration
import cache_router
import cache_router_v3
import optimization_runner as runner
import optimization_workload as workload
from test_length_workload import Words, make_anchor


class OptimizationDriverTests(unittest.TestCase):
    def setUp(self):
        self.cfg=runner.cache_suite.configuration(runner.KIT/'config.musa.json')
        self.ref,self.sources=runner.verify_local(self.cfg)

    def fingerprint(self):
        fp={name:copy.deepcopy(self.ref[name]) for name in runner.proven.ENVIRONMENT_FIELDS}
        fp.update(protocol=runner.confirm.PROTOCOL,candidate_sources=self.sources)
        return fp

    def test_fixed_plan_has_nine_cells_and_one_position_per_arm(self):
        with runner.adapted_runtime(self.ref,self.sources):
            a=runner.engine_arguments(runner.arguments(['plan']))
            plan=runner.confirm.make_plan(a,self.cfg)
        self.assertEqual(a.seeds,[33962]);self.assertEqual(a.rounds,1)
        self.assertEqual(a.concurrency,[16]);self.assertEqual(a.requests,128)
        self.assertEqual(plan['cells'],9)
        self.assertEqual(plan['expected_phase_rows'],27)
        self.assertEqual(plan['total_learning_plus_measured_requests'],1728)
        for arm in runner.ARMS:
            self.assertEqual(sorted(j['order_position'] for j in plan['execution_order'] if j['arm']==arm),[1,2,3])
        self.assertEqual(list(plan['profiles']),list(workload.PROFILES))
        self.assertNotIn('nh_mean_bounded',[x[0] for x in plan['arms']])

    def test_plan_cannot_start_hardware_or_models(self):
        with contextlib.redirect_stdout(io.StringIO()) as out,patch.object(runner.confirm,'execute') as execute:
            self.assertEqual(runner.main(['plan']),0)
        self.assertEqual(json.loads(out.getvalue())['cells'],9)
        execute.assert_not_called()

    def test_baseline_change_is_rejected(self):
        baseline=runner.read(runner.KIT/'baseline_sources.json')
        modified=dict(self.sources);modified['cache_router.py']='changed'
        with patch.object(runner.proven,'verify_local',return_value=(self.ref,modified)):
            with self.assertRaisesRegex(ValueError,'successful length run'):
                runner.verify_local(self.cfg)
        self.assertTrue(baseline)

    def test_context_restores_every_registry_on_failure(self):
        old_profiles=dict(runner.cache_workload.PROFILES)
        old_arms=runner.cache_suite.ARMS
        old_map=dict(runner.confirm.ARM_MAP)
        old_cell=runner.cache_suite.run_cell
        old_router=cache_router.Router
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with runner.adapted_runtime(self.ref,self.sources):
                self.assertIn(runner.V3_ARM[0],runner.confirm.ARM_MAP)
                raise RuntimeError('fixture')
        self.assertEqual(runner.cache_workload.PROFILES,old_profiles)
        self.assertIs(runner.cache_suite.ARMS,old_arms)
        self.assertEqual(runner.confirm.ARM_MAP,old_map)
        self.assertIs(runner.cache_suite.run_cell,old_cell)
        self.assertIs(cache_router.Router,old_router)

    def test_v3_cell_only_selects_v3_and_restores_after_error(self):
        seen=[]
        original=cache_router.Router
        async def fake_cell(cfg,root,pool,case,c,arm,client,model,opts):
            seen.append((arm[0],cache_router.Router))
            folder=root/'cells'/f'{case["profile"]}_s33962_c16_{arm[0]}'
            folder.mkdir(parents=True)
            if arm[0]==runner.V3_ARM[0]:
                raise RuntimeError('cell failure fixture')
            return []
        with TemporaryDirectory() as tmp,patch.object(runner.cache_suite,'run_cell',new=fake_cell):
            with runner.adapted_runtime(self.ref,self.sources):
                case=dict(profile='p8192_o32',seed=33962)
                for name in runner.ARMS:
                    args=(self.cfg,Path(tmp),None,case,16,runner.confirm.ARM_MAP[name],None,{},None)
                    if name==runner.V3_ARM[0]:
                        with self.assertRaisesRegex(RuntimeError,'fixture'):
                            asyncio.run(runner.cache_suite.run_cell(*args))
                    else:
                        asyncio.run(runner.cache_suite.run_cell(*args))
                    self.assertIs(cache_router.Router,original)
                    receipt=runner.read(Path(tmp)/'cells'/f'p8192_o32_s33962_c16_{name}'/'router_source_selection.json')
                    expected='cache_router_v3' if name==runner.V3_ARM[0] else 'cache_router' if name=='nh_cache_bounded_v2' else 'mean_router'
                    self.assertEqual(receipt['router_class'],expected+'.Router')
                    self.assertEqual(receipt['source_sha256'],self.sources[expected+'.py'])
        self.assertIs(seen[0][1],cache_router_v3.Router)
        self.assertTrue(all(cls is original for _,cls in seen[1:]))

    def test_runtime_drift_stops_before_builder_or_calibration(self):
        fp=self.fingerprint();fp['nh_git']={}
        with TemporaryDirectory() as tmp,runner.adapted_runtime(self.ref,self.sources):
            a=runner.engine_arguments(runner.arguments(['run']))
            plan=runner.confirm.make_plan(a,self.cfg)
            with patch.object(runner.harness,'preflight',new=AsyncMock(return_value=fp)), \
                 patch.object(runner.cache_suite,'hardware_fingerprint',new=AsyncMock(return_value=self.ref['hardware'])), \
                 patch.object(runner.harness,'run_logged',new_callable=AsyncMock) as builder, \
                 patch.object(cache_calibration,'calibrate',new_callable=AsyncMock) as calibration, \
                 self.assertRaisesRegex(ValueError,'nh_git'):
                asyncio.run(runner.execute(a,self.cfg,Path(tmp),plan,self.ref,self.sources))
            builder.assert_not_awaited();calibration.assert_not_awaited()

    def test_check_verifies_workload_without_loading_models(self):
        with TemporaryDirectory() as tmp,runner.adapted_runtime(self.ref,self.sources):
            a=runner.engine_arguments(runner.arguments(['check']))
            plan=runner.confirm.make_plan(a,self.cfg)
            with patch.object(runner.harness,'preflight',new=AsyncMock(return_value=self.fingerprint())), \
                 patch.object(runner.cache_suite,'hardware_fingerprint',new=AsyncMock(return_value=self.ref['hardware'])), \
                 patch.object(runner.confirm,'prepare_workloads',new_callable=AsyncMock) as prepare, \
                 patch.object(cache_calibration,'calibrate',new_callable=AsyncMock) as calibration, \
                 patch.object(runner.confirm.rt,'Pool') as pool:
                result=asyncio.run(runner.execute(a,self.cfg,Path(tmp),plan,self.ref,self.sources))
            self.assertTrue(result['no_models_loaded'])
            self.assertTrue(result['tokenizer_workload_verified'])
            prepare.assert_awaited_once();calibration.assert_not_awaited();pool.assert_not_called()

    def test_actual_execution_engine_runs_nine_serial_cells_one_calibration(self):
        async def prepare(a,cfg,root,fp):
            (root/'workloads').mkdir()
            return {'cells':[{'profile':p,'seed':33962} for p in workload.PROFILES]}
        async def job(cfg,root,case,job,model,opts,client):
            return [dict(valid=True,round=job['round']) for _ in range(3)]
        with TemporaryDirectory() as tmp,runner.adapted_runtime(self.ref,self.sources):
            a=runner.engine_arguments(runner.arguments(['run']))
            plan=runner.confirm.make_plan(a,self.cfg)
            with patch.object(runner.harness,'preflight',new=AsyncMock(return_value=self.fingerprint())), \
                 patch.object(runner.cache_suite,'hardware_fingerprint',new=AsyncMock(return_value=self.ref['hardware'])), \
                 patch.object(runner.confirm,'prepare_workloads',new=AsyncMock(side_effect=prepare)) as loader, \
                 patch.object(cache_calibration,'calibrate',new=AsyncMock(return_value={})) as calibration, \
                 patch('phase_router.PhaseModel'),patch('admission_telemetry.install'), \
                 patch.object(runner.confirm,'execute_job',new=AsyncMock(side_effect=job)) as bench, \
                 patch.object(runner.confirm,'validate_manifest'),patch.object(runner.cache_suite,'report'), \
                 patch.object(runner.confirm.rt,'csv_write'),contextlib.redirect_stdout(io.StringIO()):
                result=asyncio.run(runner.execute(a,self.cfg,Path(tmp),plan,self.ref,self.sources))
            self.assertEqual(result['completed_cells'],9)
            self.assertEqual(result['valid_phase_rows'],27)
            self.assertEqual(bench.await_count,9)
            self.assertEqual([call.args[3]['serial'] for call in bench.await_args_list],list(range(1,10)))
            loader.assert_awaited_once();calibration.assert_awaited_once()

    def test_failure_always_archives_without_claiming_success(self):
        with TemporaryDirectory() as tmp,runner.adapted_runtime(self.ref,self.sources):
            a=runner.engine_arguments(runner.arguments(['run']))
            plan=runner.confirm.make_plan(a,self.cfg)
            root=Path(tmp)/'failure'
            with patch.object(runner,'execute',new=AsyncMock(side_effect=RuntimeError('fixture'))), \
                 contextlib.redirect_stdout(io.StringIO()) as out:
                rc=runner.execute_locked(a,self.cfg,root,plan,self.ref,self.sources)
            self.assertEqual(rc,2)
            archive=Path(str(root)+'_report.zip')
            self.assertIn('OPTIMIZATION_REPORT_ARCHIVE='+str(archive),out.getvalue())
            with zipfile.ZipFile(archive) as z:
                self.assertFalse(json.loads(z.read('failure/optimization_status.json'))['complete'])
                self.assertIn('failure/source/cache_router_v3.py',z.namelist())
                self.assertIn('failure/baseline_sources.json',z.namelist())

    def test_final_archive_distinguishes_report_failure_from_completed_engine(self):
        for report_complete in (True,False):
            with self.subTest(report_complete=report_complete),TemporaryDirectory() as tmp, \
                 runner.adapted_runtime(self.ref,self.sources):
                a=runner.engine_arguments(runner.arguments(['run']))
                plan=runner.confirm.make_plan(a,self.cfg)
                root=Path(tmp)/'completed'
                def post(folder):
                    runner.harness.dump(folder/'postprocess_status.json',{'complete':True})
                def report(folder):
                    self.assertTrue(runner.read(folder/'status.json')['complete'])
                    (folder/'optimization_report.md').write_text('CPU archival fixture')
                    return {'complete':report_complete}
                with patch.object(runner,'execute',new=AsyncMock(return_value={'status':'EXPERIMENT_COMPLETE','complete':True})), \
                     patch.object(runner,'conventional_reports',side_effect=post), \
                     patch('optimization_report.analyze',side_effect=report), \
                     contextlib.redirect_stdout(io.StringIO()):
                    rc=runner.execute_locked(a,self.cfg,root,plan,self.ref,self.sources)
                self.assertEqual(rc,0 if report_complete else 2)
                with zipfile.ZipFile(str(root)+'_report.zip') as z:
                    self.assertTrue(json.loads(z.read('completed/status.json'))['complete'])
                    self.assertEqual(json.loads(z.read('completed/optimization_status.json'))['complete'],report_complete)
                    self.assertIn('completed/optimization_report.md',z.namelist())

    def test_workload_builder_preserves_all_inputs_and_original_globals(self):
        old=workload.proven.INPUTS,workload.proven.OUTPUTS,workload.proven.PROFILES
        with TemporaryDirectory() as tmp:
            anchor=Path(tmp)/'anchor';out=Path(tmp)/'workloads';tokenizer=Words()
            make_anchor(anchor,tokenizer)
            manifest=workload.build(tokenizer,out,anchor)
            self.assertEqual(list(workload.PROFILES),[c['profile'] for c in manifest['cells']])
            for phase,filename in workload.proven.FILENAMES.items():
                load=workload.proven._load if phase=='smoke' else workload.proven._lines
                base=load(out/'p8192_o32_seed33962'/filename)
                self.assertEqual((out/'p8192_o32_seed33962'/filename).read_bytes(),
                                 (anchor/'reuse8_seed33962'/filename).read_bytes())
                for output in (64,128):
                    items=load(out/f'p8192_o{output}_seed33962'/filename)
                    self.assertEqual(items,[workload.proven._with_output(x,output) for x in base])
                    self.assertTrue(all(len(workload.proven._actual_ids(tokenizer,x))==8192 for x in items))
            with self.assertRaises(FileExistsError):workload.build(tokenizer,out,anchor)
        self.assertEqual((workload.proven.INPUTS,workload.proven.OUTPUTS,workload.proven.PROFILES),old)

    def test_workload_rejects_tokenizer_drift_and_restores_globals(self):
        old=workload.proven.INPUTS,workload.proven.OUTPUTS,workload.proven.PROFILES
        with TemporaryDirectory() as tmp:
            out=Path(tmp)/'workloads'
            with self.assertRaisesRegex(ValueError,'tokenizer does not reproduce'):
                workload.build(Words(),out)
            self.assertFalse(out.exists())
        self.assertEqual((workload.proven.INPUTS,workload.proven.OUTPUTS,workload.proven.PROFILES),old)


class OptimizationHTTPTests(unittest.IsolatedAsyncioTestCase):
    """Exercise real local HTTP/SSE and queued warm-copy wakeup, using fake engines."""
    async def asyncSetUp(self):
        from aiohttp import ClientSession,ClientTimeout,TCPConnector
        from gateway import Gateway
        from test_gateway_integration import FakeEngine
        from test_phase_router import MODEL
        self.engines=[await FakeEngine(i).start() for i in range(2)]
        for engine in self.engines:engine.cache_mode='prefix_lru'
        self.client=ClientSession(timeout=ClientTimeout(total=5),connector=TCPConnector(force_close=True))
        self.router=cache_router_v3.Router([e.url for e in self.engines],
            phase_model=dict(MODEL,prefill_coef=[0,1000,0]),warm_wait_ms=200,
            max_inflight=1,queue_timeout_s=2)
        self.gateway=await Gateway([e.url for e in self.engines],['fixture0','fixture1'],
                                   self.router,0,'nh').start()
        self.url='http://127.0.0.1:'+str(self.gateway.runner.addresses[0][1])
        self.tasks=[]

    async def asyncTearDown(self):
        for engine in self.engines:
            for gate in engine.gates.values():gate.set()
        for task in self.tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*self.tasks,return_exceptions=True)
        await self.client.close()
        await self.gateway.close()
        self.router.close()
        for engine in self.engines:await engine.close()

    def item(self,name,mode='ok'):
        from test_gateway_integration import make_item
        item=make_item(name,mode);item['shared_tokens']=7936
        return item

    async def until(self,predicate):
        deadline=time.monotonic()+2
        while not predicate():
            if time.monotonic()>=deadline:self.fail('HTTP integration fixture timed out')
            await asyncio.sleep(.005)

    def start_request(self,item):
        from streaming import request_one
        task=asyncio.create_task(request_one(self.client,self.url,item,'client-alias'))
        self.tasks.append(task)
        return task

    async def test_actual_http_waits_releases_and_reuses_verified_prefix(self):
        from test_phase_router import place
        held=self.item('http-held','hold');place(self.router,held,0)
        first=self.start_request(held)
        await self.until(lambda:'http-held' in self.engines[0].gates and
                         any(x.started for x in self.router.active.values()))
        second=self.start_request(self.item('http-waiter'))
        await self.until(lambda:self.router.deferred_decisions>0)
        self.assertFalse(second.done())
        self.assertEqual(self.engines[1].requests,[])
        await asyncio.sleep(.025)
        self.engines[0].gates['http-held'].set()
        records=await asyncio.gather(first,second)
        self.assertTrue(all(r['success'] for r in records),records)
        self.assertEqual(records[1]['usage']['prompt_tokens_details']['cached_tokens'],7936)
        route=self.router.history[-1]
        self.assertEqual(route['chosen'],0)
        self.assertTrue(route['v3_warm_wait']['waited_for_warm_copy'])
        self.assertGreaterEqual(route['gateway_wait_ms'],20)
        self.assertGreaterEqual(records[1]['ttft_ms'],route['gateway_wait_ms'])
        self.assertEqual(route['kv_reserved_tokens'],8448)
        self.assertEqual(route['upstream_http_attempts'],1)
        self.assertTrue(route['upstream_force_close'])
        for field in ('event_payload_sha256','event_payload_count','event_payload_bytes'):
            self.assertEqual(route[field],records[1][field])
        self.assertEqual(self.router.snapshot()['policy'],'cache_bounded_v3')
        self.assertEqual(self.gateway.errors,[])
        self.assertEqual(sum(self.router.reserved),0)
        self.assertEqual(self.router.active,{})

    async def test_actual_http_error_is_not_retried_and_reservation_drains(self):
        record=await self.start_request(self.item('http-failed','http_error'))
        self.assertFalse(record['success'])
        self.assertEqual(record['http_status'],502)
        self.assertEqual(sum(len(e.requests) for e in self.engines),1)
        route=self.router.history[0]
        self.assertFalse(route['success'])
        self.assertEqual(route['upstream_http_attempts'],1)
        self.assertTrue(route['released'])
        self.assertEqual(sum(self.router.reserved),0)
        self.assertFalse(self.router._wait_evidence)


if __name__=='__main__':unittest.main()
