"""Real aiohttp gateway integration with deterministic CPU-only fake engines.

Run with an interpreter containing aiohttp:
    python -m unittest -v test_gateway_integration.py

The fake engine's usage is intentional: SSE chunk count is not token count.
These tests establish transport/accounting behavior, not GPU performance.
"""
import asyncio
import hashlib
import json
import struct
import time
import unittest

from aiohttp import ClientSession, ClientTimeout, web

from affinity_router import PrefillModel, Router
from gateway import Gateway
from streaming import request_one


def make_item(request_id, mode='ok'):
    return dict(id=str(request_id), key='shared-document', prompt_tokens=8192,
                output_tokens=64, shared_tokens=7680, prefix_digest='prefix-digest',
                prompt_sha256='fixture-prompt-sha',
                body=dict(messages=[dict(role='user', content='CPU fixture')],
                          max_tokens=64, test_mode=mode, test_request_id=str(request_id),
                          _fourway_prefix={'digest': 'untrusted-client-value', 'tokens': 1}))


def payload_signature(payloads):
    h = hashlib.sha256()
    byte_count = 0
    for data in payloads:
        raw = data.encode('utf-8')
        h.update(struct.pack('>Q', len(raw)))
        h.update(raw)
        byte_count += len(raw)
    return h.hexdigest(), len(payloads), byte_count


class FakeEngine:
    def __init__(self, index):
        self.index = index
        self.requests = []
        self.payloads = {}
        self.gates = {}
        self.prefixes = {}
        self.epoch = 0
        self.runner = None
        self.url = None

    async def start(self):
        app = web.Application()
        app.router.add_post('/v1/chat/completions', self.forward)
        self.runner = web.AppRunner(app, access_log=None, shutdown_timeout=1)
        await self.runner.setup()
        await web.TCPSite(self.runner, '127.0.0.1', 0).start()
        self.url = 'http://127.0.0.1:'+str(self.runner.addresses[0][1])
        return self

    async def close(self):
        for gate in self.gates.values():
            gate.set()
        if self.runner:
            await self.runner.cleanup()

    async def forward(self, request):
        body = await request.json()
        self.requests.append(body)
        request_id = body['test_request_id']
        mode = body.get('test_mode', 'ok')
        if mode == 'http_error':
            return web.json_response({'error': 'fixture engine unavailable'}, status=503)
        metadata = body.get('_fourway_prefix')
        if metadata is not None and (not metadata.get('digest')
                                     or not isinstance(metadata.get('tokens'), int)
                                     or metadata['tokens'] <= 0 or metadata['tokens'] % 256):
            return web.json_response({'error': 'invalid prefix registration'}, status=422)
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        payloads = self.payloads[request_id] = []

        async def emit(obj):
            # Noncanonical whitespace, multiline JSON, non-ASCII content, CRLF
            # and SSE comments exercise payload preservation independently of
            # the gateway's intentionally normalized framing.
            data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2)
            payloads.append(data)
            wire = ': fixture comment\r\n'+''.join('data: '+line+'\r\n' for line in data.split('\n'))+'\r\n'
            await response.write(wire.encode('utf-8'))

        try:
            await emit(dict(choices=[dict(delta=dict(role='assistant'), finish_reason=None)]))
            if mode == 'malformed':
                await emit('{"choices":')
                await response.write_eof()
                return response
            if mode == 'truncated':
                await response.write(b'data: {"unfinished": true}')
                await response.write_eof()
                return response
            if mode == 'engine_error':
                await emit(dict(choices=[dict(delta=dict(content='fixture error'), finish_reason='error')]))
                await emit('[DONE]')
                await response.write_eof()
                return response

            telemetry = {}
            cached = 0
            prefix = body.get('_fourway_prefix')
            if prefix:
                cached = self.prefixes.get(prefix['digest'], 0)
                self.prefixes[prefix['digest']] = prefix['tokens']
                self.epoch += 1
                telemetry = dict(prefix_caching=True, generated_tokens=1,cache_policy=getattr(self,'cache_mode','native'),
                                 cache_snapshot=dict(epoch=self.epoch, prefixes=dict(self.prefixes)))
            first = dict(choices=[dict(delta=dict(content='a猫\n'), finish_reason=None)])
            if telemetry:
                first['x_infinilm'] = telemetry
            await emit(first)

            if mode == 'hold':
                gate = self.gates.setdefault(request_id, asyncio.Event())
                await gate.wait()

            prompt_tokens = body.get('test_prompt_tokens', 8192)
            usage = dict(prompt_tokens=prompt_tokens, completion_tokens=63 if mode == 'wrong_usage' else 64,
                         total_tokens=prompt_tokens+64, prompt_tokens_details=dict(cached_tokens=cached))
            final = dict(choices=[dict(delta=dict(content='z'), finish_reason='length')], usage=usage)
            if mode == 'missing_usage':
                final.pop('usage')
            if telemetry:
                self.epoch += 1
                final['x_infinilm'] = dict(prefix_caching=True, generated_tokens=64,cache_policy=getattr(self,'cache_mode','native'),
                                          cache_snapshot=dict(epoch=self.epoch, prefixes=dict(self.prefixes)))
            await emit(final)
            if mode != 'missing_done':
                await emit('[DONE]')
            await response.write_eof()
        except (ConnectionError, asyncio.CancelledError):
            # A cancelled client is intentionally tested; fake work is released
            # in test teardown even when aiohttp keeps a disconnected handler.
            pass
        return response


class GatewayIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engines = [await FakeEngine(i).start() for i in range(2)]
        self.gateways = []
        self.client = ClientSession(timeout=ClientTimeout(total=5))

    async def asyncTearDown(self):
        for engine in self.engines:
            for gate in engine.gates.values():
                gate.set()
        await self.client.close()
        for gateway in self.gateways:
            await gateway.close()
        for engine in self.engines:
            await engine.close()

    async def make_gateway(self, policy='bounded', engine='nh', single=False, max_inflight=3):
        backends = [x.url for x in self.engines[:1 if single else 2]]
        model = PrefillModel((1, 100, 0), {1: 10, 2: 12, 3: 15})
        router = Router(backends, policy, model=model, max_inflight=max_inflight,
                        queue_timeout_s=2)
        gateway = Gateway(backends, ['actual-model-'+str(i) for i in range(len(backends))],
                          router, 0, engine)
        gateway.test_handler_tasks = {}
        original_forward = gateway.forward

        async def traced_forward(request):
            request_id = json.loads(request.headers['X-Fourway-Item'])['id']
            gateway.test_handler_tasks[request_id] = asyncio.current_task()
            return await original_forward(request)

        gateway.forward = traced_forward
        await gateway.start()
        gateway.test_url = 'http://127.0.0.1:'+str(gateway.runner.addresses[0][1])
        self.gateways.append(gateway)
        return gateway, router

    async def until(self, predicate, timeout=2):
        deadline = time.monotonic()+timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail('Timed out waiting for integration condition')
            await asyncio.sleep(.005)

    def assert_drained(self, router):
        state = router.snapshot()
        self.assertEqual(sum(state['active']), 0)
        self.assertEqual(sum(state['kv_reserved_tokens']), 0)
        self.assertEqual(state['queue_waiters'], 0)
        self.assertTrue(all(record['released'] for record in state['events']))
        self.assertEqual(sum(state['selections']), len(state['events']))
        self.assertEqual(len({record['serial'] for record in state['events']}), len(state['events']))

    async def test_three_policies_both_engines_exact_usage_and_payload(self):
        for policy in ('least_active', 'bounded', 'cost_bounded'):
            for engine in ('nh', 'vllm'):
                with self.subTest(policy=policy, engine=engine):
                    gateway, router = await self.make_gateway(policy, engine)
                    request_id = policy+'-'+engine
                    request = make_item(request_id)
                    result = await request_one(self.client, gateway.test_url, request, 'client-alias')
                    self.assertTrue(result['success'], result)
                    self.assertEqual(result['usage']['completion_tokens'], 64)
                    self.assertGreaterEqual(result['ttft_ms'], 0)
                    self.assertEqual(result['output_sha256'], hashlib.sha256('a猫\nz'.encode()).hexdigest())
                    route = router.history[0]
                    chosen = self.engines[route['chosen']]
                    forwarded = next(x for x in chosen.requests if x['test_request_id'] == request_id)
                    self.assertEqual(forwarded['model'], 'actual-model-'+str(route['chosen']))
                    if engine == 'nh':
                        self.assertEqual(forwarded['_fourway_prefix'], {'digest': 'prefix-digest', 'tokens': 7680})
                        self.assertTrue(result['engine_telemetry']['prefix_caching'])
                        self.assertTrue(route['first_content_observed'])
                    else:
                        self.assertNotIn('_fourway_prefix', forwarded)
                        self.assertEqual(result['engine_telemetry'], {})
                    expected = payload_signature(chosen.payloads[request_id])
                    for record in (result, route):
                        actual = tuple(record[key] for key in ('event_payload_sha256', 'event_payload_count', 'event_payload_bytes'))
                        self.assertEqual(actual, expected)
                    self.assertEqual(result['event_payload_count'], 4)
                    self.assertTrue(route['success'])
                    self.assertEqual(gateway.errors, [])
                    self.assert_drained(router)

    async def test_failed_streams_release_once_and_cannot_be_valid_measurements(self):
        for mode in ('http_error', 'malformed', 'truncated', 'missing_done', 'missing_usage',
                     'wrong_usage', 'engine_error'):
            with self.subTest(mode=mode):
                gateway, router = await self.make_gateway(single=True)
                result = await request_one(self.client, gateway.test_url, make_item(mode, mode), 'client-alias')
                self.assertFalse(result['success'])
                self.assertIn('error', result)
                self.assertEqual(len(router.history), 1)
                self.assertFalse(router.history[0]['success'])
                self.assertEqual(len(gateway.errors), 1)
                if mode == 'engine_error':
                    self.assertFalse(router.history[0].get('first_content_observed', False))
                self.assert_drained(router)

    async def test_ordinary_nh_requests_fall_back_without_invalid_prefix_metadata(self):
        cases = [('no-shared', 8192, 0, 'prefix-digest'),
                 ('no-digest', 8192, 7680, ''),
                 ('unaligned', 8192, 7679, 'prefix-digest'),
                 ('short-no-shared', 2048, 0, 'prefix-digest'),
                 ('short-no-digest', 2048, 1536, ''),
                 ('short-valid-prefix', 2048, 1536, 'prefix-digest')]
        for policy in ('bounded', 'cost_bounded'):
            gateway, router = await self.make_gateway(policy=policy)
            for index, (label, prompt, shared, digest) in enumerate(cases):
                with self.subTest(policy=policy, case=label):
                    request = make_item(policy+'-'+label)
                    request.update(prompt_tokens=prompt, shared_tokens=shared, prefix_digest=digest)
                    request['body']['test_prompt_tokens'] = prompt
                    result = await request_one(self.client, gateway.test_url, request, 'client-alias')
                    self.assertTrue(result['success'], result)
                    route = router.history[-1]
                    self.assertEqual(route['branch'], 'least_active_outside_affinity')
                    self.assertFalse(route['affinity_eligible'])
                    self.assertEqual(route['chosen'], index % 2)
                    forwarded = next(body for body in self.engines[route['chosen']].requests
                                     if body['test_request_id'] == request['id'])
                    if label == 'short-valid-prefix':
                        self.assertEqual(forwarded['_fourway_prefix'], {'digest': digest, 'tokens': shared})
                    else:
                        self.assertNotIn('_fourway_prefix', forwarded)
                    self.assertEqual(result['usage']['prompt_tokens'], prompt)
                    self.assert_drained(router)

    async def test_capacity_wait_is_inside_client_ttft(self):
        gateway, router = await self.make_gateway(single=True, max_inflight=1)
        first = make_item('capacity-holder', 'hold')
        first_client = asyncio.create_task(request_one(self.client, gateway.test_url, first, 'client-alias'))
        await self.until(lambda: router.active and next(iter(router.active.values())).started)
        second = make_item('capacity-waiter')
        second_client = asyncio.create_task(request_one(self.client, gateway.test_url, second, 'client-alias'))
        await self.until(lambda: router.snapshot()['queue_waiters'] == 1)
        self.engines[0].gates[first['id']].set()
        completed, waited = await asyncio.gather(first_client, second_client)
        self.assertTrue(completed['success'], completed)
        self.assertTrue(waited['success'], waited)
        route = next(x for x in router.history if x['request_id'] == second['id'])
        self.assertGreater(route['gateway_wait_ms'], 0)
        self.assertGreaterEqual(waited['ttft_ms'], route['gateway_wait_ms'])
        self.assert_drained(router)

    async def test_gateway_handler_cancellation_releases_active_lease(self):
        gateway, router = await self.make_gateway(single=True)
        request = make_item('cancel-active', 'hold')
        client_task = asyncio.create_task(request_one(self.client, gateway.test_url, request, 'client-alias'))
        await self.until(lambda: router.active and next(iter(router.active.values())).started)
        handler = gateway.test_handler_tasks[request['id']]
        handler.cancel()
        try:
            await handler
        except asyncio.CancelledError:
            pass
        await self.until(lambda: not router.active)
        result = await asyncio.wait_for(client_task, 2)
        self.assertFalse(result['success'])
        self.assertFalse(router.history[0]['success'])
        self.assert_drained(router)

    async def test_cancelled_capacity_waiter_never_selects_a_worker(self):
        gateway, router = await self.make_gateway(single=True, max_inflight=1)
        first = make_item('holder', 'hold')
        first_client = asyncio.create_task(request_one(self.client, gateway.test_url, first, 'client-alias'))
        await self.until(lambda: router.active and next(iter(router.active.values())).started)
        queued = make_item('cancel-queued')
        queued_client = asyncio.create_task(request_one(self.client, gateway.test_url, queued, 'client-alias'))
        await self.until(lambda: router.snapshot()['queue_waiters'] == 1)
        handler = gateway.test_handler_tasks[queued['id']]
        handler.cancel()
        try:
            await handler
        except asyncio.CancelledError:
            pass
        result = await asyncio.wait_for(queued_client, 2)
        self.assertFalse(result['success'])
        self.assertEqual(router.snapshot()['queue_cancellations'], 1)
        self.assertEqual(router.snapshot()['queue_waiters'], 0)
        self.assertEqual(len(router.history), 1)
        self.engines[0].gates[first['id']].set()
        completed = await asyncio.wait_for(first_client, 2)
        self.assertTrue(completed['success'], completed)
        self.assert_drained(router)

    async def test_client_disconnect_releases_after_upstream_next_write(self):
        gateway, router = await self.make_gateway(single=True)
        request = make_item('disconnect', 'hold')
        client_task = asyncio.create_task(request_one(self.client, gateway.test_url, request, 'client-alias'))
        await self.until(lambda: router.active and next(iter(router.active.values())).started)
        client_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await client_task
        # Default aiohttp does not cancel server handlers on disconnect. A
        # resumed upstream write exposes the disconnect and runs release.
        self.engines[0].gates[request['id']].set()
        await self.until(lambda: not router.active)
        self.assertFalse(router.history[0]['success'])
        self.assert_drained(router)


if __name__ == '__main__':
    unittest.main()
