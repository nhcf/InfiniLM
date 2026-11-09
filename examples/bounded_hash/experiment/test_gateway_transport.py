"""Real TCP regression tests. No GPU, replay of POSTs, or mocked connectors.

The reuse-only reset fixture demonstrates the failure mode removed by fresh
connections; it does not claim to prove the origin of the historical reset.
"""
import asyncio
import json
import socket
import struct
import unittest

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

from gateway import Gateway
from mean_router import Router as MeanRouter
from cache_router import Router as CacheRouter
from streaming import request_one
from test_gateway_integration import FakeEngine, make_item
from test_phase_router import MODEL


class ResetEngine(FakeEngine):
    def __init__(self, behavior):
        super().__init__(0)
        self.behavior = behavior
        self.connection_requests = {}
        self.received_requests = []
        self.partial_gate = asyncio.Event()

    @staticmethod
    def reset(transport):
        # Actual local TCP RST, rather than replacing ClientSession.post with a
        # mock. The backend has received the POST, so automatic replay is unsafe.
        sock = transport.get_extra_info('socket')
        if sock is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        transport.abort()

    async def forward(self, request):
        body = await request.json()
        transport = request.transport
        count = self.connection_requests.get(transport, 0) + 1
        self.connection_requests[transport] = count
        self.received_requests.append(body['test_request_id'])
        if self.behavior == 'reset_always' or (self.behavior == 'reset_on_reuse' and count > 1):
            self.reset(transport)
            return web.Response(status=500)
        if self.behavior == 'partial_reset':
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            await response.write(('data: '+json.dumps({'choices': [{'delta': {'content': 'first'}, 'finish_reason': None}]})+'\n\n').encode())
            await self.partial_gate.wait()
            self.reset(transport)
            return response
        return await super().forward(request)

    async def close(self):
        self.partial_gate.set()
        await super().close()


class GatewayTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engines = []
        self.gateways = []
        self.client = ClientSession(timeout=ClientTimeout(total=5),
                                    connector=TCPConnector(force_close=True))

    async def asyncTearDown(self):
        for engine in self.engines:
            engine.partial_gate.set()
        await self.client.close()
        for gateway in self.gateways:
            await gateway.close()
            if hasattr(gateway.router, 'close'):
                gateway.router.close()
        for engine in self.engines:
            await engine.close()

    async def start(self, behavior, policy='least_active', engine_kind='nh', legacy_pool=False):
        engine = await ResetEngine(behavior).start()
        self.engines.append(engine)
        backends = [engine.url]
        router = (CacheRouter(backends, phase_model=MODEL) if policy == 'cache_bounded_v2'
                  else MeanRouter(backends, policy))
        gateway = await Gateway(backends, ['actual-model-0'], router, 0, engine_kind).start()
        self.gateways.append(gateway)
        if legacy_pool:
            # Deliberately reconstruct the previous transport behavior; all
            # parsing, routing, and audit behavior remains the real Gateway.
            await gateway.session.close()
            gateway.session = ClientSession(timeout=ClientTimeout(total=5),
                                            connector=TCPConnector(limit=0))
        url = 'http://127.0.0.1:'+str(gateway.runner.addresses[0][1])
        return engine, gateway, router, url

    async def request(self, url, name, mode='ok'):
        return await request_one(self.client, url, make_item(name, mode), 'client-alias')

    def assert_drained(self, router, requests):
        self.assertEqual(sum(router.reserved), 0)
        self.assertEqual(sum(w.active_load for w in router.workers), 0)
        self.assertFalse(router.active)
        self.assertEqual(len(router.history), requests)
        self.assertTrue(all(r['released'] for r in router.history))

    async def test_previous_pooled_post_reproduces_connection_reset(self):
        engine, gateway, router, url = await self.start('reset_on_reuse', legacy_pool=True)
        first = await self.request(url, 'pooled-first')
        second = await self.request(url, 'pooled-second')
        self.assertTrue(first['success'], first)
        self.assertFalse(second['success'], second)
        self.assertEqual(second['http_status'], 502)
        self.assertEqual(second['event_payload_count'], 0)
        self.assertEqual(engine.received_requests, ['pooled-first', 'pooled-second'])
        self.assertEqual(list(engine.connection_requests.values()), [2])
        self.assertEqual(len(gateway.errors), 1)
        self.assertEqual(gateway.errors[0]['stage'], 'send_request_or_wait_upstream_headers')
        self.assertFalse(gateway.errors[0]['upstream_force_close'])
        self.assert_drained(router, 2)

    async def test_fresh_connections_pass_same_reuse_reset_fixture_for_all_five_arms(self):
        arms = [('cache_bounded_v2', 'nh'), ('mean_bounded', 'nh'),
                ('least_active', 'nh'), ('round_robin', 'nh'), ('least_active', 'vllm')]
        for ordinal, (policy, kind) in enumerate(arms):
            with self.subTest(policy=policy, engine=kind):
                engine, gateway, router, url = await self.start('reset_on_reuse', policy, kind)
                self.assertTrue(gateway.session.connector.force_close)
                for n in range(2):
                    record = await self.request(url, f'fresh-{ordinal}-{n}')
                    self.assertTrue(record['success'], record)
                    self.assertEqual(record['usage']['completion_tokens'], 64)
                    route = router.history[-1]
                    for field in ('event_payload_sha256', 'event_payload_count', 'event_payload_bytes'):
                        self.assertEqual(route[field], record[field])
                    self.assertEqual(route['upstream_http_attempts'], 1)
                self.assertEqual(sorted(engine.connection_requests.values()), [1, 1])
                self.assertFalse(gateway.errors)
                self.assert_drained(router, 2)

    async def test_real_reset_on_new_connection_stays_failure_without_replay(self):
        engine, gateway, router, url = await self.start('reset_always')
        record = await self.request(url, 'real-failure')
        self.assertFalse(record['success'], record)
        self.assertEqual(record['http_status'], 502)
        self.assertEqual(engine.received_requests, ['real-failure'])
        error = gateway.errors[0]
        self.assertEqual(error['stage'], 'send_request_or_wait_upstream_headers')
        self.assertTrue(error['upstream_force_close'])
        self.assertEqual(error['upstream_attempts'], 1)
        self.assertFalse(error['retried'])
        self.assertEqual(error['observed_upstream_events'], 0)
        self.assertEqual(error['worker'], 0)
        self.assertEqual(error['backend'], engine.url)
        self.assertTrue(error['error_type'])
        self.assertEqual(router.history[0]['transport_error'], error)
        self.assertFalse(router.history[0]['success'])
        self.assert_drained(router, 1)

    async def test_partial_stream_reset_remains_invalid_without_replay(self):
        engine, gateway, router, url = await self.start('partial_reset')
        pending = asyncio.create_task(self.request(url, 'partial-failure'))
        try:
            for _ in range(400):
                if any(lease.started for lease in router.active.values()):
                    break
                await asyncio.sleep(.005)
            else:
                self.fail('Gateway did not observe the actual first SSE content')
            engine.partial_gate.set()
            record = await pending
        finally:
            engine.partial_gate.set()
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        self.assertFalse(record['success'], record)
        self.assertEqual(engine.received_requests, ['partial-failure'])
        self.assertEqual(gateway.errors[0]['stage'], 'stream_upstream')
        self.assertGreaterEqual(gateway.errors[0]['observed_upstream_events'], 1)
        self.assertEqual(gateway.errors[0]['upstream_attempts'], 1)
        self.assertFalse(gateway.errors[0]['retried'])
        self.assertFalse(router.history[0]['success'])
        self.assert_drained(router, 1)

    async def test_http_failure_is_not_retried_or_accepted(self):
        engine, gateway, router, url = await self.start('normal')
        record = await self.request(url, 'http-failure', 'http_error')
        self.assertFalse(record['success'], record)
        self.assertEqual(record['http_status'], 502)
        self.assertEqual(engine.received_requests, ['http-failure'])
        self.assertEqual(gateway.errors[0]['upstream_http_status'], 503)
        self.assertEqual(gateway.errors[0]['upstream_attempts'], 1)
        self.assertFalse(gateway.errors[0]['retried'])
        self.assert_drained(router, 1)


if __name__ == '__main__':
    unittest.main()
