"""Single-loop gateway; capacity wait is inside client-observed TTFT."""
import json
import time
from aiohttp import ClientSession, ClientTimeout, TCPConnector, web
from streaming import Observation, events


class Gateway:
    def __init__(self, backends, models, router, port, engine):
        self.backends, self.models, self.router = backends, models, router
        self.port, self.engine = port, engine
        self.errors = []
        self.runner = self.session = None

    async def start(self):
        # The same connection policy is used for every engine and routing arm.
        # Do not reuse idle upstream sockets across requests: a backend can
        # retire one while the client still has it in its keep-alive pool.
        # Connecting remains inside client-observed TTFT. POSTs are never
        # replayed here, even if no response bytes have arrived.
        self.session = ClientSession(timeout=ClientTimeout(total=1800),
                                     connector=TCPConnector(limit=0, force_close=True), read_bufsize=2**20)
        app = web.Application(client_max_size=4*1024*1024)
        app.router.add_post('/v1/chat/completions', self.forward)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        await web.TCPSite(self.runner, '127.0.0.1', self.port).start()
        return self

    async def close(self):
        if self.runner:
            await self.runner.cleanup()
        if self.session:
            await self.session.close()

    async def forward(self, request):
        lease = response = None
        observation = Observation()
        success = False
        item = {}
        began = time.perf_counter()
        stage = 'read_request'
        upstream_status = None
        attempts = 0
        backend = None
        force_close = bool(getattr(getattr(self.session, 'connector', None), 'force_close', False))
        try:
            item = json.loads(request.headers['X-Fourway-Item'])
            body = await request.json()
            if body.get('stream') is not True or body.get('max_tokens') != item['output_tokens']:
                raise ValueError('Gateway request does not match its declared token contract')
            stage = 'acquire'
            lease = await self.router.acquire(item)
            body['model'] = self.models[lease.worker]
            body.pop('_fourway_prefix', None)
            # Ordinary requests need no prefix telemetry registration. In
            # particular zero, missing and partial-block prefixes are valid
            # least-active traffic, but are not valid engine cache entries.
            if (self.engine == 'nh' and lease.shared > 0
                    and lease.shared % self.router.page_size == 0 and lease.digest):
                body['_fourway_prefix'] = dict(digest=lease.digest, tokens=lease.shared)
            backend = self.backends[lease.worker]
            stage = 'send_request_or_wait_upstream_headers'
            attempts = 1
            async with self.session.post(backend+'/v1/chat/completions', json=body) as upstream:
                upstream_status = upstream.status
                if upstream.status != 200:
                    raise RuntimeError('Upstream HTTP '+str(upstream.status)+': '+(await upstream.text())[:1200])
                stage = 'prepare_downstream'
                response = web.StreamResponse(status=200, headers={'Content-Type':'text/event-stream', 'Cache-Control':'no-cache'})
                await response.prepare(request)
                stage = 'stream_upstream'
                async for data in events(upstream.content):
                    obj = observation.accept(data, time.perf_counter())
                    if obj is not None:
                        self.router.progress(lease, obj)
                    # The SSE payload is unchanged; only framing is normalized.
                    await response.write((''.join('data: '+line+'\n' for line in data.split('\n'))+'\n').encode())
                stage = 'validate_stream'
                observation.validate(item)
                stage = 'finish_downstream'
                await response.write_eof()
                success = True
        except Exception as exc:
            failure = dict(id=item.get('id'), error=repr(exc), error_type=type(exc).__name__,
                           errno=getattr(exc, 'errno', None), stage=stage,
                           worker=lease.worker if lease is not None else None,
                           backend=backend, upstream_http_status=upstream_status,
                           upstream_attempts=attempts, upstream_force_close=force_close,
                           observed_upstream_events=observation.payload_events,
                           observed_upstream_bytes=observation.payload_bytes,
                           elapsed_ms=1000*(time.perf_counter()-began),
                           retried=False)
            self.errors.append(failure)
            if lease is not None:
                lease.route['transport_error'] = failure
            if response is None:
                response = web.json_response({'error': repr(exc)}, status=502)
            else:
                try:
                    await response.write(('data: '+json.dumps({'error':repr(exc)})+'\n\n').encode())
                    await response.write_eof()
                except Exception:
                    pass
        finally:
            if lease is not None:
                lease.route.update(event_payload_sha256=observation.payload_hasher.hexdigest(),
                                   event_payload_count=observation.payload_events,
                                   event_payload_bytes=observation.payload_bytes,
                                   upstream_http_attempts=attempts,
                                   upstream_force_close=force_close)
                self.router.release(lease, success, observation.usage)
        return response
