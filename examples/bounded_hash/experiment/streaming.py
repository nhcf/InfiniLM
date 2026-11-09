"""Shared SSE parser and strict streaming measurements for both engines."""
import hashlib
import json
import time


async def events(content):
    fields = []
    while True:
        raw = await content.readline()
        if not raw:
            if fields:
                raise ValueError('Truncated SSE event')
            break
        line = raw.decode('utf-8').rstrip('\r\n')
        if line == '':
            if fields:
                yield '\n'.join(fields)
                fields = []
        elif line.startswith('data:'):
            field = line[5:]
            # SSE permits removing exactly one optional separator space. Any
            # further leading spaces belong to the event payload.
            fields.append(field[1:] if field.startswith(' ') else field)


class Observation:
    def __init__(self):
        self.first = None
        self.text = []
        self.usage = None
        self.finish = None
        self.done = False
        self.info = {}
        self.payload_hasher = hashlib.sha256()
        self.payload_events = 0
        self.payload_bytes = 0

    def accept(self, data, now):
        encoded = data.encode('utf-8')
        self.payload_hasher.update(len(encoded).to_bytes(8, 'big'))
        self.payload_hasher.update(encoded)
        self.payload_events += 1
        self.payload_bytes += len(encoded)
        if data == '[DONE]':
            self.done = True
            return None
        if self.done:
            raise ValueError('SSE data after DONE')
        obj = json.loads(data)
        if obj.get('error'):
            raise ValueError(str(obj['error']))
        if obj.get('usage'):
            self.usage = obj['usage']
        self.info.update(obj.get('x_infinilm') or {})
        for choice in obj.get('choices', []):
            if choice.get('finish_reason') in ('error','timeout','cancelled','canceled'):
                raise ValueError('Engine error finish_reason: '+str(choice['finish_reason']))
            content = choice.get('delta', {}).get('content')
            if content:
                if self.first is None:
                    self.first = now
                self.text.append(content)
            finish = choice.get('finish_reason')
            if finish:
                self.finish = finish
        return obj

    def validate(self, item):
        if not self.done or self.first is None or self.finish != 'length':
            raise ValueError(f'Incomplete stream: done={self.done}, content={self.first is not None}, finish={self.finish}')
        if not self.usage:
            raise ValueError('Missing exact usage; SSE chunk count is not a token count')
        if (self.usage.get('prompt_tokens'), self.usage.get('completion_tokens')) != (item['prompt_tokens'], item['output_tokens']):
            raise ValueError(f'Token count mismatch: {self.usage}')
        if self.info and self.info.get('prefix_caching') is not True:
            raise ValueError('InfiniLM prefix caching telemetry is absent or disabled')


async def request_one(session, url, item, model, through_gateway=True):
    body = dict(item['body'], model=model, stream=True, stream_options={'include_usage': True})
    headers = {'X-Fourway-Item': json.dumps({k: item[k] for k in
               ('id', 'key', 'prompt_tokens', 'output_tokens', 'shared_tokens', 'prefix_digest')}, separators=(',', ':'))} if through_gateway else {}
    start = time.perf_counter()
    obs = Observation()
    result = dict(id=item['id'], prompt_sha256=item['prompt_sha256'], key=item['key'], success=False)
    try:
        async with session.post(url + '/v1/chat/completions', json=body, headers=headers) as response:
            result['http_status'] = response.status
            if response.status != 200:
                raise RuntimeError((await response.text())[:1200])
            async for data in events(response.content):
                obs.accept(data, time.perf_counter())
        obs.validate(item)
        result.update(success=True, usage=obs.usage, engine_telemetry=obs.info,
                      ttft_ms=1000*(obs.first-start), output_sha256=hashlib.sha256(''.join(obs.text).encode()).hexdigest())
    except Exception as exc:
        result['error'] = repr(exc)
    result['event_payload_sha256'] = obs.payload_hasher.hexdigest()
    result['event_payload_count'] = obs.payload_events
    result['event_payload_bytes'] = obs.payload_bytes
    result['e2e_ms'] = 1000*(time.perf_counter()-start)
    return result
