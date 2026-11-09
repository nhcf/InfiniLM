"""Opt-in, process-local idle-block LRU for the exact measured BlockManager.

Native mode delegates unchanged. Mode switches reset metadata only while idle;
no checkout file, tensor, model, scheduler or native library is edited.
"""
from collections import OrderedDict
import functools
import hashlib
import inspect
from pathlib import Path

EXPECTED_SHA = 'd7792e9118683a9c5476158d14849dee8390c9b2dbfdf095cfcaca4051356987'
MODES = {'native', 'prefix_lru'}


def install_manager(cls, verify=True):
    if getattr(cls, '_matrix_lru_installed', False):
        raise RuntimeError('Cache policy already installed')
    if verify and hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest() != EXPECTED_SHA:
        raise RuntimeError('BlockManager differs from the measured source; refusing an unreviewed patch')
    original = {n: getattr(cls, n) for n in
                ('__init__', 'get_computed_blocks', 'free_blocks', 'try_free_blocks', '_deallocate_block')}

    @functools.wraps(original['__init__'])
    def init(self, *args, **kwargs):
        original['__init__'](self, *args, **kwargs)
        self._matrix_mode = 'native'
        self._matrix_idle = OrderedDict()
        self._matrix_evictions = 0

    @functools.wraps(original['get_computed_blocks'])
    def get(self, *args, **kwargs):
        table, tokens = original['get_computed_blocks'](self, *args, **kwargs)
        if self._matrix_mode == 'prefix_lru':
            for block_id in table:
                self._matrix_idle.pop(block_id, None)
        return table, tokens

    @functools.wraps(original['free_blocks'])
    def free(self, table):
        if self._matrix_mode == 'native':
            return original['free_blocks'](self, table)
        # Tails enter the idle queue before the reusable early prefix blocks.
        # A shared block stays protected until its final request releases it.
        for block_id in reversed(table):
            block = self.blocks[block_id]
            if block.ref_count <= 0:
                raise RuntimeError('Duplicate block release')
            block.ref_count -= 1
            if block.ref_count == 0:
                self._matrix_idle.pop(block_id, None)
                self._matrix_idle[block_id] = None

    @functools.wraps(original['_deallocate_block'])
    def deallocate(self, block_id):
        original['_deallocate_block'](self, block_id)
        self._matrix_idle.pop(block_id, None)

    @functools.wraps(original['try_free_blocks'])
    def evict(self, num_required):
        if self._matrix_mode == 'native':
            return original['try_free_blocks'](self, num_required)
        while not self.can_allocate(num_required) and self._matrix_idle:
            block_id = next(iter(self._matrix_idle))
            if block_id not in self.used_block_ids or self.blocks[block_id].ref_count != 0:
                raise RuntimeError('Idle LRU ownership invariant failed')
            self._deallocate_block(block_id)
            self._matrix_evictions += 1
        if not self.can_allocate(num_required):
            # Missing bookkeeping must be visible, never silently evict pins.
            untracked = [i for i in self.used_block_ids if self.blocks[i].ref_count == 0]
            if untracked:
                raise RuntimeError('Untracked idle blocks: '+str(untracked[:8]))
        return self.can_allocate(num_required)

    cls.__init__, cls.get_computed_blocks = init, get
    cls.free_blocks, cls.try_free_blocks, cls._deallocate_block = free, evict, deallocate
    cls._matrix_lru_installed = True


def reset_idle(scheduler, mode, http_active=0):
    if mode not in MODES:
        raise ValueError('Unknown cache mode')
    state = scheduler._fourway_observations
    with state.lock:
        manager = scheduler.cache_manager
        busy = http_active or any(b.ref_count for b in manager.blocks)
        for name in ('waiting_queue', 'running_queue'):
            queue = getattr(scheduler, name, None)
            if queue is not None:
                busy = busy or not queue.sync_q.empty()
        for name in ('remote_kv_requests', 'pending_free_blocks'):
            busy = busy or bool(getattr(scheduler, name, None))
        if busy:
            return {'success': False, 'reason': 'engine_not_idle'}
        # Keep manager identity; the GPU KV tensor remains allocated. Empty
        # metadata makes old tensor contents unreachable until overwritten.
        manager.__init__(manager.num_blocks, manager.block_size)
        manager._matrix_mode = mode
        state.known.clear()
        state.epoch += 1
        return {'success': True, 'cache_mode': mode, 'resident_blocks': 0,
                'free_blocks': manager.num_blocks, 'epoch': state.epoch}


def install_control(server_module, BlockManager):
    install_manager(BlockManager)
    cls = server_module.InferenceServer
    old_create, old_stream = cls._create_app, cls._stream_chat

    @functools.wraps(old_stream)
    async def stream(self, *args, **kwargs):
        self._matrix_http_active = getattr(self, '_matrix_http_active', 0) + 1
        upstream = old_stream(self, *args, **kwargs)
        try:
            async for event in upstream:
                yield event
        finally:
            try:
                await upstream.aclose()
            finally:
                self._matrix_http_active -= 1

    @functools.wraps(old_create)
    def create(self, *args, **kwargs):
        from fastapi import Body
        from fastapi.responses import JSONResponse
        app = old_create(self, *args, **kwargs)

        @app.post('/_matrix/reset_cache')
        async def reset(data: dict = Body(...)):
            if self.engine is None:
                return JSONResponse({'success': False, 'reason': 'not_ready'}, status_code=503)
            try:
                result = reset_idle(self.engine.engine.scheduler, data.get('mode'),
                                    getattr(self, '_matrix_http_active', 0))
            except ValueError as exc:
                return JSONResponse({'success': False, 'reason': str(exc)}, status_code=400)
            return JSONResponse(result, status_code=200 if result['success'] else 409)

        return app

    cls._create_app, cls._stream_chat = create, stream
