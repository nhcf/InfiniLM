"""Backend lifecycle and strict measurements reused from the verified 32B kit."""
from __future__ import annotations


import argparse


import asyncio


from collections import defaultdict


import csv


import datetime


import fcntl


import hashlib


import itertools


import json


import os


from pathlib import Path


import shutil


import signal


import socket


import statistics


import subprocess


import sys


import tempfile


import time


import traceback


KIT = Path(__file__).resolve().parent


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    x = (len(values)-1)*q
    lo, hi = int(x), min(len(values)-1, int(x)+1)
    return values[lo] + (values[hi]-values[lo])*(x-lo)


def csv_write(path, records):
    if not records:
        return
    columns = list(dict.fromkeys(k for row in records for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(records)


def fingerprint(cfg):
    repo = Path(cfg['repo'])
    files = ['python/infinilm/llm/scheduler.py', 'python/infinilm/llm/cache_manager.py',
             'python/infinilm/server/inference_server.py', 'python/infinilm/llm/llm.py']
    model_files = [p for p in Path(cfg['model']).iterdir() if p.name in
                   ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json')]
    native_files = sorted(Path(cfg['native_dir']).glob('_infinilm*.so'))
    native_files += sorted((Path(cfg['core_python'])/'infinicore/lib').glob('_infinicore*.so'))
    git_info = {k: subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
                for k,args in [('head',['rev-parse','HEAD']),('origin',['remote','get-url','origin'])]}
    return dict(config=cfg, nh_git=git_info, nh_files={name: sha(repo/name) for name in files},
                native_bindings={str(p): sha(p) for p in native_files},
                tokenizer_files={p.name: sha(p) for p in sorted(model_files)},
                router_sha256=sha(KIT/'cost_router.py'), telemetry_sha256=sha(KIT/'nh_stream_usage.py'))


def check_config(cfg):
    repo, model = Path(cfg['repo']), Path(cfg['model'])
    if not (repo/'python/infinilm/server/inference_server.py').is_file() or not (model/'config.json').is_file():
        raise ValueError('Set repo/model to the working InfiniLM checkout and /mnt/zj-data/models/Qwen3-32B/')
    origin = subprocess.check_output(['git','-C',str(repo),'remote','get-url','origin'],text=True).strip().rstrip('/')
    if origin.removesuffix('.git') not in ('https://github.com/nhcf/InfiniLM','git@github.com:nhcf/InfiniLM',
                                         'ssh://git@github.com/nhcf/InfiniLM'):
        raise ValueError('This experiment requires nhcf/InfiniLM; actual origin='+origin)
    mc = json.loads((model/'config.json').read_text())
    expected = dict(model_type='qwen3', num_hidden_layers=64, num_key_value_heads=8, head_dim=128, hidden_size=5120)
    if any(mc.get(k) != v for k, v in expected.items()) or mc.get('quantization_config') or mc.get('quantization'):
        raise ValueError('This preset requires unquantized Qwen3-32B; model config does not match')
    for key in ('native_dir', 'core_python', 'core_root'):
        if not Path(cfg[key]).is_dir():
            raise ValueError('Verified native environment directory missing: '+key+'='+str(cfg[key]))
    expected_native = list(Path(cfg['native_dir']).glob('_infinilm*.so'))
    if len(expected_native) != 1:
        raise ValueError('native_dir must contain exactly one _infinilm*.so')
    for local in (repo/'python/infinilm/lib').glob('_infinilm*.so'):
        if local.resolve() != expected_native[0].resolve():
            raise ValueError('A checkout-local binding would override native_dir: '+str(local)+
                             '; select the already verified matching binding before running')
    for key in ('nh_python', 'vllm_python'):
        if not Path(cfg[key]).is_file():
            raise ValueError('Python executable missing: '+key)
    if len(cfg['gpu_ids']) != 8 or len(set(cfg['gpu_ids'])) != 8:
        raise ValueError('This experiment requires eight distinct full replicas, TP=1')
    if cfg['nh_blocks']*cfg['nh_block_size'] != cfg['vllm_gpu_blocks']*cfg['vllm_block_size']:
        raise ValueError('Match KV token capacity between engines before this controlled comparison')
    if not 0 < cfg['max_batch'] <= 8 or cfg['token_budget'] < cfg['prompt_tokens']:
        raise ValueError('Require batch 1..8 and token budget >= full prompt length')
    if cfg['max_context'] < cfg['prompt_tokens'] + cfg['output_tokens']:
        raise ValueError('Context must fit the full input plus output')
    if cfg['max_context'] > cfg['nh_blocks'] * cfg['nh_block_size']:
        raise ValueError('KV token capacity must accommodate max_context')
    required_tokens = cfg['max_batch'] * (cfg['prompt_tokens'] + cfg['output_tokens'])
    if required_tokens > cfg['nh_blocks'] * cfg['nh_block_size']:
        raise ValueError('KV token capacity is too small for max_batch full sequences in this preset')
    if not 0 < cfg['shadow_capacity_fraction'] < 1:
        raise ValueError('shadow_capacity_fraction must be between zero and one')
    from cost_router import load_manager
    load_manager()
    local_manager = repo/'python/infinilm/server/routing/manager.py'
    if local_manager.exists():
        load_manager(local_manager)  # Do not silently compare a different baseline.
    ports = list(range(cfg['port'], cfg['port']+9)) + list(range(cfg['port']+1001, cfg['port']+1009))
    for port in ports:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))


def backend_env(cfg, engine, gpu, rpc):
    env = dict(os.environ)
    # Keep the working image's driver libraries, but isolate the two engines.
    env.pop('PYTHONPATH', None)
    for key in ('PYTHONHOME', 'VIRTUAL_ENV', '__PYVENV_LAUNCHER__'):
        env.pop(key, None)
    for key in ('INFINILM_DIAGNOSTICS_DIR', 'INFINILM_DIAGNOSTICS_LOGITS', 'INFINILM_DIAGNOSTICS_KV',
                'VLLM_USE_V1', 'VLLM_PLUGINS', 'VLLM_WORKER_MULTIPROC_METHOD'):
        env.pop(key, None)
    env.update({k: str(v) for k, v in cfg.get('common_env', {}).items()})
    env.update({k: str(v) for k, v in cfg.get(engine+'_env', {}).items()})
    if env.get('PYTHONHOME'):
        raise ValueError('Engine PYTHONHOME override is not permitted: use its explicit Python executable')
    for entry in env.get('PYTHONPATH', '').split(os.pathsep):
        if entry and any(part.startswith('.controller') for part in Path(entry).parts):
            raise ValueError('Controller packages must not be on an engine PYTHONPATH')
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS=str(cfg['cpu_threads']), MKL_NUM_THREADS=str(cfg['cpu_threads']))
    if cfg['device'] == 'moore':
        env['MUSA_VISIBLE_DEVICES'] = str(gpu)
    if engine == 'nh':
        env['PYTHONPATH'] = os.pathsep.join([str(Path(cfg['repo'])/'python'), cfg['native_dir'], cfg['core_python']]
                                           + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
        env['INFINI_ROOT'] = cfg['core_root']
        env['FOURWAY_EXPECT_NATIVE_DIR'] = cfg['native_dir']
        env['LD_LIBRARY_PATH'] = str(Path(cfg['core_root'])/'lib') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
        env['INFINILM_MAX_NUM_BATCHED_TOKENS'] = str(cfg['token_budget'])
    else:
        env['VLLM_RPC_BASE_PATH'] = str(rpc)
    return env


def command(cfg, engine, port):
    if engine == 'nh':
        return [cfg['nh_python'], str(KIT/'nh_stream_usage.py'), '--model', cfg['model'],
                '--device', cfg['device'], '--dtype', 'bfloat16', '--tp', '1', '--dp', '1',
                '--master-addr', '127.0.0.1', '--master-port', str(port+1000),
                '--enable-paged-attn', '--num-blocks', str(cfg['nh_blocks']), '--block-size', str(cfg['nh_block_size']),
                '--max-cache-len', str(cfg['max_context']), '--max-batch-size', str(cfg['max_batch']),
                '--max-new-tokens', str(cfg['output_tokens']), '--attn', 'flash-attn', '--temperature', '0', '--top-p', '1',
                '--top-k', '1', '--ignore-eos', '--host', '127.0.0.1', '--port', str(port)]
    return [cfg['vllm_python'], '-m', 'vllm.entrypoints.openai.api_server', '--model', cfg['model'],
            '--served-model-name', 'fourway-qwen3-32b', '--dtype', 'bfloat16', '--tensor-parallel-size', '1',
            '--max-model-len', str(cfg['max_context']), '--max-num-seqs', str(cfg['max_batch']),
            '--max-num-batched-tokens', str(cfg['token_budget']), '--gpu-memory-utilization', str(cfg['vllm_memory']),
            '--block-size', str(cfg['vllm_block_size']), '--num-gpu-blocks-override', str(cfg['vllm_gpu_blocks']),
            '--enable-prefix-caching', '--enable-prompt-tokens-details', '--enable-chunked-prefill',
            '--enforce-eager', '--generation-config', 'vllm',
            '--host', '127.0.0.1', '--port', str(port)]


class Pool:
    def __init__(self, cfg, engine, folder, count=8):
        self.cfg, self.engine, self.folder, self.count = cfg, engine, folder, count
        self.children, self.logs = [], []
        self.rpc = tempfile.mkdtemp(prefix='fw.', dir='/tmp')
        self.urls, self.models = [], []

    async def start(self):
        from aiohttp import ClientSession, ClientTimeout, TCPConnector
        self.folder.mkdir(parents=True, exist_ok=True)
        launch = []
        for i, gpu in enumerate(self.cfg['gpu_ids'][:self.count]):
            port = self.cfg['port']+1+i
            argv = command(self.cfg, self.engine, port)
            env = backend_env(self.cfg, self.engine, gpu, self.rpc)
            log = (self.folder/f'worker{i}.log').open('w')
            self.logs.append(log)
            process = subprocess.Popen(argv, cwd=self.cfg['repo'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.children.append(process)
            self.urls.append(f'http://127.0.0.1:{port}')
            launch.append(dict(worker=i, gpu=gpu, pid=process.pid, argv=argv,
                               environment={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'MUSA_VISIBLE_DEVICES',
                                 'PYTHONPATH', 'LD_LIBRARY_PATH', 'INFINI_ROOT', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                 'INFINILM_MAX_NUM_BATCHED_TOKENS', 'VLLM_RPC_BASE_PATH') if k in env}))
        dump(self.folder/'launch.json', launch)
        deadline = time.monotonic()+self.cfg['startup_timeout_s']
        pending = set(range(self.count))
        heartbeat = 0.0
        async with ClientSession(timeout=ClientTimeout(total=5), connector=TCPConnector(limit=0, force_close=True)) as client:
            while pending:
                self.assert_alive()
                if time.monotonic() >= heartbeat:
                    print(f'WORKER_STARTUP engine={self.engine} ready={self.count-len(pending)}/{self.count} logs={self.folder}', flush=True)
                    heartbeat = time.monotonic()+20
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Backend startup timeout; inspect {self.folder}')
                for i in list(pending):
                    try:
                        async with client.get(self.urls[i]+'/health') as r:
                            if r.status == 200:
                                pending.remove(i)
                    except Exception:
                        pass
                if pending:
                    await asyncio.sleep(1)
            for url in self.urls:
                async with client.get(url+'/v1/models') as r:
                    obj = await r.json()
                    self.models.append(obj['data'][0]['id'])
        return self

    def assert_alive(self):
        dead = [(i, p.returncode) for i, p in enumerate(self.children) if p.poll() is not None]
        if dead:
            raise RuntimeError(f'Worker exited: {dead}; inspect {self.folder}')

    async def close(self):
        for process in self.children:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic()+20
        while any(p.poll() is None for p in self.children) and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        for process in self.children:
            # Terminate remaining descendants in this runner-owned group too.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for log in self.logs:
            log.close()
        shutil.rmtree(self.rpc, ignore_errors=True)


def nnls_small(x, y):
    import numpy as np
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    best_loss, best = float('inf'), None
    for bits in itertools.product([False, True], repeat=x.shape[1]):
        cols = [i for i, use in enumerate(bits) if use]
        coef = np.zeros(x.shape[1])
        if cols:
            fit = np.linalg.lstsq(x[:, cols], y, rcond=None)[0]
            if np.any(fit < 0):
                continue
            coef[cols] = fit
        loss = float(((x@coef-y)**2).sum())
        if loss < best_loss:
            best_loss, best = loss, coef
    return best.tolist()


def statistics_row(records, elapsed, meta):
    good = [r for r in records if r['success']]
    values = [r['ttft_ms'] for r in good]
    output_tokens = sum(r['usage']['completion_tokens'] for r in good)
    cached = [r['usage'].get('prompt_tokens_details') or {} for r in good]
    available = [d['cached_tokens'] for d in cached if d.get('cached_tokens') is not None]
    return dict(meta, requests=len(records), success=len(good), failures=len(records)-len(good),
                wall_s=elapsed, output_tokens=output_tokens, output_tok_s=output_tokens/elapsed,
                request_s=len(good)/elapsed, ttft_mean_ms=statistics.mean(values) if values else None,
                ttft_p50_ms=quantile(values, .5), ttft_p95_ms=quantile(values, .95), ttft_p99_ms=quantile(values, .99),
                e2e_mean_ms=statistics.mean([r['e2e_ms'] for r in good]) if good else None,
                cached_tokens=sum(available) if len(available) == len(good) else None,
                cache_usage_coverage=len(available)/len(good) if good else 0,
                cache_fraction=sum(available)/sum(r['usage']['prompt_tokens'] for r in good)
                if good and len(available) == len(good) else None)


