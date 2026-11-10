"""Unit tests for SLA / memory-aware dynamic batching in InfiniLM.

GPU-free and dependency-free: ``infinilm/__init__.py`` eagerly imports the
compiled ``infinicore`` backend, so the modules under test are loaded directly
by path with lightweight stubs for the third-party packages that are missing
on a control node (``janus``, ``xxhash``, ``numpy``).  When those packages are
installed the real ones are used.

Run either way:

    cd /workspace/InfiniLM
    python test/service/test_sla_scheduler.py            # built-in runner
    python -m pytest test/service/test_sla_scheduler.py -v   # if pytest exists

Coverage:
  * EngineConfig validation for the new CLI surface
  * paper Algorithm 2 branches, midpoint search, clamps, running-count floor
  * record_decode_step bookkeeping (disabled no-op, empty batch, window bound)
  * schedule() integration: prefill cap, decode cap, B_max hard cap,
    running-count floor, disabled-mode regression, cache-exhaustion deferral
  * paper Algorithm 1 memory bound (b_mem) and min(b_sla, b_mem)
"""

import hashlib
import importlib.util
import pathlib
import queue
import random
import sys
import types

# ---------------------------------------------------------------------------
# Minimal pytest-compatible shim (pytest is optional in this environment)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - preference for the real thing
    import pytest  # noqa: F401

    _HAS_PYTEST = True
except ImportError:  # pragma: no cover
    _HAS_PYTEST = False


class _Raises:
    """Minimal stand-in for ``pytest.raises`` usable as a context manager."""

    def __init__(self, exc_type, match=None):
        self.exc_type = exc_type
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            raise AssertionError(f"expected {self.exc_type.__name__} to be raised")
        if not issubclass(exc_type, self.exc_type):
            return False
        self.value = exc_value
        if self.match is not None and self.match not in str(exc_value):
            raise AssertionError(
                f"{self.match!r} not found in error message {str(exc_value)!r}"
            )
        return True


def raises(exc_type, match=None):
    return _Raises(exc_type, match)


def approx(value, rel=1e-6, abs_=1e-9):
    class _Approx:
        def __init__(self, expected):
            self.expected = expected

        def __eq__(self, other):
            return abs(other - self.expected) <= max(abs_, rel * abs(self.expected))

        def __repr__(self):  # pragma: no cover
            return f"approx({self.expected})"

    return _Approx(value)


# ---------------------------------------------------------------------------
# Load the modules under test without the compiled infinicore backend
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"


def _install_missing_dep(name, factory):
    try:
        __import__(name)
        return
    except ImportError:
        sys.modules[name] = factory()


def _janus_stub():
    mod = types.ModuleType("janus")

    class _SyncAsyncQueue:
        def __init__(self):
            # The scheduler only ever touches the thread-safe side.
            self.sync_q = queue.Queue()
            self.async_q = queue.Queue()

    mod.Queue = _SyncAsyncQueue
    return mod


def _xxhash_stub():
    mod = types.ModuleType("xxhash")

    class _Hasher:
        def __init__(self, *args, **kwargs):
            self._buf = bytearray()

        def update(self, data):
            self._buf.extend(bytes(data))

        def digest(self):
            # Real xxh3_128 digests are 16 bytes; md5 matches that length.
            return hashlib.md5(bytes(self._buf)).digest()

        def intdigest(self):
            # Older revisions of cache_manager.py hash with xxh64() and read
            # the result back as an integer.
            return int.from_bytes(self.digest()[:8], "little")

    mod.xxh3_128 = _Hasher
    mod.xxh64 = _Hasher
    return mod


def _numpy_stub():
    mod = types.ModuleType("numpy")

    class _Array:
        def __init__(self, seq):
            self._buf = b"".join(int(x).to_bytes(4, "little", signed=True) for x in seq)

        def __len__(self):
            return len(self._buf) // 4

        def tobytes(self):
            return self._buf

    def asarray(seq, dtype=None):
        return _Array(seq)

    mod.asarray = asarray
    # Older revisions of cache_manager.py call np.array(..., dtype=np.int32).
    mod.array = asarray
    mod.int32 = "int32"
    return mod


def _make_parent_package(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
    return mod


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    parent_name, _, child = name.rpartition(".")
    if parent_name:
        setattr(sys.modules[parent_name], child, module)
    return module


def _bootstrap():
    _install_missing_dep("janus", _janus_stub)
    _install_missing_dep("xxhash", _xxhash_stub)
    _install_missing_dep("numpy", _numpy_stub)

    for pkg in ("infinilm", "infinilm.llm", "infinilm.config"):
        if pkg not in sys.modules:
            _make_parent_package(pkg)

    # NOTE: prefix_cache.py exists only in newer upstream revisions; this
    # checkout keeps the prefix hashing inside cache_manager.py, so the module
    # is loaded only when present.
    _prefix_cache_path = _PYTHON_DIR / "infinilm/llm/prefix_cache.py"
    if _prefix_cache_path.exists():
        _load_module("infinilm.llm.prefix_cache", _prefix_cache_path)
    _load_module(
        "infinilm.llm.nsa_paged_owner", _PYTHON_DIR / "infinilm/llm/nsa_paged_owner.py"
    )
    _load_module(
        "infinilm.llm.sampling_params", _PYTHON_DIR / "infinilm/llm/sampling_params.py"
    )
    _load_module("infinilm.llm.request", _PYTHON_DIR / "infinilm/llm/request.py")
    _load_module(
        "infinilm.llm.cache_manager", _PYTHON_DIR / "infinilm/llm/cache_manager.py"
    )
    _load_module(
        "infinilm.config.kv_transfer", _PYTHON_DIR / "infinilm/config/kv_transfer.py"
    )
    _load_module(
        "infinilm.config.engine_config",
        _PYTHON_DIR / "infinilm/config/engine_config.py",
    )
    _load_module("infinilm.llm.scheduler", _PYTHON_DIR / "infinilm/llm/scheduler.py")

    return (
        sys.modules["infinilm.llm.scheduler"].Scheduler,
        sys.modules["infinilm.config.engine_config"].EngineConfig,
        sys.modules["infinilm.llm.request"].InferenceRequest,
        sys.modules["infinilm.llm.request"].RequestStatus,
        sys.modules["infinilm.llm.sampling_params"].SamplingParams,
    )


Scheduler, EngineConfig, InferenceRequest, RequestStatus, SamplingParams = _bootstrap()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_scheduler(**kwargs):
    params = dict(
        max_batch_size=64,
        num_blocks=512,
        block_size=256,
        max_num_batched_tokens=100000,
    )
    params.update(kwargs)
    return Scheduler(**params)


def sla_scheduler(**kwargs):
    params = dict(
        enable_sla_constraint=True,
        sla_latency=0.05,
        sla_epsilon=0.005,
        batch_size_lower=4,
        batch_size_upper=64,
        sla_alpha=8,
        sla_sigma=1,
        sla_history_window=100,
    )
    params.update(kwargs)
    return make_scheduler(**params)


def feed(sched, latency, batch, times=1):
    for _ in range(times):
        sched.record_decode_step(latency, batch)


def make_waiting_request(sched, prompt_len=8, request_id="w"):
    req = InferenceRequest(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=16),
    )
    sched.add_request(req)
    return req


def make_decode_request(sched, prompt_len=8, request_id="d", allocate=True):
    """Build a RUNNING request with a real block table, as schedule() expects."""
    req = InferenceRequest(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=64),
    )
    if allocate:
        block = sched.cache_manager._allocate_block()
        req.block_table = [block.block_id]
    req.status = RequestStatus.RUNNING
    req.num_local_cached_tokens = prompt_len
    # Producer-side submission marker: _claim_request refuses requests that
    # were never submitted through the scheduler's intake path.
    req._paged_submitted = True
    sched.running_queue.sync_q.put(req)
    return req


# ---------------------------------------------------------------------------
# EngineConfig validation
# ---------------------------------------------------------------------------


class TestEngineConfigValidation:
    def test_defaults_keep_the_feature_off(self):
        cfg = EngineConfig(model_path="/tmp/model")
        assert cfg.enable_sla_constraint is False
        assert cfg.enable_memory_aware_batching is False
        assert cfg.sla_latency == 0.05
        assert cfg.sla_epsilon == 0.005
        assert cfg.batch_size_lower == 1
        assert cfg.batch_size_upper == 0
        assert cfg.sla_alpha == 8
        assert cfg.sla_sigma == 1
        assert cfg.sla_history_window == 100
        assert cfg.memory_usage_factor == 0.95

    def test_sla_requires_paged_cache(self):
        with raises(ValueError, match="paged"):
            EngineConfig(
                model_path="/tmp/model",
                cache_type="static",
                enable_sla_constraint=True,
            )

    def test_memory_aware_requires_paged_cache(self):
        with raises(ValueError, match="paged"):
            EngineConfig(
                model_path="/tmp/model",
                cache_type="static",
                enable_memory_aware_batching=True,
            )

    def test_sla_latency_must_be_positive(self):
        with raises(ValueError, match="sla_latency"):
            EngineConfig(
                model_path="/tmp/model",
                enable_sla_constraint=True,
                sla_latency=0.0,
            )

    def test_epsilon_must_be_smaller_than_latency(self):
        with raises(ValueError, match="sla_epsilon"):
            EngineConfig(
                model_path="/tmp/model",
                enable_sla_constraint=True,
                sla_latency=0.05,
                sla_epsilon=0.05,
            )

    def test_negative_epsilon_rejected(self):
        with raises(ValueError, match="sla_epsilon"):
            EngineConfig(
                model_path="/tmp/model",
                enable_sla_constraint=True,
                sla_epsilon=-0.001,
            )

    def test_batch_size_lower_must_be_positive(self):
        with raises(ValueError, match="batch_size_lower"):
            EngineConfig(
                model_path="/tmp/model",
                enable_sla_constraint=True,
                batch_size_lower=0,
            )

    def test_batch_size_upper_must_not_exceed_max_batch_size(self):
        # CUDA graphs are only captured up to max_batch_size.
        with raises(ValueError, match="max_batch_size"):
            EngineConfig(
                model_path="/tmp/model",
                enable_sla_constraint=True,
                max_batch_size=32,
                batch_size_upper=64,
            )

    def test_batch_size_upper_zero_follows_max_batch_size(self):
        cfg = EngineConfig(
            model_path="/tmp/model",
            enable_sla_constraint=True,
            max_batch_size=32,
            batch_size_upper=0,
        )
        assert cfg.batch_size_upper == 0

    def test_alpha_sigma_and_window_must_be_positive(self):
        for field in ("sla_alpha", "sla_sigma", "sla_history_window"):
            with raises(ValueError, match=field):
                EngineConfig(
                    model_path="/tmp/model",
                    enable_sla_constraint=True,
                    **{field: 0},
                )

    def test_memory_usage_factor_must_be_in_range(self):
        for bad in (0.0, -0.5, 1.5):
            with raises(ValueError, match="memory_usage_factor"):
                EngineConfig(
                    model_path="/tmp/model",
                    enable_memory_aware_batching=True,
                    memory_usage_factor=bad,
                )


# ---------------------------------------------------------------------------
# Scheduler state / default cap
# ---------------------------------------------------------------------------


class TestSchedulerDynamicCapState:
    def test_disabled_keeps_static_cap(self):
        sched = make_scheduler(max_batch_size=16)
        assert sched.enable_sla is False
        assert sched.enable_mem_aware is False
        assert sched.current_max_batch_size == 16

    def test_sla_initialises_bounds_to_b_min_b_max(self):
        sched = sla_scheduler(batch_size_lower=4, batch_size_upper=64)
        assert sched.b_lower_hard == 4
        assert sched.b_upper_hard == 64
        assert sched.b_low == 4
        assert sched.b_high == 64
        assert sched.current_max_batch_size == 64

    def test_batch_size_upper_zero_follows_max_batch_size(self):
        sched = sla_scheduler(max_batch_size=32, batch_size_upper=0)
        assert sched.b_upper_hard == 32

    def test_b_upper_hard_never_exceeds_max_batch_size(self):
        sched = sla_scheduler(max_batch_size=8, batch_size_upper=4096)
        assert sched.b_upper_hard == 8

    def test_sla_band_is_latency_plus_minus_epsilon(self):
        sched = sla_scheduler(sla_latency=0.05, sla_epsilon=0.005)
        assert sched.sla_low == approx(0.045)
        assert sched.sla_high == approx(0.055)


# ---------------------------------------------------------------------------
# Paper Algorithm 2: bounds update
# ---------------------------------------------------------------------------


class TestSlaBoundsAlgorithm:
    def test_above_sla_lowers_the_upper_bound(self):
        sched = sla_scheduler()
        feed(sched, 0.2, 32, times=100)
        result = sched._update_sla_bounds(32)
        assert sched.b_high == 32
        assert sched.b_high < sched.b_upper_hard
        assert result == 32
        assert result < sched.b_upper_hard

    def test_above_sla_walks_the_lower_bound_down(self):
        sched = sla_scheduler(batch_size_lower=4, sla_sigma=1)
        sched.b_low = 10
        sched.b_high = 64
        feed(sched, 0.2, 40, times=100)
        sched._update_sla_bounds(0)
        assert sched.b_low == 9

    def test_below_sla_raises_the_lower_bound_and_upper_bound(self):
        sched = sla_scheduler(batch_size_lower=4, batch_size_upper=64, sla_sigma=1)
        sched.b_high = 60
        feed(sched, 0.001, 10, times=100)
        result = sched._update_sla_bounds(10)
        assert sched.b_low == 10
        assert sched.b_high == 61  # grew by sigma
        assert result == 35  # floor((10 + 61) / 2)

    def test_below_sla_upper_bound_clamped_by_b_max(self):
        sched = sla_scheduler(batch_size_lower=1, batch_size_upper=64)
        feed(sched, 0.001, 2, times=100)
        sched._update_sla_bounds(0)
        assert sched.b_high == 64

    def test_inside_band_contracts_around_avg_batch(self):
        sched = sla_scheduler(batch_size_lower=4, batch_size_upper=64, sla_alpha=8)
        feed(sched, 0.05, 40, times=100)
        result = sched._update_sla_bounds(40)
        assert sched.b_high == 44  # avg_batch + floor(alpha/2)
        assert sched.b_low == 36  # avg_batch - floor(alpha/2)
        assert result == 40

    def test_inside_band_respects_bounds(self):
        sched = sla_scheduler(batch_size_lower=4, batch_size_upper=64, sla_alpha=64)
        feed(sched, 0.05, 40, times=100)
        sched._update_sla_bounds(0)
        assert sched.b_high == 64
        assert sched.b_low == 8  # 40 - 32, above B_min

    def test_no_history_yields_midpoint_of_range(self):
        sched = sla_scheduler(batch_size_lower=1, batch_size_upper=64)
        assert sched._update_sla_bounds(0) == 32

    def test_result_is_at_least_running_decode_count(self):
        sched = sla_scheduler(batch_size_lower=1, batch_size_upper=64)
        feed(sched, 0.5, 1, times=100)
        result = sched._update_sla_bounds(50)
        assert result == 50

    def test_result_never_exceeds_b_upper_hard(self):
        sched = sla_scheduler(batch_size_lower=1, batch_size_upper=64)
        sched.b_high = 10**6
        assert sched._update_sla_bounds(0) <= 64

    def test_average_stats_use_the_sliding_window(self):
        sched = sla_scheduler(sla_history_window=4)
        for lat, batch in ((0.1, 1), (0.2, 2), (0.3, 3), (0.4, 4)):
            sched.record_decode_step(lat, batch)
        avg_latency, avg_batch = sched._get_recent_decode_stats()
        assert avg_latency == approx(0.25)
        assert avg_batch == approx(2.5)

    def test_random_sequence_keeps_the_batch_invariants(self):
        rng = random.Random(20260914)
        sched = sla_scheduler(batch_size_lower=1, batch_size_upper=64)
        for _ in range(500):
            latency = rng.uniform(0.0, 0.5)
            batch = rng.randint(1, 64)
            running = rng.randint(0, 200)
            sched.record_decode_step(latency, batch)
            result = sched._update_sla_bounds(running)
            assert result <= sched.b_upper_hard
            assert result >= min(running, sched.b_upper_hard)


# ---------------------------------------------------------------------------
# record_decode_step bookkeeping
# ---------------------------------------------------------------------------


class TestRecordDecodeStep:
    def test_disabled_is_a_noop(self):
        sched = make_scheduler()
        sched.record_decode_step(0.1, 8)
        assert len(sched.decode_latency_history) == 0
        assert len(sched.decode_batch_history) == 0
        assert sched.num_decode_steps == 0

    def test_empty_batch_is_ignored(self):
        sched = sla_scheduler()
        sched.record_decode_step(0.1, 0)
        assert len(sched.decode_latency_history) == 0

    def test_window_is_bounded(self):
        sched = sla_scheduler(sla_history_window=3)
        feed(sched, 0.1, 8, times=10)
        assert len(sched.decode_latency_history) == 3
        assert len(sched.decode_batch_history) == 3

    def test_decode_steps_are_counted(self):
        sched = sla_scheduler()
        feed(sched, 0.1, 8, times=5)
        assert sched.num_decode_steps == 5

    def test_stats_default_when_history_is_empty(self):
        sched = sla_scheduler(batch_size_lower=4)
        avg_latency, avg_batch = sched._get_recent_decode_stats()
        assert avg_latency == 0.0
        assert avg_batch == 4.0

    def test_dynamic_batch_stats_reports_state(self):
        sched = sla_scheduler()
        feed(sched, 0.02, 12, times=10)
        stats = sched.get_dynamic_batch_stats()
        assert stats["enable_sla_constraint"] is True
        assert stats["num_decode_steps"] == 10
        assert stats["avg_decode_batch"] == approx(12.0)
        assert stats["avg_decode_latency"] == approx(0.02)
        assert stats["batch_size_upper"] == 64


# ---------------------------------------------------------------------------
# Paper Algorithm 1: memory bound
# ---------------------------------------------------------------------------


class TestMemoryBound:
    def test_without_running_requests_returns_upper_bound(self):
        sched = sla_scheduler(batch_size_upper=64)
        assert sched._compute_mem_bound(0) == 64

    def test_scales_with_blocks_per_sequence(self):
        sched = make_scheduler(num_blocks=100, memory_usage_factor=0.95)
        for _ in range(20):
            sched.cache_manager._allocate_block()
        # used = 20, running = 4 -> 5 blocks/seq -> floor(100*0.95/5) = 19
        assert sched._compute_mem_bound(4) == 19

    def test_never_below_running_requests(self):
        sched = make_scheduler(num_blocks=100, memory_usage_factor=0.95)
        for _ in range(100):
            sched.cache_manager._allocate_block()
        assert sched._compute_mem_bound(1) == 1

    def test_counts_pending_remote_kv_blocks(self):
        sched = make_scheduler(num_blocks=100, memory_usage_factor=0.95)
        sched.pending_kv_decode_blocks = 20
        assert sched._compute_mem_bound(4) == 19

    def test_memory_usage_factor_is_applied(self):
        sched = make_scheduler(num_blocks=100, memory_usage_factor=0.5)
        for _ in range(20):
            sched.cache_manager._allocate_block()
        # floor(100*0.5/5) = 10
        assert sched._compute_mem_bound(4) == 10


# ---------------------------------------------------------------------------
# schedule() integration
# ---------------------------------------------------------------------------


class TestScheduleIntegration:
    def test_prefill_batch_respects_the_dynamic_cap(self):
        sched = sla_scheduler(batch_size_lower=1, batch_size_upper=64)
        feed(sched, 0.2, 1, times=100)  # -> b_high = max(1, 1 + 8) = 9, cap = 5
        for i in range(20):
            make_waiting_request(sched, request_id=f"w{i}")
        out = sched.schedule()
        assert out is not None and out.is_prefill is True
        assert sched.current_max_batch_size == 5
        assert out.num_requests == 5

    def test_disabled_mode_uses_max_batch_size(self):
        sched = make_scheduler(max_batch_size=8)
        for i in range(100):
            make_waiting_request(sched, request_id=f"w{i}")
        out = sched.schedule()
        assert out is not None and out.is_prefill is True
        assert sched.current_max_batch_size == 8
        assert out.num_requests == 8

    def test_decode_batch_is_capped_by_b_max(self):
        sched = sla_scheduler(batch_size_lower=1, batch_size_upper=64)
        for i in range(100):
            make_decode_request(sched, request_id=f"d{i}")
        out = sched.schedule()
        assert out is not None and out.is_prefill is False
        assert sched.current_max_batch_size == 64
        assert out.num_requests == 64

    def test_decode_batch_is_not_cut_below_running_count(self):
        sched = sla_scheduler(batch_size_lower=1, batch_size_upper=64)
        feed(sched, 0.5, 1, times=100)  # would give a tiny cap (5) on its own
        for i in range(20):
            make_decode_request(sched, request_id=f"d{i}")
        out = sched.schedule()
        assert out is not None and out.is_prefill is False
        # Algorithm 2 line 15: never schedule fewer than the running requests.
        assert sched.current_max_batch_size == 20
        assert out.num_requests == 20

    def test_degenerate_bounds_still_allow_progress(self):
        sched = sla_scheduler(batch_size_lower=1, batch_size_upper=64)
        sched.b_low = -50
        sched.b_high = 3
        for i in range(4):
            make_waiting_request(sched, request_id=f"w{i}")
        out = sched.schedule()
        assert sched.current_max_batch_size >= 1
        assert out is not None and out.num_requests >= 1

    def test_combined_cap_is_min_of_sla_and_memory(self):
        sched = sla_scheduler(
            batch_size_lower=1,
            batch_size_upper=64,
            enable_memory_aware_batching=True,
            memory_usage_factor=0.95,
        )
        sched.cache_manager = sched.cache_manager.__class__(
            num_blocks=8, block_size=256
        )
        for i in range(2):
            make_decode_request(sched, request_id=f"d{i}")
        for _ in range(2):
            sched.cache_manager._allocate_block()
        out = sched.schedule()
        # used = 4 blocks / 2 running = 2 per seq -> b_mem = floor(8*0.95/2) = 3
        assert sched._compute_mem_bound(2) == 3
        assert sched.current_max_batch_size == 3
        assert out is not None and out.num_requests == 2

    def test_decode_is_deferred_when_the_cache_is_exhausted(self):
        sched = make_scheduler(max_batch_size=4, num_blocks=1, block_size=4)
        req = make_decode_request(sched, prompt_len=5, request_id="d0")
        # (5 - 1) % 4 == 0 -> append_slot needs a new block, but the single
        # block is already owned by this request (ref_count == 1).
        out = sched.schedule()
        assert out is None
        assert sched.running_queue.sync_q.qsize() == 1
        assert sched.running_queue.sync_q.get_nowait() is req
        assert req.block_table == [0]

    def test_memory_aware_only_does_not_touch_sla_bounds(self):
        sched = make_scheduler(max_batch_size=64, enable_memory_aware_batching=True)
        assert sched.enable_sla is False
        assert sched.current_max_batch_size == 64
        out = sched.schedule()
        assert out is None
        stats = sched.get_dynamic_batch_stats()
        assert stats["enable_sla_constraint"] is False
        assert stats["enable_memory_aware_batching"] is True


# ---------------------------------------------------------------------------
# Cap semantics: active-population bound (admission budget) + measurement-only mode
# ---------------------------------------------------------------------------


class TestActiveSetBudget:
    def test_prefill_budget_equals_cap_when_controller_is_off(self):
        sched = make_scheduler(max_batch_size=8)
        # Controller off: prefill admits up to max_batch_size per step
        # (bit-identical to before the change).
        assert sched._compute_prefill_budget(0) == 8
        assert sched._compute_prefill_budget(5) == 8

    def test_prefill_budget_subtracts_running_requests(self):
        sched = sla_scheduler(batch_size_lower=1, batch_size_upper=64)
        sched.current_max_batch_size = 8
        assert sched._compute_prefill_budget(0) == 8
        assert sched._compute_prefill_budget(5) == 3
        assert sched._compute_prefill_budget(8) == 0
        # Over-subscribed: drain back under the cap first; admit none this step.
        assert sched._compute_prefill_budget(20) == 0

    def test_running_requests_consume_part_of_the_cap(self):
        sched = sla_scheduler(batch_size_lower=1, batch_size_upper=64)
        feed(sched, 0.2, 1, times=100)  # high-pressure history -> cap = 5
        for i in range(3):
            make_decode_request(sched, request_id=f"d{i}")
        for i in range(10):
            make_waiting_request(sched, request_id=f"w{i}")
        out = sched.schedule()
        assert sched.current_max_batch_size == 5
        assert out is not None and out.is_prefill is True
        # 3 running requests already hold slots; this step admits only 5 - 3 = 2.
        assert out.num_requests == 2
        assert sched.waiting_queue.sync_q.qsize() == 8
        assert 3 + out.num_requests <= sched.current_max_batch_size

    def test_saturated_active_set_still_serves_decode(self):
        sched = sla_scheduler(batch_size_lower=1, batch_size_upper=64)
        feed(sched, 0.2, 1, times=100)  # high-pressure history -> cap = 5
        for i in range(5):
            make_decode_request(sched, request_id=f"d{i}")
        for i in range(4):
            make_waiting_request(sched, request_id=f"w{i}")
        out = sched.schedule()
        # Active set fills the cap (5) -> admission budget 0: no new request
        # this step, but decode must still run (no deadlock).
        assert sched.current_max_batch_size == 5
        assert sched.prefill_budget == 0
        assert out is not None and out.is_prefill is False
        assert out.num_requests == 5
        assert sched.waiting_queue.sync_q.qsize() == 4

    def test_stats_expose_running_and_budget(self):
        sched = sla_scheduler(batch_size_lower=1, batch_size_upper=64)
        assert sched.get_dynamic_batch_stats()["enable_batch_stats"] is False
        make_decode_request(sched, request_id="d0")
        out = sched.schedule()
        assert out is not None and out.is_prefill is False
        stats = sched.get_dynamic_batch_stats()
        assert stats["num_running"] == 1
        assert stats["prefill_budget"] == max(0, stats["current_max_batch_size"] - 1)


class TestMeasurementOnlyBatchStats:
    def test_off_by_default(self):
        sched = make_scheduler(max_batch_size=8)
        sched.record_decode_step(0.02, 8)
        assert sched.num_decode_steps == 0
        assert len(sched.decode_latency_history) == 0

    def test_records_without_the_control_loop(self):
        sched = make_scheduler(max_batch_size=8, enable_batch_stats=True)
        assert sched.enable_sla is False
        assert sched.enable_mem_aware is False
        sched.record_decode_step(0.02, 8)
        sched.record_decode_step(0.04, 4)
        assert sched.num_decode_steps == 2
        avg_latency, avg_batch = sched._get_recent_decode_stats()
        assert avg_latency == approx(0.03)
        assert avg_batch == approx(6.0)
        # Measurement only: the scheduling path is completely unaffected.
        assert sched.current_max_batch_size == 8
        assert sched._compute_prefill_budget(3) == 8
        stats = sched.get_dynamic_batch_stats()
        assert stats["enable_batch_stats"] is True
        assert stats["enable_sla_constraint"] is False


class TestPrefillStarvationCap:
    """Tier 1: prefill starvation cap (default 0 = off, bit-identical to before)."""

    def test_off_by_default_keeps_prefill_priority(self):
        sched = sla_scheduler(batch_size_lower=1, batch_size_upper=64)
        assert sched.max_prefill_starvation_steps == 0
        make_decode_request(sched, request_id="d0")
        for step in range(5):
            make_waiting_request(sched, request_id=f"w{step}")
            out = sched.schedule()
            assert out is not None and out.is_prefill is True
        assert sched.num_forced_decode_steps == 0
        assert sched.prefill_starvation_steps == 5

    def test_forces_decode_after_k_prefill_steps(self):
        sched = sla_scheduler(
            batch_size_lower=1, batch_size_upper=64, max_prefill_starvation_steps=2
        )
        make_decode_request(sched, request_id="d0")
        for step in range(2):
            make_waiting_request(sched, request_id=f"w{step}")
            out = sched.schedule()
            assert out is not None and out.is_prefill is True
        assert sched.prefill_starvation_steps == 2
        # Step 3: waiting is still non-empty, but must yield to decode
        make_waiting_request(sched, request_id="w-late")
        out = sched.schedule()
        assert out is not None and out.is_prefill is False
        assert out.num_requests == 1
        assert sched.num_forced_decode_steps == 1
        assert sched.prefill_starvation_steps == 0

    def test_no_force_without_running_requests(self):
        sched = sla_scheduler(
            batch_size_lower=1, batch_size_upper=64, max_prefill_starvation_steps=1
        )
        for step in range(3):
            make_waiting_request(sched, request_id=f"w{step}")
            out = sched.schedule()
            assert out is not None and out.is_prefill is True
        assert sched.num_forced_decode_steps == 0

    def test_stats_expose_the_probe(self):
        sched = sla_scheduler(
            batch_size_lower=1, batch_size_upper=64, max_prefill_starvation_steps=3
        )
        make_waiting_request(sched, request_id="w0")
        sched.schedule()
        stats = sched.get_dynamic_batch_stats()
        assert stats["max_prefill_starvation_steps"] == 3
        assert stats["prefill_starvation_steps"] == 1
        assert stats["num_forced_decode_steps"] == 0

    def test_config_rejects_negative_value(self):
        with raises(ValueError, match="max_prefill_starvation_steps"):
            EngineConfig(
                model_path="/tmp/model",
                enable_sla_constraint=True,
                max_prefill_starvation_steps=-1,
            )


# ---------------------------------------------------------------------------
# Standalone runner (works without pytest)
# ---------------------------------------------------------------------------


def _run_all():
    passed, failures = 0, []
    for name in sorted(list(globals())):
        obj = globals()[name]
        if name.startswith("Test") and isinstance(obj, type):
            instance = obj()
            for attr in sorted(dir(instance)):
                if not attr.startswith("test_"):
                    continue
                label = f"{name}.{attr}"
                try:
                    getattr(instance, attr)()
                    passed += 1
                except Exception as exc:  # noqa: BLE001 - report and continue
                    failures.append((label, repr(exc)))
        elif name.startswith("test_") and callable(obj):
            try:
                obj()
                passed += 1
            except Exception as exc:  # noqa: BLE001
                failures.append((name, repr(exc)))

    print(f"\n{passed} passed, {len(failures)} failed")
    for label, err in failures:
        print(f"  FAIL {label}: {err}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Queue refill (--sla-queue-refill): do not let a burst throttle admission to a trickle
# ---------------------------------------------------------------------------


class TestSlaQueueRefill:
    """Measured basis (pois:128 with a burst of 384, engine-side sampling):

    Without this refill the lower bound is anchored to the *observed* batch,
    which the cap itself limits, so b_t hugs num_running and the per-step
    admission budget (= cap - num_running) stays ~0: on a burst the tail
    requests can only enter as earlier ones finish (384 requests took 11 s to
    drain vs 6 s for the control arm), TTFT tail 8.3 s vs 4.6 s.
    Off by default = previous behaviour, bit-identical.
    """

    def test_default_is_off(self):
        cfg = EngineConfig(model_path="/tmp/model")
        assert cfg.sla_queue_refill is False
        assert sla_scheduler().sla_queue_refill is False

    def test_config_accepts_the_flag(self):
        cfg = EngineConfig(
            model_path="/tmp/model", enable_sla_constraint=True, sla_queue_refill=True
        )
        assert cfg.sla_queue_refill is True

    def test_off_keeps_the_previous_behaviour(self):
        """Off: the target batch still hugs the running count (admission ~0)."""
        sched = sla_scheduler(
            batch_size_lower=4, batch_size_upper=64, sla_queue_refill=False
        )
        feed(sched, 0.05, 32, times=100)  # inside the band, observed batch 32
        assert sched._update_sla_bounds(32, 400) == 32

    def test_on_raises_the_bound_to_the_backlog(self):
        sched = sla_scheduler(
            batch_size_lower=4, batch_size_upper=64, sla_queue_refill=True
        )
        feed(sched, 0.05, 32, times=100)
        assert sched._update_sla_bounds(32, 400) == 64  # min(32+400, B_max=64)
        assert sched.b_low == 64
        assert sched.b_high >= sched.b_low  # invariant: low <= high

    def test_on_admits_the_whole_backlog_when_it_fits(self):
        sched = sla_scheduler(
            batch_size_lower=4, batch_size_upper=256, sla_queue_refill=True
        )
        feed(sched, 0.05, 20, times=100)
        assert sched._update_sla_bounds(20, 40) == 60  # 20 running + 40 waiting

    def test_off_leaves_the_cap_below_the_backlog(self):
        """Same load, off: cap stays at 20, so 0 new requests can be admitted."""
        sched = sla_scheduler(
            batch_size_lower=4, batch_size_upper=256, sla_queue_refill=False
        )
        feed(sched, 0.05, 20, times=100)
        assert sched._update_sla_bounds(20, 40) == 20

    def test_throttling_still_wins_when_above_the_target(self):
        """Above target the demand term must not fire: still throttle (the SLA point)."""
        on = sla_scheduler(
            batch_size_lower=4, batch_size_upper=64, sla_queue_refill=True
        )
        off = sla_scheduler(
            batch_size_lower=4, batch_size_upper=64, sla_queue_refill=False
        )
        feed(on, 0.2, 32, times=100)  # far above sla_high = 0.055
        feed(off, 0.2, 32, times=100)
        assert on._update_sla_bounds(32, 400) == off._update_sla_bounds(32, 400) == 32

    def test_integrated_schedule_opens_admission(self):
        """End to end: waiting non-empty + latency inside the band => cap rises to demand and admission budget > 0."""
        sched = sla_scheduler(
            batch_size_lower=4, batch_size_upper=64, sla_queue_refill=True
        )
        for i in range(8):
            make_decode_request(sched, request_id=f"d{i}")
        for i in range(20):
            make_waiting_request(sched, request_id=f"w{i}")
        feed(sched, 0.05, 8, times=100)
        sched.schedule()
        assert sched.current_max_batch_size == 28  # 8 running + 20 waiting
        assert sched.prefill_budget == 20

    def test_integrated_schedule_stays_old_without_the_flag(self):
        sched = sla_scheduler(
            batch_size_lower=4, batch_size_upper=64, sla_queue_refill=False
        )
        for i in range(8):
            make_decode_request(sched, request_id=f"d{i}")
        for i in range(20):
            make_waiting_request(sched, request_id=f"w{i}")
        feed(sched, 0.05, 8, times=100)
        sched.schedule()
        assert sched.current_max_batch_size == 8  # old: cap hugs running count
        assert sched.prefill_budget == 0

    def test_flag_is_inert_without_the_controller(self):
        """With SLA off the knob is inert: cap stays at max_batch_size (control-arm behaviour unchanged)."""
        sched = make_scheduler(
            max_batch_size=32,
            enable_sla_constraint=False,
            enable_memory_aware_batching=False,
            sla_queue_refill=True,
        )
        assert sched.current_max_batch_size == 32
        out = sched.schedule()
        assert out is None
        assert sched.current_max_batch_size == 32
        assert sched.prefill_budget == 32


# ---------------------------------------------------------------------------
# Memory-bound floor (--sla-mem-bound-floor): the cap must not be pushed
# below B_min
# ---------------------------------------------------------------------------


class TestSlaMemBoundFloor:
    """Measured basis: ``_compute_mem_bound`` estimates blocks_per_seq as
    "used blocks / running requests"; when few requests run while many blocks
    are held (prefix cache), the ratio is inflated and the bound collapses far
    below B_min -- measured cap sampled at 15 with B_min=64, so cap == running
    count and the per-step admission budget welds to 0 (self-sealing).
    Off by default = previous behaviour, bit-identical.
    """

    def test_default_is_off(self):
        assert EngineConfig(model_path="/tmp/model").sla_mem_bound_floor is False
        assert sla_scheduler().sla_mem_bound_floor is False

    def test_config_accepts_the_flag(self):
        cfg = EngineConfig(
            model_path="/tmp/model",
            enable_sla_constraint=True,
            enable_memory_aware_batching=True,
            sla_mem_bound_floor=True,
        )
        assert cfg.sla_mem_bound_floor is True

    def _collapsed_scheduler(self, floor):
        sched = sla_scheduler(
            batch_size_lower=4,
            batch_size_upper=64,
            enable_memory_aware_batching=True,
            sla_mem_bound_floor=floor,
        )
        sched.cache_manager = sched.cache_manager.__class__(
            num_blocks=8, block_size=256
        )
        make_decode_request(sched, request_id="d0")  # holds 1 block
        for _ in range(3):
            sched.cache_manager._allocate_block()  # prefix cache holds 3 more
        return sched

    def test_off_keeps_the_collapse_below_b_min(self):
        sched = self._collapsed_scheduler(False)
        # used=4 / running=1 -> blocks_per_seq=4 -> floor(8*0.95/4)=1  (< B_min=4)
        assert sched._compute_mem_bound(1) == 1

    def test_on_floors_it_at_b_min(self):
        sched = self._collapsed_scheduler(True)
        assert sched._compute_mem_bound(1) == 4

    def test_on_is_a_noop_when_the_bound_is_already_above_b_min(self):
        sched = self._collapsed_scheduler(True)
        plain = self._collapsed_scheduler(False)
        # used=4 / running=4 -> blocks_per_seq=1 -> floor(8*0.95/1)=7  (> B_min)
        assert sched._compute_mem_bound(4) == plain._compute_mem_bound(4) == 7

    def test_flag_is_inert_without_memory_awareness(self):
        sched = make_scheduler(
            max_batch_size=32,
            enable_sla_constraint=False,
            enable_memory_aware_batching=False,
            sla_mem_bound_floor=True,
        )
        assert sched.current_max_batch_size == 32
        sched.schedule()
        assert sched.current_max_batch_size == 32


# ---------------------------------------------------------------------------
# Accurate memory-bound caliber (--sla-mem-bound-accurate, default off)
#   Background: at fixed concurrency 256 the pool was measured full (0 free
#   blocks) with only 35 running requests; the pool-wide caliber computed
#   "111 blocks per request", collapsing the bound to 34 (the algorithm
#   wanted a batch of 384), so the per-step admission budget stayed ~0 and
#   no new request could enter. The accurate caliber estimates only from the
#   blocks each running request actually holds.
# ---------------------------------------------------------------------------


class TestSlaMemBoundAccurate:
    def _setup(self, **kw):
        sched = sla_scheduler(enable_memory_aware_batching=True, num_blocks=32, **kw)
        for i in range(4):
            make_decode_request(sched, prompt_len=8, request_id=f"d{i}")
        # Occupy 24 more blocks: simulates blocks reserved by queued requests
        # for their full output and blocks the prefix cache holds for others.
        for _ in range(24):
            sched.cache_manager._allocate_block()
        return sched

    def test_default_off_uses_pool_wide_ratio_and_collapses(self):
        sched = self._setup()
        assert sched.sla_mem_bound_accurate is False
        assert sched.cache_manager.get_num_free_blocks() == 4
        # Pool-wide caliber: 28 used / 4 running = 7 blocks per request -> bound 4
        assert sched._compute_mem_bound(4) == 4

    def test_on_uses_each_running_requests_own_blocks(self):
        sched = self._setup(sla_mem_bound_accurate=True)
        # Per running request: 1 block (allocated) + 1 block (remaining output
        # 64 tokens / 256 per block) = 2 blocks
        assert sched._running_request_blocks() == (8, 4)
        assert sched._compute_mem_bound(4) == 15  # floor(32 * 0.95 / 2)

    def test_on_ignores_blocks_held_for_other_requests(self):
        sched = self._setup(sla_mem_bound_accurate=True)
        before = sched._compute_mem_bound(4)
        for _ in range(2):  # hold 2 more for queued requests
            sched.cache_manager._allocate_block()
        assert sched._compute_mem_bound(4) == before

    def test_flag_is_inert_without_memory_aware_batching(self):
        plain, flagged = sla_scheduler(), sla_scheduler(sla_mem_bound_accurate=True)
        for sched in (plain, flagged):
            make_waiting_request(sched, request_id="w")
            feed(sched, 0.05, 8, times=3)
            sched.schedule()
        assert plain.current_max_batch_size == flagged.current_max_batch_size


class TestSlaAdmissionFallback:
    """Admission-budget fallback (--sla-admission-fallback, default off)."""

    def _setup(self, **kw):
        sched = sla_scheduler(enable_memory_aware_batching=True, **kw)
        for i in range(4):
            make_decode_request(sched, prompt_len=8, request_id=f"d{i}")
        sched.current_max_batch_size = (
            4  # simulate the memory bound pushing the cap down to the running count
        )
        return sched

    def test_default_off_keeps_the_budget_at_zero(self):
        sched = self._setup()
        assert sched.sla_admission_fallback is False
        assert sched._compute_prefill_budget(4) == 0

    def test_on_opens_when_the_pool_still_has_room(self):
        sched = self._setup(sla_admission_fallback=True)
        assert sched.cache_manager.get_num_free_blocks() > 0
        assert (
            sched._compute_prefill_budget(4) == 60
        )  # pool is wide: hits the algorithm bound b_upper_hard(64) - 4 running

    def test_on_stays_zero_when_the_pool_is_full(self):
        sched = self._setup(sla_admission_fallback=True)
        while sched.cache_manager.get_num_free_blocks() > 0:
            sched.cache_manager._allocate_block()
        assert (
            sched._compute_prefill_budget(4) == 0
        )  # the memory safety bound is untouched

    def test_on_leaves_a_healthy_budget_untouched(self):
        sched = self._setup(sla_admission_fallback=True)
        sched.current_max_batch_size = 60
        assert (
            sched._compute_prefill_budget(4) == 56
        )  # budget already healthy: untouched

    def test_flag_is_inert_without_the_controller(self):
        plain, flagged = make_scheduler(), make_scheduler(sla_admission_fallback=True)
        for sched in (plain, flagged):
            make_waiting_request(sched, request_id="w")
            sched.schedule()
        assert plain._compute_prefill_budget(0) == flagged._compute_prefill_budget(0)


# ---------------------------------------------------------------------------
# SLA channel 2: adaptive prefill chunking
# ---------------------------------------------------------------------------


class TestSlaPrefillCtrl:
    def test_config_defaults_keep_channel2_off(self):
        cfg = EngineConfig(model_path="/tmp/model")
        assert cfg.enable_sla_prefill_ctrl is False
        assert cfg.sla_prefill_budget_s == approx(4.0)
        assert cfg.sla_prefill_chunk_min == 1024

    def test_config_validation(self):
        base = dict(model_path="/tmp/model", enable_sla_prefill_ctrl=True)
        with raises(ValueError, match="sla_prefill_budget_s"):
            EngineConfig(sla_prefill_budget_s=0.0, **base)
        with raises(ValueError, match="sla_prefill_chunk_min"):
            EngineConfig(sla_prefill_chunk_min=0, **base)
        EngineConfig(**base)  # valid surface accepted

    def test_disabled_keeps_static_token_cap(self):
        sched = make_scheduler()
        assert sched.enable_sla_prefill_ctrl is False
        assert sched._prefill_token_cap() == sched.max_num_batched_tokens

    def test_record_disabled_is_a_noop(self):
        sched = make_scheduler()
        sched.record_prefill_step(1000, 1.0)
        assert len(sched.prefill_step_history) == 0
        assert sched.prefill_rate_est == 0.0

    def test_measurement_only_records_but_does_not_adapt(self):
        sched = make_scheduler(enable_batch_stats=True, max_num_batched_tokens=8192)
        sched.record_prefill_step(2000, 1.0)
        assert sched.prefill_rate_est == approx(2000.0)
        # measurement only: the chunk stays pinned at the static budget
        assert sched._prefill_token_cap() == 8192

    def _ctrl_scheduler(self, **kwargs):
        params = dict(
            enable_sla_prefill_ctrl=True,
            sla_prefill_budget_s=4.0,
            sla_prefill_chunk_min=1024,
            max_num_batched_tokens=8192,
        )
        params.update(kwargs)
        return make_scheduler(**params)

    def test_chunk_converges_to_budget_times_rate(self):
        sched = self._ctrl_scheduler()
        sched.record_prefill_step(2000, 1.0)  # rate 2000 tok/s
        assert sched._prefill_token_cap() == 8000

    def test_chunk_is_clamped_to_chunk_min(self):
        sched = self._ctrl_scheduler()
        sched.record_prefill_step(100, 1.0)  # rate 100 -> 400 < chunk_min
        assert sched._prefill_token_cap() == 1024

    def test_chunk_is_clamped_to_static_max(self):
        sched = self._ctrl_scheduler()
        sched.record_prefill_step(100000, 1.0)  # rate far above the cap
        assert sched._prefill_token_cap() == 8192

    def test_rate_uses_the_sliding_window(self):
        sched = self._ctrl_scheduler(sla_history_window=2)
        sched.record_prefill_step(1000, 1.0)
        sched.record_prefill_step(3000, 1.0)
        sched.record_prefill_step(2000, 2.0)  # evicts the first sample
        # window holds (3000, 1.0) + (2000, 2.0): rate = 5000 / 3
        assert sched.prefill_rate_est == approx(5000.0 / 3.0)

    def test_auto_sets_starvation_cap_to_four(self):
        sched = make_scheduler(enable_sla_prefill_ctrl=True)
        assert sched.max_prefill_starvation_steps == 4
        explicit = make_scheduler(
            enable_sla_prefill_ctrl=True, max_prefill_starvation_steps=8
        )
        assert explicit.max_prefill_starvation_steps == 8

    def test_stats_report_channel2_state(self):
        sched = self._ctrl_scheduler()
        sched.record_prefill_step(2000, 1.0)
        stats = sched.get_dynamic_batch_stats()
        assert stats["enable_sla_prefill_ctrl"] is True
        assert stats["num_prefill_steps_recorded"] == 1
        assert stats["prefill_rate_est"] == approx(2000.0)
        assert stats["current_prefill_chunk"] == 8000

    def test_schedule_uses_adaptive_token_cap(self):
        sched = make_scheduler(
            enable_sla_prefill_ctrl=True,
            sla_prefill_budget_s=1.0,
            sla_prefill_chunk_min=128,
            max_num_batched_tokens=100000,
        )
        sched.record_prefill_step(128, 1.0)  # rate 128 -> chunk = 128
        assert sched._prefill_token_cap() == 128
        make_waiting_request(sched, prompt_len=100, request_id="a")
        make_waiting_request(sched, prompt_len=100, request_id="b")
        out = sched.schedule()
        assert out is not None and out.is_prefill
        # 100 + 100 > 128: the second request is deferred to a later step
        assert out.num_requests == 1
        assert out.num_scheduled_tokens == 100

    def test_schedule_disabled_admits_both(self):
        sched = make_scheduler(max_num_batched_tokens=100000)
        make_waiting_request(sched, prompt_len=100, request_id="a")
        make_waiting_request(sched, prompt_len=100, request_id="b")
        out = sched.schedule()
        assert out is not None and out.num_requests == 2


if __name__ == "__main__":
    sys.exit(_run_all())
