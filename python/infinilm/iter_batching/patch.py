"""Injection layer of iter_batching: ``enable()`` / ``disable()`` wiring.

Injection points (all are runtime-resolved Python references; no original file
is modified):

1. ``infinilm.llm.llm.Scheduler``
   Replaced with a factory function: ``LLMEngine.__init__`` (llm.py:104)
   resolves the module global at runtime, and the factory returns
   ``ChunkedScheduler`` per config; mamba models get the original ``Scheduler``
   (fallback).
2. ``LLMEngine.step`` / ``LLMEngine._update_requests``
   Replaced with the per-request versions from `iter_batching.engine`
   (intermediate chunks drop the sampled token, commit the prefix cache at
   chunk boundaries, and requeue to waiting).
3. ``BasicLLMProcessor.build_model_inputs``
   Replaced with the unified span-based builder from `iter_batching.processor`.
4. ``ModelRunner.__init__`` (guard only; behavior unchanged)
   Post-construction check: unsupported processor family / speculative
   decoding / pipeline parallelism are handled per the ``fallback`` config —
   warn and fully restore (``disable()``), or refuse with an error.

``enable()`` is idempotent; ``disable()`` restores every patch (for in-process
A/B comparison).
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import replace

from .config import IBConfig, load_config

_state: dict = {"active": False, "config": None, "originals": {}}

# Processor families supported in the first version; others fall back in the
# `ModelRunner` guard.
_SUPPORTED_PROCESSORS = ("BasicLLMProcessor",)


def mate_lib_active() -> bool:
    """Detect whether the loaded InfiniCore is the MATE flash-prefill build.

    Mixed batching also routes decode tokens through the prefill operator path;
    the slow prefill kernel of the old library would make mixed batching a net
    loss, so the new library must be confirmed before enabling.
    """
    try:
        import infinicore  # noqa: F401  # Ensure the .so is loaded in this process.
    except ImportError:
        return False
    try:
        with open("/proc/self/maps") as f:
            maps = f.read()
    except OSError:
        return False
    for line in maps.splitlines():
        if "libinfiniop.so" in line:
            return ".infini-mate" in line
    return False


def get_config() -> IBConfig | None:
    """The currently effective config; None when not enabled."""
    return _state["config"]


def is_active() -> bool:
    return _state["active"]


def enable(**overrides) -> IBConfig:
    """Activate iter_batching (idempotent). ``overrides`` may set any `IBConfig` field."""
    if _state["active"]:
        return _state["config"]
    cfg = load_config()
    if overrides:
        cfg = replace(cfg, **overrides).validated()
    cfg = replace(cfg, enabled=True).validated()

    if not mate_lib_active():
        msg = (
            "iter_batching: MATE flash-prefill library not loaded "
            "(LD_LIBRARY_PATH does not point to .infini-mate/lib). "
            "Mixed batching is a net loss with the old library."
        )
        if cfg.fallback == "error":
            raise RuntimeError(msg)
        warnings.warn(msg + " Enabling anyway per fallback=warn.")

    _install(cfg)
    _state.update(active=True, config=cfg)
    print(cfg.describe(), flush=True)
    return cfg


def disable() -> None:
    """Restore all patches and return to the original scheduling path."""
    if not _state["active"]:
        return
    _uninstall()
    _state.update(active=False, config=None)


def _install(cfg: IBConfig) -> None:
    import infinilm.llm.llm as llm_mod
    import infinilm.llm.scheduler as sched_mod
    from infinilm.llm.model_runner import model_runner as runner_mod
    from infinilm.processors.basic_llm_processor import BasicLLMProcessor

    from . import engine as ib_engine
    from .processor import build_model_inputs
    from .scheduler import ChunkedScheduler

    orig = _state["originals"]
    orig["Scheduler"] = llm_mod.Scheduler
    orig["LLMEngine.step"] = llm_mod.LLMEngine.step
    orig["LLMEngine._update_requests"] = llm_mod.LLMEngine._update_requests
    orig["ModelRunner.__init__"] = runner_mod.ModelRunner.__init__
    orig["BasicLLMProcessor.build_model_inputs"] = BasicLLMProcessor.build_model_inputs

    OriginalScheduler = sched_mod.Scheduler

    def scheduler_factory(*args, **kwargs):
        if kwargs.get("has_mamba_cache"):
            msg = (
                "iter_batching: mamba models do not support chunked/mixed "
                "scheduling, falling back to the original scheduler."
            )
            if cfg.fallback == "error":
                raise RuntimeError(msg)
            warnings.warn(msg)
            return OriginalScheduler(*args, **kwargs)
        return ChunkedScheduler(*args, _ib_config=cfg, **kwargs)

    @functools.wraps(orig["ModelRunner.__init__"])
    def guarded_runner_init(self, *args, **kwargs):
        orig["ModelRunner.__init__"](self, *args, **kwargs)
        reasons = []
        if type(self.processor).__name__ not in _SUPPORTED_PROCESSORS:
            reasons.append(f"processor {type(self.processor).__name__} is not adapted")
        if getattr(self, "speculative_runner", None) is not None:
            reasons.append("speculative decoding is incompatible with mixed batching")
        if getattr(self, "pipeline_control", None) is not None:
            reasons.append("pipeline parallelism (PP>1) is not adapted")
        if not reasons:
            return
        msg = "iter_batching: " + "; ".join(reasons) + "."
        if cfg.fallback == "error":
            raise RuntimeError(msg)
        warnings.warn(msg + " This process falls back to the original scheduling path.")
        disable()

    llm_mod.Scheduler = scheduler_factory
    llm_mod.LLMEngine.step = ib_engine.step
    llm_mod.LLMEngine._update_requests = ib_engine._update_requests
    runner_mod.ModelRunner.__init__ = guarded_runner_init
    BasicLLMProcessor.build_model_inputs = build_model_inputs


def _uninstall() -> None:
    import infinilm.llm.llm as llm_mod
    from infinilm.llm.model_runner import model_runner as runner_mod
    from infinilm.processors.basic_llm_processor import BasicLLMProcessor

    orig = _state["originals"]
    llm_mod.Scheduler = orig.pop("Scheduler")
    llm_mod.LLMEngine.step = orig.pop("LLMEngine.step")
    llm_mod.LLMEngine._update_requests = orig.pop("LLMEngine._update_requests")
    runner_mod.ModelRunner.__init__ = orig.pop("ModelRunner.__init__")
    BasicLLMProcessor.build_model_inputs = orig.pop(
        "BasicLLMProcessor.build_model_inputs"
    )
