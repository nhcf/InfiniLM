"""Configuration layer of iter_batching: the single source of all switches.

Everything is controlled through environment variables (no code changes needed
on the script side), and can also be overridden programmatically via
``infinilm.iter_batching.enable(**overrides)``.

| Environment variable | Default | Meaning |
|---|---|---|
| INFINILM_ITER_BATCHING | 0 | Master switch: importing the package with 1 activates it |
| INFINILM_IB_CHUNKED_PREFILL | 1 | Chunked prefill (per-request input slicing) |
| INFINILM_IB_MIXED_BATCH | 1 | Prefill+decode mixed batching (with chunked_prefill=0, whole prefill joins mixing) |
| INFINILM_IB_MAX_BATCHED_TOKENS | 8192 | Per-step token budget (chunk size cap) |
| INFINILM_IB_MIXED_STEP_TOKENS | 0 | Prefill token cap within a mixed step; 0 = unlimited (best throughput / TTFT tail in measurements) |
| INFINILM_IB_PREFILL_PRIORITY | decode-first | Mixed-batch budget policy: decode-first / prefill-first |
| INFINILM_IB_FALLBACK | warn | Handling of unsupported configs: warn (fall back to original scheduling) / error |
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_PREFILL_PRIORITIES = ("decode-first", "prefill-first")
_FALLBACK_MODES = ("warn", "error")


@dataclass(frozen=True)
class IBConfig:
    """Effective configuration of iter_batching (immutable)."""

    enabled: bool = False
    chunked_prefill: bool = True
    mixed_batch: bool = True
    max_batched_tokens: int = 8192
    mixed_step_tokens: int = 0
    prefill_priority: str = "decode-first"
    fallback: str = "warn"

    def validated(self) -> "IBConfig":
        """Validate values, resolve switch dependencies, and return the fixed config."""
        cfg = self
        if cfg.prefill_priority not in _PREFILL_PRIORITIES:
            raise ValueError(
                f"INFINILM_IB_PREFILL_PRIORITY must be one of {_PREFILL_PRIORITIES}, "
                f"got {cfg.prefill_priority!r}"
            )
        if cfg.fallback not in _FALLBACK_MODES:
            raise ValueError(
                f"INFINILM_IB_FALLBACK must be one of {_FALLBACK_MODES}, "
                f"got {cfg.fallback!r}"
            )
        if cfg.max_batched_tokens < 256:
            raise ValueError(
                f"INFINILM_IB_MAX_BATCHED_TOKENS must be >= 256, "
                f"got {cfg.max_batched_tokens}"
            )
        if cfg.mixed_step_tokens < 0:
            raise ValueError(
                f"INFINILM_IB_MIXED_STEP_TOKENS must be >= 0 (0 = unlimited), "
                f"got {cfg.mixed_step_tokens}"
            )
        if 0 < cfg.mixed_step_tokens < 64:
            raise ValueError(
                f"INFINILM_IB_MIXED_STEP_TOKENS must be 0 or >= 64, "
                f"got {cfg.mixed_step_tokens}"
            )
        if cfg.mixed_batch and not cfg.chunked_prefill:
            import warnings

            warnings.warn(
                "iter_batching: MIXED_BATCH=1 with CHUNKED_PREFILL=0 schedules "
                "whole prefill segments inside mixed steps (long prompts inflate "
                "step time and ITL), and an over-budget whole prompt is deferred "
                "while decodes occupy the batch. Intended for A/B experiments only."
            )
        return cfg

    def describe(self) -> str:
        return (
            f"iter_batching config: enabled={self.enabled} "
            f"chunked_prefill={self.chunked_prefill} "
            f"mixed_batch={self.mixed_batch} "
            f"max_batched_tokens={self.max_batched_tokens} "
            f"mixed_step_tokens={self.mixed_step_tokens} "
            f"prefill_priority={self.prefill_priority!r} "
            f"fallback={self.fallback!r}"
        )


def _flag(env: dict[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def load_config(env: dict[str, str] | None = None) -> IBConfig:
    """Load configuration from environment variables (`os.environ` by default)."""
    env = os.environ if env is None else env
    cfg = IBConfig(
        enabled=_flag(env, "INFINILM_ITER_BATCHING", False),
        chunked_prefill=_flag(env, "INFINILM_IB_CHUNKED_PREFILL", True),
        mixed_batch=_flag(env, "INFINILM_IB_MIXED_BATCH", True),
        max_batched_tokens=int(env.get("INFINILM_IB_MAX_BATCHED_TOKENS", "8192")),
        mixed_step_tokens=int(env.get("INFINILM_IB_MIXED_STEP_TOKENS", "0")),
        prefill_priority=env.get("INFINILM_IB_PREFILL_PRIORITY", "decode-first"),
        fallback=env.get("INFINILM_IB_FALLBACK", "warn"),
    )
    return cfg.validated()
