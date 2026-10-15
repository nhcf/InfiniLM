"""iter_batching: iterative scheduling + selective batching for InfiniLM.

Non-invasive implementation: no existing InfiniLM file is modified; everything
takes effect through runtime injection. Disabled by default; importing this
package with the environment variable ``INFINILM_ITER_BATCHING=1`` activates it
automatically, or call ``enable()`` / ``disable()`` explicitly.

Usage:

    import infinilm.iter_batching as ib
    ib.enable()                    # Activate with env-var / default config.
    ib.enable(mixed_batch=False)   # Activate with programmatic overrides.
    ib.disable()                   # Restore the original path.

See the docstring of `config.py` for configuration.
"""

from .config import IBConfig, load_config
from .patch import disable, enable, get_config, is_active, mate_lib_active

__all__ = [
    "IBConfig",
    "load_config",
    "enable",
    "disable",
    "get_config",
    "is_active",
    "mate_lib_active",
]

# Master switch: importing this package with `INFINILM_ITER_BATCHING=1`
# activates it automatically.
if load_config().enabled:
    enable()
