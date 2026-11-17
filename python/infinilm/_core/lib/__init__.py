# Compatibility facade: all native types are registered once by the LM module.
import os
import sys

# The linked MetaX profile needs Torch to own its Python teardown lifecycle.
if os.environ.get("INFINILM_PRELOAD_TORCH") == "1":
    import torch  # noqa: F401

from _infinilm import _core as _infinicore

sys.modules[__name__ + "._infinicore"] = _infinicore

__all__ = ["_infinicore"]
