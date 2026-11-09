"""Integrity-only loader reused for the audited vendor ring; no routing policy here."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

MANAGER_SHA = 'ec65247a78c01272bef9c4d51ebdc2bc0b6a1e1383a53d61e1ba5b5c0de56118'


def load_manager(path=None):
    path = Path(path or Path(__file__).parent / 'vendor/routing_manager.py').resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != MANAGER_SHA:
        raise ValueError('Baseline routing manager differs from the previously audited version: ' + str(path))
    spec = importlib.util.spec_from_file_location('_fourway_original_manager', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


