"""Verified deployment evidence for bounded-hash calibration compatibility.

This module has no native imports. GPU evidence is gathered only when an
opted-in backend starts; CPU tests exercise the file and compatibility rules.
"""

import ctypes
import hashlib
import importlib
import json
import os
import struct
import uuid
from pathlib import Path


MANIFEST_ENV = "INFINILM_BOUNDED_HASH_BUILD_MANIFEST"
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json",
    "merges.txt", "chat_template.jinja",
)


def digest(value):
    """Hash canonical JSON without nonfinite values."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()


def file_sha256(path):
    """Hash files in bounded memory, including large native libraries."""
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def model_metadata(model_dir):
    """Fingerprint model headers without reading tensor payloads."""
    directory = Path(model_dir)
    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards = {}
    for name in sorted(set(index["weight_map"].values())):
        path = directory / name
        with path.open("rb") as source:
            size = struct.unpack("<Q", source.read(8))[0]
            header_bytes = source.read(size)
        header = json.loads(header_bytes)
        tensor_bytes = sum(
            value["data_offsets"][1] - value["data_offsets"][0]
            for key, value in header.items() if key != "__metadata__"
        )
        shards[name] = {
            "file_size": path.stat().st_size,
            "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
            "tensor_bytes": tensor_bytes,
        }
    return {
        "config_sha256": file_sha256(directory / "config.json"),
        "model_index_sha256": file_sha256(index_path),
        "shards": shards,
    }


def tokenizer_metadata(model_dir):
    """Fingerprint every present tokenizer and chat-template artifact."""
    directory = Path(model_dir)
    result = {
        name: file_sha256(directory / name)
        for name in TOKENIZER_FILES if (directory / name).is_file()
    }
    if not result:
        raise ValueError("Tokenizer artifacts are required for routing identity.")
    return result


def metax_device_evidence():
    """Query the assigned runtime device without trusting a hardware label."""
    import torch

    maca = Path(os.environ.get("MACA_PATH", "/opt/maca")).resolve()
    runtime_path = next((path for path in (
        maca / "lib/libmcruntime.so", maca / "lib64/libmcruntime.so"
    ) if path.is_file()), None)
    if runtime_path is None:
        raise ValueError("The MetaX runtime library is missing.")
    runtime = ctypes.CDLL(str(runtime_path))

    def call(name, *arguments):
        status = getattr(runtime, name)(*arguments)
        if status:
            raise ValueError(f"MetaX identity query `{name}` failed: {status}.")

    count = ctypes.c_int()
    call("mcGetDeviceCount", ctypes.byref(count))
    if count.value != 1 or torch.cuda.device_count() != 1:
        raise ValueError("A TP1 replica must see exactly one physical GPU.")
    bus = ctypes.create_string_buffer(64)
    device_uuid = (ctypes.c_ubyte * 16)()
    total, free = ctypes.c_size_t(), ctypes.c_size_t()
    call("mcDeviceGetPCIBusId", bus, len(bus), 0)
    call("mcDeviceGetUuid", device_uuid, 0)
    call("mcMemGetInfo", ctypes.byref(free), ctypes.byref(total))
    properties = torch.cuda.get_device_properties(0)
    return {
        "common": {
            "name": properties.name,
            "total_memory_bytes": total.value,
            "torch_total_memory_bytes": properties.total_memory,
        },
        "device_uuid": "GPU-" + str(uuid.UUID(bytes=bytes(device_uuid))),
        "pci_bus": bus.value.decode(),
        "logical_device": 0,
        "runtime_path": str(runtime_path.resolve()),
        "runtime_sha256": file_sha256(runtime_path),
    }


def validate_build_manifest(filename, expected, hardware):
    """Reject stale files, unloaded native DSOs, or incompatible deployments.

    `identity` contains only common compatibility data. Absolute paths and
    each replica's UUID are kept outside it, so eight identical builds share
    one identity even though their physical devices differ.
    """
    if not filename:
        raise ValueError(f"Set `{MANIFEST_ENV}` to a verified build manifest.")
    manifest = json.loads(Path(filename).read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported bounded-hash build manifest schema.")
    identity = manifest.get("identity")
    required = {
        "sources", "native_sha256", "model", "tokenizer", "runtime",
        "hardware", "execution_config", "dtype", "device", "tensor_parallel_size",
    }
    if not isinstance(identity, dict) or not required <= identity.keys():
        raise ValueError("The build manifest has incomplete common identity.")
    if any(key in identity["hardware"] for key in ("uuid", "device_uuid", "pci_bus")):
        raise ValueError("Per-device identifiers cannot enter common compatibility.")
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f"The loaded backend differs from build identity `{key}`.")
    if identity["hardware"] != hardware["common"]:
        raise ValueError("The assigned GPU differs from verified build hardware.")
    sources = identity["sources"]
    for field in ("infinilm_main_sha", "pr616_sha", "pr616_patch_sha256",
                  "infiniops_pin_sha", "pr1033_sha", "infiniops_integration_sha",
                  "pr1033_patch_sha256", "source_lock_sha256", "source_tree_sha256"):
        value = sources.get(field)
        if not isinstance(value, str) or len(value) not in (40, 64):
            raise ValueError(f"The build manifest omitted source evidence `{field}`.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("The build manifest requires verified artifacts.")
    maps = Path("/proc/self/maps").read_text()
    kinds = set()
    native = {}
    for artifact in artifacts:
        path = Path(artifact["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"A build artifact is missing: {path}.")
        if file_sha256(path) != artifact.get("sha256"):
            raise ValueError(f"A build artifact changed: {path}.")
        kind = artifact["kind"]
        kinds.add(kind)
        if kind == "native":
            resolved = str(path.resolve())
            if resolved not in maps:
                raise ValueError(f"A required native artifact is not loaded: {path}.")
            native[artifact["name"]] = artifact["sha256"]
    if not {"native", "source", "model_metadata", "source_lock"} <= kinds:
        raise ValueError("The build manifest omitted required artifact classes.")
    if native != identity["native_sha256"]:
        raise ValueError("Native identity and verified artifact hashes disagree.")
    module_path = Path(manifest["core_module_path"]).resolve()
    module = importlib.import_module("_infinilm")
    if getattr(module, "_core", None) is None:
        raise ValueError("The loaded extension does not expose `_infinilm._core`.")
    if Path(module.__file__).resolve() != module_path:
        raise ValueError("The loaded `_infinilm._core` is not the frozen build.")
    if str(module_path) not in maps or file_sha256(module_path) not in native.values():
        raise ValueError("The core extension is absent from native build evidence.")
    if hardware["runtime_sha256"] != identity["runtime"].get("mcruntime_sha256"):
        raise ValueError("The loaded MetaX runtime differs from build evidence.")
    return identity, digest(identity)
