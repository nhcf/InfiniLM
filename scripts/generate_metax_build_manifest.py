#!/usr/bin/env python3
"""Record the native and model metadata for a local MetaX TP1 deployment."""

import argparse
import json
import subprocess
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = ROOT / "docs/bounded_hash/metax/sources.json"
SOURCE_SUFFIXES = {".py", ".cc", ".c", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".yaml"}
EXECUTION_FIELDS = {
    "attn_backend", "enable_graph", "pre_transpose", "max_batch_size",
    "ignore_eos", "weight_load_mode", "chunked_prefill_size", "cache_type",
    "enable_prefix_caching", "use_mla", "max_cache_len", "max_tokens",
    "max_num_batched_tokens",
}


def mapped_libraries(core, runtime, native_root):
    """Use the same library selection as the frozen build-identity generator."""
    paths = {core, runtime}
    bundled = ROOT / "python/infinilm/_lib"
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5]).resolve()
        if path.is_file() and (
            path.is_relative_to(native_root)
            or path.is_relative_to(bundled)
            or any(part in str(path) for part in (
                "mcoplib/", "flash_attn_2_cuda", "/torch/lib/", "/libpython3."
            ))
        ):
            paths.add(path)
    for library in ("libinfinirt", "libinfiniops", "libinfiniccl", "libinfinilm_runtime"):
        if not any(path.name.startswith(library) for path in paths):
            raise RuntimeError(f"Required loaded native dependency missing: {library}")
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--execution-json", required=True, type=Path)
    parser.add_argument("--native-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    execution = json.loads(args.execution_json.read_text())
    if set(execution) != EXECUTION_FIELDS:
        raise ValueError("Execution JSON must contain exactly the backend execution fields.")

    # Torch owns the linked providers' lifecycle and must initialize first.
    import torch

    core_module = import_module("_infinilm")
    identity = import_module("infinilm.server.bounded_hash_identity")
    core = Path(core_module.__file__).resolve()
    if core.parent != ROOT / "python" or not hasattr(core_module, "_core"):
        raise RuntimeError("Install the native extension from this checkout in editable mode.")
    if Path(identity.__file__).resolve() != ROOT / "python/infinilm/server/bounded_hash_identity.py":
        raise RuntimeError("The imported InfiniLM package comes from another checkout.")

    hardware = identity.metax_device_evidence()
    lock = json.loads(SOURCE_LOCK.read_text())
    model_dir = args.model.resolve()
    model = identity.model_metadata(model_dir)
    tokenizer = identity.tokenizer_metadata(model_dir)
    sources = {"xmake.lua": identity.file_sha256(ROOT / "xmake.lua")}
    for directory in ("python", "csrc"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                sources[str(path.relative_to(ROOT))] = identity.file_sha256(path)
    artifacts = [
        {"kind": "source", "path": str(ROOT / path), "sha256": digest}
        for path, digest in sources.items()
    ]
    lock_digest = identity.file_sha256(SOURCE_LOCK)
    artifacts.append({"kind": "source_lock", "path": str(SOURCE_LOCK), "sha256": lock_digest})
    for name in ("config.json", "model.safetensors.index.json", *tokenizer):
        path = model_dir / name
        artifacts.append({
            "kind": "model_metadata", "path": str(path), "sha256": identity.file_sha256(path),
        })

    native = {}
    for path in mapped_libraries(
        core, Path(hardware["runtime_path"]).resolve(), args.native_root.resolve()
    ):
        digest = identity.file_sha256(path)
        if path.name in native and native[path.name] != digest:
            raise RuntimeError(f"Two different native DSOs have the same basename: {path.name}")
        native[path.name] = digest
        artifacts.append({
            "kind": "native", "name": path.name, "path": str(path), "sha256": digest,
        })

    source = lock["infinilm"]
    ops = lock["infiniops"]
    expected = {
        "model": model,
        "tokenizer": tokenizer,
        "dtype": "bfloat16",
        "device": "metax",
        "tensor_parallel_size": 1,
        "execution_config": execution,
    }
    manifest = {
        "schema_version": 1,
        "identity": {
            **expected,
            "sources": {
                "infinilm_main_sha": source["upstream_base"],
                "pr616_sha": source["pr616_head"],
                "pr616_patch_sha256": source["pr616_original_patch_sha256"],
                "infinilm_integration_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "infiniops_pin_sha": lock["dependencies"][
                    "third_party/InfiniCore/submodules/InfiniOps"
                ]["public_commit"],
                "pr1033_sha": ops["pr1033_head"],
                "pr1033_patch_sha256": ops["pr1033_original_patch_sha256"],
                "infiniops_integration_sha": ops["apply_in_order"][0]["tested_commit"],
                "source_lock_sha256": lock_digest,
                "source_tree_sha256": identity.digest(sources),
            },
            "native_sha256": native,
            "runtime": {
                "torch_version": torch.__version__,
                "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                "mcruntime_sha256": hardware["runtime_sha256"],
                "torch_preload_requested": hardware["torch_preload_requested"],
                "packages": {name: version(name) for name in ("torch", "mcoplib", "flash-attn")},
            },
            "hardware": hardware["common"],
        },
        "artifacts": artifacts,
        "core_module_path": str(core),
        "builder_device_evidence": hardware,
        "header_only_weight_fingerprint": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=args.out.parent, delete=False) as stream:
        pending = Path(stream.name)
        stream.write(json.dumps(manifest, indent=2) + "\n")
    try:
        _, digest = identity.validate_build_manifest(pending, expected, hardware)
        pending.replace(args.out)
    finally:
        pending.unlink(missing_ok=True)
    print(f"Build identity SHA256 {digest}")


if __name__ == "__main__":
    main()
