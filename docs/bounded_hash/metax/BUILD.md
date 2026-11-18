# Build the pinned MetaX stack

## Recover the source

Run the commands in order in Bash and stop if any command fails.

```bash
git clone --branch bound-hash-w64 --single-branch https://github.com/nhcf/InfiniLM.git
cd InfiniLM
git submodule update --init --recursive

ops=third_party/InfiniCore/submodules/InfiniOps
test "$(git -C "$ops" rev-parse HEAD)" = a4b11fb3730d0d39a93072abfe17925671b9d402
for patch_file in docs/bounded_hash/metax/patches/000*.patch; do
    git -C "$ops" apply --check --index --whitespace=error-all "$PWD/$patch_file"
    git -C "$ops" apply --index --whitespace=error-all "$PWD/$patch_file"
done
test "$(git -C "$ops" write-tree)" = f1617e0850ef0319702def4807bf17a23ebf3def
```

The ordered [source manifest](sources.json) records the public commits and
tested trees for InfiniCore, InfiniRT, InfiniOps, InfiniCCL, InfiniOps' CI
submodule, nlohmann/json and spdlog. InfiniCore's gitlink preserves the
public InfiniOps pin. The three patches reproduce the tested Ops tree
while keeping that pin available to new checkouts. Apply them after
initializing submodules and build from the patched working tree.

| Order | Patch and author | Source |
| --- | --- | --- |
| 1 | [MetaX operators](patches/0001-metax-operators-pr1033.patch), rubik | [InfiniOps PR 1033](https://github.com/InfiniTensor/InfiniOps/pull/1033/commits/a6213df839ca8d96509d9b4edaa8872273c76260), cherry-picked on the public pin in the run |
| 2 | [Linked Torch host adapters](patches/0002-metax-linked-torch-host-adapters.patch), Codex | Common experimental fix separating host implementations from MetaX device dispatch |
| 3 | [Separate-table RoPE and embedded Python](patches/0003-metax-separate-rope-and-python-embed.patch), Codex | Common experimental fix for the device RoPE API and the linked Torch/Python lifecycle |

The patches retain the original author headers and apply under
[InfiniOps' Apache-2.0 license](patches/LICENSE.Apache-2.0). The upstream checkout
provides the component notices under `LICENSES/`. The resulting tree hash
above matches the historical local Ops commit `423ba98eb779545b6d2086bf0c3cfc8f85654b41`.
The three local Ops commit IDs in the manifest identify the historical stages.
The patch files and tree hashes provide the recoverable source.

## Build and install

Use Linux with the MetaX MACA SDK, MCCL, Python 3.12, CMake, Ninja, GCC, xmake,
and the [InfiniLM Python dependencies](../../../pyproject.toml), including the
`bounded-hash` extra. Use an environment with the vendor's ABI-1 MetaX Torch,
mcoplib and FlashAttention packages. The tested SDK, driver, vendor package
versions and wheel digests are in [sources.json](sources.json). The vendor
distributions are separate prerequisites. The CMake linked provider resolver
used FlashAttention and both mcoplib and mcoplib_moe from those packages.
The original build also used `libclang==18.1.1` and xmake's pinned
`pybind11 3.0.4`.

Set `MACA_PATH` to the SDK installation. Put the SDK tools and the intended
Python environment on `PATH` before configuring. Use a fresh writable build
directory and install all three native libraries into the same prefix.

```bash
export INFINI_ROOT="$PWD/.metax-install"
export INFINI_RT_ROOT="$INFINI_ROOT"
export INFINILM_CXX11_ABI=1
export INFINILM_PRELOAD_TORCH=1
export CMAKE_PREFIX_PATH="$INFINI_ROOT"
export PATH="$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$PATH"
python_bin=$(command -v python)
torch_lib=$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$torch_lib:$MACA_PATH/lib:$MACA_PATH/lib64:$MACA_PATH/mxgpu_llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

native_src="$PWD/third_party/InfiniCore/submodules"
native_build="$PWD/.metax-build"
common_flags=(
    -G Ninja -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_INSTALL_PREFIX=$INFINI_ROOT"
    "-DPython_EXECUTABLE=$python_bin" "-DPython3_EXECUTABLE=$python_bin"
    -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1
    -DWITH_METAX=ON -DAUTO_DETECT_DEVICES=OFF -DAUTO_DETECT_BACKENDS=OFF
    -DWITH_NVIDIA=OFF -DWITH_ILUVATAR=OFF -DWITH_HYGON=OFF -DWITH_THEAD=OFF
    -DWITH_MOORE=OFF -DWITH_CAMBRICON=OFF -DWITH_ASCEND=OFF
)
build_component() {
    local component=$1
    shift
    cmake -S "$native_src/$component" -B "$native_build/$component" "${common_flags[@]}" "$@" &&
        cmake --build "$native_build/$component" --parallel &&
        cmake --install "$native_build/$component"
}
build_component InfiniRT -DWITH_CPU=ON -DWITH_MARS=OFF -DINFINI_RT_BUILD_TESTING=OFF
build_component InfiniOps -DWITH_CPU=ON -DWITH_MARS=OFF -DWITH_TORCH=ON -DWITH_LINKED=ON \
    -DWITH_NINETOOTHED=OFF -DWITH_TRITON=OFF -DGENERATE_PYTHON_BINDINGS=OFF \
    -DINFINI_OPS_SMOKE_BUILD=OFF -DINFINI_OPS_TORCH_COMPILE_JOBS=8 \
    "-DINFINI_RT_ROOT=$INFINI_ROOT"
build_component InfiniCCL -DWITH_MCCL=ON -DWITH_NCCL=OFF -DWITH_OMPI=OFF \
    -DWITH_MPICH=OFF -DWITH_CNCL=OFF -DBUILD_EXAMPLES=OFF

xmake f -y -m release --cxx11-abi=1 --builddir="$native_build/InfiniLM"
xmake build -y _infinilm
python -m pip install -e '.[bounded-hash]' --no-build-isolation --no-deps
```

The editable install command assumes the Python dependencies are already
provisioned. The full InfiniOps build includes the CPU provider for public
operator API coverage. Replica runtime commands explicitly select MetaX.
Keep the linked-provider DSO and symbol checks enabled and initialize Torch
before direct imports of the native extension.

## Generate a local build identity

Choose the deployment's `MODEL_DIR`, `DEPLOY_DIR`, and one assigned `GPU_INDEX`.
Use the [example execution settings](execution.example.json) with the
[corresponding launch flags](../README.md#launch), or supply the execution
settings for your deployment. Changing the execution settings also changes
the contract identity.

```bash
export MACA_VISIBLE_DEVICES="$GPU_INDEX"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export INFINILM_BOUNDED_HASH_BUILD_MANIFEST="$DEPLOY_DIR/build_identity.json"
python scripts/generate_metax_build_manifest.py --model "$MODEL_DIR" \
    --execution-json docs/bounded_hash/metax/execution.example.json \
    --native-root "$INFINI_ROOT" --out "$INFINILM_BOUNDED_HASH_BUILD_MANIFEST"
```

The generator records the loaded libraries, the checkout, and model and
tokenizer metadata. Safetensors fingerprinting covers headers and sizes.
Keep generated deployment manifests local. They contain native and model
paths and per-device evidence. Use the same verified build and configuration
across replicas, then configure the gateway's [compatibility input](../README.md#launch).
