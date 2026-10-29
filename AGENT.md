# AGENT.md — InfiniLM 项目指南

> 面向 AI Agent（以及新成员）的 InfiniLM 工作手册。描述项目概览、架构分层、构建依赖、模型接入、测试流程与编码约定。

---

## 1. 项目概览

**InfiniLM** 是基于 [InfiniCore](https://github.com/InfiniTensor/InfiniCore) 的通用 LLM 推理引擎。它提供：

- 一套 **C++17 推理框架**（`csrc/`），涵盖 Runtime、KV Cache、Attention、MoE、量化、图编译、分布式等模块，通过 pybind11 暴露为 `_infinilm` Python 扩展。
- 一套 **Python 上层引擎**（`python/infinilm/`），包含 Scheduler、ModelRunner、Processor、HTTP Server 与 LLM/AsyncLLMEngine 两个高级接口。
- 对多种硬件后端的支持：`cpu` / `nvidia` / `metax` / `moore` / `cambricon(mlu)` / `hygon` / `qy` / `ali` / `iluvatar`。
- 对多种 LLM 架构的支持：Llama / Mistral / Baichuan / ChatGLM / GPT-2 / Qwen2 / Qwen3 / Qwen3.5 / Qwen3-Next / DeepSeek / DeepSeek-V2 / Kimi-K2 / Kimi-K3 / InternLM3 / MiniCPM / FM9G / Mamba / VideoNSA 等（详见 `MODELS.md` / `MODELS_ZH.md`）。

**不做什么**：InfiniLM 不隐式构建 InfiniCore 的依赖栈（InfiniRT/InfiniOps/InfiniCCL），使用者需要先安装 InfiniCore 并通过 `INFINI_ROOT` 指向同一前缀。InfiniLM 只打包自身扩展 + InfiniCore 提供的 `.so` 运行时库。

---

## 2. 目录结构

```
InfiniLM/
├── csrc/                          # C++ 核心
│   ├── runtime/                   #   Tensor / Runtime / Device / Memory 抽象
│   ├── execution/                 #   Graph / Trace（调用 InfiniOps API）
│   ├── config/                    #   ModelConfig / QuantConfig / ConfigFactory
│   ├── cache/                     #   KV Cache / Mamba Cache
│   ├── layers/                    #   Attention / MLP / Norm / Linear / MoE / Quant / RoPE / Embedding
│   │   └── causal_lm_templates/   #   TextCausalLM / TextModel / TextDecoderLayer 模板
│   ├── models/                    #   各架构实现 + ModelFactory + ModelsRegistry
│   │   ├── <family>/              #     每个模型家族一个子目录
│   │   ├── infinilm_model.*       #     InfinilmModel 抽象基类
│   │   ├── model_factory.*        #     由 model_type 创建实例
│   │   └── models_registry.*      #     静态注册宏 INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL
│   ├── engine/                    #   InferEngine + RankWorker + Distributed + Compiler
│   │   ├── compiler/              #     Paged / Static Batching 图编译
│   │   └── distributed/           #     TP / PP / MoE-EP
│   ├── weights/                   #   ModelWeights 加载 / 遍历
│   ├── pybind11/                  #   Python 绑定
│   └── backends/                   #   AttentionBackend 枚举 (default/paged-attn/flash-attn)
├── python/infinilm/               # Python 包
│   ├── _core/                     #   Tensor / Device / Dtype 封装 (面向 _infinilm)
│   ├── _lib/                      #   InfiniCore .so 运行时 + infinilm_runtime.so
│   ├── llm/                       #   Scheduler / CacheManager / ModelRunner / Request
│   ├── cache/                     #   PagedKVCacheConfig 等
│   ├── config/                    #   EngineConfig / KVTransferConfig / DistConfig
│   ├── distributed/               #   PipelineTransport 等
│   ├── processors/                #   各模型家族的 tokenizer/chat template 封装
│   ├── server/                    #   FastAPI 推理服务
│   ├── kv_connector/              #   Mooncake 跨节点 KV 传输
│   ├── infer_engine.py            #   InferEngine（薄封装 C++ InferEngine，含 generate）
│   └── llm/llm.py                 #   LLM / AsyncLLMEngine 高层接口
├── examples/                      # test_infer.py / bench.py 等入口
├── scripts/                       # format.py / test_perf.py / test_ppl.py
├── test/
│   ├── native/                    # C++ 单元测试
│   ├── runtime/ / scripts/        # Python 单测
│   ├── bench/                     # C-Eval / MMLU 基准
│   └── service/                   # 服务请求示例
├── .github/workflows/             # CI 配置（格式、宿主检查、ruff、ci_test）
├── ci_config.yaml / ci_config_nvidia.yaml   # 多平台 CI 矩阵
├── xmake.lua                      # C++ 构建（xmake）
├── setup.py                       # Python 包构建（调 xmake + 打包 InfiniCore 运行时）
├── pyproject.toml                 # PEP 621 元数据 + ruff 规则
├── CONTRIBUTING.md                # 编码风格 / commit 规范 / PR 要求
└── MODELS_ZH.md / MODELS.md       # 新增模型参考
```

---

## 3. 构建与依赖

### 3.1 外部依赖链（必须先装好）

| 组件 | 用途 | 安装 |
|---|---|---|
| **InfiniCore** (子模块 `third_party/InfiniCore`) | 提供 InfiniRT / InfiniOps / InfiniCCL | 按子模块 README 原生构建入口，全部安装到同一前缀 |
| `INFINI_ROOT` | 指向上面那个前缀 | `export INFINI_ROOT=/abs/prefix` |
| `pybind11 3.0.4` | C++ → Python 绑定 | xmake 自动拉 |
| `spdlog` / `nlohmann/json` | 日志 / JSON | 本地子模块 |
| `CUDA_HOME` / `NEUWARE_HOME` / `ASCEND_HOME_PATH` / `MUSA_ROOT` | 各硬件 SDK | 按需设置 |

### 3.2 编译 InfiniLM

```bash
export INFINI_ROOT=/absolute/path/to/coherent-prefix
# 若跟 CoreX PyTorch 配套，需与 PyTorch/libstdc++ ABI 一致：
export INFINILM_CXX11_ABI=0

xmake f -y -m release --cxx11-abi=0 --capability-profile=release
pip install -e . --no-build-isolation
```

xmake 产生两个目标：
- `infinilm_runtime`（`libinfinilm_runtime.so`）— 纯 C++ Runtime，拷到 `python/infinilm/_lib/`。
- `_infinilm`（`_infinilm*.so`）— pybind11 扩展，拷到仓库根 `python/`。

`pip install` 通过 `setup.py` 的 `stage_runtime_dependencies` 把 InfiniCore 的 `infiniops`/`infiniccl`/`infinirt` 连同 InfiniLM 自己的 `.so` 一起复制到 `python/infinilm/_lib/`，并通过 `readelf` 做 ELF 类/机器校验和许可证拷贝。

### 3.3 开发期常用脚本

```bash
# 格式化（clang-format 按 .clang-format；ruff 按 pyproject.toml；末尾换行统一）
python scripts/format.py --ref origin/main --check   # 只检查
python scripts/format.py                             # 格式化暂存文件

# 单卡推理
python examples/test_infer.py --device nvidia --model=/models/TinyLlama-1.1B-Chat-v1.0

# 分布式 TP=4
python examples/test_infer.py --device nvidia --model=<path> --backend=cpp --tp=4

# PP=2（两个终端分别启动 stage 0/1）
python python/infinilm/server/inference_server.py --device=nvidia --model=<path> --tp=1 --pp=2 --node-rank=0 --master-addr=... --master-port=... --port=8000   # coordinator
python python/infinilm/server/inference_server.py --device=nvidia --model=<path> --tp=1 --pp=2 --node-rank=1 --master-addr=... --master-port=...               # worker

# 基准（C-Eval / MMLU）
python test/bench/test_benchmark.py --device nvidia /models/9g_8b_thinking --bench mmlu --backend cpp --max-new-tokens 5 --cache-dir ~/.cache/huggingface/datasets/

# 服务压测
python scripts/test_perf.py --verbose
```

---

## 4. 架构总览

### 4.1 分层

```
┌──────────────────────────────────────────────────────────┐
│          Python API (LLM / AsyncLLMEngine)               │   python/infinilm/llm/llm.py
├──────────────────────────────────────────────────────────┤
│   EngineConfig  ──►  LLMEngine  ──►  ModelRunner          │
│                            │            │                │   python/infinilm/config/, llm/, llm/model_runner/
│                     Scheduler + CacheManager              │   python/infinilm/llm/scheduler.py
│                                          │                │
│                            InferEngine(pybind11)           │   python/infinilm/infer_engine.py
├──────────────────────────────────────────────────────────┤
│   InferEngine (csrc/engine)  ──►  RankWorker[]            │   csrc/engine/infer_engine.*
│        │                                 │                │
│   CommunicationGroup          ModelFactory                 │   csrc/engine/distributed/, models/model_factory.*
│        │                          │                       │
│   DistConfig             ModelsRegistry (INFINILM_REGISTER)│   csrc/models/models_registry.*
│                                    │                      │
│                            InfinilmModel (abstract)        │   csrc/models/infinilm_model.hpp
│                            ├── Input  (token_ids, pos_ids,│
│                            │           block_tables, ...)  │
│                            └── Output (logits, hidden)     │
├──────────────────────────────────────────────────────────┤
│   layers/ (Attention / MLP / Norm / Linear / RoPE / MoE) │   csrc/layers/
│   cache/  (KVCache / MambaCache / PagedKVCache)           │   csrc/cache/
│   weights/ ModelWeights  weights/                         │   csrc/weights/
│   execution/ Graph / Trace  (直接调 InfiniOps)             │   csrc/execution/
│   runtime/  Tensor / Runtime / Device / Memory             │   csrc/runtime/
└──────────────────────────────────────────────────────────┘
                   ▲                        ▲
              InfiniRT                  InfiniOps/InfiniCCL
            (third_party/InfiniCore)
```

### 4.2 关键类型（C++）

- `infinilm::InfinilmModel`（`csrc/models/infinilm_model.hpp`）— 所有架构的抽象基类。要求实现 `forward(Input, ForwardContext)` 与可选的 `reset_cache(...)`。
- `infinilm::engine::InferEngine`（`csrc/engine/infer_engine.hpp`）— 持有 `vector<RankWorker>`、`RankBarrier`、`CommunicationGroup`；负责 load_params、forward、compile、reset_cache。它会从 `ModelsRegistry` 按 `model_type` 构造具体的 `InfinilmModel`。
- `infinilm::config::ModelConfig` — 由 HuggingFace `config.json` 反序列化得到，支持 `get<T>(key)` / `get_rotary_dim()` / `get_kv_cache_dtype()` 等。
- `infinilm::models::register_causal_lm_model / register_model_config` — 静态注册入口；通过 `INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(model_type, ModelT, create_model_config)` 在 translation unit 的匿名命名空间里触发。
- `ForwardContext` — 线程/请求级全局状态，持有当前 `kv_cache_vec` 等（`csrc/global_state/`）。

### 4.3 关键类型（Python）

- `infinilm.llm.LLM` / `AsyncLLMEngine` — 高层离线 / 流式接口。内部构造 `EngineConfig` → `LLMEngine` → `ModelRunner`。
- `infinilm.llm.LLMEngine`（`python/infinilm/llm/llm.py`）— 核心调度循环：`scheduler.schedule()` → `model_runner.execute_model(...)` → `_update_requests(...)`。Async 版本把循环放进独立 step 线程，通过 janus.Queue 向 asyncio 侧分发 token。
- `infinilm.infer_engine.InferEngine`（`python/infinilm/infer_engine.py`）— C++ `_infinilm.InferEngine` 的薄封装，负责 HF config 读入、`_PagedDecodeMetadataBuffers`、`generate()` 循环、多模态 `pixel_values` 等。
- `infinilm.llm.scheduler.Scheduler` — Paged Attention 下的 block 调度、prefix caching、speculative verification reservation。配合 `BlockManager` / `MambaCacheManager`。
- `infinilm.llm.model_runner.ModelRunner` — 把调度产出的 `SchedulerOutput` 翻译成传给 `InferEngine.forward` 的输入，处理 TP/PP、KV connector；内部可选择装配 `SpeculativeRunner` 接 Eagle/MiniCPM-Eagle draft 模型做 speculative decoding。
- **多模态**：`infinilm.multimodal.resolve_multimodal_inputs` 从 chat messages 的 `image_url` / `video_url` / `audio_url` 字段解析出媒体路径与占位 ID；各模型家族 `processors/<name>_processor.py` 负责 chat template + tokenizer + 图像/视频预处理，返回 HF-style `processed_inputs` 与 `mm_token_index_mappings`，在 `LLM.generate` / `AsyncLLMEngine.add_request` 里一路透传给 `InferEngine.forward` 的 `pixel_values` / `image_bound` / `tgt_sizes` / `image_grid_thw` / `visual_token_ranges`。

### 4.4 Attention Backend 与 KV Cache

- `AttentionBackend`：`Default` / `PagedAttn` / `FlashAttn`；选择逻辑在 `csrc/layers/attention/backends/`。
- `CacheConfig`：基类 `CacheConfig`（`csrc/cache/base_cache.hpp`），子类 `KVCacheConfig`（静态，`max_cache_len`）与 `PagedKVCacheConfig`（`num_blocks`、`block_size`、`has_mamba_cache` 等）。Python 侧通过 `cache_type="paged"/"static"` 决定 Scheduler 类型。

### 4.5 MoE 与分布式

- TP（Tensor Parallel）/ PP（Pipeline Parallel）在 `csrc/engine/distributed/`，由 `DistConfig`（python: `infinilm.distributed.DistConfig`）驱动；PP 采用 rendezvous 机制（`TcpRendezvous`），stage 0 为 coordinator，stage N-1 是最后一个 worker。
- MoE Expert Parallel（EP）在 `csrc/layers/moe/ep/`，支持 `AllgatherReduceScatter`、`DeepEP`、`LocalAllreduce` 等 dispatcher。

### 4.6 图编译

`csrc/engine/compiler/` 下两个编译路径：
- `paged_compiler` — 针对 Paged Attention + CUDA Graph；
- `static_batching_compiler` — 针对静态 batching。

Python 侧通过 `enable_graph=True` 打开；C++ `InferEngine::compile()` 在权重加载完后调用。

---

## 5. 新增模型（快速路径）

详细步骤见 `MODELS_ZH.md`。这里只给出最短清单：

1. 在 `csrc/models/<model_type>/` 新建目录，命名与 HF `config.json` 中的 `model_type` 完全一致。
2. 优先组合已有模板（`TextCausalLM` / `TextModel` / `TextDecoderLayer`）与公共层（`MLP` / `Attention` / `RMSNorm` / `RoPE` 等，见 `csrc/layers/common_modules.hpp`）。只有在通用实现不适配时才自定义 `Attention` / `DecoderLayer` / `KV 分配`。
3. 核心文件（按需要增删）：
   - `<name>_for_causal_lm.hpp/.cpp` — 顶层 ForCausalLM 组合；末尾匿名命名空间放注册宏。
   - `<name>_attention.hpp/.cpp` — 自定义 Attention。
   - `<name>_decoderLayer.hpp/.cpp` — 自定义 DecoderLayer。
   - `<name>_allocate_kv_cache_tensors.cpp` — 自定义 KV 分配（默认走 `InfinilmModel::default_allocate_kv_cache_tensors`）。
4. 必须提供 `create_<name>_model_config(std::shared_ptr<config::ModelConfig>)`，用于校验 `model_type` 并补全（如 `head_dim`、`layer_types`）。
5. 静态注册：
   ```cpp
   // <name>_for_causal_lm.cpp 末尾
   #include "../models_registry.hpp"
   namespace {
   INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
       <model_type>,
       infinilm::models::<model_type>::<ModelClassName>,
       infinilm::models::<model_type>::create_<model_type>_model_config);
   }
   ```
6. Python 侧：如有特殊 tokenizer / chat template / 多模态预处理，在 `python/infinilm/processors/` 新增 `<model>_processor.py`，并在 `python/infinilm/infer_engine.py::read_hf_config` 里必要的 model_type 分支补 config 归一化逻辑。
7. 尽量不修改 `csrc/layers/`、`models_registry.*`、`model_factory.*`；这些是框架级公共路径。

**参照模型选型**（摘自 `MODELS_ZH.md`）：
- 纯组合 → `fm9g`
- 自定义 Attention + 标准 MLP → `qwen3`
- 自定义 KV 分配 → `minicpm_sala`
- MoE + 多模态 → `qwen3_next` / `videonsa` / `kimi_k3`

---

## 6. 测试与 CI

### 6.1 本地

- **格式化（必跑）**：`python scripts/format.py --ref origin/main --check`。
- **宿主单测**：`python -m unittest discover -s test/scripts -v`（目前是 `test_base_config_device_detection.py`）。
- **C++ 单测**：通过 xmake 的 `test_rank_barrier` / `test_tensor_empty_view` / `test_post_load_traversal` / `test_parallel_topology` 目标（`xmake build test_rank_barrier && xmake run test_rank_barrier`）。
- **推理测试**：`examples/test_infer.py`。
- **基准**：`test/bench/test_benchmark.py --bench ceval|mmlu`；可选 `--cache-dir` 指向 HuggingFace 本地缓存避免网络。
- **服务**：`python python/infinilm/server/inference_server.py ...` + `python scripts/test_perf.py --verbose`，或单请求 `python test/service/request.py ...`。

### 6.2 GitHub Actions（`.github/workflows/`）

- **ci_test.yml** — `host-checks`（ubuntu-latest + unittest）、`check-format`（clang-format 21 + ruff 0.15 + black）、`dependency-identity`（workflow_dispatch）。
- **ruff.yml** — 独立 ruff check workflow。
- **ci_test_cleanup.yml** — 清理。
- **chatops.yml** — PR 状态评论。

### 6.3 多平台 CI（`ci_config.yaml` / `ci_config_nvidia.yaml`）

自有 CI 矩阵覆盖 nvidia / metax / moore / cambricon / hygon（ascend 被注释）。每个平台都跑 inference / bench / accuracy（C-Eval/MMLU）/ service 四类 job，`pip install .[dev] --no-build-isolation` 统一。

---

## 7. 编码约定

来自 `CONTRIBUTING.md`：

### 7.1 通用

- 改动最小化，注释不必越多越好；代码应自解释。
- 文件以换行结尾。
- 标识符用 Markdown 反引号；注释、error 消息、commit message 全部 **英文**。
- Error / warning 消息风格遵循 LLVM Coding Standards（完整句子、首字母大写、句号结尾）。

### 7.2 C++

- **Google C++ Style Guide**；格式化使用仓库提供的 `.clang-format`（`scripts/format.py --c clang-format`）。
- 初始化列表顺序必须与成员声明一致。
- 命名空间：`namespace infinilm::models::<model_type> { ... }`，避免跨模型名冲突。
- 新增模型不要改 `csrc/layers/`、`models_registry.*`、`model_factory.*` 等公共文件，除非框架能力确实不够。
- **严禁**引用/修改 `csrc/models/llama_legacy/`（老路径，计划删除）。

### 7.3 Python

- 主风格：**PEP 8**；细节补充参考 GDScript Style Guide（虽不同语言，但规则通用）。
- Docstring 遵循 PEP 257。
- 格式化：`ruff check --fix` + `ruff format`（`scripts/format.py --py ruff`）。
- ruff 规则集（`pyproject.toml`）：`select = ["E4", "E7", "E9", "F", "I"]`。
- Python ≥ 3.10，可用 `X | None`、`list[int]` 等内建泛型。

### 7.4 Commit / PR

- Conventional Commits 消息格式：`<type>(<scope>): <summary>`。
- 小 PR squash；大 PR 多 commit 但每笔应有意义。
- 标题应包含 issue 编号与 case 摘要。
- 合并前（或阶段性改动后）在主要平台构建 + 测试，结果贴 PR。
- 分支命名：`<type>/xxx-yyyy-zzzz`（与 commit type 对齐，连字符连接）。

---

## 8. 常见坑与注意事项

- **`INFINI_ROOT` 是硬要求**，缺少会在 `xmake f` / `setup.py` 里直接 raise。
- **libstdc++ ABI 必须一致**：PyTorch / InfiniCore / InfiniLM 三方对齐 `_GLIBCXX_USE_CXX11_ABI`；`INFINILM_CXX11_ABI` 环境变量可覆盖。
- **`_infinilm` 返回的 Tensor 是可变借用视图**，下一次 forward 会复用缓冲区；调用方必须在读或复制完再触发下一轮。
- **PP stage 0 才是 HTTP server 拥有者**；其他 stage 仅加入 rendezvous 并等待 coordinator。两个进程的模型/并行/cache 参数必须完全一致。
- **MoE + prefix caching 目前不兼容**，开了会在 `LLMEngine.__init__` raise。
- **Kimi-K3 的 RESET_CACHE** 会主动 drain 所有 worker-owned stream，Python 侧 `reset_cache` 前不手动 `sync_device()`（代码里有注释）。
- **`config.json` vs `generation_config.json`**：EOS / max_new_tokens 等生成参数优先从后者读，与 HF 保持一致（见 `InferEngine.eos_token_id`）。
- **`readelf -d` 自检**：`setup.py` 会校验 InfiniCore .so 的 ELF class/machine 与是否携带 `sampling.so` / `libtvm_ffi.so` 等辅助依赖；打包前务必 clean staging。
- **CI 上不要假设 clang-format 默认版本**，`check-format` workflow 锁死 `clang-format==21.1.8` + `ruff==0.15.20` + `black`。

---

## 9. 快速参考

| 要做什么 | 看哪里 |
|---|---|
| 跑起来一个模型 | `examples/test_infer.py` + `README.md` |
| 开推理服务 | `python/infinilm/server/inference_server.py` |
| 看底层 forward 路径 | `csrc/engine/infer_engine.cpp` → `csrc/engine/rank_worker.cpp` → `InfinilmModel::forward` |
| 新增模型 | `MODELS_ZH.md` + `csrc/models/<参考>/` |
| 加一种 Attention 后端 | `csrc/backends/attention_backends.hpp` + `csrc/layers/attention/backends/` |
| 调试 Python → C++ 参数 | `python/infinilm/infer_engine.py::_build_input` 和 `InferEngine.forward_raw` |
| 看 Scheduler 状态机 | `python/infinilm/llm/scheduler.py`（含 speculative reservation） |
| MoE EP 改动 | `csrc/layers/moe/ep/` + `EPConfig` |
| 图编译 debug | `csrc/engine/compiler/paged_compiler.*` + `execution/graph/` |
| KV connector（Mooncake） | `python/infinilm/kv_connector/` |
| CI 矩阵 | `.github/workflows/ci_test.yml`、`ci_config.yaml`、`ci_config_nvidia.yaml` |
| 打包出包失败 | `setup.py::stage_runtime_dependencies` + ELF 校验错误信息 |
| 编码风格争议 | `CONTRIBUTING.md` + `scripts/format.py` 跑一遍 |
