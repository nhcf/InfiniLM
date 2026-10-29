# InfiniLM 总体架构（精简版）

InfiniLM 参照 SGLang 的 **Server → Scheduler → Worker** 三角色心智模型，但 **CPU 侧不做进程隔离** —— HTTP 入口、Scheduler、ModelRunner、Python InferEngine 全在同一个 Python 进程里，靠直接函数调用 + `janus.Queue` 做 sync/async 桥接；真正多进程的只有 Pipeline Parallel 的远端节点（原生 TCP + InfiniCCL），以及 TP 下 C++ 内部的 `RankWorker[]`（同进程内）。

---

## 模块划分

| # | 模块 | 职责 | 代码目录 |
|---|---|---|---|
| ① | **Server** | 对外 HTTP/CLI 入口：解析请求 → tokenize → chat template → 多模态预处理 → 构造 `InferenceRequest` → 入队 | `python/infinilm/server/`、`python/infinilm/processors/`、`python/infinilm/multimodal/`、离线入口 `examples/test_infer.py` / `LLM.generate()` |
| ② | **Scheduler** | 请求生命周期：维护 `waiting`/`running`/`WAITING_FOR_REMOTE_KVS` 队列；`schedule()` 单函数内按 waiting 优先的顺序做 prefill → decode 两阶段；KV block 分配（`BlockManager`）、prefix cache hash、speculative verification reservation | `python/infinilm/llm/scheduler.py`、`request.py`、`cache_manager.py`、`prefix_cache.py`、`nsa_paged_owner.py`、`llm.py::LLMEngine.step()` |
| ③ | **ModelRunner** | batch 组装与执行桥接：把 `SchedulerOutput` 翻译成传给 InferEngine 的 model_input dict（block_tables / slot_mapping / position_ids / pixel_values / sampling params）；可选 speculative decoding（`SpeculativeRunner`）、PP 控制（`PipelineControlServer`）、KV connector 同步点 | `python/infinilm/llm/model_runner/`、`infer_engine.py`（pybind11 薄封装 + HF config 读入 + `generate()` 循环） |
| ④ | **C++ Engine** | `InferEngine` 持有 `RankWorker[0..TP-1]` + `RankBarrier` + `InfiniCCL Communicator`，分发 Input 到每个 TP rank 并聚合采样结果；图编译（paged / static batching） | `csrc/engine/`（infer_engine、rank_worker、compiler、distributed）、`csrc/runtime/`、`csrc/execution/`、`csrc/pybind11/` |
| ⑤ | **C++ Model + Layers** | 各架构的真正 forward 实现（继承 `InfinilmModel`），Attention/MLP/Norm/Linear/MoE/Quant/RoPE/Embedding 等算子层 | `csrc/models/`（每家族一子目录）、`csrc/layers/`、`csrc/weights/`、`csrc/cache/`（KV tensor 数据面） |
| ⑥ | **KV Connector**（可选） | 跨节点 KV 迁移（Mooncake）：Scheduler 侧查远端命中、ModelRunner 侧 forward 前后触发 recv/send | `python/infinilm/kv_connector/` |

---

## 模块交互关系

```
      ┌─────────┐  InferenceRequest   ┌────────────┐  SchedulerOutput   ┌──────────────┐
 HTTP │ Server  │ ─────────────────► │ Scheduler  │ ─────────────────► │ ModelRunner  │
      │         │  janus.Queue       │            │  (block_tables,   │              │
      └─────────┘                    │            │   slot_mapping,    │              │
                                     │            │   position_ids…)  │              │
                                     │            │                   │              │
                                     │            │◄──────────────────│              │
                                     │ BlockManager│  ModelRunnerOutput │              │
                                     │ (Python    │   (sampled_tokens) │              │
                                     │  control)  │                    └──────┬───────┘
                                     │            │                           │
                                     └────────────┘                           │ pybind11
                                                                              ▼
                                                                     ┌──────────────┐
                                                                     │ Python        │
                                                                     │ InferEngine   │
                                                                     │ (thin wrap)   │
                                                                     └──────┬───────┘
                                                                            │ direct call
                                                                            ▼
                                                                     ┌──────────────┐
                                                                     │ C++ Engine    │  RankWorker[TP-1]
                                                                     │ InferEngine   │  RankWorker[0]
                                                                     │              │  InfiniCCL Communicator
                                                                     └──────┬───────┘
                                                                            │ forward()
                                                                            ▼
                                                                     ┌──────────────┐
                                                                     │ C++ Model    │  InfinilmModel::forward
                                                                     │ + Layers      │  Attention / MLP / Norm / RoPE
                                                                     └──────────────┘
                                                                            ▼
                                                                     KVCache tensor
                                                                     (csrc/cache/   │
                                                                      数据面)       │
                                                                                    │
                                            可选跨节点 ───────────────────────────────┘
                                            KV 迁移
                                            ┌──────────────┐
                                            │ KV Connector │
                                            │ Mooncake     │
                                            └──────────────┘
```

关键交互语义：

- Server → Scheduler 是 **janus.Queue**（asyncio 端 put，step 线程端 get），这是 async 和 sync 世界的唯一交接点。
- Scheduler → ModelRunner 是 **直接函数调用**，没有序列化；`SchedulerOutput` 里已经包含了 Python 计算好的 block_tables 和 slot_mapping。
- ModelRunner → InferEngine (Python) 也是 **直接函数调用**（pybind11 方法），数据通过 `runtime::Tensor` 零拷贝视图传递。
- C++ Engine 内部 TP 用 **InfiniCCL** 做 all-reduce/all-gather；PP 时 ModelRunner 额外通过 **原生 TCP `PipelineControlServer`** 广播 forward 元数据（tensor 激活仍然走 InfiniCCL）。

---

## 一条请求的调用链示例

**场景**：`LLM.generate("Hello")` —— 单卡 TP=1、Paged Attention、无 PP、无 speculative、无 KV transfer

```
────────────────────────────────────── 模块 ① Server ───────────────────────────
examples/test_infer.py::test()
  └─ LLM.generate(prompts=["Hello"])
       └─ self.engine.add_request(InferenceRequest)
            ├─ tokenizer.encode("Hello")
            ├─ 构造 InferenceRequest(prompt_token_ids, sampling_params)
            └─ Scheduler.add_request(req)        ──► 写入 waiting_queue.sync_q


────────────────────────────────────── 模块 ② Scheduler ──────────────────────
LLMEngine.step()   ←── while not finished: engine.step()
  └─ Scheduler.schedule()
       ├─ 从 waiting_queue 取 req                    # 首次在 waiting
       ├─ BlockManager.get_computed_blocks()          # 查 prefix cache hash
       ├─ BlockManager.allocate_slots(...)            # 分配 prefill block_table + slot_mapping
       ├─ req.block_table = [0, 1, 2, ...]            # Python 控制面决定
       ├─ req.slot_mapping = [具体 token 位置]
       ├─ req.status = RUNNING
       └─ SchedulerOutput(scheduled_requests=[req], is_prefill=True)


────────────────────────────────────── 模块 ③ ModelRunner ────────────────────
LLMEngine.step() 继续
  └─ ModelRunner.execute_model(scheduler_output)
       └─ ModelRunner._model_forward(scheduler_output)
            ├─ processor.build_model_inputs(...)      # 组装 dict: block_tables, slot_mapping,
            │                                         #   position_ids=[0,1,...,5], past_kv_lengths=[0], ...
            └─ self.model_engine.forward(**model_input)   # 直接调 Python InferEngine


────────────────────────────────────── Python InferEngine ───────────────────
python/infinilm/infer_engine.py::InferEngine.forward()
  └─ super().forward(Input)                            # pybind11 → C++


────────────────────────────────────── 模块 ④ C++ Engine ────────────────────
csrc/engine/infer_engine.cpp::InferEngine::forward(Input)
  ├─ workers_[0]->forward(input)                       # 单 TP rank
  └─ RankBarrier 同步，聚合 sampled token


────────────────────────────────────── 模块 ⑤ C++ Model ─────────────────────
csrc/models/qwen3/qwen3_for_causal_lm.cpp::Qwen3ForCausalLM::forward(Input, ForwardContext)
  ├─ 逐层 TextDecoderLayer::forward
  │    ├─ Attention::forward (paged_attn)             # 用 block_table / slot_mapping 读写 KVCache tensor
  │    ├─ MLP::forward
  │    └─ RMSNorm::forward
  └─ lm_head::forward → logits → sampling → sampled token_id=37


────────────────────────────────────── 回到 Scheduler ────────────────────
LLMEngine.step() 继续
  ├─ scheduler.accept_model_output(req, ...)           # block_table 可能更新
  ├─ _update_requests(req, sampled_token_ids=[37])
  │    ├─ req.append_generated_token_id(37)
  │    ├─ tokenizer.decode([37]) → token_text=" there"
  │    └─ 判 EOS / max_tokens → 未结束，回到 running_queue
  └─ scheduler.complete_requests([req])                # 本次没完成，不放行

# while 循环继续...

# decode 第 N 轮，request 从 running_queue 出来
Scheduler.schedule() 第二次
  ├─ 从 running_queue 取 req                          # 现在在 running
  ├─ BlockManager.allocate_slots(1, past_seq_len, ...) # 追加 1 个 decode block
  └─ SchedulerOutput(is_prefill=False)

# 重复 ModelRunner → InferEngine → C++ Engine → C++ Model 路径，每次只跑 1 个 token 的 decode

# 直到 token 37 之后又生成了若干个，最后一个是 EOS token_id
_update_requests 里 finish_reason = EOS_TOKEN → req.mark_finished()
Scheduler.complete_requests([req])
  └─ BlockManager.free_blocks(req.block_table)         # 释放所有 KV blocks
LLMEngine.step() 下一轮 scheduler.schedule() 返回 None（waiting 和 running 都空了）
LLM.generate() 的 while 循环退出，返回 [RequestOutput(generated_text="Hello there...")]
```

---

## 一句话小结

InfiniLM 的 CPU 侧是单 Python 进程内的 **Server(HTTP) → Scheduler(调度+KV 控制面) → ModelRunner(batch 组装) → InferEngine(pybind11)** 四层直接调用链，没有 ZMQ/IPC；TP 在 C++ 内部由 `InferEngine → RankWorker[] → InfiniCCL Communicator` 完成，PP 跨节点 tensor 激活走 InfiniCCL、forward 元数据走原生 TCP `PipelineControlServer`；KV Cache 控制面（block_table/slot_mapping/ref_count）在 Python Scheduler，数据面（tensor）在 C++ KVCache，两侧零拷贝衔接。
