# 有界哈希缓存感知路由实验集成

本目录保存已在 MUSA 平台完成对比实验的有界哈希实现（V3），包括完整源码、测试、合成请求与固定环境配置。优化位置是完整模型副本前的 HTTP 网关：每张卡独立完成 Prefill 和 Decode，网关结合缓存复用、容量约束和预计等待成本选择副本。

这是**已经验证的实验集成归档**。放入 InfiniLM 仓库不会自动改变默认推理服务，也不代表已经适配仓库最新主线的全部后端接口。

## 目录与代码

`experiment/` 原样保留源包的 98 个文件；源码、测试和固定校验均未修改。`.gitattributes` 禁止原件换行转换，仅对两个原文件已有的末尾空行保留精确的格式检查例外。部分文件名称仍含旧方案，是最新版的实际依赖或固定测试清单成员，不能仅凭名称删除。

| 文件 | 作用 |
| --- | --- |
| `experiment/cache_router_v3.py` | 可复用副本的有限等待决策 |
| `experiment/cache_router.py` | 缓存预测、归属及成本评分基础 |
| `experiment/affinity_router.py`、`mean_router.py`、`stable_router.py` | 容量准入、状态、队列及基础路由 |
| `experiment/phase_router.py` | 经校准的 Prefill / Decode 成本模型 |
| `experiment/vendor/routing_manager.py` | 一致性哈希环；许可证见同目录 `LICENSE` |
| `experiment/gateway.py`、`streaming.py` | HTTP 流式代理、协议检查与资源释放 |
| `experiment/nh_stream_usage.py`、`cache_policy.py` | 后端状态观测、前缀优先保留和实验控制 |
| `experiment/optimization_runner.py`、`run_optimized.sh` | 已验证的 V3 / V2 / vLLM 对比入口 |
| `experiment/optimization_workload.py`、`optimization_report.py` | 工作负载和结果分析 |

其余运行器基础模块、27 个根目录测试文件、`anchor_data/`、`vendor/nhcf_cache/` 及固定清单一并保留。`vendor/nhcf_cache/` 是 CPU 测试夹具，不会替换实际 GPU 后端。保留 `vendor/LICENSE`；本次整理未重新授权第三方代码。

## 方法

公共前缀通过一致性哈希获得稳定归属；热点可增加受限的次归属。请求必须满足每卡在途数量和完整序列 KV 预留限制，然后比较待处理输入、当前输入和一轮解码干扰的预计成本。V3 在健康、有复用依据、没有待处理 Prefill 且暂时容量不足的副本上评估短暂等待是否更有利。

主动等待预算为 `min(1000 ms, 冷暖 Prefill 预计节省的 25%)`，截止点固定，不会随重试续期。该预算不是所有请求的总排队上限。预测复用与实际缓存命中分开记录；算法不进行跨卡 KV 搬运、不修改模型权重，也不改写 GPU 算子。

## CPU 检查

使用独立的 CPU 环境，避免修改已跑通的 MUSA 环境：

```bash
python3 -m venv /tmp/infinilm-bounded-hash-cpu
/tmp/infinilm-bounded-hash-cpu/bin/python -m pip install -r examples/bounded_hash/requirements-cpu.txt
cd examples/bounded_hash/experiment
/tmp/infinilm-bounded-hash-cpu/bin/python -B -m unittest discover -q -b
```

本次整理实际执行了 262 项 CPU 测试，结果见 `PACKAGING_VALIDATION.json`。这些测试包含本地 HTTP/SSE 假后端测试，不加载模型，不验证 GPU 性能。`requirements-cpu.txt` 仅用于这组 CPU 测试，不是 GPU 环境安装清单。

## GPU 实验及兼容性

原实验入口是 `run_optimized.sh`，它依次调用固定校验、独立成本校准、V3 / V2 / vLLM 同轮测试和分析。V3 通过入口选择，单独执行 `cache_router_v3.py` 不会启动服务。完整复现需要原 MUSA 软件栈、Qwen3-32B 权重及匹配的 InfiniLM 底座；本目录不包含模型、原生动态库或整个底座工作区。

`experiment/config.musa.json`、`reference_environment.json` 及旧脚本含原服务器目录和解释器路径，作为环境契约保留，不能当作通用配置。不要直接全局替换路径，也不要修改摘要常量来跳过校验。原包环境检查同时比较后端源码、编译库、分词器、硬件和配置。其历史底座记录还包含本地改动，仅检出同一基线提交并不足以重建完整运行环境。

已检查的仓库主线为 `eb97954b518cf33fbf3ee2dc22f13ecf2141400b`。其 `python/infinilm/llm/cache_manager.py` 与原补丁要求的文件不同：原补丁要求 SHA256 `d7792e9118683a9c5476158d14849dee8390c9b2dbfdf095cfcaca4051356987`，该主线文件为 `449cb2c73a966a46c42cf6166023056a643c1910307965b6ace97a1e5782d856`。在未适配前，原保护检查会拒绝不匹配的后端。上传此目录不改变或绕过该保护。

网关当前要求 `X-Fourway-Item` 请求头中的 token 数及前缀契约，并要求流式请求；这些信息由实验工作负载构造。直接供通用 OpenAI 客户端使用，还需要独立服务入口、实际请求分词及前缀契约生成，并在目标底座上验证。旧 `run_cache.sh` 等入口属于保留依赖，不应替代 V3 的 `run_optimized.sh`。

## 实验结果和原文件状态

后续成功 MUSA 实验及其局限见 `RESULTS.md` 与 `results/summary.csv`。短回答场景有吞吐和平均 TTFT 优势；输出更长时不保证优势，尾延迟仍存在退步。

原 `experiment/README_优化测试.md` 的“尚无 GPU 性能结果”和 `experiment/VALIDATION.json` 的 `NOT_RUN` 描述的是候选包制作时的状态；为保持原件一致，本次未修改。它们不代表后续实验没有完成。当前整理也没有新跑 GPU 实验，不能把 CPU 测试通过或上传成功作为新的 GPU 性能证据。
