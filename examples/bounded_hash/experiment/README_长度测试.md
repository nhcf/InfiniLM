# InfiniLM 八文档输入与输出长度实验

本轮以输出吞吐为主指标。旧版 mean、V2 和 vLLM 在相同输入和到达顺序下逐组比较，并记录平均首响应时间及 p95。上一份报告表 3 的正式阶段按数值严格高于 vLLM 计数，V2 为 3/12、旧 mean 为 2/12。若门槛设为至少高 2%，则分别为 2/12 和 1/12。旧 mean 多出的一次小幅领先发生在长输出、并发 16，约高 0.60%。

## 默认面板

| 输入 token | 可共享的前缀 token | 共享占比 | 每种输入分别生成 |
|---:|---:|---:|---|
| 4,096 | 3,840 | 93.75% | 32、128 token |
| 6,144 | 5,888 | 95.83% | 32、128 token |
| 8,192 | 7,936 | 96.88% | 32、128 token |

每组的输入长度包含聊天模板，固定私有部分的声明长度为 256 token。共享长度按 InfiniLM 的 256 token 页对齐。本轮所选长度均满足旧 mean 的长输入、短输出和共享占比要求，也适用于 V2 的前缀亲和路径。输出从 32 增至 128，可观察生成阶段占比增加后吞吐领先幅度如何变化。

默认 C16，固定种子 33962、单轮、8 篇文档，正式请求中的热点文档占 25%。六种长度组合各跑三种方法，合计 18 组。每组 64 条学习请求、128 条正式请求，共 3,456 条；另有校准及探测。相同并发下运行次序随长度组合轮换，每种方法各排第一、第二、第三两次。

输入以成功报告归档的同一批合成文档请求为基础。同一输入长度的两档输出使用相同文本、请求 ID 和文档键，只有生成长度变化。缩短输入时沿用同一逻辑文档键和到达顺序，使同一文档的哈希环遍历顺序一致。最终选卡由当时负载和缓存决定。8,192 输入、32 输出为历史工作负载的对照点，新实验会在本轮重新计时。每组独立重启模型池，学习阶段通过实际网关建立缓存。

## 在 MUSA 服务器的 VS Code 终端运行

把 ZIP 传到此前跑通测试的服务器，在终端执行以下命令。

~~~bash
cd /mnt/zj-data/data/chengcp/infinilm
python3 -m zipfile -e infinilm_reuse8_lengths_20261103.zip .
cd infinilm_reuse8_lengths_20261103
bash run_lengths.sh plan
nohup bash run_lengths.sh run > lengths_c16.log 2>&1 < /dev/null &
echo $! > lengths_c16.pid
tail -f lengths_c16.log
~~~

`plan` 只显示实验计划。`run` 包含 CPU 自检、运行身份核对、工作负载准备、校准和正式测量。原控制端环境位于 `qwen3_32b_fourway_20261021/.controller/bin/python`。如需单独做环境和分词器检查，可执行 `bash run_lengths.sh check`。模型目录为 `/mnt/zj-data/models/Qwen3-32B/`。

若需验证此前 V2 在 8 文档 C32 下小幅领先的现象，可在 C16 结束后单独运行以下命令。这会增加 18 组，结果单独保存。

~~~bash
nohup bash run_lengths.sh run --concurrency 32 > lengths_c32.log 2>&1 < /dev/null &
echo $! > lengths_c32.pid
tail -f lengths_c32.log
~~~

## 阅读结果

日志末尾的 `LENGTH_REPORT_ARCHIVE=` 指向本轮报告包。先看 `throughput_report.md` 的正式阶段表，再看包含学习开销的表。每种长度独立计算“V2 / vLLM”“旧 mean / vLLM”和“V2 / 旧 mean”的吞吐变化，同时区分任何正收益和至少 2% 的收益。各方案的 TTFT 用来解释响应体验。

结论适用于这次固定种子、单轮、八文档高复用的合成负载。18 组的正式测量和校准均须在八卡服务器完成。一次 CPU 自检或打印计划只表明脚本走到了相应步骤。
