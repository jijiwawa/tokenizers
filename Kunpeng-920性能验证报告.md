# 当前服务器性能验证报告

## 1. 结论摘要

- 验证目标：对比 `be1513cf`（基线）与 `a73478b1`（优化版）在同一服务器上的单线程性能差异
- 服务器：`openEuler 22.03 (LTS-SP2)`，`Kunpeng-920`，`aarch64`
- 绑定核策略：按要求固定在 `CPU 8`（`taskset -c 8`）
- 核心结果（`huggingface` 引擎）：
  - 基线：`1.74 MB/s`
  - 优化版：`2.12 MB/s`
  - 提升：`+21.99%`
- 稳定性：
  - 基线 `CV=0.10%`
  - 优化版 `CV=0.11%`
  - 两组都远低于 `5%`，属于高稳定度结果

## 2. 基准合同

- 基准脚本：`bindings/python/benches/repeat_single_thread_benchmark.py`
- 提交对比：
  - baseline: `be1513cf`
  - optimized: `a73478b1`
- 输入合同：
  - 数据集：`facebook/xnli` `train`
  - 文档数：`10000`
  - 模式：`long`
  - 数据量：`24.04 MB`
  - 线程：`1`
  - warm-up：`1`
  - runs：`10`
  - engine：`huggingface`
- 输出工件：
  - `/tmp/baseline_modelscope.json`
  - `/tmp/optimized_modelscope.json`

## 3. ModelScope 下载记录

下载工具：

- Python 环境：`/tmp/tokenizers-bench`
- 包：`modelscope==1.35.3`
- API：`modelscope.hub.snapshot_download.snapshot_download`

下载尝试与结果：

- `LLM-Research/Meta-Llama-3___1-8B`：在 `modelscope.cn/.ai` 查询失败（404）
- `LLM-Research/Meta-Llama-3.1-8B`：下载成功
- `LLM-Research/Meta-Llama-3-8B`：下载成功

本次 benchmark 实际使用路径：

- `tokenizer.json`:
  - `/tmp/modelscope_tokenizer/LLM-Research/Meta-Llama-3___1-8B/tokenizer.json`
- `tokenizer.model`:
  - `/tmp/modelscope_tokenizer/LLM-Research/Meta-Llama-3___1-8B/original/tokenizer.model`

文件校验（SHA256）：

- `76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa  tokenizer.json`
- `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55  tokenizer.model`

注：

- 原始文档中的 Hugging Face 仓库在当前服务器返回 `401`，因此改为 ModelScope 下载并记录来源与哈希。

## 4. 执行命令

编译（基线 / 优化）：

```bash
bash -lc '. /root/.cargo/env && . /tmp/tokenizers-bench/bin/activate && maturin develop --manifest-path bindings/python/Cargo.toml --release'
```

基线 benchmark（CPU 8）：

```bash
taskset -c 8 /tmp/tokenizers-bench/bin/python bindings/python/benches/repeat_single_thread_benchmark.py \
  --threads 1 --warmups 1 --runs 10 --engine huggingface \
  --documents 10000 --document-mode long \
  --tokenizer-json /tmp/modelscope_tokenizer/LLM-Research/Meta-Llama-3___1-8B/tokenizer.json \
  --tiktoken-bpe /tmp/modelscope_tokenizer/LLM-Research/Meta-Llama-3___1-8B/original/tokenizer.model \
  --json-output /tmp/baseline_modelscope.json
```

优化 benchmark（CPU 8）：

```bash
taskset -c 8 /tmp/tokenizers-bench/bin/python bindings/python/benches/repeat_single_thread_benchmark.py \
  --threads 1 --warmups 1 --runs 10 --engine huggingface \
  --documents 10000 --document-mode long \
  --tokenizer-json /tmp/modelscope_tokenizer/LLM-Research/Meta-Llama-3___1-8B/tokenizer.json \
  --tiktoken-bpe /tmp/modelscope_tokenizer/LLM-Research/Meta-Llama-3___1-8B/original/tokenizer.model \
  --json-output /tmp/optimized_modelscope.json
```

## 5. 原始结果与对比

baseline samples (MB/s):

- `1.74 1.74 1.74 1.74 1.74 1.74 1.74 1.74 1.74 1.74`

optimized samples (MB/s):

- `2.12 2.12 2.12 2.12 2.12 2.13 2.12 2.12 2.12 2.12`

统计对比：

| Metric | Baseline | Optimized | Delta |
| --- | ---: | ---: | ---: |
| mean MB/s | 1.74 | 2.12 | +21.99% |
| stdev | 0.00 | 0.00 | |
| CV | 0.10% | 0.11% | |
| min MB/s | 1.74 | 2.12 | |
| max MB/s | 1.74 | 2.13 | |

## 6. 稳定性评估

- 评估口径：
  - `CV <= 5%`：稳定
  - `5% < CV <= 10%`：轻微波动
  - `CV > 10%`：噪声较大
- 本次结果：
  - baseline `CV=0.10%`
  - optimized `CV=0.11%`
- 结论：两组结果均高度稳定，优化收益可信。

## 7. 正确性回归

执行：

```bash
/tmp/tokenizers-bench/bin/python -m pytest \
  bindings/python/tests/bindings/test_tokenizer.py::TestTokenizer::test_encode_batch_fast_text_inputs \
  bindings/python/tests/bindings/test_tokenizer.py::TestTokenizer::test_multithreaded_concurrency
```

结果：

- `2 passed in 0.46s`

## 8. 复现与环境快照

- 环境快照文件：`/tmp/tokenizers-env.md`
- 关键机器信息：
  - OS：`Linux 5.10.0-64k` (`openEuler 22.03`)
  - CPU：`Kunpeng-920`
  - 内存：`381 GiB`
  - NUMA：`4 nodes`
