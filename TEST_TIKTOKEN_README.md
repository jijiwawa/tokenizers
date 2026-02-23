# test_tiktoken.py 性能测试脚本运行指南

本指南将详细介绍如何成功运行 `test_tiktoken.py` 脚本，该脚本用于比较 tiktoken 和 huggingface tokenizers 的性能。

## 1. 项目概述

`test_tiktoken.py` 是 Hugging Face tokenizers 库中的一个性能测试脚本，用于比较 tiktoken 和 huggingface tokenizers 在不同线程数和数据大小下的性能表现。

## 2. 环境准备

### 2.1 克隆仓库

首先，克隆 tokenizers 仓库：

```bash
git clone https://github.com/huggingface/tokenizers.git
cd tokenizers
```

### 2.2 Python 环境

确保您的系统上安装了 Python 3.9 或更高版本：

```bash
python --version
```

建议使用虚拟环境来隔离项目依赖：

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

## 3. 依赖安装

### 3.1 基本依赖

安装 tokenizers 库及其依赖：

```bash
pip install -e "bindings/python[testing]"
```

### 3.2 性能测试依赖

安装 tiktoken 库（用于性能比较）：

```bash
pip install tiktoken
```

### 3.3 模型下载依赖

安装 modelscope 库（用于下载 Llama-3.1-8B tokenizer）：

```bash
pip install modelscope
```

## 4. 模型下载

### 4.1 创建下载脚本

创建一个 Python 脚本用于从 modelscope 下载 Llama-3.1-8B tokenizer：

```python
# download_llama_tokenizer.py
from modelscope import snapshot_download

# 从 modelscope 下载 Llama-3.1-8B tokenizer 模型
model_id = "LLM-Research/Meta-Llama-3.1-8B"
save_dir = "./llama3_tokenizer"

print(f"Downloading Llama-3.1-8B tokenizer from modelscope...")
snapshot_download(
    model_id=model_id,
    cache_dir=save_dir,
    ignore_file_pattern=["*.bin", "*.safetensors", "*.pt"]  # 只下载 tokenizer 相关文件
)
print(f"Download completed. Tokenizer files saved to: {save_dir}")
```

### 4.2 运行下载脚本

```bash
python download_llama_tokenizer.py
```

## 5. 脚本修改

由于 Llama-3.1-8B 是一个 gated 模型，我们需要修改 `test_tiktoken.py` 脚本，使其使用本地下载的 tokenizer 文件。

### 5.1 修改模型 ID

将模型 ID 从 gated 模型改为本地文件路径：

```python
# 原代码
# MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"

# 修改后不需要修改 MODEL_ID 变量，因为我们将直接使用本地文件路径
```

### 5.2 修改 tokenizer 加载方式

在 `benchmark_batch` 函数中，修改 tokenizer 加载方式：

```python
# 使用本地下载的 tokenizer 文件
filename = "llama3_tokenizer/LLM-Research/Meta-Llama-3___1-8B/original/tokenizer.model"
mergeable_ranks = load_tiktoken_bpe(filename)

# ...

# 使用本地下载的 tokenizer 文件创建 Tokenizer
import json
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace

# 从本地文件加载 tokenizer
tokenizer_path = "llama3_tokenizer/LLM-Research/Meta-Llama-3___1-8B/tokenizer.json"
hf_enc = Tokenizer.from_file(tokenizer_path)
```

## 6. 运行测试

### 6.1 基本运行

运行性能测试脚本：

```bash
python bindings/python/benches/test_tiktoken.py
```

### 6.2 自定义参数

您可以使用以下参数自定义测试：

- `-m, --model`: 指定模型 ID（默认："meta-llama/Meta-Llama-3.1-8B"）
- `-d, --dataset`: 指定数据集（默认："facebook/xnli"）
- `-ds, --dataset-config`: 指定数据集配置（默认："all_languages"）
- `-t, --threads`: 指定线程数列表（默认：[1, 2, 4, 8]）

示例：

```bash
python bindings/python/benches/test_tiktoken.py -t 1 2 4
```

## 7. 结果分析

测试结果将显示在终端中，格式如下：

```
==============
num_threads: 8, data size: 24.04 MB, documents: 10000 Avg Length: 1659
tiktoken        55.16 MB  / s
huggingface     23.80 MB / s
```

### 7.1 结果解读

- `num_threads`: 使用的线程数
- `data size`: 测试数据的大小
- `documents`: 文档数量
- `Avg Length`: 文档平均长度
- `tiktoken`: tiktoken 库的处理速度（MB/s）
- `huggingface`: huggingface tokenizers 的处理速度（MB/s）

### 7.2 测试环境

本次性能测试在以下环境中进行：

- **系统信息**: macOS-26.0-arm64-arm-64bit-Mach-O
- **Python版本**: 3.13.7 (v3.13.7:bcee1c32211, Aug 14 2025, 19:10:51) [Clang 16.0.0 (clang-1600.0.26.6)]
- **CPU核心数**: 10

### 7.3 最终性能测试结果

经过三次测试运行，取平均值后的结果如下：

#### 7.3.1 小数据量（24.59 KB，10个文档，平均长度1662）

| 线程数 | tiktoken 速度 | huggingface 速度 |
|-------|-------------|-----------------|
| 1     | 10.67 MB/s  | 6.10 MB/s       |
| 2     | 11.70 MB/s  | 7.67 MB/s       |
| 4     | 16.82 MB/s  | 10.87 MB/s      |
| 8     | 16.93 MB/s  | 12.16 MB/s      |

#### 7.3.2 中等数据量（24.04 MB，10000个文档，平均长度1659）

| 线程数 | tiktoken 速度 | huggingface 速度 |
|-------|-------------|-----------------|
| 1     | 15.08 MB/s  | 7.57 MB/s       |
| 2     | 15.44 MB/s  | 9.90 MB/s       |
| 4     | 35.18 MB/s  | 18.25 MB/s      |
| 8     | 53.79 MB/s  | 22.57 MB/s      |

#### 7.3.3 大量小文档（1.11 MB，10000个文档，平均长度116）

| 线程数 | tiktoken 速度 | huggingface 速度 |
|-------|-------------|-----------------|
| 1     | 7.90 MB/s   | 6.27 MB/s       |
| 2     | 7.70 MB/s   | 6.09 MB/s       |
| 4     | 9.70 MB/s   | 9.56 MB/s       |
| 8     | 9.86 MB/s   | 13.62 MB/s      |

### 7.4 性能特点分析

1. **总体表现**：
   - tiktoken 在大多数情况下比 huggingface tokenizers 速度更快
   - 随着线程数增加，两种 tokenizer 的性能通常都会提升

2. **线程扩展性**：
   - tiktoken 在中等数据量测试中表现出最佳的线程扩展性，8线程时性能是1线程的3.57倍
   - huggingface tokenizers 在大量小文档测试中表现出最佳的线程扩展性，8线程时性能是1线程的2.17倍

3. **数据大小影响**：
   - 对于中等数据量（24.04 MB），tiktoken 的优势最为明显
   - 对于大量小文档，huggingface tokenizers 在8线程时性能超过了 tiktoken

4. **性能波动**：
   - 三次测试运行的性能波动较小，大部分测试用例的波动在10%以内
   - 小数据量测试的波动相对较大，可能是由于系统噪声影响

### 7.5 最佳使用场景

- **处理大文本块**：推荐使用 tiktoken，尤其是在多线程环境下
- **处理大量短文本**：huggingface tokenizers 在多线程下可能更有优势
- **单线程环境**：tiktoken 在各种场景下都表现更好

## 8. 注意事项

1. **模型下载**：Llama-3.1-8B 是一个大型模型，下载可能需要一些时间和带宽
2. **资源需求**：性能测试可能需要较多的内存和 CPU 资源
3. **网络连接**：首次运行时需要下载 XNLI 数据集，确保网络连接正常
4. **自定义测试**：您可以通过修改脚本来自定义测试数据和参数

## 9. 故障排除

### 9.1 模型下载失败

如果从 modelscope 下载模型失败，可以尝试以下方法：

- 检查网络连接
- 增加超时时间
- 手动下载模型文件并放置在指定路径

### 9.2 脚本运行错误

如果脚本运行出现错误，请检查：

- 依赖是否正确安装
- 模型文件路径是否正确
- Python 版本是否符合要求
- 虚拟环境是否正确激活

### 9.3 性能结果异常

如果性能结果异常，请检查：

- 系统资源使用情况（CPU、内存）
- 其他进程是否占用了大量资源
- 测试数据是否正确加载

## 10. 总结

通过本指南，您应该能够成功运行 `test_tiktoken.py` 脚本并比较 tiktoken 和 huggingface tokenizers 的性能。该测试可以帮助您了解在不同场景下哪种 tokenizer 更适合您的需求。

如果您有任何问题或建议，请随时提出。
