# `encode_batch_fast` Benchmark 关键路径与输入输出说明

配套流程图：`benchmark_encode_batch_fast_critical_path.excalidraw`

## 1. 这份分析覆盖什么

这里分析的对象有两类 benchmark：

- Python 端到端 benchmark：
  [bindings/python/benches/repeat_single_thread_benchmark.py](bindings/python/benches/repeat_single_thread_benchmark.py)
- Rust 局部热点 benchmark：
  [tokenizers/benches/aarch64_hotpath_benchmark.rs](tokenizers/benches/aarch64_hotpath_benchmark.rs)

其中 Python benchmark 真正计时的只有这一句：

```python
hf_tokenizer.encode_batch_fast(batch)
```

也就是说，计时范围不包含：

- `load_dataset(...)`
- `Tokenizer.from_file(...)`
- `tiktoken` 初始化
- sanity check

## 2. Benchmark 对应的 tokenizer 配置

本次 benchmark 使用的 tokenizer 来自：

- [llama3_tokenizer/LLM-Research/Meta-Llama-3___1-8B/tokenizer.json](llama3_tokenizer/LLM-Research/Meta-Llama-3___1-8B/tokenizer.json)

关键配置如下：

- `normalizer = null`
- `pre_tokenizer = Sequence(Split regex + ByteLevel(use_regex=false))`
- `post_processor = Sequence(ByteLevel + TemplateProcessing)`
- `decoder = ByteLevel`
- `model = BPE`
- `ignore_merges = true`
- `added_tokens = 256`

这意味着当前 benchmark 的主路径会稳定经过：

1. Python 批量输入编组
2. AddedVocabulary 特殊 token 检查
3. Split regex 分词切片
4. ByteLevel 字节映射
5. BPE tokenize/cache/merge
6. `OffsetType::None` 的 fast `Encoding` 组装
7. post-process 加 special token
8. Python `Encoding` 返回

## 3. 关键路径总览

端到端调用链可以概括为：

```text
repeat_single_thread_benchmark.py
  -> tokenizers.Tokenizer.encode_batch_fast(batch)
  -> bindings/python/src/tokenizer.rs::build_owned_encode_inputs
  -> tokenizers/src/tokenizer/mod.rs::encode_batch_fast
  -> tokenizers/src/tokenizer/mod.rs::encode_fast
  -> tokenizers/src/tokenizer/mod.rs::encode_single_sequence
  -> AddedVocabulary::extract_and_normalize
  -> do_pre_tokenize
  -> PreTokenizer: Split regex + ByteLevel.map_bytes
  -> do_tokenize
  -> BPE::tokenize / tokenize_with_cache / merge_word
  -> PreTokenizedString::into_encoding(OffsetType::None)
  -> post_process
  -> Python PyEncoding
```

## 4. 各阶段输入输出

| 阶段 | 关键函数 | 输入数据 | 输出数据 | 说明 |
| --- | --- | --- | --- | --- |
| Python benchmark 入口 | `benchmark_engine(..., lambda batch: hf_tokenizer.encode_batch_fast(batch))` | `documents: list[str]` | 传给 Python binding 的 `list[str]` | benchmark 用批量长文本驱动真实 `encode_batch_fast` 路径 |
| Python 输入提取 | `build_owned_encode_inputs` | `Vec<Bound<PyAny>>`，当前场景主要是 `PyString` | `Vec<tk::EncodeInput<'static>>` | 这里会把 Python `str` 转成 Rust 自有 `String`，pair 场景会变成 `EncodeInput::Dual` |
| Rust batch 驱动 | `TokenizerImpl::encode_batch_fast` | `Vec<EncodeInput<'s>>` | `Vec<Encoding>` | 逐条样本调用 `encode_fast`，单线程 benchmark 下不会从线程调度拿收益 |
| 单条样本入口 | `encode_fast` | `EncodeInput::Single/ Dual` | 单条或 pair 的 `Encoding` | 这里显式走 `OffsetType::None`，所以 fast path 不跟踪真实 offsets |
| 特殊 token 预处理 | `AddedVocabulary::extract_and_normalize` | `&str` + `Option<&Normalizer>` | `PreTokenizedString` | 把原始句子包成可切分对象；若命中特殊 token，会把部分 split 直接标为已匹配 token |
| AddedVocabulary 预筛选 | `MatchingPrefilter::can_skip` + `contains_any_ascii_byte` | 原始 `&str`，以及“所有 added token 的必要 ASCII 字节” | 布尔结果：是否可以跳过 trie 匹配 | 当前 Llama 3 special token 都是 `<|...|>` 风格，普通文本没有 `<` 时，这一段可以快速返回 |
| Pre-tokenize 切分 | `do_pre_tokenize` -> `Split` regex | `PreTokenizedString` | 切成多个 `Split` 的 `PreTokenizedString` | 每个 split 内部仍然保存 `NormalizedString`，此时通常还没有最终 `Token` |
| Byte-level 映射 | `NormalizedString::map_bytes` | `NormalizedString { normalized: String, alignments: Vec<(usize, usize)> }` | 映射后的 `NormalizedString` | 对每个字节执行 GPT-2 byte-to-unicode 映射，并同步重建 alignment |
| 模型 tokenize | `do_tokenize` -> `BPE::tokenize` | 每个 split 的 `NormalizedString` 文本视图 | `Vec<Token>` | `Token { id, value, offsets }`；当前模型 `ignore_merges=true`，会先尝试整词 vocab 命中，再走 cache/merge |
| BPE cache / merge | `tokenize_with_cache` / `merge_word` | `&str` split 文本 | `Vec<Token>` 或中间 `Word` | 重复文本或热 cache 会降低这段成本，因此多轮 benchmark 后这里的相对占比通常下降 |
| Encoding 组装 | `PreTokenizedString::into_encoding(OffsetType::None)` | 已带 `Vec<Token>` 的 `PreTokenizedString` | `Encoding` | fast path 会预统计 token 数并一次性分配；这里 push 的正文 token 字符串是 `String::new()`，offsets 是 `(0, 0)` |
| Post process | `post_process` | `Encoding` / `pair_encoding` + `add_special_tokens` | 最终 `Encoding` | 会插入 `<|begin_of_text|>` 等 special token，并应用 post-processor 逻辑 |
| Python 返回 | `Vec<Encoding> -> Vec<PyEncoding>` | Rust `Encoding` | Python `Encoding` 对象列表 | `PyEncoding` 带 `freelist = 1024`，缓解大批量返回时的 Python 对象分配成本 |

## 5. 需要特别解释的数据结构

### `EncodeInput`

定义位置：

- [tokenizers/src/tokenizer/mod.rs](tokenizers/src/tokenizer/mod.rs)

语义：

- `EncodeInput::Single(InputSequence)`
- `EncodeInput::Dual(InputSequence, InputSequence)`

当前 benchmark 主要是：

- `EncodeInput::Single(InputSequence::Raw(Cow<str>))`

### `NormalizedString`

定义位置：

- [tokenizers/src/tokenizer/normalizer.rs](tokenizers/src/tokenizer/normalizer.rs)

语义：

- 保存“当前归一化字符串”
- 同时保存“归一化后字符位置 -> 原始字符串位置”的 alignment

这就是为什么 `map_bytes(...)` 不只是改字符串内容，还要同步重建 alignment。

### `PreTokenizedString`

定义位置：

- [tokenizers/src/tokenizer/pre_tokenizer.rs](tokenizers/src/tokenizer/pre_tokenizer.rs)

语义：

- 保存原始字符串 `original`
- 保存多个 `Split`
- 每个 `Split` 内含：
  - 一个 `NormalizedString`
  - 一个可选 `Vec<Token>`

所以它本质上是“等待被切分、归一化、tokenize，最后再组装成 `Encoding`”的中间结构。

### `Token`

定义位置：

- [tokenizers/src/tokenizer/mod.rs](tokenizers/src/tokenizer/mod.rs)

结构：

```text
Token {
  id: u32,
  value: String,
  offsets: (usize, usize)
}
```

它是 BPE 模型输出给 `PreTokenizedString::into_encoding(...)` 的直接产物。

### `Encoding`

定义位置：

- [tokenizers/src/tokenizer/encoding.rs](tokenizers/src/tokenizer/encoding.rs)

核心字段：

- `ids`
- `type_ids`
- `tokens`
- `words`
- `offsets`
- `special_tokens_mask`
- `attention_mask`
- `overflowing`

但是对 `encode_batch_fast` 来说，要特别注意：

- 正文 token 在 `OffsetType::None` 组装阶段会写入 `token = ""`
- 正文 token 的 `offsets` 会是 `(0, 0)`
- 也就是说 fast path 的设计重点是吞吐，不是完整字符级元数据

## 6. 为什么当前 benchmark 的热点主要在前半段

### `byte-level` 一定在主路径上

因为 tokenizer.json 的 pre-tokenizer 明确是：

- 先 `Split` regex
- 再 `ByteLevel`

所以只要 benchmark 输入是正文文本，`map_bytes(...)` 就会被执行。

### AddedVocabulary 会稳定经过

因为 tokenizer.json 里有 256 个 `added_tokens`，而 `encode_single_sequence` 在 pre-tokenize 前总会先经过：

- `AddedVocabulary::extract_and_normalize(...)`

所以即使文本没有 special token，判断“有没有可能命中 special token”本身也是固定成本。

### BPE 是中后段热点，但更容易被 cache 稀释

当前模型配置：

- `ignore_merges = true`
- `dropout = null`
- 默认带 cache

所以重复文本或多轮 benchmark 后，BPE 一部分成本会被 cache 吸收；相比之下，前面的 Python 输入提取、AddedVocabulary 检查、regex split、ByteLevel 映射更像是每轮都稳定发生的“前置税”。

## 7. 新增 Rust microbenchmark 分别对应哪段路径

新增 benchmark 文件：

- [tokenizers/benches/aarch64_hotpath_benchmark.rs](tokenizers/benches/aarch64_hotpath_benchmark.rs)

映射关系如下：

| benchmark case | 对应代码路径 | 想单独观察的热点 |
| --- | --- | --- |
| `byte-level-normalize` | `ByteLevelNormalizer::normalize -> map_bytes` | 正文 UTF-8 -> byte-level 映射 |
| `extract-and-normalize-no-match` | `AddedVocabulary::extract_and_normalize` | 文本没有 special token 时的预筛选和跳过逻辑 |
| `bpe-tokenize-cached` | `BPE::tokenize_with_cache` | 热 cache 情况下的 BPE 开销 |
| `bpe-tokenize-no-cache` | `BPE::merge_word` | 冷 cache / merge 主体开销 |

## 8. 结论

基于 benchmark 文件本身和 tokenizer 配置，可以把当前主路径总结成一句话：

> 这条 benchmark 测到的不是“抽象的 tokenizer 性能”，而是 Python 批量文本输入经过 AddedVocabulary、Split regex、ByteLevel、BPE、fast `Encoding` 组装，再返回 Python `Encoding` 对象的整条端到端路径。

因此，当前最值得重点盯住的关键路径顺序是：

1. Python 文本输入提取
2. AddedVocabulary 无命中快速判定
3. Split regex
4. ByteLevel `map_bytes`
5. BPE `tokenize_with_cache / merge_word`
6. `OffsetType::None` 的 fast `Encoding` 构造
7. Python 结果对象回传
