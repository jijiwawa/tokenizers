#!/usr/bin/env python3
"""Repeatable benchmark for the local Llama 3 tokenization use case."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
import tiktoken  # type: ignore[import]
from tiktoken.load import load_tiktoken_bpe  # type: ignore[import]

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_SRC = REPO_ROOT / "bindings" / "python" / "py_src"
TOKENIZER_JSON = (
    REPO_ROOT
    / "llama3_tokenizer"
    / "LLM-Research"
    / "Meta-Llama-3___1-8B"
    / "tokenizer.json"
)
TIKTOKEN_BPE = (
    REPO_ROOT
    / "llama3_tokenizer"
    / "LLM-Research"
    / "Meta-Llama-3___1-8B"
    / "original"
    / "tokenizer.model"
)
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
DATASET = "facebook/xnli"
DATASET_CONFIG = "all_languages"
DATASET_SPLIT = "train"
DEFAULT_DOCUMENTS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeatable single-thread benchmark for local tokenizers builds.",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--documents", type=int, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--dataset-split", default=DATASET_SPLIT)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--document-mode",
        choices=("long", "short"),
        default="long",
        help="Use concatenated multilingual premise values or a single language premise.",
    )
    parser.add_argument(
        "--engine",
        choices=("huggingface", "tiktoken", "both"),
        default="both",
    )
    parser.add_argument("--tokenizer-json", type=Path, default=TOKENIZER_JSON)
    parser.add_argument("--tiktoken-bpe", type=Path, default=TIKTOKEN_BPE)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def format_byte_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def summarize(samples: list[float]) -> dict[str, float]:
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    min_value = min(samples)
    max_value = max(samples)
    return {
        "mean_mb_s": mean,
        "stdev_mb_s": stdev,
        "cv_percent": (stdev / mean * 100.0) if mean else 0.0,
        "min_mb_s": min_value,
        "max_mb_s": max_value,
        "range_mb_s": max_value - min_value,
    }


def load_documents(args: argparse.Namespace) -> list[str]:
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    documents: list[str] = []
    use_long_documents = args.document_mode == "long"
    for index, item in enumerate(dataset):
        if index >= args.documents:
            break
        premise = item["premise"]
        if use_long_documents:
            documents.append("".join(premise.values()))
        else:
            documents.append(premise[args.language])
    return documents


def build_tiktoken_encoder(model: str, bpe_path: Path) -> tiktoken.Encoding:
    mergeable_ranks = load_tiktoken_bpe(str(bpe_path))
    pat_str = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
        r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
    num_reserved_special_tokens = 256
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",
    ] + [
        f"<|reserved_special_token_{i}|>"
        for i in range(5, num_reserved_special_tokens - 5)
    ]
    num_base_tokens = len(mergeable_ranks)
    special_tokens_map = {
        token: num_base_tokens + i for i, token in enumerate(special_tokens)
    }
    return tiktoken.Encoding(
        name=model,
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens_map,
    )


def benchmark_engine(
    name: str,
    runner: Any,
    documents: list[str],
    total_bytes: int,
    warmups: int,
    runs: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        runner(documents)

    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        runner(documents)
        end = time.perf_counter_ns()
        elapsed_s = (end - start) / 1e9
        samples.append(total_bytes / elapsed_s / 1024.0 / 1024.0)

    return {"engine": name, "samples_mb_s": samples, "summary": summarize(samples)}


def main() -> None:
    args = parse_args()

    if not args.tokenizer_json.exists():
        raise FileNotFoundError(f"Missing tokenizer json: {args.tokenizer_json}")
    if not args.tiktoken_bpe.exists():
        raise FileNotFoundError(f"Missing tiktoken BPE file: {args.tiktoken_bpe}")
    if not PY_SRC.exists():
        raise FileNotFoundError(f"Missing local python source directory: {PY_SRC}")

    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    sys.path.insert(0, str(PY_SRC))

    from tokenizers import Tokenizer

    documents = load_documents(args)
    total_bytes = sum(len(doc.encode("utf-8")) for doc in documents)

    print("==============")
    print(
        f"threads: {args.threads}, documents: {len(documents)}, "
        f"data size: {format_byte_size(float(total_bytes))}, "
        f"mode: {args.document_mode}"
    )
    print(f"tokenizer_json: {args.tokenizer_json}")
    print(f"tiktoken_bpe: {args.tiktoken_bpe}")

    hf_tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    tk_encoder = build_tiktoken_encoder(args.model, args.tiktoken_bpe)

    sanity_text = "This is a test"
    tk_output = tk_encoder.encode(sanity_text)
    hf_output = hf_tokenizer.encode(sanity_text, add_special_tokens=False).ids
    if tk_output != hf_output:
        raise AssertionError("Sanity check failed: tiktoken and tokenizers outputs differ")

    results: dict[str, Any] = {
        "config": {
            "threads": args.threads,
            "warmups": args.warmups,
            "runs": args.runs,
            "documents": len(documents),
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "document_mode": args.document_mode,
            "language": args.language,
            "total_bytes": total_bytes,
            "tokenizer_json": str(args.tokenizer_json),
            "tiktoken_bpe": str(args.tiktoken_bpe),
            "py_src": str(PY_SRC),
        },
        "results": [],
    }

    if args.engine in ("tiktoken", "both"):
        tk_result = benchmark_engine(
            "tiktoken",
            lambda batch: tk_encoder.encode_ordinary_batch(batch, num_threads=args.threads),
            documents,
            total_bytes,
            args.warmups,
            args.runs,
        )
        results["results"].append(tk_result)

    if args.engine in ("huggingface", "both"):
        hf_result = benchmark_engine(
            "huggingface",
            lambda batch: hf_tokenizer.encode_batch_fast(batch),
            documents,
            total_bytes,
            args.warmups,
            args.runs,
        )
        results["results"].append(hf_result)

    for result in results["results"]:
        summary = result["summary"]
        sample_text = " ".join(f"{value:.2f}" for value in result["samples_mb_s"])
        print(f"{result['engine']} samples_mb_s: {sample_text}")
        print(
            f"{result['engine']} summary: "
            f"mean={summary['mean_mb_s']:.2f} MB/s, "
            f"stdev={summary['stdev_mb_s']:.2f}, "
            f"cv={summary['cv_percent']:.2f}%, "
            f"min={summary['min_mb_s']:.2f}, "
            f"max={summary['max_mb_s']:.2f}, "
            f"range={summary['range_mb_s']:.2f}"
        )

    if args.json_output:
        args.json_output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"json_output: {args.json_output}")


if __name__ == "__main__":
    main()
