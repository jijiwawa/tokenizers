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
