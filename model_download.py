from huggingface_hub import snapshot_download

# This downloads the files to a specific local folder
model_path = snapshot_download(
    repo_id="Qwen/Qwen2.5-Coder-0.5B",
    local_dir="./qwen_model_local",
    ignore_patterns=["*.msgpack", "*.h5"] # Skip non-pytorch weights to save time
)
print(f"Model downloaded to: {model_path}")