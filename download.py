from huggingface_hub import snapshot_download

repo_id = "ByteDance-Seed/BAGEL-7B-MoT"

path = snapshot_download(
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

print(path)
