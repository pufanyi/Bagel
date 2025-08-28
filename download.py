import os
from huggingface_hub import snapshot_download

repo_id = "ByteDance-Seed/BAGEL-7B-MoT"

path = snapshot_download(repo_id, local_dir_use_symlinks=False)

print("[VAE path]")
print(os.path.join(path, "ae.safetensors"))
# print("[VIT path]")
# print(vit_path)
# print("[LLM path]")
# print(llm_path)
