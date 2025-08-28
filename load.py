from safetensors.torch import safe_open

path = "/mnt/raid10/pufanyi/hf/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/570026eca23479ee7df5a6ce9fb50a835530da30/ae.safetensors"
# path = "ae.safetensors"
with safe_open(path, framework="pt") as f:
    print(f.keys())
