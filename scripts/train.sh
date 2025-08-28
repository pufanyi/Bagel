# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

vae_path=/mnt/raid10/pufanyi/hf/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/570026eca23479ee7df5a6ce9fb50a835530da30/ae.safetensors
vit_path=HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
llm_path=Qwen/Qwen2.5-0.5B-Instruct

. ".venv/bin/activate"

export PYTHONPATH=$PYTHONPATH:.

CUDA_VISIBLE_DEVICES=4 torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=29501 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --vae_path $vae_path \
  --vit_path $vit_path \
  --llm_path $llm_path \
  --use_flex True \
  --results_dir ./results \
  --checkpoint_dir ./temp/checkpoints \
  --max_latent_size 64 \
  --num_shard 1 \
  --cpu_offload True \
  --expected_num_tokens 4096 \
  --num_workers 1 \
  --visual_und False