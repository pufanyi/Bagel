# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# replace the variables with your own
torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=10.10.10.10 \
  --master_port=29500 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --vae_path $vae_path \
  --vit_path $vit_path \
  --llm_path $llm_path \
  --use_flex True \
  --resume_from $resume_from \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --max_latent_size 64  \
  --num_workers 1 # use small num_workers since the num_used_data (10) are not enough to split