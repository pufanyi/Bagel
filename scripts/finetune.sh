# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

. ".venv/bin/activate"

export PYTHONPATH=$PYTHONPATH:.

# CUDA_VISIBLE_DEVICES=7

python -m torch.distributed.run \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=2 \
  --master_addr=localhost \
  --master_port=29511 \
  train/pretrain_unified_navit.py \
  --data_path pufanyi/BLIP3o-60k-top100 \
  --dataset_config_file ./data/configs/example.yaml \
  --model_path /mnt/aigc/users/pufanyi/workspace/lmms-engine-mini/playground/models/BAGEL-7B-MoT \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from /mnt/aigc/users/pufanyi/workspace/lmms-engine-mini/playground/models/BAGEL-7B-MoT \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240
