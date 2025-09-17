# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}


DATASET_INFO = {
    # 't2i_pretrain': {
    #     't2i': {
    #         'data_dir': '/mnt/aigc/users/pufanyi/workspace/playground/blip3o/blip3o-60k-top100-data', # path of the parquet files
    #         'num_files': 10, # number of data units to be sharded across all ranks and workers
    #         'num_total_samples': 1000, # number of total samples in the dataset
    #     },
    # },
    "vlm_sft": {
        "llava_ov": {
            "data_dir": "/mnt/aigc/users/pufanyi/workspace/playground/bagel/official/data/bagel_example/vlm/llava_ov_si.jsonl",
            "num_files": 10,
            "num_total_samples": 1000,
        },
    },
}