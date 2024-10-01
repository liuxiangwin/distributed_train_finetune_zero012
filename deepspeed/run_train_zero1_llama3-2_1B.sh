deepspeed --num_gpus=2 experiment-zero1_empty-cache-finetune_llama_3-2_1B.py

## Zero-1: ds_config1.json
# {
#   "train_micro_batch_size_per_gpu": "auto",
#   "zero_optimization": {
#     "stage": 1
#   }
# }
