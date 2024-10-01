
deepspeed --num_gpus=2 experiment-zero2-empty-cache_finetune_llama_3-2_1B.py

##Zero-2:  ds_config2.json
# {
#   "train_micro_batch_size_per_gpu": "auto",
#   "zero_optimization": {
#     "stage": 2
#   },
#     "bf16": {
#         "enabled": true
#     },
#     "optimizer": {
#         "type": "AdamW",
#         "params": {
#         "lr": "auto",
#         "betas": "auto",
#         "eps": 1e-8,
#         "weight_decay": "auto"
#         }
#     }
# }

