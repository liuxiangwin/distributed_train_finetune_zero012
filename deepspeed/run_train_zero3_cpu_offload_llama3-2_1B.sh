
deepspeed --num_gpus=2 experiment-zero3-empty-cache_finetune_llama_3-2_1B.py

##Zero-3:  ds_config3.json
# {
#     "train_micro_batch_size_per_gpu": "auto",
#     "zero_optimization": {
#         "stage": 3,
#         "offload_optimizer": {
#             "device": "cpu",
#             "pin_memory": true
#         }
#     },
#     "zero_force_ds_cpu_optimizer": false,
#     "fp16": {
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
#     },
# }

