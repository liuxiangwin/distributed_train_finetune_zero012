### BASE Training Script QWEN
## 1xnode with 2xGPU - each GPU vRAM 16GB
## SUCCESS

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
import torch

dataset_source = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_source)

base_model = "Qwen/Qwen1.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(base_model)
## Change-1 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

## Change-4
compute_dtype = torch.bfloat16
if torch.cuda.is_bf16_supported():
    #os.system('pip install flash_attn')
    compute_dtype = torch.bfloat16
    attn_implementation = 'flash_attention_2'
else:
    compute_dtype = torch.float16
    attn_implementation = 'sdpa'
print("-------------------------")
print("**compute_dtype=",compute_dtype)
print("**attn_implementation=",attn_implementation)
print("-------------------------")

model = AutoModelForCausalLM.from_pretrained(base_model, 
                                             torch_dtype=compute_dtype,
                                             attn_implementation=attn_implementation,
                                             device_map="cuda")

## Change-6 enable gradient checkpoint
model.gradient_checkpointing_enable()
model.config.use_cache=False

batch_size = 1
args = TrainingArguments(
    'outputs',
    learning_rate=8e-5,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    eval_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    weight_decay=0.01,
    report_to='none',
    ## Change-2
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),    
    remove_unused_columns=True    
)


def tokenize_function(examples):
    ## Change-3 remove max_length from 2048 to 1024
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized


tokenized_dataset = dataset.map(tokenize_function, batched=True)

trainer = Trainer(
    model, args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
)
trainer.train()


# RESULT
# ======
#  deepspeed --num_gpus=2 experiment-base-finetune_qwen_1.5-0_5B.py
# [2024-10-01 01:17:59,171] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
# [2024-10-01 01:18:00,213] [WARNING] [runner.py:212:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
# [2024-10-01 01:18:00,213] [INFO] [runner.py:585:main] cmd = /tools/miniconda3/envs/finetune/bin/python -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMV19 --master_addr=127.0.0.1 --master_port=29500 --enable_each_rank_log=None experiment-base-finetune_qwen_1.5-0_5B.py
# [2024-10-01 01:18:01,109] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
# [2024-10-01 01:18:02,139] [INFO] [launch.py:146:main] WORLD INFO DICT: {'localhost': [0, 1]}
# [2024-10-01 01:18:02,139] [INFO] [launch.py:152:main] nnodes=1, num_local_procs=2, node_rank=0
# [2024-10-01 01:18:02,139] [INFO] [launch.py:163:main] global_rank_mapping=defaultdict(<class 'list'>, {'localhost': [0, 1]})
# [2024-10-01 01:18:02,139] [INFO] [launch.py:164:main] dist_world_size=2
# [2024-10-01 01:18:02,139] [INFO] [launch.py:168:main] Setting CUDA_VISIBLE_DEVICES=0,1
# [2024-10-01 01:18:02,140] [INFO] [launch.py:256:main] process 199296 spawned with command: ['/tools/miniconda3/envs/finetune/bin/python', '-u', 'experiment-base-finetune_qwen_1.5-0_5B.py', '--local_rank=0']
# [2024-10-01 01:18:02,140] [INFO] [launch.py:256:main] process 199297 spawned with command: ['/tools/miniconda3/envs/finetune/bin/python', '-u', 'experiment-base-finetune_qwen_1.5-0_5B.py', '--local_rank=1']
# Repo card metadata block was not found. Setting CardData to empty.
# Repo card metadata block was not found. Setting CardData to empty.
# -------------------------
# **compute_dtype= torch.bfloat16
# **attn_implementation= flash_attention_2
# -------------------------
# -------------------------
# **compute_dtype= torch.bfloat16
# **attn_implementation= flash_attention_2
# -------------------------
# [W1001 01:18:06.108184462 CUDAAllocatorConfig.h:28] Warning: expandable_segments not supported on this platform (function operator())
# [W1001 01:18:06.108367019 CUDAAllocatorConfig.h:28] Warning: expandable_segments not supported on this platform (function operator())
# [2024-10-01 01:18:07,202] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
# [2024-10-01 01:18:07,264] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
#   0%|                                                                                                                      | 0/9846 [00:00<?, ?it/s]/tools/miniconda3/envs/finetune/lib/python3.11/site-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
#   with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
# /tools/miniconda3/envs/finetune/lib/python3.11/site-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
#   with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
#   1%|▊                                                                                                                          | 63/9846 [01:2  1%|▋                                                                                                      | 64/9846 [01:22<3:29:02,  1.28s/it  1%|▋                                                                                                     | 65/9846 [01:23<3:29:01,  1.28s/it]  1%|▋     1%|▊                                                                                                        | 74/9846 [01:35<3:29:12,  1.28s/it]