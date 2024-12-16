### BASE Training Script LLAMA 3-2
## 1xnode with 2xGPU - each GPU vRAM 16GB
## OOM

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
import torch

# from huggingface_hub import login
# login(
#   token="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", # ADD YOUR TOKEN HERE
# )


# dataset_source = "timdettmers/openassistant-guanaco"
# dataset = load_dataset(dataset_source)
dataset = load_dataset("timdettmers/openassistant-guanaco")
base_model = "meta-llama/Llama-3.2-1B-Instruct"

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
# [rank0]:   File "/tools/miniconda3/envs/finetune/lib/python3.11/site-packages/torch/optim/adamw.py", line 600, in _multi_tensor_adamw
# [rank0]:     exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
# [rank0]:                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# [rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB. GPU 0 has a total capacity of 15.70 GiB of which 18.81 MiB is free. Process 198907 has 2.42 GiB memory in use. Including non-PyTorch memory, this process has 13.25 GiB memory in use. Of the allocated memory 13.00 GiB is allocated by PyTorch, and 7.33 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
#   0%|          | 0/9846 [00:03<?, ?it/s]                                                                                                            
# [2024-10-01 01:17:14,554] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 198906
# [2024-10-01 01:17:14,554] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 198907
# [2024-10-01 01:17:14,772] [ERROR] [launch.py:325:sigkill_handler] ['/tools/miniconda3/envs/finetune/bin/python', '-u', 'experiment-base-finetune_llama_3-2_1B.py', '--local_rank=1'] exits with return code = 1
