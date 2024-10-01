
from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

result = estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)

print(result)
