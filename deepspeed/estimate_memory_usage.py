from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

#model_id = "meta-llama/Llama-3.2-3B-Instruct"
model_id = "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModel.from_pretrained(model_id)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)
