# distributed_train_finetune
Experiement with LLM Distributed Train and Fine-Tuning 


## DeepSpeed Zero's Experiments

### setup
export:
`conda env export --no-builds > environment.yml`

Import:
`conda env create --name finetune --file=environments.yml`

#### check model memory requirements
- calculate LLM model memory requirement

##### Train Llama 3.2- 1B and 3B
- Zero-1
- Zero-2
- Zero-3
  