---
library_name: peft
license: other
base_model: Qwen/Qwen2-0.5B
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: qwen-0.5b-cpt
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen-0.5b-cpt

This model is a fine-tuned version of [Qwen/Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B) on the test_corpus_large dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 2
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 1

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.49.0
- Pytorch 2.9.1
- Datasets 3.2.0
- Tokenizers 0.21.0