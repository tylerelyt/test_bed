# 指令数据与监督微调

[项目说明](../../../../README.md) · [当前模块说明](../../README.md)

监督微调使用输入与期望回答样本训练模型遵循指令。数据质量包含任务覆盖、回答正确性、格式一致性和重复控制，不能用样本条数替代质量判断。

当前页面的指令生成调用 `generate_instructions(..., use_mock=True)`。生成器的 `_api_generate` 也仍返回模拟数据，因此这些输出用于联通数据格式与训练流程，不应描述成已完成真实大模型扩写。

## 怎样验证

1. 抽查 instruction、input、output 或导出的对话结构，确认内容确实符合任务。
2. 核对数据集注册与训练模板，避免把模拟样本当作正式训练集。
3. 完成训练后在未参与训练的指令上对照基础模型。

## 使用边界

当前模拟生成不会仅因配置了 API 密钥就变为真实模型生成；需要补充实现并验证该调用后，文档才能更新此项能力。

## 实现入口

- [src/search_engine/training_tab/self_instruct_generator.py](../../../../src/search_engine/training_tab/self_instruct_generator.py)
- [src/search_engine/training_tab/llmops_tab.py](../../../../src/search_engine/training_tab/llmops_tab.py)
- [src/search_engine/training_tab/llama_factory_config.py](../../../../src/search_engine/training_tab/llama_factory_config.py)
- [src/search_engine/training_tab/llamafactory_trainer.py](../../../../src/search_engine/training_tab/llamafactory_trainer.py)

## 公式参考

以下保留原有公式。变量、特征与数据分布须结合当前实现解释，公式本身不代表已经完成对应评估。

$$
\mathcal{L}_{\text{SFT}} = -\sum_{i=1}^{N} \log P(y_i \mid x_i; \theta)
$$

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```mermaid
graph LR
    A[基础模型] --> B[CPT<br/>可选]
    B --> C[SFT<br/>指令微调]
    C --> D[对话模型]
    D --> E[DPO<br/>可选]
    E --> F[对齐后的模型]

    style C fill:#90EE90
```

### 示例 2

```json
{
  "instruction": "Write a Python function to calculate factorial",
  "input": "",
  "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n-1)"
}
```

### 示例 3

```json
{
  "conversations": [
    {"from": "user", "value": "What is machine learning?"},
    {"from": "assistant", "value": "Machine learning is a branch of AI..."}
  ]
}
```

### 示例 4

```json
{
  "conversations": [
    {"from": "user", "value": "Hello!"},
    {"from": "assistant", "value": "Hi! How can I help you today?"},
    {"from": "user", "value": "Tell me about Python"},
    {"from": "assistant", "value": "Python is a high-level programming language..."}
  ]
}
```

### 示例 5

```yaml
# SFT Configuration Example
model_name_or_path: checkpoints/cpt/domain-model  # or base model
stage: sft  # supervised fine-tuning
do_train: true
dataset: alpaca_en,self_cognition
template: qwen  # chat template

# Data processing
cutoff_len: 2048
max_samples: 50000
overwrite_cache: true

# Training hyperparameters
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1

# Optimization
fp16: true
gradient_checkpointing: true

# LoRA configuration
finetuning_type: lora
lora_target: all
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.1

# Evaluation
val_size: 0.1
eval_strategy: steps
eval_steps: 500

# Output
output_dir: checkpoints/sft/chat-model
logging_steps: 10
save_steps: 1000
save_total_limit: 3
```

### 示例 6

```python
import json
from openai import OpenAI

client = OpenAI()

def generate_instruction_data(seed_examples, num_samples=100):
    """Generate instruction-following examples using GPT-4"""
    
    generated_data = []
    
    for i in range(num_samples):
        # Create prompt with seed examples
        prompt = f"""Generate a diverse instruction-following example.

Seed examples:
{json.dumps(seed_examples[:3], indent=2)}

Create a NEW example with:
1. A clear instruction
2. Optional input context
3. A helpful, detailed output

Format as JSON with keys: instruction, input, output"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0
        )
        
        try:
            example = json.loads(response.choices[0].message.content)
            generated_data.append(example)
        except:
            continue
    
    return generated_data

# Generate data
seed_data = [
    {
        "instruction": "Explain quantum computing",
        "input": "",
        "output": "Quantum computing uses quantum mechanics..."
    },
    # More seed examples...
]

sft_data = generate_instruction_data(seed_data, num_samples=1000)

# Save to file
with open('data/llmops/sft/generated_instructions.json', 'w') as f:
    json.dump(sft_data, f, indent=2, ensure_ascii=False)
```

### 示例 7

```json
{
  "custom_instructions": {
    "file_name": "sft/generated_instructions.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

### 示例 8

```bash
llamafactory-cli train \
    --config_file config/sft_config.yaml
```

### 示例 9

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
llamafactory-cli train \
    --config_file config/sft_config.yaml \
    --deepspeed config/ds_zero2.json
```

### 示例 10

```bash
tensorboard --logdir=checkpoints/sft/chat-model/runs
```

### 示例 11

```python
def assess_quality(example):
    scores = {}
    
    # Instruction clarity (1-5)
    scores['clarity'] = rate_clarity(example['instruction'])
    
    # Response helpfulness (1-5)
    scores['helpfulness'] = rate_helpfulness(example['output'])
    
    # Formatting correctness (1-5)
    scores['formatting'] = check_format(example)
    
    # Overall quality
    return sum(scores.values()) / len(scores)

# Filter high-quality examples
quality_threshold = 4.0
filtered_data = [
    ex for ex in sft_data 
    if assess_quality(ex) >= quality_threshold
]
```

### 示例 12

```python
# Evaluate on held-out test set
eval_results = trainer.evaluate()

print(f"Eval Loss: {eval_results['eval_loss']:.4f}")
print(f"Eval Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
```

### 示例 13

```python
evaluation_template = {
    "instruction": "...",
    "model_output": "...",
    "ratings": {
        "helpfulness": 0,  # 1-5
        "accuracy": 0,
        "clarity": 0,
        "safety": 0
    },
    "preferred_response": "..."  # Optional
}
```

### 示例 14

```yaml
# Increase regularization
lora_dropout: 0.15

# Early stopping
save_strategy: steps
eval_steps: 500
load_best_model_at_end: true

# More diverse data
max_samples: null  # Use full dataset
```

### 示例 15

   ```yaml
   dataset: alpaca_en,sharegpt,general_qa
   ```

### 示例 16

   ```yaml
   learning_rate: 2.0e-5  # Instead of 5e-5
   ```

### 示例 17

   ```yaml
   num_train_epochs: 1.0  # Instead of 3.0
   ```

### 示例 18

   ```yaml
   template: qwen  # or llama3, chatml
   ```

### 示例 19

   ```json
   {
     "conversations": [
       {"from": "system", "value": "You are a helpful assistant."},
       {"from": "user", "value": "..."},
       {"from": "assistant", "value": "..."}
     ]
   }
   ```

### 示例 20

```yaml
dataset: >
  alpaca_en,  # General instructions
  code_alpaca,  # Code tasks
  math_reasoning,  # Math problems
  qa_pairs  # Question answering
```

### 示例 21

```python
# Sort data by complexity
sorted_data = sorted(sft_data, key=lambda x: estimate_complexity(x))

# Create curriculum splits
easy_split = sorted_data[:len(sorted_data)//3]
medium_split = sorted_data[len(sorted_data)//3:2*len(sorted_data)//3]
hard_split = sorted_data[2*len(sorted_data)//3:]

# Train in stages
for split_name, split_data in [("easy", easy_split), 
                                ("medium", medium_split), 
                                ("hard", hard_split)]:
    print(f"Training on {split_name} examples...")
    train_sft(split_data)
```

### 示例 22

```python
def augment_instruction(example):
    """Create variations of the instruction"""
    variations = []
    
    # Rephrase instruction
    variations.append({
        "instruction": rephrase(example['instruction']),
        "input": example['input'],
        "output": example['output']
    })
    
    # Add constraints
    variations.append({
        "instruction": f"{example['instruction']} (Be concise)",
        "input": example['input'],
        "output": shorten(example['output'])
    })
    
    return variations
```

</details>
