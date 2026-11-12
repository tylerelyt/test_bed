# 从偏好反馈到模型迭代

[项目说明](../../../../README.md) · [当前模块说明](../../README.md)

一次完整迭代应能追溯：原模型与问题 → 回答及人工偏好 → 导出数据 → 训练配置与产物 → 独立评估。先让这一条链路可复现，再讨论自动化频率。

工作台提供各步骤的操作入口，但训练和评估仍需显式执行。CTR 的新增点击触发机制与大语言模型 DPO 流程独立，不能把前者描述成全平台自动自进化。

## 怎样验证

1. 为一次迭代保留输入模型、数据文件与输出目录。
2. 比较候选模型与原模型，记录错误类型及选择依据。
3. 确认要使用哪个模型后，再在相应推理入口加载；保留前一版本。

## 使用边界

每周训练、模型胜率自动提升、灰度发布和自动回滚是可设计的流程，不是本仓库已验证的运行事实。

## 实现入口

- [src/search_engine/training_tab/llmops_tab.py](../../../../src/search_engine/training_tab/llmops_tab.py)
- [src/search_engine/training_tab/preference_collector.py](../../../../src/search_engine/training_tab/preference_collector.py)
- [src/search_engine/training_tab/llamafactory_trainer.py](../../../../src/search_engine/training_tab/llamafactory_trainer.py)
- [src/search_engine/data_service.py](../../../../src/search_engine/data_service.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```mermaid
graph TB
    A[模型 v1.0<br/>生产环境] --> B[用户交互]
    B --> C[收集反馈]
    C --> D[偏好数据集]
    D --> E[DPO 训练]
    E --> F[模型 v1.1]
    F --> G{A/B 测试}
    G -->|更好| H[部署 v1.1]
    G -->|更差| I[分析与调试]
    H --> B
    I --> E
    
    style H fill:#90EE90
    style G fill:#FFD700
```

### 示例 2

```mermaid
graph TB
    subgraph Production["生产环境"]
        A[负载均衡器] --> B[模型 A<br/>当前版本]
        A --> C[模型 B<br/>候选版本]
        B --> D[日志记录器]
        C --> D
    end
    
    subgraph DataPipeline["数据处理流程"]
        D --> E[交互数据库]
        E --> F[采样]
        F --> G[标注队列]
    end
    
    subgraph Training["训练流程"]
        G --> H[偏好数据]
        H --> I[DPO 训练]
        I --> J[新模型]
    end
    
    subgraph Evaluation["评估与部署"]
        J --> K[离线评估]
        K --> L[在线 A/B 测试]
        L --> M{决策}
        M -->|部署| C
        M -->|拒绝发布| N[归档]
    end
```

### 示例 3

```python
interaction_log = {
    "interaction_id": "uuid-string",
    "timestamp": "2024-01-15T10:30:00Z",
    "model_version": "v1.0",
    "user_id": "user-123",
    "conversation": [
        {"role": "user", "content": "Explain quantum computing"},
        {"role": "assistant", "content": "Quantum computing uses..."}
    ],
    "metadata": {
        "latency_ms": 250,
        "tokens_generated": 150
    }
}
```

### 示例 4

```python
import json
import sqlite3
from datetime import datetime

class InteractionLogger:
    def __init__(self, db_path="data/llmops/interactions.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                timestamp DATETIME,
                model_version TEXT,
                user_id TEXT,
                conversation TEXT,
                feedback_score INTEGER,
                metadata TEXT
            )
        """)
    
    def log_interaction(self, interaction):
        self.conn.execute("""
            INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            interaction['interaction_id'],
            interaction['timestamp'],
            interaction['model_version'],
            interaction['user_id'],
            json.dumps(interaction['conversation']),
            interaction.get('feedback_score'),
            json.dumps(interaction.get('metadata', {}))
        ))
        self.conn.commit()
    
    def get_interactions(self, start_date, end_date, model_version=None):
        query = """
            SELECT * FROM interactions 
            WHERE timestamp BETWEEN ? AND ?
        """
        params = [start_date, end_date]
        
        if model_version:
            query += " AND model_version = ?"
            params.append(model_version)
        
        return self.conn.execute(query, params).fetchall()
```

### 示例 5

```python
def collect_implicit_feedback(interactions):
    """Infer quality from user behavior"""
    
    scored_interactions = []
    
    for interaction in interactions:
        score = 0
        
        # Positive signals
        if interaction.get('user_continued'):  # User kept chatting
            score += 2
        if interaction.get('session_duration') > 300:  # > 5 minutes
            score += 1
        if interaction.get('copied_response'):  # User copied output
            score += 2
        
        # Negative signals
        if interaction.get('regenerated'):  # User hit regenerate
            score -= 2
        if interaction.get('session_duration') < 30:  # < 30 seconds
            score -= 1
        if interaction.get('reported'):  # User reported
            score -= 5
        
        interaction['implicit_score'] = score
        scored_interactions.append(interaction)
    
    return scored_interactions
```

### 示例 6

```python
import gradio as gr

def create_feedback_ui():
    """UI for collecting explicit user feedback"""
    
    def save_feedback(conversation_id, rating, comment):
        feedback = {
            "conversation_id": conversation_id,
            "rating": rating,  # 1-5 stars
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }
        
        save_to_db(feedback)
        return "Thank you for your feedback!"
    
    with gr.Blocks() as demo:
        gr.Markdown("## Rate this conversation")
        
        rating = gr.Slider(1, 5, step=1, label="Rating")
        comment = gr.Textbox(label="Comments (optional)", lines=3)
        submit_btn = gr.Button("Submit Feedback")
        
        output = gr.Textbox(label="Status")
        
        submit_btn.click(
            save_feedback,
            inputs=[rating, comment],
            outputs=[output]
        )
    
    return demo
```

### 示例 7

```python
def sample_for_annotation(interactions, target_count=1000):
    """Sample diverse, high-value interactions for annotation"""
    
    # Priority factors
    def priority_score(interaction):
        score = 0
        
        # Diverse topics
        if is_new_topic(interaction):
            score += 3
        
        # User engagement
        if interaction['implicit_score'] > 0:
            score += 2
        
        # Model uncertainty (if available)
        if interaction.get('uncertainty', 0) > 0.5:
            score += 2
        
        # Edge cases
        if is_edge_case(interaction):
            score += 3
        
        return score
    
    # Score and sort
    scored = [(priority_score(i), i) for i in interactions]
    scored.sort(reverse=True, key=lambda x: x[0])
    
    # Sample top + random for diversity
    top_n = int(target_count * 0.7)
    random_n = target_count - top_n
    
    sampled = [i for _, i in scored[:top_n]]
    sampled += random.sample([i for _, i in scored[top_n:]], random_n)
    
    return sampled
```

### 示例 8

```python
def create_annotation_ui(sampled_interactions):
    """UI for annotators to create preference pairs"""
    
    def generate_alternative(prompt, original_response):
        """Generate alternative response for comparison"""
        # Could use different model or different sampling params
        alternative = stronger_model.generate(prompt, temperature=0.7)
        return alternative
    
    def save_preference(interaction_id, prompt, response_a, response_b, choice):
        if choice == "A":
            chosen, rejected = response_a, response_b
        else:
            chosen, rejected = response_b, response_a
        
        preference = {
            "conversations": [{"from": "user", "value": prompt}],
            "chosen": chosen,
            "rejected": rejected,
            "source": "human_annotation",
            "interaction_id": interaction_id
        }
        
        with open('data/llmops/dpo/online_preferences.jsonl', 'a') as f:
            f.write(json.dumps(preference) + '\n')
    
    with gr.Blocks() as demo:
        state = gr.State({"current_idx": 0})
        
        with gr.Row():
            prompt_display = gr.Textbox(label="User Prompt", interactive=False)
        
        with gr.Row():
            with gr.Column():
                response_a = gr.Textbox(label="Response A", lines=10)
                choose_a = gr.Button("Choose A")
            
            with gr.Column():
                response_b = gr.Textbox(label="Response B", lines=10)
                choose_b = gr.Button("Choose B")
        
        equal_btn = gr.Button("Both Equal / Both Bad")
        skip_btn = gr.Button("Skip")
        
        progress = gr.Textbox(label="Progress")
        
        # Button callbacks...
    
    return demo
```

### 示例 9

```python
import subprocess
import yaml

def trigger_dpo_training(preference_file, base_model):
    """Automatically trigger DPO training with new preferences"""
    
    # Create config
    config = {
        "model_name_or_path": base_model,
        "stage": "dpo",
        "dataset": "online_preferences",
        "output_dir": f"checkpoints/dpo/{datetime.now().strftime('%Y%m%d_%H%M')}",
        "learning_rate": 5e-7,
        "num_train_epochs": 1,
        # ... other params
    }
    
    config_path = "config/auto_dpo.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Launch training
    subprocess.run([
        "llamafactory-cli", "train",
        "--config_file", config_path
    ])
    
    return config['output_dir']
```

### 示例 10

```python
class ABTestRouter:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a  # Current version
        self.model_b = model_b  # New candidate
        self.split_ratio = split_ratio
    
    def route_request(self, user_id, prompt):
        """Route user to model A or B"""
        
        # Consistent hashing for user stickiness
        user_hash = hash(user_id)
        
        if (user_hash % 100) < (self.split_ratio * 100):
            model = self.model_a
            version = "A"
        else:
            model = self.model_b
            version = "B"
        
        response = model.generate(prompt)
        
        # Log for analysis
        log_ab_test(user_id, version, prompt, response)
        
        return response, version
```

### 示例 11

```python
def analyze_ab_test(start_date, end_date):
    """Analyze A/B test results"""
    
    results_a = get_interactions(start_date, end_date, model_version="A")
    results_b = get_interactions(start_date, end_date, model_version="B")
    
    metrics = {
        "model_a": calculate_metrics(results_a),
        "model_b": calculate_metrics(results_b)
    }
    
    # Statistical significance
    metrics['significance'] = ttest_ind(
        [r['feedback_score'] for r in results_a],
        [r['feedback_score'] for r in results_b]
    )
    
    return metrics

def calculate_metrics(interactions):
    """Calculate performance metrics"""
    return {
        "avg_rating": mean([i['feedback_score'] for i in interactions]),
        "engagement_rate": sum(i['user_continued'] for i in interactions) / len(interactions),
        "avg_session_duration": mean([i['session_duration'] for i in interactions]),
        "regeneration_rate": sum(i['regenerated'] for i in interactions) / len(interactions)
    }
```

### 示例 12

```python
def quality_gate_check(new_model, baseline_model):
    """Check if new model passes quality gates"""
    
    checks = {
        "offline_eval": offline_eval(new_model) > offline_eval(baseline_model),
        "safety_check": safety_score(new_model) > 0.95,
        "ab_test": ab_test_winner(new_model, baseline_model),
        "regression_test": no_regressions(new_model, baseline_model)
    }
    
    passed = all(checks.values())
    
    return passed, checks
```

### 示例 13

```bash
# Export last week's interactions
python scripts/export_interactions.py \
    --start-date 2024-01-08 \
    --end-date 2024-01-14 \
    --output data/week_01_interactions.jsonl

# Output: 15,234 interactions collected
```

### 示例 14

```bash
# Sample for annotation
python scripts/sample_for_annotation.py \
    --input data/week_01_interactions.jsonl \
    --output data/week_01_sampled.jsonl \
    --count 1000

# Launch annotation UI
python scripts/annotation_ui.py \
    --input data/week_01_sampled.jsonl \
    --output data/llmops/dpo/week_01_preferences.jsonl

# Annotators work throughout the day
# Output: 856 preference pairs created
```

### 示例 15

```bash
# Train DPO
llamafactory-cli train \
    --config_file config/online_dpo.yaml

# Training takes ~6 hours on 4x A100
# Output: checkpoints/dpo/v1.1
```

### 示例 16

```bash
# Offline evaluation
python scripts/evaluate.py \
    --model checkpoints/dpo/v1.1 \
    --benchmarks mtbench,alpacaeval

# Results:
# MT-Bench: 7.8 → 8.1 (+0.3) ✓
# AlpacaEval: 82% → 85% (+3%) ✓

# Start A/B test
python scripts/start_ab_test.py \
    --model-a checkpoints/v1.0 \
    --model-b checkpoints/dpo/v1.1 \
    --split 50/50 \
    --duration 48h
```

### 示例 17

```bash
# Analyze A/B test
python scripts/analyze_ab_test.py \
    --test-id week_01

# Results:
# Model B win rate: 58% (p < 0.01) ✓
# User engagement: +12% ✓
# Safety score: 0.97 (no change) ✓

# Deploy
python scripts/deploy.py \
    --model checkpoints/dpo/v1.1 \
    --version v1.1 \
    --traffic 100

# v1.1 now serving 100% of traffic
```

</details>
