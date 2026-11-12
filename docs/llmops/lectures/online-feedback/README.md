# A/B 选择与偏好采集

[项目说明](../../../../README.md) · [当前模块说明](../../README.md)

当前 A/B 页面用同一个已加载模型，按两个 temperature 分别生成回答，再随机打乱 A/B 展示位置。用户选择后，页面把更偏好的回答写入 chosen，另一条写入 rejected。

这验证的是“从回答比较到训练数据”的采集流程，不是生产环境中对两个独立模型版本分配用户流量的在线实验。采集文件默认位于 `data/llmops/dpo/prefs.jsonl`，导出的训练文件是 JSON 数组。

## 怎样验证

1. 先加载推理模型，记录问题与两个采样温度。
2. 检查显示的温度标签与回答，再选择更好的回答。
3. 核对保存的 prompt、chosen、rejected、时间戳及元数据，并抽查导出文件。

## 使用边界

同一问题多次选择不是独立用户效果证据。采集条数、偏好比例和后续模型胜率是不同指标，不能相互替代。

## 实现入口

- [src/search_engine/training_tab/llmops_tab.py](../../../../src/search_engine/training_tab/llmops_tab.py)
- [src/search_engine/training_tab/preference_collector.py](../../../../src/search_engine/training_tab/preference_collector.py)
- [src/search_engine/training_tab/inference_model.py](../../../../src/search_engine/training_tab/inference_model.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```mermaid
graph LR
    A[对话模型 v1.0] --> B[生产部署]
    B --> C[用户交互]
    C --> D[A/B 测试界面]
    D --> E{用户选择}
    E -->|模型 A 更好| F[收集偏好]
    E -->|模型 B 更好| F
    F --> G{数据是否充足？}
    G -->|否| C
    G -->|是| H[DPO 训练]
    H --> I[对话模型 v1.1]
    I --> J{效果检查}
    J -->|效果提升| B
    J -->|效果未提升| K[调整并重试]
    K --> H
```

### 示例 2

```python
# Core components
FEEDBACK_LOOP_COMPONENTS = {
    "inference_service": {
        "purpose": "Serve multiple model versions",
        "models": ["chat-v1.0", "chat-v1.1", "chat-v1.2"],
        "api": "generate(model_id, prompt)"
    },
    "ab_testing_ui": {
        "purpose": "Compare model outputs",
        "interface": "Gradio web interface",
        "features": ["side-by-side comparison", "user voting"]
    },
    "preference_collector": {
        "purpose": "Store user preferences",
        "storage": "data/llmops/dpo/prefs.jsonl",
        "format": "chosen/rejected pairs"
    },
    "dpo_trainer": {
        "purpose": "Train on collected preferences",
        "framework": "LLaMA-Factory",
        "frequency": "weekly/monthly"
    }
}
```

### 示例 3

```
问题：Python 和 Java 哪个更好？

┌──────────────────────┬──────────────────────────────┐
│ 模型 A（v1.0）        │ 模型 B（v1.1）                │
├──────────────────────┼──────────────────────────────┤
│ Python 更好。         │ Python 和 Java 各有优势。     │
│                      │ Python 擅长数据科学和 AI，    │
│                      │ Java 常用于企业应用……        │
│                      │                              │
│ [选择模型 A]          │ [选择模型 B]                  │
└──────────────────────┴──────────────────────────────┘

今日已收集偏好：47 条；目标：100 条。
```

### 示例 4

```python
COLLECTION_PLAN = {
    # Minimum viable
    "minimum": {
        "samples_per_week": 100,
        "training_frequency": "monthly",
        "expected_improvement": "small"
    },
    
    # Target goal
    "target": {
        "samples_per_week": 300,
        "training_frequency": "bi-weekly",
        "expected_improvement": "moderate"
    },
    
    # Optimal
    "optimal": {
        "samples_per_week": 500,
        "training_frequency": "weekly",
        "expected_improvement": "significant"
    }
}
```

### 示例 5

```python
QUESTION_CATEGORIES = {
    "factual_qa": [
        "What is the capital of France?",
        "Explain quantum computing."
    ],
    "comparison": [
        "Compare Python and Java.",
        "What's better: tabs or spaces?"
    ],
    "creative": [
        "Write a poem about AI.",
        "Create a story about a robot."
    ],
    "problem_solving": [
        "How to debug a segfault?",
        "Optimize this SQL query..."
    ],
    "explanation": [
        "Explain like I'm 5: blockchain",
        "How does HTTP work?"
    ]
}
```

### 示例 6

```python
def validate_preference_quality(chosen, rejected):
    """Ensure preference data is high quality"""
    
    checks = {
        # Chosen should be substantially better
        "length_difference": len(chosen) > len(rejected) * 0.5,
        
        # Responses should be different
        "sufficiently_different": similarity(chosen, rejected) < 0.8,
        
        # Both should be valid responses
        "both_valid": len(chosen) > 10 and len(rejected) > 10,
        
        # Chosen should be more detailed
        "more_detailed": count_sentences(chosen) > count_sentences(rejected)
    }
    
    return all(checks.values())
```

### 示例 7

```mermaid
gantt
    title Monthly DPO Iteration Cycle
    dateFormat YYYY-MM-DD
    section Week 1-2
    Collect preferences (target: 300)     :2023-11-01, 14d
    section Week 3
    Train DPO model                       :2023-11-15, 5d
    Evaluate new model                    :2023-11-20, 2d
    section Week 4
    Deploy to staging                     :2023-11-22, 2d
    Gradual production rollout            :2023-11-24, 5d
    Monitor metrics                       :2023-11-24, 7d
```

### 示例 8

```python
MODEL_NAMING = {
    "base": "Qwen/Qwen2-1.5B",
    "v1.0": "checkpoints/sft/sft-lora",           # SFT baseline
    "v1.1": "checkpoints/dpo/dpo-iter1-lora",     # First DPO iteration
    "v1.2": "checkpoints/dpo/dpo-iter2-lora",     # Second DPO iteration
    "v1.3": "checkpoints/dpo/dpo-iter3-lora"      # Third DPO iteration
}
```

### 示例 9

```python
# Store model metadata
MODEL_REGISTRY = {
    "v1.0": {
        "type": "SFT",
        "base_model": "Qwen/Qwen2-1.5B",
        "training_date": "2023-10-15",
        "training_data": "10K SFT samples",
        "metrics": {
            "user_satisfaction": 3.5,
            "avg_response_length": 120,
            "deployment_date": "2023-10-20"
        }
    },
    "v1.1": {
        "type": "DPO",
        "base_model": "v1.0",
        "training_date": "2023-11-15",
        "training_data": "300 preferences",
        "metrics": {
            "user_satisfaction": 3.8,
            "avg_response_length": 150,
            "improvement": "+8.6%",
            "deployment_date": "2023-11-22"
        }
    }
}
```

### 示例 10

```python
ROLLOUT_SCHEDULE = {
    "stage_1_canary": {
        "traffic": 0.05,      # 5% of users
        "duration": "24h",
        "rollback_if": "error_rate > 5% OR satisfaction < baseline"
    },
    "stage_2_small": {
        "traffic": 0.20,      # 20% of users
        "duration": "48h",
        "rollback_if": "error_rate > 3% OR satisfaction < baseline"
    },
    "stage_3_half": {
        "traffic": 0.50,      # 50% of users
        "duration": "72h",
        "rollback_if": "error_rate > 2% OR satisfaction < baseline"
    },
    "stage_4_full": {
        "traffic": 1.00,      # 100% of users
        "duration": "stable",
        "monitor": "continuously"
    }
}
```

### 示例 11

```python
def should_rollback(new_model_metrics, baseline_metrics):
    """Automatic rollback decision"""
    
    conditions = {
        "error_rate_high": new_model_metrics['error_rate'] > baseline_metrics['error_rate'] * 1.5,
        "latency_high": new_model_metrics['p99_latency'] > 2000,  # ms
        "satisfaction_drop": new_model_metrics['satisfaction'] < baseline_metrics['satisfaction'] * 0.9,
        "crash_rate_high": new_model_metrics['crash_rate'] > 0.01
    }
    
    return any(conditions.values())
```

### 示例 12

```python
TRACKING_METRICS = {
    # User Experience
    "user_satisfaction": {
        "source": "user ratings",
        "target": "> 4.0 / 5.0",
        "trend": "increasing"
    },
    
    # Model Performance
    "win_rate_vs_baseline": {
        "source": "A/B testing",
        "target": "> 55%",
        "measurement": "weekly"
    },
    
    # System Health
    "error_rate": {
        "source": "logs",
        "target": "< 1%",
        "alert_threshold": 2%
    },
    
    # Engagement
    "avg_session_length": {
        "source": "analytics",
        "target": "increasing",
        "measurement": "daily"
    }
}
```

### 示例 13

```python
def generate_monitoring_report(current_week):
    """Weekly monitoring report"""
    
    report = {
        "preferences_collected": 287,
        "target": 300,
        "progress": "95.7%",
        
        "model_comparison": {
            "v1.1_vs_v1.0": {
                "win_rate": 0.58,
                "user_preference": "+16%",
                "avg_length": "150 vs 120 tokens"
            }
        },
        
        "data_quality": {
            "clear_winners": 0.85,  # 85% have clear preference
            "category_coverage": 0.92,  # 92% categories covered
            "avg_quality_score": 4.2
        },
        
        "next_actions": [
            "Continue collecting (13 more preferences)",
            "DPO training scheduled for Nov 22",
            "Prepare staging environment"
        ]
    }
    
    return report
```

### 示例 14

```python
RECOMMENDED_CADENCE = {
    "startup_phase": {
        "frequency": "weekly",
        "reason": "Rapid iteration to find what works",
        "min_samples": 100
    },
    "growth_phase": {
        "frequency": "bi-weekly",
        "reason": "Balance iteration speed and stability",
        "min_samples": 200
    },
    "mature_phase": {
        "frequency": "monthly",
        "reason": "Stable model, incremental improvements",
        "min_samples": 300
    }
}
```

### 示例 15

```python
# Automate collection
AUTOMATION_STRATEGIES = [
    "Daily reminder to team members",
    "Batch testing sessions (30 min daily)",
    "Integrate into product workflow",
    "Incentivize user participation"
]
```

### 示例 16

```python
def diagnose_stagnation():
    checks = {
        "preference_quality": "Are preferences clear?",
        "data_diversity": "Covering all task types?",
        "sufficient_volume": "Have 100+ samples?",
        "hyperparameters": "Beta too high? LR too low?",
        "base_model": "Is SFT model good enough?"
    }
    return checks
```

### 示例 17

```python
# Match training to production
PRODUCTION_ALIGNMENT = {
    "use_real_queries": "Sample from production logs",
    "test_edge_cases": "Include unusual queries",
    "load_testing": "Ensure can handle traffic",
    "shadow_mode": "Deploy in parallel, don't serve yet"
}
```

### 示例 18

```python
# Test more than 2 models simultaneously
def multi_model_comparison(question, models):
    """Compare multiple model versions"""
    
    responses = {}
    for model_id in models:
        responses[model_id] = generate(model_id, question)
    
    # Pairwise comparisons
    pairs = [(m1, m2) for m1 in models for m2 in models if m1 < m2]
    
    for model_a, model_b in pairs:
        winner = user_vote(question, responses[model_a], responses[model_b])
        record_preference(question, responses, winner)
```

### 示例 19

```python
# Supplement human evaluation with automated metrics
from transformers import pipeline

evaluator = pipeline("text-classification", model="sentiment-analysis")

def auto_evaluate(response):
    """Automated quality scoring"""
    
    scores = {
        "length": len(response.split()),
        "sentiment": evaluator(response)[0]['score'],
        "coherence": calculate_coherence(response),
        "factuality": check_facts(response)
    }
    
    return weighted_average(scores)
```

### 示例 20

```python
# Generate synthetic preferences when real data is scarce
def augment_preferences(real_preferences):
    """Carefully augment with synthetic data"""
    
    synthetic = []
    
    for pref in real_preferences:
        # Paraphrase question
        para_question = paraphrase(pref['question'])
        
        # Generate new responses
        response_a = generate(model_a, para_question)
        response_b = generate(model_b, para_question)
        
        # Auto-evaluate (use cautiously!)
        if auto_evaluate(response_a) > auto_evaluate(response_b):
            synthetic.append({
                "question": para_question,
                "chosen": response_a,
                "rejected": response_b
            })
    
    # Mix: 80% real, 20% synthetic
    return real_preferences + synthetic[:len(real_preferences)//4]
```

### 示例 21

```python
IMPROVEMENT_TRAJECTORY = {
    "v1.0_baseline": {
        "satisfaction": 3.5,
        "win_rate": None,
        "avg_length": 120
    },
    "v1.1_week1": {
        "satisfaction": 3.8,
        "win_rate": 0.58,
        "avg_length": 150,
        "improvement": "+8.6%"
    },
    "v1.2_week2": {
        "satisfaction": 4.0,
        "win_rate": 0.62,
        "avg_length": 165,
        "improvement": "+14.3%"
    },
    "v1.3_week3": {
        "satisfaction": 4.2,
        "win_rate": 0.65,
        "avg_length": 175,
        "improvement": "+20%"
    }
}
```

</details>
