#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTR演示数据生成器
生成更真实、多样化的CTR数据用于模型训练
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

def generate_realistic_ctr_data(num_records: int = 20) -> List[Dict[str, Any]]:
    """生成更真实的CTR数据"""
    
    # 真实的查询词
    queries = [
        "人工智能", "机器学习", "深度学习", "神经网络", "自然语言处理",
        "计算机视觉", "强化学习", "知识图谱", "数据挖掘", "推荐系统",
        "搜索引擎", "图像识别", "语音识别", "机器翻译", "情感分析"
    ]
    
    # 真实的文档ID
    doc_ids = [f"doc{i}" for i in range(1, 21)]
    
    # 真实的摘要内容（避免模板化）
    summaries = [
        "人工智能技术正在快速发展，在各个领域都有广泛应用。",
        "机器学习算法能够从数据中学习模式，实现自动化决策。",
        "深度学习通过多层神经网络模拟人脑的学习过程。",
        "神经网络是机器学习的重要基础，具有强大的模式识别能力。",
        "自然语言处理让计算机能够理解和生成人类语言。",
        "计算机视觉技术使机器能够看到并理解图像内容。",
        "强化学习通过试错机制让智能体学会最优策略。",
        "知识图谱将信息组织成结构化的知识网络。",
        "数据挖掘从大量数据中发现有价值的信息和模式。",
        "推荐系统根据用户偏好提供个性化的内容推荐。",
        "搜索引擎技术帮助用户快速找到相关信息。",
        "图像识别技术能够识别和分类图像中的对象。",
        "语音识别将人类语音转换为可处理的文本数据。",
        "机器翻译技术实现不同语言之间的自动翻译。",
        "情感分析技术能够识别文本中的情感倾向。",
        "深度学习在图像处理领域取得了突破性进展。",
        "神经网络在自然语言处理任务中表现优异。",
        "机器学习算法在推荐系统中发挥重要作用。",
        "人工智能技术在医疗诊断中具有广阔前景。",
        "计算机视觉在自动驾驶技术中至关重要。"
    ]
    
    ctr_data = []
    base_time = datetime.now() - timedelta(hours=2)
    
    for i in range(num_records):
        # 随机选择查询和文档
        query = random.choice(queries)
        doc_id = random.choice(doc_ids)
        summary = random.choice(summaries)
        
        # 位置特征（1-10，但更真实）
        position = random.choices(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            weights=[0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01, 0.01]  # 位置越靠前权重越高
        )[0]
        
        # 相似度分数（更真实的范围）
        score = round(random.uniform(0.1, 0.95), 4)
        
        # 点击概率（基于位置和分数的真实CTR模型）
        position_decay = 1.0 / (position ** 0.5)  # 位置衰减
        score_boost = score * 0.3  # 分数提升
        base_ctr = 0.1  # 基础点击率
        
        click_prob = min(0.8, base_ctr + score_boost + position_decay * 0.2)
        
        # 添加随机噪声
        click_prob += random.uniform(-0.1, 0.1)
        click_prob = max(0.01, min(0.9, click_prob))
        
        # 根据概率决定是否点击
        clicked = 1 if random.random() < click_prob else 0
        
        # 时间戳（递增）
        timestamp = base_time + timedelta(minutes=i*5 + random.randint(0, 10))
        
        record = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "doc_id": doc_id,
            "position": position,
            "score": score,
            "clicked": clicked,
            "summary": summary
        }
        
        ctr_data.append(record)
    
    return ctr_data

def save_demo_data(data: List[Dict[str, Any]], filename: str = 'data/ctr_data.json'):
    """保存演示数据到文件"""
    output_data = {
        'records': data,
        'total_records': len(data),
        'total_clicks': sum(record['clicked'] for record in data),
        'overall_ctr': sum(record['clicked'] for record in data) / len(data) if data else 0
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 演示数据已保存到 {filename}")

def main():
    """主函数"""
    print("🎯 CTR演示数据生成器")
    print("=" * 40)
    
    try:
        # 获取用户输入
        user_input = input("请输入要生成的记录数 (默认20): ").strip()
        if user_input:
            num_records = int(user_input)
        else:
            num_records = 20
        
        if num_records < 10:
            print("⚠️  建议至少生成10条记录以确保数据质量")
        num_records = 10
    
    print(f"\n🔄 正在生成 {num_records} 条演示数据...")
    
    # 生成数据
        demo_data = generate_realistic_ctr_data(num_records)
    
    # 保存数据
    save_demo_data(demo_data)
    
        # 统计信息
        total_clicks = sum(record['clicked'] for record in demo_data)
        overall_ctr = total_clicks / len(demo_data)
        
        print(f"📊 数据统计:")
        print(f"   - 总记录数: {len(demo_data)}")
        print(f"   - 点击次数: {total_clicks}")
        print(f"   - 整体CTR: {overall_ctr:.4f}")
        
        # 数据质量检查
        unique_queries = len(set(record['query'] for record in demo_data))
        unique_docs = len(set(record['doc_id'] for record in demo_data))
        unique_positions = len(set(record['position'] for record in demo_data))
        
        print(f"📈 数据质量:")
        print(f"   - 不同查询数: {unique_queries}")
        print(f"   - 不同文档数: {unique_docs}")
        print(f"   - 不同位置数: {unique_positions}")
        
        if unique_queries >= 3 and unique_docs >= 3 and unique_positions >= 3:
            print("✅ 数据质量良好，可以用于模型训练")
        else:
            print("⚠️  数据多样性不足，建议生成更多数据")
        
    print("\n💡 使用说明:")
        print("1. 启动UI界面: python ui/ui_interface.py")
    print("2. 查看历史记录，确认数据已加载")
    print("3. 点击'训练CTR模型'按钮")
    print("4. 观察特征权重和模型性能")
    
    print("\n🎉 演示数据生成完成！")
        
    except ValueError:
        print("❌ 请输入有效的数字")
    except Exception as e:
        print(f"❌ 生成数据失败: {e}")

if __name__ == "__main__":
    main() 