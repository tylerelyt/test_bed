#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本使用示例
演示如何使用Intelligent Search Engine的核心功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from search_engine.service_manager import (
    get_data_service,
    get_index_service,
    get_model_service
)
from search_engine.data_utils import (
    record_search_impression,
    record_document_click,
    get_data_statistics,
    analyze_click_patterns
)


def main():
    """基本使用示例"""
    print("🔍 Intelligent Search Engine - 基本使用示例")
    print("=" * 50)
    
    # 1. 获取服务实例
    print("\n1. 初始化服务...")
    data_service = get_data_service()
    index_service = get_index_service()
    model_service = get_model_service()
    print("✅ 服务初始化完成")
    
    # 2. 添加示例文档
    print("\n2. 添加示例文档...")
    documents = {
        "doc1": "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "doc2": "机器学习是人工智能的一个子领域，专注于算法的设计，这些算法可以从数据中学习并做出预测或决策。",
        "doc3": "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。",
        "doc4": "自然语言处理是人工智能的一个重要分支，旨在让计算机能够理解、解释和生成人类语言。",
        "doc5": "计算机视觉是人工智能的一个领域，致力于让计算机能够识别和理解图像和视频内容。"
    }
    
    # 批量添加文档
    added_count = index_service.batch_add_documents(documents)
    print(f"✅ 成功添加 {added_count} 个文档")
    
    # 3. 执行搜索
    print("\n3. 执行搜索...")
    query = "人工智能"
    request_id = "example_request_001"
    
    # 召回阶段
    doc_ids = index_service.retrieve(query, top_k=10)
    print(f"📋 召回文档: {doc_ids}")
    
    # 排序阶段
    ranked_results = index_service.rank(query, doc_ids, top_k=5)
    print(f"📊 排序结果: {len(ranked_results)} 个文档")
    
    # 4. 记录用户行为
    print("\n4. 记录用户行为...")
    
    # 记录展示事件
    for position, result in enumerate(ranked_results, 1):
        doc_id, score, summary = result[:3]  # 解析结果
        
        # 使用工具函数记录展示
        record_search_impression(
            query=query,
            doc_id=doc_id,
            position=position,
            score=score,
            summary=summary,
            request_id=request_id
        )
        
        print(f"  📄 位置{position}: {doc_id} (分数: {score:.3f})")
    
    # 模拟用户点击第一个结果
    clicked_doc_id = ranked_results[0][0]
    success = record_document_click(clicked_doc_id, request_id)
    print(f"👆 用户点击文档: {clicked_doc_id} {'✅' if success else '❌'}")
    
    # 5. 查看数据统计
    print("\n5. 数据统计...")
    stats = get_data_statistics()
    print(f"📊 总样本数: {stats['total_samples']}")
    print(f"📊 总点击数: {stats['total_clicks']}")
    print(f"📊 点击率: {stats['click_rate']:.2%}")
    print(f"📊 唯一查询数: {stats['unique_queries']}")
    print(f"📊 唯一文档数: {stats['unique_docs']}")
    
    # 6. 点击模式分析
    print("\n6. 点击模式分析...")
    patterns = analyze_click_patterns()
    if 'error' not in patterns:
        print(f"🔍 整体CTR: {patterns['overall_ctr']:.2%}")
        print(f"🔍 总展示数: {patterns['total_impressions']}")
        print(f"🔍 总点击数: {patterns['total_clicks']}")
        
        # 位置分析
        if patterns.get('position_analysis'):
            print("📈 位置CTR分析:")
            for pos, stats in patterns['position_analysis'].items():
                if isinstance(stats, dict) and 'mean' in stats:
                    print(f"  位置{pos}: CTR={stats['mean']:.2%}")
    
    # 7. 模型训练
    print("\n7. 模型训练...")
    
    # 验证训练数据
    validation = model_service.validate_training_data(data_service)
    if validation['valid']:
        print("✅ 训练数据验证通过")
        
        # 训练模型
        result = model_service.train_model(data_service)
        if result['success']:
            print("✅ 模型训练成功")
            
            # 获取模型信息
            model_info = model_service.get_model_info()
            print(f"🤖 模型状态: {'已训练' if model_info['is_trained'] else '未训练'}")
            print(f"🤖 训练时间: {model_info.get('training_time', 'N/A')}")
        else:
            print(f"❌ 模型训练失败: {result.get('error', '未知错误')}")
    else:
        print("❌ 训练数据验证失败:")
        for issue in validation['issues']:
            print(f"  - {issue}")
        print("💡 建议:")
        for rec in validation['recommendations']:
            print(f"  - {rec}")
    
    # 8. 获取索引统计
    print("\n8. 索引统计...")
    index_stats = index_service.get_stats()
    print(f"📚 总文档数: {index_stats.get('total_documents', 0)}")
    print(f"📚 总词汇数: {index_stats.get('total_terms', 0)}")
    print(f"📚 平均文档长度: {index_stats.get('avg_doc_length', 0):.1f}")
    
    print("\n🎉 示例执行完成！")
    print("💡 提示: 运行 'python start_system.py' 启动Web界面")


if __name__ == "__main__":
    main() 