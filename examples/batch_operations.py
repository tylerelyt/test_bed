#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量操作示例
演示如何使用批量操作提高数据处理效率
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from search_engine.service_manager import get_data_service
from search_engine.data_utils import get_data_statistics
import uuid
from datetime import datetime


def generate_sample_data(num_queries=5, num_docs_per_query=10):
    """生成示例数据"""
    queries = [
        "人工智能",
        "机器学习", 
        "深度学习",
        "自然语言处理",
        "计算机视觉"
    ]
    
    impressions = []
    clicks = []
    
    for i, query in enumerate(queries[:num_queries]):
        request_id = f"batch_req_{i}_{uuid.uuid4().hex[:8]}"
        
        # 为每个查询生成多个文档展示
        for j in range(num_docs_per_query):
            doc_id = f"doc_{i}_{j}"
            position = j + 1
            score = 1.0 - (j * 0.1)  # 分数递减
            summary = f"关于{query}的文档{j+1}的摘要内容"
            
            impressions.append({
                "query": query,
                "doc_id": doc_id,
                "position": position,
                "score": score,
                "summary": summary,
                "request_id": request_id
            })
            
            # 模拟点击（前几个位置有更高的点击概率）
            if j < 3 and (j + i) % 2 == 0:  # 简单的点击模拟
                clicks.append({
                    "doc_id": doc_id,
                    "request_id": request_id
                })
    
    return impressions, clicks


def main():
    """批量操作示例"""
    print("🔍 Intelligent Search Engine - 批量操作示例")
    print("=" * 50)
    
    # 1. 获取数据服务
    print("\n1. 初始化数据服务...")
    data_service = get_data_service()
    print("✅ 数据服务初始化完成")
    
    # 2. 生成示例数据
    print("\n2. 生成示例数据...")
    impressions, clicks = generate_sample_data(num_queries=3, num_docs_per_query=8)
    print(f"📊 生成 {len(impressions)} 个展示事件")
    print(f"👆 生成 {len(clicks)} 个点击事件")
    
    # 3. 批量记录展示事件
    print("\n3. 批量记录展示事件...")
    start_time = datetime.now()
    
    result = data_service.batch_record_impressions(impressions)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"⏱️  批量展示耗时: {duration:.3f}秒")
    print(f"✅ 成功记录: {result['success_count']} 个展示")
    print(f"❌ 失败记录: {result['error_count']} 个展示")
    
    if result['errors']:
        print("错误详情:")
        for error in result['errors'][:5]:  # 只显示前5个错误
            print(f"  - {error}")
    
    # 4. 批量记录点击事件
    print("\n4. 批量记录点击事件...")
    start_time = datetime.now()
    
    result = data_service.batch_record_clicks(clicks)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"⏱️  批量点击耗时: {duration:.3f}秒")
    print(f"✅ 成功记录: {result['success_count']} 个点击")
    print(f"❌ 失败记录: {result['error_count']} 个点击")
    
    if result['errors']:
        print("错误详情:")
        for error in result['errors'][:5]:
            print(f"  - {error}")
    
    # 5. 查看数据统计
    print("\n5. 数据统计...")
    stats = get_data_statistics()
    print(f"📊 总样本数: {stats['total_samples']}")
    print(f"📊 总点击数: {stats['total_clicks']}")
    print(f"📊 点击率: {stats['click_rate']:.2%}")
    print(f"📊 唯一查询数: {stats['unique_queries']}")
    print(f"📊 唯一文档数: {stats['unique_docs']}")
    
    # 6. 性能对比示例
    print("\n6. 性能对比...")
    
    # 生成小批量数据用于对比
    small_impressions, _ = generate_sample_data(num_queries=1, num_docs_per_query=10)
    
    # 方式1: 批量操作
    print("方式1: 批量操作")
    start_time = datetime.now()
    data_service.batch_record_impressions(small_impressions)
    batch_duration = (datetime.now() - start_time).total_seconds()
    print(f"  批量操作耗时: {batch_duration:.3f}秒")
    
    # 方式2: 逐个操作（仅用于演示，实际不推荐）
    print("方式2: 逐个操作（演示用）")
    start_time = datetime.now()
    for impression in small_impressions[:5]:  # 只测试前5个
        try:
            data_service.record_impression(
                impression['query'],
                impression['doc_id'] + "_single",  # 避免重复
                impression['position'],
                impression['score'],
                impression['summary'],
                impression['request_id'] + "_single"
            )
        except Exception as e:
            print(f"    错误: {e}")
    single_duration = (datetime.now() - start_time).total_seconds()
    print(f"  逐个操作耗时: {single_duration:.3f}秒")
    
    if batch_duration > 0:
        speedup = single_duration / batch_duration * 2  # 乘以2因为只测试了一半
        print(f"  性能提升: {speedup:.1f}倍")
    
    # 7. 数据健康检查
    print("\n7. 数据健康检查...")
    health = data_service.get_data_health_check()
    print(f"📊 总样本数: {health['total_samples']}")
    print(f"📊 待保存变更: {health['pending_changes']}")
    print(f"📊 缓存状态: {health['cache_status']}")
    
    if health['data_issues']:
        print("⚠️  数据问题:")
        for issue in health['data_issues']:
            print(f"  - {issue}")
    
    if health['recommendations']:
        print("💡 建议:")
        for rec in health['recommendations']:
            print(f"  - {rec}")
    
    # 8. 强制保存数据
    print("\n8. 强制保存数据...")
    data_service.force_save()
    print("✅ 数据已强制保存到磁盘")
    
    print("\n🎉 批量操作示例完成！")
    print("💡 提示: 批量操作比逐个操作效率更高，建议在处理大量数据时使用")


if __name__ == "__main__":
    main() 