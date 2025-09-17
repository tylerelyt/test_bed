#!/usr/bin/env python3
"""
测试模型服务集成功能
"""

import sys
import os
import requests
import time

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_model_service_integration():
    """测试模型服务集成功能"""
    print("🎯 测试模型服务集成功能")
    print("=" * 50)
    
    # 1. 测试模型服务检查功能
    print("\n1️⃣ 测试模型服务检查功能...")
    try:
        from start_system import check_and_start_model_service
        
        # 检查并启动模型服务
        result = check_and_start_model_service()
        if result:
            print("✅ 模型服务检查功能正常")
        else:
            print("❌ 模型服务检查功能失败")
            return False
            
    except Exception as e:
        print(f"❌ 模型服务检查功能异常: {e}")
        return False
    
    # 2. 等待服务启动
    print("\n2️⃣ 等待服务启动...")
    time.sleep(3)
    
    # 3. 测试API接口
    print("\n3️⃣ 测试API接口...")
    base_url = "http://localhost:8501"
    
    # 健康检查
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ 健康检查: {health_data}")
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False
    
    # 模型列表
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            models = models_data['model']
            print(f"✅ 模型列表: {len(models)} 个模型")
            for model in models:
                print(f"   - {model['name']}: {model['status']} ({model['type']})")
        else:
            print(f"❌ 模型列表失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 模型列表异常: {e}")
        return False
    
    # 4. 测试预测功能
    print("\n4️⃣ 测试预测功能...")
    test_data = {
        'inputs': {
            'query': '人工智能',
            'doc_id': 'test_doc_001',
            'position': 1,
            'score': 0.8,
            'summary': '人工智能技术介绍'
        }
    }
    
    # LR模型预测
    try:
        response = requests.post(f"{base_url}/v1/models/logistic_regression:predict", 
                               json=test_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            ctr_score = result['outputs']['ctr_score']
            print(f"✅ LR模型预测: CTR = {ctr_score:.6f}")
        else:
            print(f"❌ LR模型预测失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ LR模型预测异常: {e}")
        return False
    
    # Wide & Deep模型预测
    try:
        response = requests.post(f"{base_url}/v1/models/wide_and_deep:predict", 
                               json=test_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            ctr_score = result['outputs']['ctr_score']
            print(f"✅ Wide & Deep模型预测: CTR = {ctr_score:.6f}")
        else:
            print(f"❌ Wide & Deep模型预测失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Wide & Deep模型预测异常: {e}")
        return False
    
    # 5. 测试批量预测
    print("\n5️⃣ 测试批量预测...")
    batch_data = {
        'inputs': [
            {'query': '机器学习', 'doc_id': 'doc1', 'position': 1, 'score': 0.9, 'summary': '机器学习介绍'},
            {'query': '深度学习', 'doc_id': 'doc2', 'position': 2, 'score': 0.7, 'summary': '深度学习介绍'}
        ]
    }
    
    try:
        response = requests.post(f"{base_url}/v1/models/logistic_regression/batch_predict", 
                               json=batch_data, timeout=5)
        if response.status_code == 200:
            results = response.json()['outputs']
            print(f"✅ 批量预测: {len(results)} 个结果")
            for i, result in enumerate(results):
                print(f"   结果{i+1}: CTR = {result['ctr_score']:.6f}")
        else:
            print(f"❌ 批量预测失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 批量预测异常: {e}")
        return False
    
    print("\n🎉 模型服务集成功能测试完成！")
    print("✅ 所有功能都正常工作")
    return True

if __name__ == "__main__":
    success = test_model_service_integration()
    if success:
        print("\n🎯 测试结果: 通过")
        sys.exit(0)
    else:
        print("\n🎯 测试结果: 失败")
        sys.exit(1)
