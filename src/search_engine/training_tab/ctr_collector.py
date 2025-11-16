#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTR收集器实现 - 实现CTR接口
负责CTR数据收集和管理
"""

import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ..search_tab.search_interface import CTRInterface
from typing import List, Dict, Any
from datetime import datetime
from .ctr_config import CTRSampleConfig, ctr_sample_config

class CTRCollector(CTRInterface):
    """CTR收集器实现类"""
    
    def __init__(self):
        self.ctr_data = []  # CTR数据存储
        # 使用绝对路径，确保在任何目录下都能正确找到文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.data_file = os.path.join(project_root, "data", "ctr_data.json")
        self.load_data()  # 启动时加载已有数据
    
    def load_data(self):
        """从文件加载CTR数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.ctr_data = data.get('records', [])
                print(f"已加载 {len(self.ctr_data)} 条CTR数据")
            except Exception as e:
                print(f"加载CTR数据失败: {e}")
                self.ctr_data = []
    
    def save_data(self):
        """保存CTR数据到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            
            data = {
                'records': self.ctr_data,
                'total_records': len(self.ctr_data),
                'total_clicks': sum(record['clicked'] for record in self.ctr_data),
                'overall_ctr': sum(record['clicked'] for record in self.ctr_data) / len(self.ctr_data) if self.ctr_data else 0
            }
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存CTR数据失败: {e}")
    
    def record_impression(self, query: str, doc_id: str, position: int, score: float, summary: str):
        """记录曝光"""
        timestamp = datetime.now()
        
        # 使用配置类创建标准化的CTR记录
        ctr_record = CTRSampleConfig.create_empty_sample()
        
        # 设置基础字段
        ctr_record.update({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'query': query,
            'doc_id': doc_id,
            'position': position,
            'score': score,
            'clicked': 0,  # 初始化为未点击
            'summary': summary,
            'doc_length': len(summary),
            'query_length': len(query),
            'summary_length': len(summary),
            'position_decay': 1.0 / (position + 1)
        })
        
        # 验证样本完整性
        errors = CTRSampleConfig.validate_sample(ctr_record)
        if errors:
            print(f"警告: CTR样本验证失败: {errors}")
        
        self.ctr_data.append(ctr_record)
        self.save_data()  # 每次记录后保存
    
    def record_click(self, query: str, doc_id: str, position: int):
        """记录点击"""
        # 找到对应的CTR记录并更新点击状态
        for record in self.ctr_data:
            if (record['doc_id'] == doc_id and 
                record['position'] == position and 
                record['query'] == query):
                record['clicked'] = 1
                self.save_data()  # 点击后保存
                # 检查是否需要触发在线训练
                self._check_and_trigger_online_training()
                break
    
    def _check_and_trigger_online_training(self):
        """检查是否需要触发在线训练"""
        # 如果没有model_service引用，则跳过
        if not self.model_service:
            return
        
        # 如果在线学习未启用，则跳过
        if not self.model_service.is_online_learning_enabled():
            return
        
        # 计算新增的数据量
        current_data_count = len(self.ctr_data)
        new_data_count = current_data_count - self.last_training_data_count
        
        # 如果新增数据达到阈值，触发在线训练
        if new_data_count >= self.online_training_trigger_threshold:
            print(f"📊 检测到{new_data_count}条新数据，触发在线训练...")
            try:
                # 创建一个简单的数据服务包装器
                class DataServiceWrapper:
                    def __init__(self, ctr_data):
                        self.ctr_data = ctr_data
                    
                    def get_all_samples(self):
                        return self.ctr_data
                
                data_service = DataServiceWrapper(self.ctr_data)
                result = self.model_service.trigger_online_training(
                    data_service, 
                    min_new_samples=self.online_training_trigger_threshold
                )
                
                if result.get('success', False):
                    # 更新训练数据计数
                    self.last_training_data_count = current_data_count
                    print(f"✅ 在线训练完成，已处理{new_data_count}条新数据")
                elif not result.get('skipped', False):
                    print(f"⚠️ 在线训练失败: {result.get('error', '未知错误')}")
            except Exception as e:
                print(f"❌ 触发在线训练时发生错误: {e}")
    
    def set_model_service(self, model_service):
        """设置模型服务引用"""
        self.model_service = model_service
        print("✅ 模型服务已关联到CTR收集器")
    
    def set_online_training_threshold(self, threshold: int):
        """设置在线训练触发阈值"""
        self.online_training_trigger_threshold = max(1, threshold)
        print(f"🔄 在线训练触发阈值已设置为: {threshold}条新数据")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取历史记录"""
        # 按时间倒序排列
        sorted_data = sorted(self.ctr_data, key=lambda r: r['timestamp'], reverse=True)
        return sorted_data
    
    def export_data(self) -> Dict[str, Any]:
        """导出CTR数据"""
        total_records = len(self.ctr_data)
        total_clicks = sum(record['clicked'] for record in self.ctr_data)
        overall_ctr = total_clicks / total_records if total_records > 0 else 0
        
        return {
            'records': self.ctr_data,
            'total_records': total_records,
            'total_clicks': total_clicks,
            'overall_ctr': overall_ctr
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取CTR统计信息"""
        total_impressions = len(self.ctr_data)
        total_clicks = sum(record['clicked'] for record in self.ctr_data)
        overall_ctr = total_clicks / total_impressions if total_impressions > 0 else 0
        
        return {
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'overall_ctr': overall_ctr
        }
    
    def clear_data(self):
        """清空所有CTR数据"""
        self.ctr_data = []
        # 删除数据文件
        if os.path.exists(self.data_file):
            try:
                os.remove(self.data_file)
                print(f"已删除CTR数据文件: {self.data_file}")
            except Exception as e:
                print(f"删除CTR数据文件失败: {e}")
        print("CTR数据已清空") 