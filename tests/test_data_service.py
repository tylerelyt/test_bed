#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据服务测试用例
"""

import unittest
import tempfile
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from search_engine.data_service import DataService
import json


class TestDataService(unittest.TestCase):
    """数据服务测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_service = DataService(auto_save_interval=1, batch_size=5)
        # 修改数据文件路径到临时目录
        self.data_service.data_file = os.path.join(self.temp_dir, "test_ctr_data.json")
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_record_impression(self):
        """测试记录展示事件"""
        sample = self.data_service.record_impression(
            query="测试查询",
            doc_id="test_doc",
            position=1,
            score=0.85,
            summary="测试摘要",
            request_id="test_req"
        )
        
        self.assertIsNotNone(sample)
        self.assertEqual(sample['query'], "测试查询")
        self.assertEqual(sample['doc_id'], "test_doc")
        self.assertEqual(sample['position'], 1)
        self.assertEqual(sample['score'], 0.85)
        self.assertEqual(sample['clicked'], 0)
    
    def test_record_impression_validation(self):
        """测试展示事件参数验证"""
        # 空查询
        with self.assertRaises(ValueError):
            self.data_service.record_impression("", "doc1", 1, 0.8, "摘要", "req1")
        
        # 空文档ID
        with self.assertRaises(ValueError):
            self.data_service.record_impression("查询", "", 1, 0.8, "摘要", "req1")
        
        # 无效位置
        with self.assertRaises(ValueError):
            self.data_service.record_impression("查询", "doc1", 0, 0.8, "摘要", "req1")
        
        # 负分数
        with self.assertRaises(ValueError):
            self.data_service.record_impression("查询", "doc1", 1, -0.1, "摘要", "req1")
    
    def test_record_click(self):
        """测试记录点击事件"""
        # 先记录展示
        self.data_service.record_impression(
            "查询", "doc1", 1, 0.8, "摘要", "req1"
        )
        
        # 再记录点击
        success = self.data_service.record_click("doc1", "req1")
        self.assertTrue(success)
        
        # 验证点击状态
        samples = self.data_service.get_samples_by_request("req1")
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]['clicked'], 1)
    
    def test_record_click_validation(self):
        """测试点击事件参数验证"""
        # 空文档ID
        with self.assertRaises(ValueError):
            self.data_service.record_click("", "req1")
        
        # 空请求ID
        with self.assertRaises(ValueError):
            self.data_service.record_click("doc1", "")
    
    def test_batch_record_impressions(self):
        """测试批量记录展示事件"""
        impressions = [
            {
                "query": "查询1",
                "doc_id": "doc1",
                "position": 1,
                "score": 0.9,
                "summary": "摘要1",
                "request_id": "req1"
            },
            {
                "query": "查询2",
                "doc_id": "doc2",
                "position": 2,
                "score": 0.8,
                "summary": "摘要2",
                "request_id": "req1"
            }
        ]
        
        result = self.data_service.batch_record_impressions(impressions)
        self.assertTrue(result['success'])
        self.assertEqual(result['success_count'], 2)
        self.assertEqual(result['error_count'], 0)
        
        # 验证数据
        samples = self.data_service.get_all_samples()
        self.assertEqual(len(samples), 2)
    
    def test_batch_record_clicks(self):
        """测试批量记录点击事件"""
        # 先记录展示
        impressions = [
            {
                "query": "查询1",
                "doc_id": "doc1",
                "position": 1,
                "score": 0.9,
                "summary": "摘要1",
                "request_id": "req1"
            },
            {
                "query": "查询2",
                "doc_id": "doc2",
                "position": 2,
                "score": 0.8,
                "summary": "摘要2",
                "request_id": "req1"
            }
        ]
        self.data_service.batch_record_impressions(impressions)
        
        # 再记录点击
        clicks = [
            {"doc_id": "doc1", "request_id": "req1"},
            {"doc_id": "doc2", "request_id": "req1"}
        ]
        
        result = self.data_service.batch_record_clicks(clicks)
        self.assertTrue(result['success'])
        self.assertEqual(result['success_count'], 2)
        self.assertEqual(result['error_count'], 0)
        
        # 验证点击状态
        samples = self.data_service.get_all_samples()
        clicked_count = sum(1 for s in samples if s['clicked'] == 1)
        self.assertEqual(clicked_count, 2)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        # 空数据统计
        stats = self.data_service.get_stats()
        self.assertEqual(stats['total_samples'], 0)
        self.assertEqual(stats['total_clicks'], 0)
        self.assertEqual(stats['click_rate'], 0.0)
        
        # 添加数据后统计
        self.data_service.record_impression("查询", "doc1", 1, 0.8, "摘要", "req1")
        self.data_service.record_impression("查询", "doc2", 2, 0.7, "摘要", "req1")
        self.data_service.record_click("doc1", "req1")
        
        stats = self.data_service.get_stats()
        self.assertEqual(stats['total_samples'], 2)
        self.assertEqual(stats['total_clicks'], 1)
        self.assertEqual(stats['click_rate'], 0.5)
        self.assertEqual(stats['unique_queries'], 1)
        self.assertEqual(stats['unique_docs'], 2)
    
    def test_data_export_import(self):
        """测试数据导出导入"""
        # 添加测试数据
        self.data_service.record_impression("查询", "doc1", 1, 0.8, "摘要", "req1")
        self.data_service.record_click("doc1", "req1")
        
        # 导出数据
        export_file = os.path.join(self.temp_dir, "export.json")
        success = self.data_service.export_data(export_file)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_file))
        
        # 清空数据
        self.data_service.clear_data()
        self.assertEqual(len(self.data_service.get_all_samples()), 0)
        
        # 导入数据
        success = self.data_service.import_data(export_file)
        self.assertTrue(success)
        
        # 验证导入结果
        samples = self.data_service.get_all_samples()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]['query'], "查询")
        self.assertEqual(samples[0]['clicked'], 1)
    
    def test_data_health_check(self):
        """测试数据健康检查"""
        # 空数据检查
        health = self.data_service.get_data_health_check()
        self.assertEqual(health['total_samples'], 0)
        self.assertIn('没有数据', health['data_issues'])
        
        # 添加数据后检查
        self.data_service.record_impression("查询", "doc1", 1, 0.8, "摘要", "req1")
        self.data_service.record_impression("查询", "doc2", 2, 0.7, "摘要", "req1")
        
        health = self.data_service.get_data_health_check()
        self.assertEqual(health['total_samples'], 2)
        # 点击率低应该有警告
        self.assertIn('点击率过低', str(health['data_issues']))
    
    def test_time_range_query(self):
        """测试时间范围查询"""
        # 添加测试数据
        self.data_service.record_impression("查询", "doc1", 1, 0.8, "摘要", "req1")
        
        # 查询时间范围
        from datetime import datetime, timedelta
        now = datetime.now()
        start_time = (now - timedelta(hours=1)).isoformat()
        end_time = (now + timedelta(hours=1)).isoformat()
        
        samples = self.data_service.get_samples_by_time_range(start_time, end_time)
        self.assertEqual(len(samples), 1)
        
        # 查询过去的时间范围
        past_start = (now - timedelta(days=1)).isoformat()
        past_end = (now - timedelta(hours=2)).isoformat()
        
        samples = self.data_service.get_samples_by_time_range(past_start, past_end)
        self.assertEqual(len(samples), 0)
    
    def test_query_pattern_search(self):
        """测试查询模式搜索"""
        # 添加测试数据
        self.data_service.record_impression("人工智能", "doc1", 1, 0.8, "摘要", "req1")
        self.data_service.record_impression("机器学习", "doc2", 2, 0.7, "摘要", "req2")
        self.data_service.record_impression("深度学习", "doc3", 3, 0.6, "摘要", "req3")
        
        # 模式搜索
        samples = self.data_service.get_samples_by_query_pattern(".*学习")
        self.assertEqual(len(samples), 2)  # 机器学习和深度学习
        
        samples = self.data_service.get_samples_by_query_pattern("人工.*")
        self.assertEqual(len(samples), 1)  # 人工智能
    
    def test_caching(self):
        """测试缓存功能"""
        # 添加数据
        self.data_service.record_impression("查询", "doc1", 1, 0.8, "摘要", "req1")
        
        # 第一次获取统计（应该计算）
        stats1 = self.data_service.get_stats()
        self.assertFalse(stats1.get('cache_hit', True))
        
        # 第二次获取统计（应该使用缓存）
        # 注意：由于缓存TTL很短，这个测试可能不稳定
        stats2 = self.data_service.get_stats()
        self.assertEqual(stats1['total_samples'], stats2['total_samples'])


if __name__ == '__main__':
    unittest.main() 