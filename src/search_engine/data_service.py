import threading
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import jieba
from .training_tab.ctr_config import CTRSampleConfig
from abc import ABC, abstractmethod
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor


class DataServiceInterface(ABC):
    """数据服务接口 - 定义标准的数据访问方法"""
    
    @abstractmethod
    def record_impression(self, query: str, doc_id: str, position: int, 
                         score: float, summary: str, request_id: str) -> Dict[str, Any]:
        """记录展示事件"""
        pass
    
    @abstractmethod
    def record_click(self, doc_id: str, request_id: str) -> bool:
        """记录点击事件"""
        pass
    
    @abstractmethod
    def get_all_samples(self) -> List[Dict[str, Any]]:
        """获取所有CTR样本"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        pass


class DataService(DataServiceInterface):
    """数据服务：负责CTR事件收集、样本状态管理和数据读写操作
    
    设计说明：
    - 位置：服务层 (Service Layer)
    - 职责：统一管理CTR数据，提供线程安全的数据访问接口
    - 使用：被多个业务模块调用，符合分层架构原则
    - 数据存储：models/ctr_data.json (与模型文件放在一起便于管理)
    
    优化特性：
    - 批量保存：减少频繁的文件IO操作
    - 延迟保存：异步保存数据，不阻塞主线程
    - 数据缓存：内存缓存提高访问速度
    """
    
    def __init__(self, auto_save_interval: int = 30, batch_size: int = 100):
        self.ctr_data: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.data_file = "models/ctr_data.json"
        
        # 优化参数
        self.auto_save_interval = auto_save_interval  # 自动保存间隔（秒）
        self.batch_size = batch_size  # 批量保存大小
        self.pending_changes = 0  # 待保存的变更数量
        self.last_save_time = time.time()
        
        # 异步保存相关
        self.save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DataSaver")
        self.is_saving = False
        
        # 数据缓存
        self._stats_cache = None
        self._stats_cache_time = 0
        self._cache_ttl = 10  # 缓存TTL（秒）
        
        self._load_existing_data()
        self._start_auto_save_timer()
    
    def _start_auto_save_timer(self):
        """启动自动保存定时器"""
        def auto_save():
            while True:
                time.sleep(self.auto_save_interval)
                if self.pending_changes > 0:
                    self._save_data_async()
        
        timer_thread = threading.Thread(target=auto_save, daemon=True)
        timer_thread.start()
    
    def _should_save_now(self) -> bool:
        """判断是否应该立即保存"""
        return (
            self.pending_changes >= self.batch_size or
            time.time() - self.last_save_time > self.auto_save_interval
        )
    
    def _save_data_async(self):
        """异步保存数据"""
        if self.is_saving:
            return
        
        self.is_saving = True
        self.save_executor.submit(self._save_data_sync)
    
    def _save_data_sync(self):
        """同步保存数据到文件"""
        try:
            import json
            import os
            
            with self.lock:
                data_to_save = self.ctr_data.copy()
                self.pending_changes = 0
                self.last_save_time = time.time()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            
            # 写入临时文件，然后原子性替换
            temp_file = self.data_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
            # 原子性替换
            os.replace(temp_file, self.data_file)
            
            print(f"✅ 数据保存成功: {len(data_to_save)}条记录")
            
        except Exception as e:
            print(f"⚠️ 保存CTR数据失败: {e}")
        finally:
            self.is_saving = False
    
    def _invalidate_cache(self):
        """清除缓存"""
        self._stats_cache = None
        self._stats_cache_time = 0
    
    def _load_existing_data(self):
        """加载已存在的CTR数据"""
        try:
            import json
            import os
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.ctr_data = json.load(f)
                print(f"✅ 加载CTR数据成功，共{len(self.ctr_data)}条记录")
        except Exception as e:
            print(f"⚠️ 加载CTR数据失败: {e}")
            self.ctr_data = []
    
    def record_impression(self, query: str, doc_id: str, position: int, 
                         score: float, summary: str, request_id: str) -> Dict[str, Any]:
        """记录展示事件"""
        with self.lock:
            try:
                # 使用内部方法创建样本
                sample = self._create_sample(query, doc_id, position, score, summary, request_id)
                
                # 检查重复记录
                duplicate_count = sum(1 for d in self.ctr_data 
                                    if d.get('request_id') == request_id.strip() and 
                                       d.get('doc_id') == doc_id.strip() and 
                                       d.get('position') == position)
                
                if duplicate_count > 0:
                    print(f"⚠️ 发现重复记录: request_id={request_id}, doc_id={doc_id}, position={position}")
                
                self.ctr_data.append(sample)
                self.pending_changes += 1
                self._invalidate_cache()  # 新增数据时清除缓存
                
                if self._should_save_now():
                    self._save_data_async()
                
                return sample
                
            except Exception as e:
                print(f"❌ 记录展示事件失败: {e}")
                raise
    
    def record_click(self, doc_id: str, request_id: str) -> bool:
        """记录点击事件"""
        # 数据验证
        if not doc_id or not doc_id.strip():
            raise ValueError("文档ID不能为空")
        
        if not request_id or not request_id.strip():
            raise ValueError("请求ID不能为空")
        
        with self.lock:
            try:
                updated_count = 0
                doc_id_clean = doc_id.strip()
                request_id_clean = request_id.strip()
                
                for sample in self.ctr_data:
                    if (sample.get('request_id') == request_id_clean and 
                        sample.get('doc_id') == doc_id_clean):
                        # 记录点击事件 - 不同次点击作为独立事件
                        if sample.get('clicked', 0) == 0:
                            # 首次点击
                            sample['clicked'] = 1
                            sample['click_time'] = datetime.now().isoformat()
                            sample['click_count'] = 1
                            updated_count += 1
                            print(f"✅ 首次点击: doc_id={doc_id_clean}, request_id={request_id_clean}")
                        else:
                            # 多次点击，递增点击计数
                            sample['click_count'] = sample.get('click_count', 1) + 1
                            sample['last_click_time'] = datetime.now().isoformat()
                            updated_count += 1
                            print(f"✅ 多次点击: doc_id={doc_id_clean}, request_id={request_id_clean}, 总计点击{sample['click_count']}次")
                
                if updated_count > 0:
                    self.pending_changes += updated_count
                    self._invalidate_cache()  # 更新数据时清除缓存
                    if self._should_save_now():
                        self._save_data_async()
                    print(f"✅ 记录点击事件成功: doc_id={doc_id_clean}, request_id={request_id_clean}, 更新{updated_count}条记录")
                    return True
                else:
                    print(f"⚠️ 未找到匹配的CTR样本: doc_id={doc_id_clean}, request_id={request_id_clean}")
                    return False
                    
            except Exception as e:
                print(f"❌ 记录点击事件失败: {e}")
                raise
    
    def get_samples_by_request(self, request_id: str) -> List[Dict[str, Any]]:
        """获取指定请求的CTR样本"""
        with self.lock:
            return [sample for sample in self.ctr_data if sample.get('request_id') == request_id]
    
    def get_all_samples(self) -> List[Dict[str, Any]]:
        """获取所有CTR样本"""
        with self.lock:
            return self.ctr_data.copy()
    
    def get_samples_dataframe(self, request_id: Optional[str] = None) -> pd.DataFrame:
        """获取CTR样本DataFrame"""
        with self.lock:
            if request_id:
                samples = [sample for sample in self.ctr_data if sample.get('request_id') == request_id]
            else:
                samples = self.ctr_data
            
            if not samples:
                return pd.DataFrame()
            
            df = pd.DataFrame(samples)
            
            # 确保DataFrame包含所有配置的列
            expected_columns = CTRSampleConfig.get_field_names()
            missing_columns = [col for col in expected_columns if col not in df.columns]
            for col in missing_columns:
                df[col] = ''
            
            # 验证DataFrame的列顺序
            field_names = CTRSampleConfig.get_field_names()
            if list(df.columns) != field_names:
                df = df.reindex(columns=field_names)
            
            return df
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据统计信息（带缓存）"""
        current_time = time.time()
        
        # 检查缓存是否有效
        if (self._stats_cache is not None and 
            current_time - self._stats_cache_time < self._cache_ttl):
            return self._stats_cache
        
        # 重新计算统计信息
        with self.lock:
            if not self.ctr_data:
                stats = {
                    'total_samples': 0,
                    'total_clicks': 0,
                    'click_rate': 0.0,
                    'unique_queries': 0,
                    'unique_docs': 0,
                    'cache_hit': False,
                    'cache_time': current_time
                }
            else:
                df = pd.DataFrame(self.ctr_data)
                total_samples = len(df)
                total_clicks = df['clicked'].sum() if 'clicked' in df.columns else 0
                click_rate = total_clicks / total_samples if total_samples > 0 else 0.0
                unique_queries = df['query'].nunique() if 'query' in df.columns else 0
                unique_docs = df['doc_id'].nunique() if 'doc_id' in df.columns else 0
                
                # 新增点击计数统计
                total_click_events = 0
                avg_clicks_per_clicked_item = 0.0
                max_clicks_per_item = 0
                if 'click_count' in df.columns:
                    click_counts = df[df['clicked'] == 1]['click_count']
                    if len(click_counts) > 0:
                        total_click_events = click_counts.sum()
                        avg_clicks_per_clicked_item = click_counts.mean()
                        max_clicks_per_item = click_counts.max()
                else:
                    # 兼容旧数据
                    total_click_events = total_clicks
                    avg_clicks_per_clicked_item = 1.0 if total_clicks > 0 else 0.0
                    max_clicks_per_item = 1 if total_clicks > 0 else 0
                
                stats = {
                    'total_samples': total_samples,
                    'total_clicks': total_clicks,
                    'total_click_events': total_click_events,
                    'click_rate': click_rate,
                    'avg_clicks_per_clicked_item': round(avg_clicks_per_clicked_item, 2),
                    'max_clicks_per_item': max_clicks_per_item,
                    'unique_queries': unique_queries,
                    'unique_docs': unique_docs,
                    'cache_hit': False,
                    'cache_time': current_time
                }
            
            # 更新缓存
            self._stats_cache = stats
            self._stats_cache_time = current_time
            
            return stats
    
    def clear_data(self):
        """清空所有CTR数据"""
        with self.lock:
            self.ctr_data = []
            self.pending_changes = 0
            self._save_data_async() # 清空后也保存一次
            print("✅ CTR数据已清空")
    
    def export_data(self, filepath: str) -> bool:
        """导出CTR数据"""
        try:
            with self.lock:
                import json
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.ctr_data, f, ensure_ascii=False, indent=2)
                print(f"✅ CTR数据导出成功: {filepath}")
                return True
        except Exception as e:
            print(f"❌ CTR数据导出失败: {e}")
            return False
    
    def import_data(self, filepath: str) -> bool:
        """导入CTR数据"""
        try:
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                imported_data = json.load(f)
            
            with self.lock:
                self.ctr_data.extend(imported_data)
                self.pending_changes += len(imported_data)
                if self._should_save_now():
                    self._save_data_async()
                print(f"✅ CTR数据导入成功: {len(imported_data)}条记录")
                return True
        except Exception as e:
            print(f"❌ CTR数据导入失败: {e}")
            return False 
    
    def batch_record_impressions(self, impressions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量记录展示事件"""
        if not impressions:
            return {'success': False, 'error': '没有数据需要记录'}
        
        results = {
            'success': True,
            'total_count': len(impressions),
            'success_count': 0,
            'error_count': 0,
            'errors': []
        }
        
        with self.lock:
            try:
                batch_samples = []
                
                for i, impression in enumerate(impressions):
                    try:
                        # 验证必要字段
                        required_fields = ['query', 'doc_id', 'position', 'score', 'summary', 'request_id']
                        for field in required_fields:
                            if field not in impression:
                                raise ValueError(f"缺少必要字段: {field}")
                        
                        # 创建样本
                        sample = self._create_sample(
                            impression['query'],
                            impression['doc_id'],
                            impression['position'],
                            impression['score'],
                            impression['summary'],
                            impression['request_id']
                        )
                        
                        batch_samples.append(sample)
                        results['success_count'] += 1
                        
                    except Exception as e:
                        results['error_count'] += 1
                        results['errors'].append(f"第{i+1}条记录错误: {str(e)}")
                
                # 批量添加到数据中
                if batch_samples:
                    self.ctr_data.extend(batch_samples)
                    self.pending_changes += len(batch_samples)
                    self._invalidate_cache()
                    
                    if self._should_save_now():
                        self._save_data_async()
                
                print(f"✅ 批量记录展示事件: 成功{results['success_count']}条, 失败{results['error_count']}条")
                
            except Exception as e:
                results['success'] = False
                results['error'] = str(e)
                print(f"❌ 批量记录展示事件失败: {e}")
        
        return results
    
    def _create_sample(self, query: str, doc_id: str, position: int, 
                      score: float, summary: str, request_id: str) -> Dict[str, Any]:
        """创建单个样本（内部方法）"""
        # 数据验证
        if not query or not query.strip():
            raise ValueError("查询不能为空")
        
        if not doc_id or not doc_id.strip():
            raise ValueError("文档ID不能为空")
        
        if position < 1:
            raise ValueError("位置必须大于0")
        
        if score < 0:
            raise ValueError("分数不能为负数")
        
        if not request_id or not request_id.strip():
            raise ValueError("请求ID不能为空")
        
        # 生成时间戳
        ts = datetime.now().isoformat()
        
        # 计算查询匹配度
        query_words = set(jieba.lcut(query.strip()))
        summary_words = set(jieba.lcut(summary or ""))
        match_ratio = 0.0
        if len(query_words) > 0:
            match_ratio = len(query_words.intersection(summary_words)) / len(query_words)
        
        # 计算历史CTR
        query_history = [d for d in self.ctr_data if d.get('query') == query.strip()]
        doc_history = [d for d in self.ctr_data if d.get('doc_id') == doc_id.strip()]
        query_ctr = sum(d.get('clicked', 0) for d in query_history) / len(query_history) if query_history else 0.1
        doc_ctr = sum(d.get('clicked', 0) for d in doc_history) / len(doc_history) if doc_history else 0.1
        
        # 创建样本
        sample = {
            'query': query.strip(),
            'doc_id': doc_id.strip(),
            'position': position,
            'score': float(score),
            'summary': summary or "",
            'request_id': request_id.strip(),
            'request_time': ts,
            'clicked': 0,
            'click_count': 0,
            'click_time': "",
            'last_click_time': "",
            'match_score': round(match_ratio, 4),
            'query_ctr': round(query_ctr, 4),
            'doc_ctr': round(doc_ctr, 4),
            'timestamp': ts,
            'doc_length': len(summary) if summary else 0,
            'query_length': len(query.strip()),
            'summary_length': len(summary) if summary else 0,
            'position_decay': round(1.0 / position, 4)
        }
        
        # 验证样本完整性
        errors = CTRSampleConfig.validate_sample(sample)
        if errors:
            raise ValueError(f"CTR样本验证失败: {errors}")
        
        return sample
    
    def batch_record_clicks(self, clicks: List[Dict[str, str]]) -> Dict[str, Any]:
        """批量记录点击事件"""
        if not clicks:
            return {'success': False, 'error': '没有数据需要记录'}
        
        results = {
            'success': True,
            'total_count': len(clicks),
            'success_count': 0,
            'error_count': 0,
            'errors': []
        }
        
        with self.lock:
            try:
                for i, click in enumerate(clicks):
                    try:
                        # 验证必要字段
                        if 'doc_id' not in click or 'request_id' not in click:
                            raise ValueError("缺少必要字段: doc_id 或 request_id")
                        
                        doc_id_clean = click['doc_id'].strip()
                        request_id_clean = click['request_id'].strip()
                        
                        # 查找并更新匹配的样本
                        updated = False
                        for sample in self.ctr_data:
                            if (sample.get('request_id') == request_id_clean and 
                                sample.get('doc_id') == doc_id_clean):
                                if sample.get('clicked', 0) == 0:
                                    # 首次点击
                                    sample['clicked'] = 1
                                    sample['click_time'] = datetime.now().isoformat()
                                    sample['click_count'] = 1
                                    updated = True
                                    break
                                else:
                                    # 多次点击，递增点击计数
                                    sample['click_count'] = sample.get('click_count', 1) + 1
                                    sample['last_click_time'] = datetime.now().isoformat()
                                    updated = True
                                    break
                        
                        if updated:
                            results['success_count'] += 1
                        else:
                            results['error_count'] += 1
                            results['errors'].append(f"第{i+1}条记录: 未找到匹配的展示记录")
                            
                    except Exception as e:
                        results['error_count'] += 1
                        results['errors'].append(f"第{i+1}条记录错误: {str(e)}")
                
                if results['success_count'] > 0:
                    self.pending_changes += results['success_count']
                    self._invalidate_cache()
                    
                    if self._should_save_now():
                        self._save_data_async()
                
                print(f"✅ 批量记录点击事件: 成功{results['success_count']}条, 失败{results['error_count']}条")
                
            except Exception as e:
                results['success'] = False
                results['error'] = str(e)
                print(f"❌ 批量记录点击事件失败: {e}")
        
        return results
    
    def get_samples_by_time_range(self, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """按时间范围获取样本"""
        with self.lock:
            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                
                filtered_samples = []
                for sample in self.ctr_data:
                    if 'timestamp' in sample:
                        sample_dt = datetime.fromisoformat(sample['timestamp'].replace('Z', '+00:00'))
                        if start_dt <= sample_dt <= end_dt:
                            filtered_samples.append(sample)
                
                return filtered_samples
                
            except Exception as e:
                print(f"❌ 按时间范围获取样本失败: {e}")
                return []
    
    def get_samples_by_query_pattern(self, pattern: str) -> List[Dict[str, Any]]:
        """按查询模式获取样本"""
        with self.lock:
            try:
                import re
                regex = re.compile(pattern, re.IGNORECASE)
                
                filtered_samples = []
                for sample in self.ctr_data:
                    if 'query' in sample and regex.search(sample['query']):
                        filtered_samples.append(sample)
                
                return filtered_samples
                
            except Exception as e:
                print(f"❌ 按查询模式获取样本失败: {e}")
                return []
    
    def force_save(self):
        """强制保存数据"""
        self._save_data_sync()
    
    def get_data_health_check(self) -> Dict[str, Any]:
        """数据健康检查"""
        with self.lock:
            try:
                health_report = {
                    'total_samples': len(self.ctr_data),
                    'pending_changes': self.pending_changes,
                    'cache_status': 'valid' if self._stats_cache else 'invalid',
                    'data_issues': [],
                    'recommendations': []
                }
                
                if not self.ctr_data:
                    health_report['data_issues'].append('没有数据')
                    health_report['recommendations'].append('进行一些搜索实验生成数据')
                    return health_report
                
                # 检查重复记录
                seen_keys = set()
                duplicates = 0
                for sample in self.ctr_data:
                    key = (sample.get('request_id'), sample.get('doc_id'), sample.get('position'))
                    if key in seen_keys:
                        duplicates += 1
                    else:
                        seen_keys.add(key)
                
                if duplicates > 0:
                    health_report['data_issues'].append(f'发现{duplicates}条重复记录')
                    health_report['recommendations'].append('考虑清理重复数据')
                
                # 检查数据完整性
                incomplete_samples = 0
                for sample in self.ctr_data:
                    required_fields = ['query', 'doc_id', 'position', 'score', 'request_id']
                    if not all(field in sample for field in required_fields):
                        incomplete_samples += 1
                
                if incomplete_samples > 0:
                    health_report['data_issues'].append(f'发现{incomplete_samples}条不完整记录')
                    health_report['recommendations'].append('检查数据收集逻辑')
                
                # 检查点击率
                total_clicks = sum(sample.get('clicked', 0) for sample in self.ctr_data)
                click_rate = total_clicks / len(self.ctr_data) if self.ctr_data else 0
                
                if click_rate < 0.01:
                    health_report['data_issues'].append(f'点击率过低: {click_rate:.2%}')
                    health_report['recommendations'].append('检查点击事件记录是否正常')
                
                return health_report
                
            except Exception as e:
                return {
                    'error': str(e),
                    'total_samples': len(self.ctr_data),
                    'pending_changes': self.pending_changes
                } 