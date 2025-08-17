import os
import json
import subprocess
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from .index_tab.index_service import InvertedIndexService
from .index_tab.kg_retrieval_service import KGRetrievalService


class IndexService:
    """索引服务：负责索引构建、文档管理、检索功能"""
    
    def __init__(self, index_file: str = "models/index_data.json"):
        self.index_file = index_file
        self.index_service = InvertedIndexService(index_file)
        # 确保KGRetrievalService使用Ollama作为默认API配置
        self.kg_retrieval_service = KGRetrievalService(
            api_type="ollama",
            default_model="qwen2.5-coder:latest"
        )
        self._ensure_index_exists()
        
    def set_ner_api_config(self, 
                          api_type: str = "ollama",
                          api_key: Optional[str] = None,
                          base_url: Optional[str] = None,
                          default_model: Optional[str] = None):
        """
        设置NER服务的API配置
        
        Args:
            api_type: API类型 ("ollama" 或 "openai")
            api_key: API密钥
            base_url: API基础URL
            default_model: 默认模型名称
        """
        # 重新初始化知识图谱检索服务
        self.kg_retrieval_service = KGRetrievalService(
            api_type=api_type,
            api_key=api_key,
            base_url=base_url,
            default_model=default_model
        )
    
    def _ensure_index_exists(self):
        """确保索引存在，如果不存在则构建"""
        if not os.path.exists(self.index_file):
            print("📦 索引文件不存在，开始构建...")
            self.build_index()
        else:
            print(f"✅ 索引文件已存在: {self.index_file}")
    
    def build_index(self) -> bool:
        """构建离线索引"""
        try:
            print("🔨 开始构建离线索引...")
            
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # 添加src目录到Python路径
            src_path = os.path.join(current_dir, 'src')
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            
            # 设置环境变量
            env = os.environ.copy()
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = src_path + os.pathsep + env['PYTHONPATH']
            else:
                env['PYTHONPATH'] = src_path
            
            # 运行离线索引构建
            result = subprocess.run(
                [sys.executable, "-m", "search_engine.index_tab.offline_index"],
                check=True,
                cwd=current_dir,
                env=env,
                capture_output=True,
                text=True
            )
            
            print("✅ 离线索引构建完成")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 离线索引构建失败: {e}")
            print(f"错误输出: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ 构建索引时发生错误: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """获取文档内容"""
        return self.index_service.get_document(doc_id)
    
    def search(self, query: str, top_k: int = 10, retrieval_mode: str = "tfidf") -> List[Tuple[str, float, str]]:
        """
        搜索文档
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            retrieval_mode: 检索模式，目前只支持 'tfidf'
            
        Returns:
            List[Tuple[str, float, str]]: (doc_id, score, reason)
        """
        # 目前只支持TF-IDF检索
        return self.index_service.search(query, top_k)
    
    def retrieve(self, query: str, top_k: int = 20) -> List[str]:
        """检索文档ID列表"""
        return self.index_service.search_doc_ids(query, top_k)
    
    def rank(self, query: str, doc_ids: List[str], top_k: int = 10, sort_mode: str = "tfidf", model_type: Optional[str] = None) -> List[Tuple[str, float, str]]:
        """对文档进行排序，支持TF-IDF和CTR排序模式"""
        if not doc_ids:
            return []
        
        # 获取所有文档的搜索结果
        all_results = self.search(query, top_k=len(doc_ids))
        
        # 过滤出指定doc_ids的结果
        filtered_results = []
        for result in all_results:
            if result[0] in doc_ids:
                filtered_results.append(result)
        
        if not filtered_results:
            return []
        
        # 如果是CTR排序模式，调用模型服务进行CTR预测
        if sort_mode == "ctr":
            try:
                # 导入模型服务
                from .service_manager import service_manager
                model_service = service_manager.model_service
                
                # 计算CTR分数
                ctr_results = []
                from datetime import datetime
                current_timestamp = datetime.now().isoformat()
                
                for position, (doc_id, tfidf_score, summary) in enumerate(filtered_results, 1):
                    # 准备特征
                    features = {
                        'query': query,
                        'doc_id': doc_id,
                        'position': position,
                        'score': tfidf_score,
                        'summary': summary,
                        'timestamp': current_timestamp  # 添加当前时间戳
                    }
                    
                    # 预测CTR，使用指定的模型类型
                    ctr_score = model_service.predict_ctr(features, model_type)
                    
                    # 返回4元组: (doc_id, tfidf_score, ctr_score, summary)
                    ctr_results.append((doc_id, tfidf_score, ctr_score, summary))
                
                # 按CTR分数排序
                sorted_results = sorted(ctr_results, key=lambda x: x[2], reverse=True)
                return sorted_results[:top_k]
                
            except Exception as e:
                print(f"❌ CTR排序失败，回退到TF-IDF排序: {e}")
                # 回退到TF-IDF排序
                sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)
                return sorted_results[:top_k]
        
        # 默认TF-IDF排序
        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def get_document_page(self, doc_id: str, request_id: str, data_service=None) -> Dict[str, Any]:
        """获取文档页面（可选记录点击事件）"""
        try:
            # 获取文档内容
            content = self.get_document(doc_id)
            if content is None:
                return {
                    'html': f"<h3>❌ 文档不存在</h3><p>文档ID: {doc_id}</p>",
                    'click_recorded': False
                }
            
            # 如果提供了数据服务，记录点击事件
            click_recorded = False
            if data_service:
                click_recorded = data_service.record_click(doc_id, request_id)
            
            # 生成HTML页面
            html_content = f"""
            <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <h2 style="margin: 0; color: #333;">📄 文档详情</h2>
                    <p style="margin: 5px 0; color: #666;">文档ID: {doc_id}</p>
                    <p style="margin: 5px 0; color: #666;">请求ID: {request_id}</p>
                    <p style="margin: 5px 0; color: #666;">点击记录: {'✅ 已记录' if click_recorded else '❌ 记录失败'}</p>
                </div>
                
                <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px;">文档内容</h3>
                    <div style="line-height: 1.6; color: #333; white-space: pre-wrap; font-family: 'Courier New', monospace;">
                        {content}
                    </div>
                </div>
            </div>
            """
            
            return {
                'html': html_content,
                'click_recorded': click_recorded,
                'doc_id': doc_id,
                'request_id': request_id
            }
            
        except Exception as e:
            error_html = f"""
            <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb;">
                    <h3>❌ 获取文档失败</h3>
                    <p><strong>文档ID:</strong> {doc_id}</p>
                    <p><strong>请求ID:</strong> {request_id}</p>
                    <p><strong>错误信息:</strong> {str(e)}</p>
                </div>
            </div>
            """
            
            return {
                'html': error_html,
                'click_recorded': False,
                'error': str(e)
            }
    
    def get_document_preview(self, doc_id: str, max_length: int = 200) -> str:
        """获取文档预览（截取前N个字符）"""
        content = self.get_document(doc_id)
        if content is None:
            return "文档不存在"
        
        if len(content) <= max_length:
            return content
        else:
            return content[:max_length] + "..."
    
    def get_documents_batch(self, doc_ids: list) -> Dict[str, str]:
        """批量获取文档内容"""
        results = {}
        for doc_id in doc_ids:
            content = self.get_document(doc_id)
            results[doc_id] = content if content else "文档不存在"
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        try:
            stats = self.index_service.get_stats()
            return {
                'total_documents': stats.get('total_documents', 0),
                'total_terms': stats.get('total_terms', 0),
                'average_doc_length': stats.get('average_doc_length', 0),
                'index_size': stats.get('index_size', 0),
                'index_file': self.index_file,
                'index_exists': os.path.exists(self.index_file)
            }
        except Exception as e:
            print(f"❌ 获取索引统计失败: {e}")
            return {
                'total_documents': 0,
                'total_terms': 0,
                'average_doc_length': 0,
                'index_size': 0,
                'index_file': self.index_file,
                'index_exists': os.path.exists(self.index_file)
            }
    
    def add_document(self, doc_id: str, content: str) -> bool:
        """添加文档到索引"""
        return self.index_service.add_document(doc_id, content)
    
    def delete_document(self, doc_id: str) -> bool:
        """从索引中删除文档"""
        return self.index_service.delete_document(doc_id)
    
    def batch_add_documents(self, documents: Dict[str, str]) -> int:
        """批量添加文档"""
        return self.index_service.batch_add_documents(documents)
    
    def get_all_documents(self) -> Dict[str, str]:
        """获取所有文档"""
        return self.index_service.get_all_documents()
    
    def clear_index(self) -> bool:
        """清空索引"""
        return self.index_service.clear_index()
    
    def save_index(self, filepath: Optional[str] = None) -> bool:
        """保存索引"""
        return self.index_service.save_index(filepath)
    
    def load_index(self, filepath: str) -> bool:
        """加载索引"""
        return self.index_service.load_index(filepath)
    
    def export_documents(self) -> Tuple[Optional[str], str]:
        """导出所有文档"""
        try:
            documents = self.get_all_documents()
            if not documents:
                return None, "❌ 没有文档可导出"
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"documents_export_{timestamp}.json"
            filepath = os.path.join("data", filename)
            
            os.makedirs("data", exist_ok=True)
            
            export_data = {
                "export_time": datetime.now().isoformat(),
                "total_documents": len(documents),
                "documents": documents
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return filepath, f"✅ 文档导出成功！\n文档数量: {len(documents)}\n文件: {filename}"
            
        except Exception as e:
            return None, f"❌ 导出文档失败: {str(e)}"
    
    # 知识图谱相关方法
    def build_knowledge_graph(self, model: Optional[str] = None) -> Dict[str, Any]:
        """构建知识图谱"""
        documents = self.get_all_documents()
        if not documents:
            return {"error": "没有文档可用于构建知识图谱"}
        
        return self.kg_retrieval_service.build_knowledge_graph(documents, model)
    
    def rebuild_knowledge_graph(self, model: Optional[str] = None) -> Dict[str, Any]:
        """重新构建知识图谱"""
        documents = self.get_all_documents()
        if not documents:
            return {"error": "没有文档可用于构建知识图谱"}
        
        return self.kg_retrieval_service.rebuild_knowledge_graph(documents, model)
    
    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """获取知识图谱统计信息"""
        return self.kg_retrieval_service.get_graph_stats()
    
    def query_entity_relations(self, entity_name: str) -> Dict[str, Any]:
        """
        查询实体的相关实体和关系
        
        Args:
            entity_name: 实体名称
            
        Returns:
            Dict: 实体关系信息
        """
        return self.kg_retrieval_service.query_entity_relations(entity_name)
    
    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        搜索实体
        
        Args:
            query: 搜索查询
            limit: 返回数量限制
            
        Returns:
            List[Dict]: 实体列表
        """
        return self.kg_retrieval_service.search_entities(query, limit)
    
    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """获取实体详细信息"""
        return self.kg_retrieval_service.get_entity_info(entity_name)
    
    def export_knowledge_graph(self) -> Tuple[Optional[str], str]:
        """导出知识图谱"""
        return self.kg_retrieval_service.export_graph()
    
    def clear_knowledge_graph(self) -> str:
        """清空知识图谱"""
        return self.kg_retrieval_service.clear_graph()
    
    def get_graph_visualization_data(self) -> Dict[str, Any]:
        """获取图谱可视化数据"""
        return self.kg_retrieval_service.get_graph_visualization_data()
    
    def analyze_query_entities(self, query: str, model: Optional[str] = None) -> Dict[str, Any]:
        """分析查询中的实体"""
        return self.kg_retrieval_service.analyze_query_entities(query, model)
    
    def import_documents(self, file_path: str) -> str:
        """导入文档"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if not isinstance(import_data, dict):
                return "❌ 文件格式错误"
            
            documents = import_data.get("documents", {})
            if not isinstance(documents, dict):
                return "❌ 文档数据格式错误"
            
            if not documents:
                return "❌ 没有文档数据"
            
            # 清空现有索引并导入新文档
            self.clear_index()
            success_count = self.batch_add_documents(documents)
            self.save_index()
            
            return f"✅ 文档导入成功！\n导入文档数: {success_count}\n总文档数: {len(documents)}"
            
        except Exception as e:
            return f"❌ 导入文档失败: {str(e)}" 