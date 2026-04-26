#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[已弃用] 知识图谱 — NetworkX 实现

生产路径已切换为 JanusGraph + Gremlin（见 janusgraph_backend.py、kg_retrieval_service.py）。
本文件仅保留供历史脚本或迁移参考，新代码请勿引用 KnowledgeGraph。
"""

import json
import os
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any, Set
from datetime import datetime
import pickle
from collections import defaultdict, Counter

class KnowledgeGraph:
    """知识图谱类"""

    def __init__(self, graph_file: str = "models/knowledge_graph.pkl"):
        """
        初始化知识图谱

        Args:
            graph_file: 图谱文件路径
        """
        self.graph_file = graph_file
        self.graph = nx.MultiDiGraph()  # 有向多重图，支持多种关系
        self.entity_docs = defaultdict(set)  # 实体->文档映射
        self.doc_entities = defaultdict(set)  # 文档->实体映射
        self.relation_types = Counter()  # 关系类型统计
        self.entity_types = Counter()  # 实体类型统计

        # 加载现有图谱
        self.load_graph()

    def add_entity(self, entity_name: str, entity_type: str, description: str = "", doc_id: Optional[str] = None):
        """
        添加实体到图谱

        Args:
            entity_name: 实体名称
            entity_type: 实体类型
            description: 实体描述
            doc_id: 来源文档ID
        """
        # 标准化实体名称
        entity_name = entity_name.strip()
        if not entity_name:
            return

        # 添加或更新实体节点
        if self.graph.has_node(entity_name):
            # 更新现有实体
            node_data = self.graph.nodes[entity_name]
            if description and len(description) > len(node_data.get("description", "")):
                node_data["description"] = description
            node_data["doc_count"] = node_data.get("doc_count", 0) + 1
        else:
            # 添加新实体
            self.graph.add_node(entity_name,
                              entity_type=entity_type,
                              description=description,
                              doc_count=1,
                              created_at=datetime.now().isoformat())

        # 更新统计
        self.entity_types[entity_type] += 1

        # 更新文档映射
        if doc_id:
            self.entity_docs[entity_name].add(doc_id)
            self.doc_entities[doc_id].add(entity_name)

    def add_relation(self, subject: str, predicate: str, object_entity: str,
                    description: str = "", doc_id: Optional[str] = None):
        """
        添加关系到图谱

        Args:
            subject: 主体实体
            predicate: 关系类型
            object_entity: 客体实体
            description: 关系描述
            doc_id: 来源文档ID
        """
        # 标准化名称
        subject = subject.strip()
        object_entity = object_entity.strip()
        predicate = predicate.strip()

        if not all([subject, predicate, object_entity]):
            return

        # 确保实体存在
        if not self.graph.has_node(subject):
            self.add_entity(subject, "未分类", "", doc_id)
        if not self.graph.has_node(object_entity):
            self.add_entity(object_entity, "未分类", "", doc_id)

        # 添加关系边
        self.graph.add_edge(subject, object_entity,
                          predicate=predicate,
                          description=description,
                          doc_id=doc_id,
                          created_at=datetime.now().isoformat())

        # 更新统计
        self.relation_types[predicate] += 1

    def remove_relation(self, subject: str, predicate: str, object_entity: str) -> bool:
        """
        删除一条关系边（若存在多条相同 predicate 则删除一条）

        Args:
            subject: 主体实体
            predicate: 关系类型
            object_entity: 客体实体

        Returns:
            bool: 是否删除了至少一条边
        """
        subject = subject.strip()
        object_entity = object_entity.strip()
        predicate = predicate.strip()
        if not self.graph.has_edge(subject, object_entity):
            return False
        removed = False
        for key in list(self.graph[subject][object_entity].keys()):
            if self.graph[subject][object_entity][key].get("predicate") == predicate:
                self.graph.remove_edge(subject, object_entity, key)
                self.relation_types[predicate] -= 1
                if self.relation_types[predicate] <= 0:
                    del self.relation_types[predicate]
                removed = True
                break
        return removed

    def remove_entity(self, entity_name: str) -> bool:
        """
        删除实体及其所有关联边，并更新文档映射与类型统计。

        Args:
            entity_name: 实体名称

        Returns:
            bool: 是否存在并删除了该实体
        """
        entity_name = entity_name.strip()
        if not self.graph.has_node(entity_name):
            return False
        node_data = self.graph.nodes[entity_name]
        entity_type = node_data.get("entity_type", "未分类")
        for target in list(self.graph.successors(entity_name)):
            for key in list(self.graph[entity_name][target].keys()):
                p = self.graph[entity_name][target][key].get("predicate", "")
                self.graph.remove_edge(entity_name, target, key)
                if p:
                    self.relation_types[p] -= 1
                    if self.relation_types[p] <= 0:
                        del self.relation_types[p]
        for source in list(self.graph.predecessors(entity_name)):
            for key in list(self.graph[source][entity_name].keys()):
                p = self.graph[source][entity_name][key].get("predicate", "")
                self.graph.remove_edge(source, entity_name, key)
                if p:
                    self.relation_types[p] -= 1
                    if self.relation_types[p] <= 0:
                        del self.relation_types[p]
        self.graph.remove_node(entity_name)
        self.entity_types[entity_type] -= 1
        if self.entity_types[entity_type] <= 0:
            del self.entity_types[entity_type]
        for doc_id in list(self.entity_docs.get(entity_name, set())):
            self.doc_entities[doc_id].discard(entity_name)
            if not self.doc_entities[doc_id]:
                del self.doc_entities[doc_id]
        if entity_name in self.entity_docs:
            del self.entity_docs[entity_name]
        return True

    def build_from_ner_results(self, ner_results: Dict[str, Any]):
        """
        从NER结果构建知识图谱

        Args:
            ner_results: NER提取结果
        """
        print("开始构建知识图谱...")

        for doc_id, doc_result in ner_results.items():
            if "error" in doc_result:
                print(f"❌ [KG-Build] 跳过文档 {doc_id}，NER提取错误: {doc_result['error']}")
                continue

            print(f"✅ [KG-Build] 处理文档 {doc_id}")
            entities = doc_result.get("entities", [])
            relations = doc_result.get("relations", [])
            print(f"📊 [KG-Build] 文档 {doc_id}: {len(entities)} 个实体, {len(relations)} 个关系")

            # 添加实体
            for entity in doc_result.get("entities", []):
                self.add_entity(
                    entity_name=entity.get("name", ""),
                    entity_type=entity.get("type", "未分类"),
                    description=entity.get("description", ""),
                    doc_id=doc_id
                )

            # 添加关系
            for relation in doc_result.get("relations", []):
                self.add_relation(
                    subject=relation.get("subject", ""),
                    predicate=relation.get("predicate", ""),
                    object_entity=relation.get("object", ""),
                    description=relation.get("description", ""),
                    doc_id=doc_id
                )

        print(f"知识图谱构建完成，共 {self.graph.number_of_nodes()} 个实体，{self.graph.number_of_edges()} 条关系")

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        搜索实体

        Args:
            query: 搜索查询
            limit: 返回数量限制

        Returns:
            List[Dict]: 匹配的实体列表
        """
        query = query.lower()
        matches = []

        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]

            # 名称匹配
            if query in node.lower():
                score = 1.0 if query == node.lower() else 0.8
                matches.append({
                    "entity": node,
                    "type": node_data.get("entity_type", "未分类"),
                    "description": node_data.get("description", ""),
                    "doc_count": node_data.get("doc_count", 0),
                    "score": score
                })
            # 描述匹配
            elif query in node_data.get("description", "").lower():
                matches.append({
                    "entity": node,
                    "type": node_data.get("entity_type", "未分类"),
                    "description": node_data.get("description", ""),
                    "doc_count": node_data.get("doc_count", 0),
                    "score": 0.6
                })

        # 按分数排序
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:limit]

    def get_entity_relations(self, entity: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取实体的所有关系

        Args:
            entity: 实体名称

        Returns:
            Dict: 实体的关系信息
        """
        if not self.graph.has_node(entity):
            return {"outgoing": [], "incoming": []}

        outgoing = []
        incoming = []

        # 出边（主体关系）
        for target in self.graph.successors(entity):
            for edge_data in self.graph[entity][target].values():
                outgoing.append({
                    "target": target,
                    "predicate": edge_data.get("predicate", ""),
                    "description": edge_data.get("description", ""),
                    "doc_id": edge_data.get("doc_id", "")
                })

        # 入边（客体关系）
        for source in self.graph.predecessors(entity):
            for edge_data in self.graph[source][entity].values():
                incoming.append({
                    "source": source,
                    "predicate": edge_data.get("predicate", ""),
                    "description": edge_data.get("description", ""),
                    "doc_id": edge_data.get("doc_id", "")
                })

        return {"outgoing": outgoing, "incoming": incoming}

    def get_related_entities(self, entity: str, max_distance: int = 2) -> List[Dict[str, Any]]:
        """
        获取相关实体（基于图距离）

        Args:
            entity: 实体名称
            max_distance: 最大距离

        Returns:
            List[Dict]: 相关实体列表
        """
        if not self.graph.has_node(entity):
            return []

        related = []

        # 使用BFS获取指定距离内的实体
        try:
            # 转换为无向图进行距离计算
            undirected_graph = self.graph.to_undirected()
            distances = nx.single_source_shortest_path_length(
                undirected_graph, entity, cutoff=max_distance
            )

            for related_entity, distance in distances.items():
                if related_entity != entity and distance <= max_distance:
                    node_data = self.graph.nodes[related_entity]
                    related.append({
                        "entity": related_entity,
                        "type": node_data.get("entity_type", "未分类"),
                        "description": node_data.get("description", ""),
                        "distance": distance,
                        "doc_count": node_data.get("doc_count", 0)
                    })

            # 按距离和文档数量排序
            related.sort(key=lambda x: (x["distance"], -x["doc_count"]))

        except Exception as e:
            print(f"获取相关实体失败: {e}")

        return related

    def get_entity_documents(self, entity: str) -> List[str]:
        """
        获取包含指定实体的文档列表

        Args:
            entity: 实体名称

        Returns:
            List[str]: 文档ID列表
        """
        return list(self.entity_docs.get(entity, set()))

    def graph_retrieval(self, query: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        """
        基于知识图谱的检索

        Args:
            query: 查询语句
            top_k: 返回结果数量

        Returns:
            List[Tuple[str, float, str]]: (doc_id, score, reason)
        """
        # 1. 搜索相关实体
        matched_entities = self.search_entities(query, limit=20)

        if not matched_entities:
            return []

        # 2. 收集相关文档
        doc_scores = defaultdict(float)
        doc_reasons = defaultdict(list)

        for entity_info in matched_entities:
            entity = entity_info["entity"]
            entity_score = entity_info["score"]

            # 直接匹配的文档
            for doc_id in self.get_entity_documents(entity):
                doc_scores[doc_id] += entity_score
                doc_reasons[doc_id].append(f"包含实体: {entity}")

            # 相关实体的文档
            related_entities = self.get_related_entities(entity, max_distance=2)
            for related_info in related_entities[:5]:  # 限制相关实体数量
                related_entity = related_info["entity"]
                distance = related_info["distance"]

                # 距离越近，分数越高
                related_score = entity_score * (1.0 / (distance + 1)) * 0.5

                for doc_id in self.get_entity_documents(related_entity):
                    doc_scores[doc_id] += related_score
                    doc_reasons[doc_id].append(f"相关实体: {related_entity} (距离: {distance})")

        # 3. 排序和返回
        results = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            reason = "; ".join(doc_reasons[doc_id][:3])  # 限制原因数量
            results.append((doc_id, score, reason))

        return results

    def save_graph(self, filepath: Optional[str] = None):
        """
        保存知识图谱

        Args:
            filepath: 保存路径
        """
        if filepath is None:
            filepath = self.graph_file

        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            graph_data = {
                "graph": self.graph,
                "entity_docs": dict(self.entity_docs),
                "doc_entities": dict(self.doc_entities),
                "relation_types": dict(self.relation_types),
                "entity_types": dict(self.entity_types),
                "saved_at": datetime.now().isoformat()
            }

            with open(filepath, 'wb') as f:
                pickle.dump(graph_data, f)

            print(f"知识图谱已保存到: {filepath}")

        except Exception as e:
            print(f"保存知识图谱失败: {e}")

    def load_from_json_file(self, filepath: str) -> bool:
        """
        从JSON文件加载预置知识图谱
        支持两种结构：
        1) 导出格式：{"entities": [...], "relations": [...]}（与export_graph_data一致）
        2) 简化三元组列表：{"triples": [{"subject":..., "predicate":..., "object":...}, ...]}
        """
        try:
            if not os.path.exists(filepath):
                print(f"预置图谱文件不存在: {filepath}")
                return False
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 清空现有图
            self.clear_graph()
            # 情况1：包含entities/relations
            if isinstance(data, dict) and 'entities' in data and 'relations' in data:
                for ent in data.get('entities', []):
                    self.add_entity(
                        entity_name=ent.get('name', ''),
                        entity_type=ent.get('type', '未分类'),
                        description=ent.get('description', ''),
                        doc_id=None
                    )
                    # 文档映射（如果提供）
                    for did in ent.get('documents', []):
                        self.entity_docs[ent.get('name', '')].add(did)
                        self.doc_entities[did].add(ent.get('name', ''))
                for rel in data.get('relations', []):
                    self.add_relation(
                        subject=rel.get('subject', ''),
                        predicate=rel.get('predicate', ''),
                        object_entity=rel.get('object', ''),
                        description=rel.get('description', ''),
                        doc_id=rel.get('doc_id', None)
                    )
            # 情况2：triples 列表
            elif isinstance(data, dict) and 'triples' in data:
                for t in data.get('triples', []):
                    s = (t.get('subject') or '').strip()
                    p = (t.get('predicate') or '').strip()
                    o = (t.get('object') or '').strip()
                    if s:
                        self.add_entity(s, '未分类')
                    if o:
                        self.add_entity(o, '未分类')
                    if s and p and o:
                        self.add_relation(s, p, o)
            else:
                print("不支持的预置图谱JSON结构")
                return False
            print(f"✅ 预置知识图谱加载完成：{self.graph.number_of_nodes()} 个实体，{self.graph.number_of_edges()} 条关系")
            return True
        except Exception as e:
            print(f"加载预置知识图谱失败: {e}")
            return False


    def load_from_openkg_triples(self, filepath: str, max_triples: int = 5000) -> bool:
        """
        从 OpenKG 三元组文件加载（TSV 格式：subject \t predicate \t object）
        仅加载前 max_triples 条，避免内存占用过大
        """
        try:
            if not os.path.exists(filepath):
                print(f"预置OpenKG三元组文件不存在: {filepath}")
                return False
            # 清空现有图
            self.clear_graph()
            loaded = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_triples and loaded >= max_triples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) < 3:
                        continue
                    subject = parts[0].strip()
                    predicate = parts[1].strip()
                    obj = parts[2].strip()
                    if subject and predicate and obj:
                        # 简单的类型标注为"未分类"
                        if not self.graph.has_node(subject):
                            self.add_entity(subject, '未分类')
                        if not self.graph.has_node(obj):
                            self.add_entity(obj, '未分类')
                        self.add_relation(subject, predicate, obj)
                        loaded += 1
            print(f"✅ 预置OpenKG图谱加载完成：{self.graph.number_of_nodes()} 个实体，{self.graph.number_of_edges()} 条关系（载入三元组 {loaded} 条）")
            return loaded > 0
        except Exception as e:
            print(f"加载OpenKG三元组失败: {e}")
            return False

    def load_graph(self, filepath: Optional[str] = None):
        """
        加载知识图谱

        Args:
            filepath: 加载路径
        """
        if filepath is None:
            filepath = self.graph_file

        if not os.path.exists(filepath):
            print(f"知识图谱文件不存在: {filepath}")
            return

        try:
            with open(filepath, 'rb') as f:
                graph_data = pickle.load(f)

            self.graph = graph_data.get("graph", nx.MultiDiGraph())
            self.entity_docs = defaultdict(set, {k: set(v) for k, v in graph_data.get("entity_docs", {}).items()})
            self.doc_entities = defaultdict(set, {k: set(v) for k, v in graph_data.get("doc_entities", {}).items()})
            self.relation_types = Counter(graph_data.get("relation_types", {}))
            self.entity_types = Counter(graph_data.get("entity_types", {}))

            print(f"知识图谱已加载: {self.graph.number_of_nodes()} 个实体，{self.graph.number_of_edges()} 条关系")

        except Exception as e:
            print(f"加载知识图谱失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识图谱统计信息

        Returns:
            Dict: 统计信息
        """
        return {
            "entity_count": self.graph.number_of_nodes(),
            "relation_count": self.graph.number_of_edges(),
            "entity_types": dict(self.entity_types),
            "relation_types": dict(self.relation_types),
            "document_count": len(self.doc_entities),
            "avg_entities_per_doc": len(self.doc_entities) and sum(len(entities) for entities in self.doc_entities.values()) / len(self.doc_entities) or 0,
            "avg_relations_per_entity": self.graph.number_of_nodes() and self.graph.number_of_edges() / self.graph.number_of_nodes() or 0
        }

    def clear_graph(self):
        """清空知识图谱"""
        self.graph.clear()
        self.entity_docs.clear()
        self.doc_entities.clear()
        self.relation_types.clear()
        self.entity_types.clear()
        print("知识图谱已清空")

    def export_graph_data(self) -> Dict[str, Any]:
        """
        导出图谱数据

        Returns:
            Dict: 图谱数据
        """
        entities = []
        relations = []

        # 导出实体
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            entities.append({
                "name": node,
                "type": node_data.get("entity_type", "未分类"),
                "description": node_data.get("description", ""),
                "doc_count": node_data.get("doc_count", 0),
                "documents": list(self.entity_docs.get(node, set()))
            })

        # 导出关系
        for source, target in self.graph.edges():
            for edge_data in self.graph[source][target].values():
                relations.append({
                    "subject": source,
                    "predicate": edge_data.get("predicate", ""),
                    "object": target,
                    "description": edge_data.get("description", ""),
                    "doc_id": edge_data.get("doc_id", "")
                })

        return {
            "entities": entities,
            "relations": relations,
            "stats": self.get_stats(),
            "exported_at": datetime.now().isoformat()
        }