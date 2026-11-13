#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱检索服务 —— 仅使用 JanusGraph（Gremlin Server）

Janus 为**唯一**图数据存储；`data/openkg_triples.tsv` 等仅当图中无业务实体时用于**启动补种**。
须安装 gremlinpython，并启动 JanusGraph（默认 ws://localhost:8182/gremlin）。
环境变量：JANUSGRAPH_URL 或 GREMLIN_SERVER_URL 覆盖地址。
"""

import logging
import os
import json
import time
import threading
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING
from datetime import datetime
from .ner_service import NERService

if TYPE_CHECKING:
    from .janusgraph_backend import JanusGraphAdapter

logger = logging.getLogger(__name__)

_DEFAULT_OPENKG_PREDICATE_FILE = os.path.join("data", "openkg_triples.tsv")
_PREDICATE_ALIAS_TO_OPENKG = {
    "主演": "演员",
    "cast": "演员",
    "starring": "演员",
    "actor": "演员",
    "actors": "演员",
    "director": "导演",
    "directors": "导演",
    "writer": "编剧",
    "writers": "编剧",
    "screenwriter": "编剧",
    "genre": "类型",
    "genres": "类型",
    "category": "类型",
    "country": "国家",
    "countries": "国家",
    "district": "地区",
    "language": "语言",
    "languages": "语言",
    "year": "上映年份",
    "showtime": "上映年份",
    "release_date": "上映日期",
    "runtime": "片长",
    "length": "片长",
    "rating": "豆瓣评分",
    "rate": "豆瓣评分",
    "douban_rating": "豆瓣评分",
    "alias": "别名",
    "aka": "别名",
    "othername": "别名",
}


def _gremlin_url(janusgraph_url: Optional[str]) -> str:
    if janusgraph_url:
        return janusgraph_url
    return os.environ.get("JANUSGRAPH_URL") or os.environ.get("GREMLIN_SERVER_URL") or "ws://localhost:8182/gremlin"


class KGRetrievalService:
    """知识图谱检索服务 —— JanusGraph / Gremlin 后端（不再使用 NetworkX）"""

    def __init__(
        self,
        graph_file: str = "data/knowledge_graph_janusgraph_export.json",
        api_type: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        janusgraph_url: Optional[str] = None,
    ):
        """
        Args:
            graph_file: 导出/备份 JSON 路径（save_graph 写入；非 pickle）
            janusgraph_url: Gremlin WebSocket 地址，缺省见环境变量或 localhost:8182
        """
        self.ner_service = NERService.try_create(
            api_type=api_type,
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
        )
        if self.ner_service is None:
            logger.warning("NER 服务未启用（无 LLM API 密钥等），图谱检索仍可用")

        self.graph_file = graph_file
        self._gremlin_url = _gremlin_url(janusgraph_url)

        from .janusgraph_backend import JanusGraphAdapter, is_janusgraph_available

        if not is_janusgraph_available():
            raise RuntimeError(
                "知识图谱需要 gremlinpython。请执行: pip install 'gremlinpython>=3.7.0' "
                "并启动 JanusGraph（或 Gremlin Server），默认地址 ws://localhost:8182/gremlin"
            )
        last_err: Optional[Exception] = None
        kg: Optional["JanusGraphAdapter"] = None
        for attempt in range(25):
            try:
                kg = JanusGraphAdapter(self._gremlin_url)
                last_err = None
                break
            except Exception as e:
                last_err = e
                wait_s = min(2 + attempt // 5, 8)
                logger.warning(
                    "Gremlin 连接重试 %s/25（%s）: %s；%ss 后再试",
                    attempt + 1,
                    self._gremlin_url,
                    e,
                    wait_s,
                )
                time.sleep(wait_s)
        if last_err is not None or kg is None:
            raise RuntimeError(
                f"无法连接 Gremlin Server（{self._gremlin_url}）。请确认 JanusGraph 已启动且端口正确: {last_err}"
            ) from last_err
        self.knowledge_graph = kg
        self._kg_reload_lock = threading.RLock()

        self._ensure_kg_in_janus_at_startup()

    def _entity_count_in_janus(self) -> int:
        try:
            st = self.knowledge_graph.get_stats()
            return int(st.get("entity_count", 0) or 0)
        except Exception as e:
            logger.warning("读取 Janus 实体数失败: %s", e)
            return 0

    def _sync_is_graph_built_from_janus(self) -> None:
        """
        以 Janus 中业务实体数为准更新 is_graph_built。避免「内存标志为 False 但库里已有图」
        时 search/query 直接短路返回空（用户表现为搜不到任何实体）。
        """
        self.is_graph_built = self._entity_count_in_janus() > 0

    def _ensure_kg_in_janus_at_startup(self) -> None:
        """
        以 Janus 为准：若其中已有业务实体（label=entity），不覆盖，仅设 is_graph_built。
        若为空头图，则依次尝试 data/openkg_triples.tsv、data/preloaded_knowledge_graph.json 导入。
        """
        n = self._entity_count_in_janus()
        if n > 0:
            self.is_graph_built = True
            logger.info(
                "知识图谱：Janus 中已有 %s 个业务实体，作为唯一数据源，跳过预置文件导入。",
                n,
            )
            return

        loaded = False
        openkg_path = os.path.join("data", "openkg_triples.tsv")
        if os.path.exists(openkg_path):
            try:
                loaded = self.knowledge_graph.load_from_openkg_triples(openkg_path, max_triples=3000)
            except Exception as e:
                logger.warning("预置 OpenKG 写入 Janus 失败: %s", e)
            if loaded:
                logger.info("已从预置补种 Janus: %s", openkg_path)
        if not loaded:
            preloaded_kg_path = os.path.join("data", "preloaded_knowledge_graph.json")
            if os.path.exists(preloaded_kg_path):
                try:
                    loaded = self.knowledge_graph.load_from_json_file(preloaded_kg_path)
                except Exception as e:
                    logger.warning("预置 JSON 知识图谱加载失败: %s", e)
                if loaded:
                    logger.info("已从预置补种 Janus: %s", preloaded_kg_path)
        n_after = self._entity_count_in_janus()
        self.is_graph_built = n_after > 0
        if not self.is_graph_built:
            if not os.path.exists(openkg_path) and not os.path.exists(
                os.path.join("data", "preloaded_knowledge_graph.json")
            ):
                logger.warning(
                    "Janus 中无业务实体且未找到 data/openkg_triples.tsv 或 data/preloaded_knowledge_graph.json，"
                    "图谱为空。请放入预置文件后重载或从 UI「从预置重新加载」。"
                )
            else:
                logger.warning(
                    "Janus 中仍无业务实体（预置导入可能失败或文件为空），请检查 Gremlin 日志与数据文件。"
                )

    def _check_graph_exists(self) -> bool:
        return self.knowledge_graph.graph.number_of_nodes() > 0

    def build_knowledge_graph(self, documents: Dict[str, str], model: Optional[str] = None) -> Dict[str, Any]:
        return {"error": "KG building is disabled in this version. A preloaded KG is used if available."}

    def rebuild_knowledge_graph(self, documents: Dict[str, str], model: Optional[str] = None) -> Dict[str, Any]:
        return {"error": "Rebuild is disabled. Use the preloaded KG or clear and reload the file."}

    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        return self.query_entity_relations(entity_name)

    def query_entity_relations(self, entity_name: str) -> Dict[str, Any]:
        entity_name = (entity_name or "").strip()
        if not entity_name:
            return {"error": "空实体名", "entity": "", "exists": False}
        self._sync_is_graph_built_from_janus()
        if not self.is_graph_built:
            return {"error": "知识图谱未构建", "entity": entity_name, "exists": False}

        if not self.knowledge_graph.graph.has_node(entity_name):
            return {"error": "实体不存在", "entity": entity_name, "exists": False}

        node_data = self.knowledge_graph.graph.nodes[entity_name]
        relations = self.knowledge_graph.get_entity_relations(entity_name)
        related_entities = self.knowledge_graph.get_related_entities(entity_name, max_distance=2)
        documents = self.knowledge_graph.get_entity_documents(entity_name)

        return {
            "entity": entity_name,
            "exists": True,
            "type": node_data.get("entity_type", "未分类"),
            "description": node_data.get("description", ""),
            "doc_count": node_data.get("doc_count", 0),
            "documents": documents,
            "relations": relations,
            "related_entities": related_entities,
            "created_at": node_data.get("created_at", ""),
        }

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        self._sync_is_graph_built_from_janus()
        if not self.is_graph_built:
            return []
        return self.knowledge_graph.search_entities(query, limit)

    def analyze_query_entities(self, query: str, model: Optional[str] = None) -> Dict[str, Any]:
        entities = self.search_entities(query, limit=10)
        return {"query": query, "entities": entities, "model": model}

    @staticmethod
    def _load_openkg_predicates(filepath: str = _DEFAULT_OPENKG_PREDICATE_FILE) -> set[str]:
        """
        从 openkg_triples.tsv 读取当前库内可用谓词集合，作为 NER 入库本体约束。
        """
        predicates: set[str] = set()
        if not os.path.exists(filepath):
            return predicates
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    p = parts[1].strip()
                    if p:
                        predicates.add(p)
        except Exception as e:
            logger.warning("读取 OpenKG 谓词集合失败: %s", e)
        return predicates

    @staticmethod
    def _normalize_predicate_to_openkg(predicate: str, openkg_predicates: set[str]) -> str:
        raw = (predicate or "").strip()
        if not raw:
            return ""
        if raw in openkg_predicates:
            return raw
        alias = _PREDICATE_ALIAS_TO_OPENKG.get(raw.lower()) or _PREDICATE_ALIAS_TO_OPENKG.get(raw)
        if alias and alias in openkg_predicates:
            return alias
        return ""

    def extract_text_triples_for_review(
        self,
        text: str,
        model: Optional[str] = None,
        max_items: int = 200,
    ) -> Dict[str, Any]:
        """
        对单段文本做 NER 抽取，返回可供 UI 勾选审阅的三元组清单。
        仅抽取，不写入图谱。
        """
        content = (text or "").strip()
        if not content:
            return {"success": False, "error": "输入文本为空", "triples": []}
        if self.ner_service is None:
            return {"success": False, "error": "NER 未启用（缺少 API Key 或配置）", "triples": []}
        openkg_predicates = self._load_openkg_predicates()
        try:
            ner_res = self.ner_service.extract_entities_and_relations(
                content,
                model=model,
                ontology_predicates=sorted(openkg_predicates),
            )
        except Exception as e:
            return {"success": False, "error": f"NER 调用失败: {e}", "triples": []}
        if ner_res.get("error"):
            return {"success": False, "error": str(ner_res.get("error")), "triples": []}

        entity_type_map: Dict[str, str] = {}
        for ent in ner_res.get("entities", []) or []:
            n = str(ent.get("name", "")).strip()
            t = str(ent.get("type", "未分类")).strip() or "未分类"
            if n and n not in entity_type_map:
                entity_type_map[n] = t

        triples: List[Dict[str, Any]] = []
        seen = set()
        filtered_out = 0
        review_total = 0
        for rel in ner_res.get("relations", []) or []:
            s = str(rel.get("subject", "")).strip()
            p_raw = str(rel.get("predicate", "")).strip()
            o = str(rel.get("object", "")).strip()
            d = str(rel.get("description", "")).strip()
            if not (s and p_raw and o):
                continue
            review_total += 1
            p_norm = self._normalize_predicate_to_openkg(p_raw, openkg_predicates)
            can_insert = bool(p_norm)
            if not can_insert:
                filtered_out += 1
            key = (s, p_raw, o)
            if key in seen:
                continue
            seen.add(key)
            triples.append(
                {
                    "selected": False,
                    "subject": s,
                    "predicate_raw": p_raw,
                    "predicate": p_norm if p_norm else p_raw,
                    "object": o,
                    "subject_type": entity_type_map.get(s, "未分类"),
                    "object_type": entity_type_map.get(o, "未分类"),
                    "description": d,
                    "can_insert": can_insert,
                    "reject_reason": "" if can_insert else "谓词不在 OpenKG 本体集合",
                }
            )
            if len(triples) >= max(1, int(max_items)):
                break
        return {
            "success": True,
            "triples": triples,
            "entities_count": len(entity_type_map),
            "relations_count": len(triples),
            "raw_relations_count": review_total,
            "filtered_out_count": filtered_out,
            "openkg_predicate_count": len(openkg_predicates),
            "model": model or self.ner_service.default_model,
        }

    def insert_selected_triples(self, triples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        将 UI 勾选后的三元组写入 JanusGraph。
        """
        if not triples:
            return {"success": False, "error": "没有可写入的三元组", "inserted": 0}
        inserted = 0
        skipped = 0
        openkg_predicates = self._load_openkg_predicates()
        for item in triples:
            if not bool(item.get("selected", False)):
                skipped += 1
                continue
            s = str(item.get("subject", "")).strip()
            p_raw = str(item.get("predicate_raw", "")).strip() or str(item.get("predicate", "")).strip()
            p = self._normalize_predicate_to_openkg(p_raw, openkg_predicates)
            o = str(item.get("object", "")).strip()
            d = str(item.get("description", "")).strip()
            st = str(item.get("subject_type", "未分类")).strip() or "未分类"
            ot = str(item.get("object_type", "未分类")).strip() or "未分类"
            if not (s and p and o):
                skipped += 1
                continue
            # 先写实体类型，再写关系，便于后续前端按 type 筛选
            self.knowledge_graph.add_entity(s, st, "", doc_id=None)
            self.knowledge_graph.add_entity(o, ot, "", doc_id=None)
            self.knowledge_graph.add_relation(s, p, o, description=d, doc_id=None)
            inserted += 1
        self._sync_is_graph_built_from_janus()
        return {"success": inserted > 0, "inserted": inserted, "skipped": skipped}

    def get_graph_stats(self) -> Dict[str, Any]:
        self._sync_is_graph_built_from_janus()
        base_stats = self.knowledge_graph.get_stats()
        return {
            **base_stats,
            "is_graph_built": self.is_graph_built,
            "graph_file": self.graph_file,
            "graph_backend": "janusgraph",
            "gremlin_url": self._gremlin_url,
            "ner_service": (
                self.ner_service.get_stats()
                if self.ner_service
                else {"disabled": True, "reason": "no_llm_api_key"}
            ),
        }

    def export_graph(self, format: str = "json") -> Tuple[Optional[str], str]:
        self._sync_is_graph_built_from_janus()
        if not self.is_graph_built:
            return None, "知识图谱未构建"

        try:
            graph_data = self.knowledge_graph.export_graph_data()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"knowledge_graph_export_{timestamp}.json"
            filepath = os.path.join("data", filename)
            os.makedirs("data", exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            return filepath, f"知识图谱导出成功！文件: {filename}"
        except Exception as e:
            return None, f"导出失败: {str(e)}"

    def clear_graph(self) -> str:
        self.knowledge_graph.clear_graph()
        self.is_graph_built = False
        return "知识图谱已清空。可从预置数据重新加载或重新构建。"

    def reload_from_preloaded(self) -> Dict[str, Any]:
        with self._kg_reload_lock:
            # 0 表示全量导入，满足“仅初始化时一次性灌入”的场景
            res = self.knowledge_graph.reimport_from_preloaded(0)
            if res.get("success"):
                self.is_graph_built = int(res.get("entity_count", 0) or 0) > 0
            else:
                self._sync_is_graph_built_from_janus()
            return res

    def add_entity(self, entity_name: str, entity_type: str = "未分类", description: str = "") -> Dict[str, Any]:
        entity_name = (entity_name or "").strip()
        if not entity_name:
            return {"success": False, "error": "实体名称不能为空"}
        self.knowledge_graph.add_entity(entity_name, entity_type, description, doc_id=None)
        self.is_graph_built = True
        return {"success": True, "entity": entity_name, "message": "实体已添加"}

    def add_relation(self, subject: str, predicate: str, object_entity: str, description: str = "") -> Dict[str, Any]:
        subject, object_entity, predicate = (subject or "").strip(), (object_entity or "").strip(), (predicate or "").strip()
        if not all([subject, predicate, object_entity]):
            return {"success": False, "error": "主体、关系类型、客体均不能为空"}
        self.knowledge_graph.add_relation(subject, predicate, object_entity, description=description, doc_id=None)
        self.is_graph_built = True
        return {"success": True, "message": "关系已添加"}

    def remove_entity(self, entity_name: str) -> Dict[str, Any]:
        entity_name = (entity_name or "").strip()
        if not entity_name:
            return {"success": False, "error": "实体名称不能为空"}
        if not self.is_graph_built or not self.knowledge_graph.graph.has_node(entity_name):
            return {"success": False, "error": "实体不存在"}
        removed = self.knowledge_graph.remove_entity(entity_name)
        return {"success": removed, "entity": entity_name, "message": "实体已删除" if removed else "实体不存在"}

    def remove_relation(self, subject: str, predicate: str, object_entity: str) -> Dict[str, Any]:
        subject, object_entity, predicate = (subject or "").strip(), (object_entity or "").strip(), (predicate or "").strip()
        if not all([subject, predicate, object_entity]):
            return {"success": False, "error": "主体、关系类型、客体均不能为空"}
        if not self.is_graph_built:
            return {"success": False, "error": "知识图谱未构建"}
        removed = self.knowledge_graph.remove_relation(subject, predicate, object_entity)
        return {"success": removed, "message": "关系已删除" if removed else "未找到该关系"}

    def save_graph(self) -> Tuple[bool, str]:
        if not self.is_graph_built:
            return False, "知识图谱未构建或为空，无需保存"
        try:
            self.knowledge_graph.save_graph(self.graph_file)
            return True, f"已导出到 {self.graph_file}（JanusGraph 数据主要由服务端持久化）"
        except Exception as e:
            return False, str(e)

    def register_financial_knowledge_ontology_services(self, index_service: Any) -> Dict[str, Any]:
        """
        将「金融知识」域的检索/索引能力注册到 JanusGraph ontology_service 顶点，
        与 default_ontology 中 ontology_domain=financial_knowledge 及 janus_service_id 对齐。
        """
        kg = self.knowledge_graph
        if not hasattr(kg, "upsert_ontology_service"):
            return {"registered": False, "reason": "no_janus_adapter"}
        meta: Dict[str, Any] = {}
        try:
            st = index_service.get_stats()
            meta = {
                "total_documents": st.get("total_documents"),
                "index_file": getattr(index_service, "index_file", ""),
            }
        except Exception:
            pass
        sid = "financial_knowledge_lexical"
        ok = bool(
            kg.upsert_ontology_service(
                service_id=sid,
                ontology_domain="financial_knowledge",
                role="lexical_search",
                interface_api="LexicalIndexedContent",
                action_apis=["LexicalIndexSearch", "RefreshFinancialKnowledgeIndex"],
                status="available",
                endpoint_hint="in_process:index_service",
                meta=meta,
            )
        )
        if ok:
            kg.heartbeat_ontology_service(sid)
        return {"registered": ok, "service_id": sid}

    def get_graph_visualization_data(self) -> Dict[str, Any]:
        self._sync_is_graph_built_from_janus()
        if not self.is_graph_built:
            return {"nodes": [], "edges": []}
        # Janus 为唯一源：从 Gremlin 后端拉全量 nodes/edges，不依赖未同步的 NetworkX 门面
        try:
            return self.knowledge_graph.janus_backend.get_visualization_data()
        except Exception as e:
            logger.warning("get_graph_visualization_data 从 Janus 拉取失败: %s", e)
            return {"nodes": [], "edges": []}
