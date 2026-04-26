#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JanusGraph / Gremlin 图后端（商业级）
通过 Gremlin Server 连接 JanusGraph 或兼容服务，提供与 KnowledgeGraph 一致的图操作接口。
支持连接超时、重试、扫描上限与统一异常处理。
"""

import os
import json
import time
import sys
import logging
import threading
from typing import List, Dict, Tuple, Optional, Any, Set
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from gremlin_python.process.anonymous_traversal import traversal
    from gremlin_python.process.graph_traversal import __
    from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
    from gremlin_python.driver import serializer as gremlin_serializer
    _GREMLIN_AVAILABLE = True
except ImportError:
    _GREMLIN_AVAILABLE = False
    gremlin_serializer = None  # type: ignore
    __ = None  # type: ignore

# 连接与扫描配置（可经环境变量覆盖）
CONNECT_TIMEOUT_SEC = int(os.environ.get("JANUSGRAPH_CONNECT_TIMEOUT", "15"))
CONNECT_RETRIES = int(os.environ.get("JANUSGRAPH_CONNECT_RETRIES", "3"))
CONNECT_RETRY_DELAY_SEC = float(os.environ.get("JANUSGRAPH_RETRY_DELAY", "1.0"))
SEARCH_MAX_VERTICES = int(os.environ.get("JANUSGRAPH_SEARCH_MAX_VERTICES", "50000"))
# 脚本/事务在 Berkeley DB Java Edition（Janus 存储）上偶发死锁/ victim，有限次退避
SCRIPT_SUBMIT_RETRIES = int(os.environ.get("JANUSGRAPH_SCRIPT_RETRIES", "4"))
SCRIPT_SUBMIT_BASE_DELAY = float(os.environ.get("JANUSGRAPH_SCRIPT_RETRY_DELAY", "0.1"))
IMPORT_PROGRESS_EVERY = int(os.environ.get("JANUSGRAPH_IMPORT_PROGRESS_EVERY", "1000"))
IMPORT_SLEEP_EVERY = int(os.environ.get("JANUSGRAPH_IMPORT_SLEEP_EVERY", "500"))
IMPORT_SLEEP_MS = float(os.environ.get("JANUSGRAPH_IMPORT_SLEEP_MS", "5"))
IMPORT_RESUME_ENABLED = os.environ.get("JANUSGRAPH_IMPORT_RESUME", "1").strip().lower() not in {"0", "false", "no"}
IMPORT_CHECKPOINT_FILE = os.environ.get(
    "JANUSGRAPH_IMPORT_CHECKPOINT_FILE",
    os.path.join("data", "janus_import_checkpoint.json"),
)
IMPORT_PROGRESS_BAR_ENABLED = os.environ.get("JANUSGRAPH_IMPORT_PROGRESS_BAR", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
IMPORT_PROGRESS_BAR_WIDTH = int(os.environ.get("JANUSGRAPH_IMPORT_PROGRESS_BAR_WIDTH", "30"))
IMPORT_PROGRESS_BAR_EVERY = int(os.environ.get("JANUSGRAPH_IMPORT_PROGRESS_BAR_EVERY", "50"))

# 顶点标签与属性
VERTEX_LABEL = "entity"
VERTEX_PROP_NAME = "name"
VERTEX_PROP_TYPE = "entity_type"
VERTEX_PROP_DESC = "description"
VERTEX_PROP_DOC_COUNT = "doc_count"
VERTEX_PROP_CREATED = "created_at"

# 边标签与属性（边 label 统一为 relation，关系类型存于 predicate 属性）
EDGE_LABEL = "relation"
EDGE_PROP_PREDICATE = "predicate"
EDGE_PROP_DESC = "description"
EDGE_PROP_DOC_ID = "doc_id"
EDGE_PROP_CREATED = "created_at"

# 本体模式（ObjectType / LinkType / Interface / Action）元数据顶点，与业务 entity 顶点分离
ONTOLOGY_DEFINITION_LABEL = "ontology_definition"
ONTOLOGY_SCHEMA_KEY_PROP = "schema_key"
ONTOLOGY_SCHEMA_JSON_PROP = "schema_json"
ONTOLOGY_SCHEMA_VERSION_PROP = "schema_version"
ONTOLOGY_SCHEMA_UPDATED_PROP = "updated_at"
DEFAULT_ONTOLOGY_SCHEMA_KEY = "default"

# 本体检索/索引等服务进程在图中的注册（与业务 entity 顶点分离）
ONTOLOGY_SERVICE_LABEL = "ontology_service"
ONT_SVC_ID = "service_id"
ONT_SVC_DOMAIN = "ontology_domain"
ONT_SVC_ROLE = "role"
ONT_SVC_STATUS = "status"
ONT_SVC_INTERFACE = "interface_api"
ONT_SVC_ACTIONS_JSON = "action_apis_json"
ONT_SVC_ENDPOINT = "endpoint_hint"
ONT_SVC_HEARTBEAT = "last_heartbeat"
ONT_SVC_INDEX_REFRESH = "last_index_refresh_at"
ONT_SVC_META_JSON = "meta_json"


def _default_url() -> str:
    return os.environ.get("JANUSGRAPH_URL", os.environ.get("GREMLIN_SERVER_URL", "ws://localhost:8182/gremlin"))


class JanusGraphBackendError(Exception):
    """JanusGraph 后端异常，用于统一向上层返回可区分错误"""
    pass


class JanusGraphBackend:
    """
    基于 Gremlin 的图后端，可连接 JanusGraph Server 或任意兼容 Gremlin Server。
    实现与 KnowledgeGraph 一致的方法签名，供 KGRetrievalService 复用。
    连接支持超时与重试；搜索支持顶点扫描上限，避免大图 OOM。
    """

    def __init__(
        self,
        gremlin_url: Optional[str] = None,
        connect_timeout_sec: Optional[int] = None,
        connect_retries: Optional[int] = None,
    ):
        if not _GREMLIN_AVAILABLE:
            raise JanusGraphBackendError(
                "gremlinpython 未安装。请执行: pip install gremlinpython>=3.7.0"
            )
        self._url = gremlin_url or _default_url()
        self._connect_timeout = connect_timeout_sec if connect_timeout_sec is not None else CONNECT_TIMEOUT_SEC
        self._connect_retries = connect_retries if connect_retries is not None else CONNECT_RETRIES
        self._g = None
        self._connection = None
        self._script_client: Any = None
        self._entity_docs: Dict[str, set] = {}
        self._doc_entities: Dict[str, set] = {}
        self._janus_lock = threading.RLock()
        self._connect()

    def _connect(self):
        last_error = None
        for attempt in range(1, self._connect_retries + 1):
            try:
                # gremlinpython 默认 GraphBinary 与 JanusGraph 返回的自定义类型不兼容时会报 DataType.custom。
                # Gremlin Server 与 JanusGraph 默认 WebSocket 配置为 GraphSON 3，与下列序列化器一致。
                msg_ser = None
                if _GREMLIN_AVAILABLE and gremlin_serializer is not None:
                    msg_ser = gremlin_serializer.GraphSONSerializersV3d0()
                self._connection = DriverRemoteConnection(
                    self._url,
                    "g",
                    message_serializer=msg_ser,
                )
                self._g = traversal().with_remote(self._connection)
                self._g.V().limit(1).to_list()
                return
            except Exception as e:
                last_error = e
                logger.warning("JanusGraph 连接第 %s 次失败: %s", attempt, e)
                if attempt < self._connect_retries:
                    time.sleep(CONNECT_RETRY_DELAY_SEC)
        raise JanusGraphBackendError(f"无法连接 Gremlin Server {self._url}: {last_error}")

    def _close(self):
        if self._script_client is not None:
            try:
                self._script_client.close()
            except Exception:
                pass
            self._script_client = None
        if self._connection:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
        self._g = None

    def _get_script_client(self):
        if self._script_client is not None:
            return self._script_client
        from gremlin_python.driver import client

        msg_ser = None
        if _GREMLIN_AVAILABLE and gremlin_serializer is not None:
            msg_ser = gremlin_serializer.GraphSONSerializersV3d0()
        self._script_client = client.Client(self._url, "g", message_serializer=msg_ser)
        return self._script_client

    @staticmethod
    def _is_je_transient_error(exc: BaseException) -> bool:
        t = f"{type(exc).__name__} {exc!s}".lower()
        keys = (
            "deadlock",
            "victim",
            " edgestore",
            "com.sleepycat",  # JE
            " lock",
            "timeout: 500ms",  # JE 常见
        )
        return any(k in t for k in keys)

    def _script_submit(self, gremlin: str, bindings: Optional[dict] = None) -> None:
        """
        在 Gremlin 服务端以 **Groovy 脚本** 执行，避免 gremlinpython **bytecode** 中
        drop/删边 在部分 Janus 上被误译为 discard 而报 599。bindings 为脚本中的绑定名。
        调用方应已持有 _janus_lock（同一线程内对图访问串行，降低 JE 层死锁）；
        对 JE 可恢复死锁/ victim 作有限次退避重试。
        """
        c = self._get_script_client()
        last: Optional[BaseException] = None
        for attempt in range(1, SCRIPT_SUBMIT_RETRIES + 1):
            try:
                if bindings is not None:
                    rs = c.submit(gremlin, bindings)
                else:
                    rs = c.submit(gremlin)
                # 必须消费结果，确保真正串行执行，避免请求堆积导致“假串行”锁冲突。
                rs.all().result()
                return
            except Exception as e:
                last = e
                if attempt >= SCRIPT_SUBMIT_RETRIES or not self._is_je_transient_error(e):
                    raise
                d = SCRIPT_SUBMIT_BASE_DELAY * (2 ** (attempt - 1))
                time.sleep(d)
        if last:
            raise last

    @staticmethod
    def _load_import_checkpoint(path: str) -> Optional[Dict[str, Any]]:
        try:
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    @staticmethod
    def _save_import_checkpoint(path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _remove_import_checkpoint(path: str) -> None:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    @staticmethod
    def _print_cli_progress(stage: str, current: int, total: int) -> None:
        if not IMPORT_PROGRESS_BAR_ENABLED:
            return
        t = max(total, 1)
        c = max(0, min(current, t))
        ratio = c / t
        width = max(10, IMPORT_PROGRESS_BAR_WIDTH)
        filled = int(ratio * width)
        bar = "=" * filled + "-" * (width - filled)
        line = f"[{stage}] {c}/{t} ({ratio * 100:6.2f}%) |{bar}|"
        try:
            # 交互终端走单行刷新；被捕获/重定向时改为逐行，确保日志可见。
            if sys.stdout.isatty():
                sys.stdout.write("\r" + line)
                sys.stdout.flush()
                if c >= t:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
            else:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
        except Exception:
            # 在部分非交互终端中写 stdout 失败时静默回退，不影响导入流程
            return

    def _v_by_name(self, name: str):
        """获取名为 name 的实体顶点 traversal（未执行）"""
        return self._g.V().has(VERTEX_LABEL, VERTEX_PROP_NAME, name)

    def _has_entity(self, name: str) -> bool:
        return self._v_by_name(name).has_next()

    def add_entity(
        self,
        entity_name: str,
        entity_type: str,
        description: str = "",
        doc_id: Optional[str] = None,
    ):
        with self._janus_lock:
            entity_name = entity_name.strip()
            if not entity_name:
                return
            if self._has_entity(entity_name):
                ncount = int(self._get_doc_count(entity_name) + 1)
                if description:
                    self._script_submit(
                        "g.V().hasLabel('entity').has('name', nm).property('doc_count', cv).property('description', dv).iterate()",
                        {"nm": entity_name, "cv": ncount, "dv": description},
                    )
                else:
                    self._script_submit(
                        "g.V().hasLabel('entity').has('name', nm).property('doc_count', cv).iterate()",
                        {"nm": entity_name, "cv": ncount},
                    )
            else:
                now = datetime.now().isoformat()
                self._script_submit(
                    "g.addV('entity').property('name', nm).property('entity_type', tv).property('description', dvv).property('doc_count', c1).property('created_at', cre).iterate()",
                    {
                        "nm": entity_name,
                        "tv": entity_type or "未分类",
                        "dvv": description or "",
                        "c1": 1,
                        "cre": now,
                    },
                )
            if doc_id:
                self._entity_docs.setdefault(entity_name, set()).add(doc_id)
                self._doc_entities.setdefault(doc_id, set()).add(entity_name)

    def _get_doc_count(self, name: str) -> int:
        try:
            return self._v_by_name(name).values(VERTEX_PROP_DOC_COUNT).next() or 0
        except (StopIteration, Exception):
            return 0

    def add_relation(
        self,
        subject: str,
        predicate: str,
        object_entity: str,
        description: str = "",
        doc_id: Optional[str] = None,
    ):
        with self._janus_lock:
            subject = subject.strip()
            object_entity = object_entity.strip()
            predicate = predicate.strip()
            if not all([subject, predicate, object_entity]):
                return
            if not self._has_entity(subject):
                self.add_entity(subject, "未分类", "", doc_id)
            if not self._has_entity(object_entity):
                self.add_entity(object_entity, "未分类", "", doc_id)
            now = datetime.now().isoformat()
            # 与实体/边 label 及边属性名用字面值，减少 bindings（Janus 每请求绑定数有上限，通常 16）
            self._script_submit(
                "g.V().hasLabel('entity').has('name', s).as('a').V().hasLabel('entity').has('name', o).as('b').addE('relation').from('a').to('b').property('predicate', pvv).property('description', dvv).property('doc_id', didv).property('created_at', cret).iterate()",
                {
                    "s": subject,
                    "o": object_entity,
                    "pvv": predicate,
                    "dvv": description or "",
                    "didv": doc_id or "",
                    "cret": now,
                },
            )

    def remove_relation(self, subject: str, predicate: str, object_entity: str) -> bool:
        with self._janus_lock:
            subject = subject.strip()
            object_entity = object_entity.strip()
            predicate = predicate.strip()
            if not self._has_entity(subject) or not self._has_entity(object_entity):
                return False
            out_v = self._v_by_name(subject).next()
            for e in self._g.V(out_v).out_e(EDGE_LABEL).to_list():
                try:
                    pred_val = self._g.E(e).values(EDGE_PROP_PREDICATE).next()
                    in_vertex = self._g.E(e).in_v().next()
                    in_name = self._g.V(in_vertex).values(VERTEX_PROP_NAME).next()
                    if pred_val == predicate and in_name == object_entity:
                        try:
                            eid = getattr(e, "id", e)
                            self._script_submit("g.E(eid).drop().iterate()", {"eid": eid})
                        except Exception as ex:
                            logger.warning("remove_relation 删边失败: %s", ex)
                            return False
                        return True
                except Exception:
                    continue
            return False

    def remove_entity(self, entity_name: str) -> bool:
        with self._janus_lock:
            entity_name = entity_name.strip()
            if not self._has_entity(entity_name):
                return False
            v = self._v_by_name(entity_name).next()
            try:
                vid = getattr(v, "id", v)
                self._script_submit("g.V(vid).bothE(lbe).drop().iterate()", {"vid": vid, "lbe": EDGE_LABEL})
                self._script_submit("g.V(vid).drop().iterate()", {"vid": vid})
            except Exception as ex:
                logger.warning("remove_entity 脚本删点失败: %s", ex)
                return False
            self._entity_docs.pop(entity_name, None)
            for doc_id, names in list(self._doc_entities.items()):
                names.discard(entity_name)
                if not names:
                    self._doc_entities.pop(doc_id, None)
            return True

    def _edge_prop(self, e, key: str, default: str = ""):
        try:
            return self._g.E(e).values(key).next()
        except (StopIteration, Exception):
            return default

    def get_entity_relations(self, entity: str) -> Dict[str, List[Dict[str, Any]]]:
        with self._janus_lock:
            if not self._has_entity(entity):
                return {"outgoing": [], "incoming": []}
            # 不绑定 g.E(边) 的 incident 步进（Janus RelationIdentifier 在 Python 驱动下常失败）。
            def v_ent():
                return self._g.V().has_label(VERTEX_LABEL).has(VERTEX_PROP_NAME, entity)

            try:
                tgt_list = v_ent().out_e(EDGE_LABEL).in_v().values(VERTEX_PROP_NAME).to_list()
                p_out = v_ent().out_e(EDGE_LABEL).values(EDGE_PROP_PREDICATE).to_list()
                d_out = v_ent().out_e(EDGE_LABEL).values(EDGE_PROP_DESC).to_list()
                doc_out = v_ent().out_e(EDGE_LABEL).values(EDGE_PROP_DOC_ID).to_list()
            except Exception as ex:
                logger.warning("get_entity_relations 出边枚举失败: %s", ex)
                tgt_list, p_out, d_out, doc_out = [], [], [], []
            m_out = min(len(tgt_list), len(p_out), len(d_out), len(doc_out))
            if len({len(tgt_list), len(p_out), len(d_out), len(doc_out)}) > 1 and m_out:
                logger.warning(
                    "get_entity_relations 出边各列条数: tgt=%s p=%s d=%s did=%s",
                    len(tgt_list), len(p_out), len(d_out), len(doc_out),
                )
            outgoing: List[Dict[str, Any]] = []
            for i in range(m_out):
                try:
                    t = str(tgt_list[i] or "").strip()
                except (IndexError, TypeError):
                    continue
                if not t:
                    continue
                outgoing.append(
                    {
                        "target": t,
                        "predicate": str(p_out[i] or "") if i < len(p_out) else "",
                        "description": str(d_out[i] or "") if i < len(d_out) else "",
                        "doc_id": str(doc_out[i] or "") if i < len(doc_out) else "",
                    }
                )
            try:
                src_list = v_ent().in_e(EDGE_LABEL).out_v().values(VERTEX_PROP_NAME).to_list()
                p_in = v_ent().in_e(EDGE_LABEL).values(EDGE_PROP_PREDICATE).to_list()
                d_in = v_ent().in_e(EDGE_LABEL).values(EDGE_PROP_DESC).to_list()
                doc_in = v_ent().in_e(EDGE_LABEL).values(EDGE_PROP_DOC_ID).to_list()
            except Exception as ex2:
                logger.warning("get_entity_relations 入边枚举失败: %s", ex2)
                src_list, p_in, d_in, doc_in = [], [], [], []
            m_in = min(len(src_list), len(p_in), len(d_in), len(doc_in))
            if len({len(src_list), len(p_in), len(d_in), len(doc_in)}) > 1 and m_in:
                logger.warning(
                    "get_entity_relations 入边各列条数: src=%s p=%s d=%s did=%s",
                    len(src_list), len(p_in), len(d_in), len(doc_in),
                )
            incoming: List[Dict[str, Any]] = []
            for i in range(m_in):
                try:
                    s = str(src_list[i] or "").strip()
                except (IndexError, TypeError):
                    continue
                if not s:
                    continue
                incoming.append(
                    {
                        "source": s,
                        "predicate": str(p_in[i] or "") if i < len(p_in) else "",
                        "description": str(d_in[i] or "") if i < len(d_in) else "",
                        "doc_id": str(doc_in[i] or "") if i < len(doc_in) else "",
                    }
                )
            return {"outgoing": outgoing, "incoming": incoming}

    def _get_vertex_props(self, v) -> Dict[str, Any]:
        def _val(key: str, default: Any = ""):
            try:
                return self._g.V(v).values(key).next()
            except (StopIteration, Exception):
                return default
        name = _val(VERTEX_PROP_NAME, "")
        return {
            "name": name,
            "entity_type": _val(VERTEX_PROP_TYPE, "未分类"),
            "description": _val(VERTEX_PROP_DESC, ""),
            "doc_count": _val(VERTEX_PROP_DOC_COUNT, 0) or 0,
            "created_at": _val(VERTEX_PROP_CREATED, ""),
        }

    def get_related_entities(self, entity: str, max_distance: int = 2) -> List[Dict[str, Any]]:
        with self._janus_lock:
            if not self._has_entity(entity):
                return []
            result = []
            seen = {entity}
            front = [entity]
            for dist in range(1, max_distance + 1):
                next_front = []
                for name in front:
                    v = self._v_by_name(name).next()
                    for nb in self._g.V(v).bothE(EDGE_LABEL).otherV().to_list():
                        nb_name = self._g.V(nb).values(VERTEX_PROP_NAME).next()
                        if nb_name in seen:
                            continue
                        seen.add(nb_name)
                        next_front.append(nb_name)
                        props = self._get_vertex_props(nb)
                        result.append({
                            "entity": nb_name,
                            "type": props.get("entity_type", "未分类"),
                            "description": props.get("description", ""),
                            "distance": dist,
                            "doc_count": props.get("doc_count", 0),
                        })
                front = next_front
            result.sort(key=lambda x: (x["distance"], -x.get("doc_count", 0)))
            return result

    def get_entity_documents(self, entity: str) -> List[str]:
        return list(self._entity_docs.get(entity, set()))

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        query = query.lower().strip()
        if not query:
            return []
        with self._janus_lock:
            matches = []
            try:
                vertices = self._g.V().has_label(VERTEX_LABEL).limit(SEARCH_MAX_VERTICES).to_list()
                if len(vertices) >= SEARCH_MAX_VERTICES:
                    logger.warning("search_entities 达到顶点扫描上限 %s，结果可能不完整", SEARCH_MAX_VERTICES)
                for v in vertices:
                    try:
                        name = self._g.V(v).values(VERTEX_PROP_NAME).next()
                    except StopIteration:
                        continue
                    if name is None:
                        continue
                    name = str(name).strip()
                    if not name:
                        continue
                    props = self._get_vertex_props(v)
                    try:
                        nlower = name.lower()
                    except Exception:
                        nlower = name
                    desc = (props.get("description") or "").lower()
                    score = 0.0
                    if query == nlower:
                        score = 1.0
                    elif query in nlower:
                        score = 0.8
                    elif query in desc:
                        score = 0.6
                    if score > 0:
                        matches.append({
                            "entity": name,
                            "type": props.get("entity_type", "未分类"),
                            "description": props.get("description", ""),
                            "doc_count": props.get("doc_count", 0),
                            "score": score,
                        })
                matches.sort(key=lambda x: x["score"], reverse=True)
                return matches[:limit]
            except Exception as ex:
                logger.warning("search_entities 扫描 Janus 失败: %s", ex)
                return []

    def upsert_ontology_service(
        self,
        service_id: str,
        ontology_domain: str,
        role: str,
        interface_api: str = "",
        action_apis: Optional[List[str]] = None,
        status: str = "available",
        endpoint_hint: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        在 JanusGraph 注册（或覆盖）本体相关服务能力：与业务 ObjectType/Interface 对应的进程侧实例。
        role 示例：lexical_search、index_refresh。
        """
        service_id = (service_id or "").strip()
        if not service_id:
            return False
        with self._janus_lock:
            now = datetime.now().isoformat()
            actions_json = json.dumps(action_apis or [], ensure_ascii=False)
            meta_json = json.dumps(meta or {}, ensure_ascii=False)
            try:
                self._script_submit(
                    "g.V().hasLabel(lb).has(pk, sid).drop().iterate()",
                    {
                        "lb": ONTOLOGY_SERVICE_LABEL,
                        "pk": ONT_SVC_ID,
                        "sid": service_id,
                    },
                )
            except Exception as e:
                logger.warning("清理旧 ontology_service 顶点: %s", e)
            try:
                self._script_submit(
                    "g.addV('ontology_service').property('service_id', sid).property('ontology_domain', odom).property('role', rol).property('interface_api', iface).property('action_apis_json', aj).property('status', st).property('endpoint_hint', eph).property('last_heartbeat', hb).property('last_index_refresh_at', ir).property('meta_json', mj).iterate()",
                    {
                        "sid": service_id,
                        "odom": ontology_domain or "",
                        "rol": role or "",
                        "iface": interface_api or "",
                        "aj": actions_json,
                        "st": status or "available",
                        "eph": endpoint_hint or "",
                        "hb": now,
                        "ir": "",
                        "mj": meta_json,
                    },
                )
                return True
            except Exception as e:
                logger.error("写入 ontology_service 失败: %s", e)
                return False

    def heartbeat_ontology_service(self, service_id: str) -> bool:
        service_id = (service_id or "").strip()
        if not service_id:
            return False
        with self._janus_lock:
            now = datetime.now().isoformat()
            try:
                verts = self._g.V().has_label(ONTOLOGY_SERVICE_LABEL).has(ONT_SVC_ID, service_id).to_list()
                if not verts:
                    return False
                self._script_submit(
                    "g.V().hasLabel(lb).has(pk, sid).property(hk, hv).iterate()",
                    {
                        "lb": ONTOLOGY_SERVICE_LABEL,
                        "pk": ONT_SVC_ID,
                        "sid": service_id,
                        "hk": ONT_SVC_HEARTBEAT,
                        "hv": now,
                    },
                )
                return True
            except Exception as e:
                logger.warning("ontology_service 心跳失败: %s", e)
                return False

    def touch_index_refresh_ontology_service(self, service_id: str) -> bool:
        """索引合并/刷新完成后更新 last_index_refresh_at。"""
        service_id = (service_id or "").strip()
        if not service_id:
            return False
        with self._janus_lock:
            now = datetime.now().isoformat()
            try:
                verts = self._g.V().has_label(ONTOLOGY_SERVICE_LABEL).has(ONT_SVC_ID, service_id).to_list()
                if not verts:
                    return False
                self._script_submit(
                    "g.V().hasLabel(lb).has(pk, sid).property(ik, iv).property(hk, hv).iterate()",
                    {
                        "lb": ONTOLOGY_SERVICE_LABEL,
                        "pk": ONT_SVC_ID,
                        "sid": service_id,
                        "ik": ONT_SVC_INDEX_REFRESH,
                        "iv": now,
                        "hk": ONT_SVC_HEARTBEAT,
                        "hv": now,
                    },
                )
                return True
            except Exception as e:
                logger.warning("ontology_service 索引时间更新失败: %s", e)
                return False

    def set_ontology_service_status(self, service_id: str, status: str) -> bool:
        service_id = (service_id or "").strip()
        if not service_id:
            return False
        with self._janus_lock:
            try:
                verts = self._g.V().has_label(ONTOLOGY_SERVICE_LABEL).has(ONT_SVC_ID, service_id).to_list()
                if not verts:
                    return False
                hb = datetime.now().isoformat()
                self._script_submit(
                    "g.V().hasLabel(lb).has(pk, sid).property(sk, sv).property(hk, hv).iterate()",
                    {
                        "lb": ONTOLOGY_SERVICE_LABEL,
                        "pk": ONT_SVC_ID,
                        "sid": service_id,
                        "sk": ONT_SVC_STATUS,
                        "sv": status,
                        "hk": ONT_SVC_HEARTBEAT,
                        "hv": hb,
                    },
                )
                return True
            except Exception as e:
                logger.warning("ontology_service 状态更新失败: %s", e)
                return False

    def _ontology_svc_prop(self, v, key: str, default: Any = ""):
        try:
            return self._g.V(v).values(key).next()
        except (StopIteration, Exception):
            return default

    def get_ontology_service(self, service_id: str) -> Optional[Dict[str, Any]]:
        service_id = (service_id or "").strip()
        if not service_id:
            return None
        with self._janus_lock:
            try:
                verts = self._g.V().has_label(ONTOLOGY_SERVICE_LABEL).has(ONT_SVC_ID, service_id).to_list()
                if not verts:
                    return None
                v = verts[0]
                raw_actions = self._ontology_svc_prop(v, ONT_SVC_ACTIONS_JSON, "")
                raw_meta = self._ontology_svc_prop(v, ONT_SVC_META_JSON, "")
                actions: List[Any] = []
                meta: Dict[str, Any] = {}
                if raw_actions:
                    try:
                        actions = json.loads(raw_actions)
                    except (TypeError, ValueError):
                        pass
                if raw_meta:
                    try:
                        meta = json.loads(raw_meta)
                    except (TypeError, ValueError):
                        pass
                return {
                    "service_id": self._ontology_svc_prop(v, ONT_SVC_ID),
                    "ontology_domain": self._ontology_svc_prop(v, ONT_SVC_DOMAIN),
                    "role": self._ontology_svc_prop(v, ONT_SVC_ROLE),
                    "interface_api": self._ontology_svc_prop(v, ONT_SVC_INTERFACE),
                    "action_apis": actions,
                    "status": self._ontology_svc_prop(v, ONT_SVC_STATUS),
                    "endpoint_hint": self._ontology_svc_prop(v, ONT_SVC_ENDPOINT),
                    "last_heartbeat": self._ontology_svc_prop(v, ONT_SVC_HEARTBEAT),
                    "last_index_refresh_at": self._ontology_svc_prop(v, ONT_SVC_INDEX_REFRESH),
                    "meta": meta,
                }
            except Exception as e:
                logger.warning("读取 ontology_service 失败: %s", e)
                return None

    def list_ontology_services(self, ontology_domain: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._janus_lock:
            out: List[Dict[str, Any]] = []
            try:
                tr = self._g.V().has_label(ONTOLOGY_SERVICE_LABEL)
                if ontology_domain:
                    tr = tr.has(ONT_SVC_DOMAIN, ontology_domain)
                for v in tr.to_list():
                    sid = self._ontology_svc_prop(v, ONT_SVC_ID, "")
                    if sid:
                        r = self.get_ontology_service(str(sid))
                        if r:
                            out.append(r)
            except Exception as e:
                logger.warning("list_ontology_services: %s", e)
            return out

    def is_ontology_service_available(self, service_id: str) -> bool:
        with self._janus_lock:
            rec = self.get_ontology_service(service_id)
            if not rec:
                return False
            return (rec.get("status") or "").lower() == "available"

    def upsert_ontology_schema(self, schema_dict: Dict[str, Any]) -> bool:
        """
        将完整 Ontology JSON 文档写入 JanusGraph（单例顶点 schema_key=default）。
        与 label=entity 的业务顶点分离，不计入 entity 统计。
        """
        key = DEFAULT_ONTOLOGY_SCHEMA_KEY
        try:
            payload = json.dumps(schema_dict, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.error("ontology schema 序列化失败: %s", e)
            return False
        version = str(schema_dict.get("version", "1.0"))
        with self._janus_lock:
            now = datetime.now().isoformat()
            try:
                self._script_submit(
                    "g.V().hasLabel(lb).has(pk, kv).drop().iterate()",
                    {
                        "lb": ONTOLOGY_DEFINITION_LABEL,
                        "pk": ONTOLOGY_SCHEMA_KEY_PROP,
                        "kv": key,
                    },
                )
            except Exception as e:
                logger.warning("清理旧 ontology_definition 顶点: %s", e)
            try:
                self._script_submit(
                    "g.addV('ontology_definition').property('schema_key', sk).property('schema_json', sj).property('schema_version', ver).property('updated_at', ut).iterate()",
                    {"sk": key, "sj": payload, "ver": version, "ut": now},
                )
                return True
            except Exception as e:
                logger.error("写入 ontology_definition 失败: %s", e)
                return False

    def fetch_ontology_schema(self) -> Optional[Dict[str, Any]]:
        """读取 JanusGraph 中的 Ontology JSON；不存在或损坏则返回 None。"""
        with self._janus_lock:
            try:
                verts = self._g.V().has_label(ONTOLOGY_DEFINITION_LABEL).has(ONTOLOGY_SCHEMA_KEY_PROP, DEFAULT_ONTOLOGY_SCHEMA_KEY).to_list()
                if not verts:
                    return None
                v = verts[0]
                raw = self._g.V(v).values(ONTOLOGY_SCHEMA_JSON_PROP).next()
                if not raw:
                    return None
                return json.loads(raw)
            except Exception as e:
                logger.warning("读取 ontology_definition 失败: %s", e)
                return None

    def get_stats(self) -> Dict[str, Any]:
        with self._janus_lock:
            try:
                vertex_count = self._g.V().has_label(VERTEX_LABEL).count().next()
                edge_count = self._g.E().has_label(EDGE_LABEL).count().next()
                ontology_schema_count = self._g.V().has_label(ONTOLOGY_DEFINITION_LABEL).count().next()
                ontology_service_count = self._g.V().has_label(ONTOLOGY_SERVICE_LABEL).count().next()
            except Exception:
                vertex_count = 0
                edge_count = 0
                ontology_schema_count = 0
                ontology_service_count = 0
            return {
                "entity_count": vertex_count,
                "relation_count": edge_count,
                "ontology_definition_count": ontology_schema_count,
                "ontology_service_count": ontology_service_count,
                "entity_types": {},
                "relation_types": {},
                "document_count": len(self._doc_entities),
                "avg_entities_per_doc": len(self._doc_entities) and sum(len(s) for s in self._doc_entities.values()) / len(self._doc_entities) or 0,
                "avg_relations_per_entity": vertex_count and edge_count / vertex_count or 0,
            }

    def clear_graph(self):
        with self._janus_lock:
            try:
                self._script_submit("g.V().hasLabel(lb).drop().iterate()", {"lb": VERTEX_LABEL})
            except Exception as e:
                print(f"JanusGraph clear_graph: {e}")
            self._entity_docs.clear()
            self._doc_entities.clear()
        print("知识图谱已清空（JanusGraph）")

    def export_graph_data(self) -> Dict[str, Any]:
        with self._janus_lock:
            entities = []
            relations = []
            try:
                for v in self._g.V().has_label(VERTEX_LABEL).to_list():
                    props = self._get_vertex_props(v)
                    entities.append({
                        "name": props["name"],
                        "type": props.get("entity_type", "未分类"),
                        "description": props.get("description", ""),
                        "doc_count": props.get("doc_count", 0),
                        "documents": list(self._entity_docs.get(props["name"], set())),
                    })
                for e in self._g.E().has_label(EDGE_LABEL).to_list():
                    out_v = self._g.E(e).out_v().next()
                    in_v = self._g.E(e).in_v().next()
                    subj = self._g.V(out_v).values(VERTEX_PROP_NAME).next()
                    obj = self._g.V(in_v).values(VERTEX_PROP_NAME).next()
                    relations.append({
                        "subject": subj,
                        "predicate": self._edge_prop(e, EDGE_PROP_PREDICATE),
                        "object": obj,
                        "description": self._edge_prop(e, EDGE_PROP_DESC),
                        "doc_id": self._edge_prop(e, EDGE_PROP_DOC_ID),
                    })
            except Exception as e:
                print(f"export_graph_data: {e}")
            return {
                "entities": entities,
                "relations": relations,
                "stats": self.get_stats(),
                "exported_at": datetime.now().isoformat(),
            }

    def load_from_openkg_triples(self, filepath: str, max_triples: int = 5000) -> bool:
        if not os.path.exists(filepath):
            print(f"预置OpenKG三元组文件不存在: {filepath}")
            return False
        # 空库时跳过 clear，避免在部分 Janus+Gremlin 版本上 drop/删点 bytecode 报 599（discard）导致无法首启导入
        with self._janus_lock:
            try:
                n_ent = int(self.get_stats().get("entity_count", 0) or 0)
            except Exception:
                n_ent = 0
            if n_ent > 0:
                self.clear_graph()
            loaded = 0
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if loaded >= max_triples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    s, p, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    if s and p and o:
                        self.add_entity(s, "未分类")
                        self.add_entity(o, "未分类")
                        self.add_relation(s, p, o)
                        loaded += 1
            print(f"JanusGraph 已从 OpenKG 加载 {loaded} 条三元组")
            return loaded > 0

    def load_from_json_file(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            print(f"预置图谱文件不存在: {filepath}")
            return False
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        with self._janus_lock:
            try:
                n_ent = int(self.get_stats().get("entity_count", 0) or 0)
            except Exception:
                n_ent = 0
            if n_ent > 0:
                self.clear_graph()
            if isinstance(data, dict) and "entities" in data and "relations" in data:
                for ent in data.get("entities", []):
                    self.add_entity(
                        ent.get("name", ""),
                        ent.get("type", "未分类"),
                        ent.get("description", ""),
                        None,
                    )
                    for did in ent.get("documents", []):
                        self._entity_docs.setdefault(ent.get("name", ""), set()).add(did)
                        self._doc_entities.setdefault(did, set()).add(ent.get("name", ""))
                for rel in data.get("relations", []):
                    self.add_relation(
                        rel.get("subject", ""),
                        rel.get("predicate", ""),
                        rel.get("object", ""),
                        rel.get("description", ""),
                        rel.get("doc_id"),
                    )
            elif isinstance(data, dict) and "triples" in data:
                for t in data.get("triples", []):
                    s = (t.get("subject") or "").strip()
                    p = (t.get("predicate") or "").strip()
                    o = (t.get("object") or "").strip()
                    if s:
                        self.add_entity(s, "未分类")
                    if o:
                        self.add_entity(o, "未分类")
                    if s and p and o:
                        self.add_relation(s, p, o)
            else:
                print("不支持的预置图谱 JSON 结构")
                return False
            print("JanusGraph 已从 JSON 加载")
            return True

    def save_graph(self, filepath: Optional[str] = None):
        """JanusGraph 数据由服务端持久化，本方法可导出为 JSON 到 filepath"""
        if not filepath:
            filepath = os.path.join("data", "janusgraph_export.json")
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.export_graph_data(), f, ensure_ascii=False, indent=2)
        print(f"图谱已导出到: {filepath}")

    def get_visualization_data(self) -> Dict[str, Any]:
        with self._janus_lock:
            nodes: List[Dict[str, Any]] = []
            edges: List[Dict[str, Any]] = []
            try:
                vlist = self._g.V().has_label(VERTEX_LABEL).to_list()
            except Exception as e:
                logger.warning("get_visualization_data 枚举顶点失败: %s", e)
                vlist = []
            for v in vlist:
                try:
                    props = self._get_vertex_props(v)
                    name = (props.get("name") or "").strip()
                    if not name:
                        continue
                    nodes.append({
                        "id": name,
                        "label": name,
                        "type": props.get("entity_type", "未分类"),
                        "description": props.get("description", ""),
                        "doc_count": props.get("doc_count", 0),
                    })
                except Exception as ex:
                    logger.debug("get_visualization_data 跳过顶点: %s", ex)
                    continue
            # 关键：边属性与端点必须在同一次遍历中读取，避免“分别查询再按下标对齐”导致错配。
            try:
                edge_rows = (
                    self._g.E()
                    .has_label(EDGE_LABEL)
                    .project("source", "target", "predicate", "description")
                    .by(__.out_v().values(VERTEX_PROP_NAME).fold())
                    .by(__.in_v().values(VERTEX_PROP_NAME).fold())
                    .by(__.values(EDGE_PROP_PREDICATE).fold())
                    .by(__.values(EDGE_PROP_DESC).fold())
                    .to_list()
                )
            except Exception as e:
                logger.warning("get_visualization_data 边投影失败，回退旧遍历: %s", e)
                edge_rows = []
            for row in edge_rows:
                try:
                    src_list = row.get("source") or []
                    tgt_list = row.get("target") or []
                    pred_list = row.get("predicate") or []
                    desc_list = row.get("description") or []
                    a = str(src_list[0] if src_list else "").strip()
                    c = str(tgt_list[0] if tgt_list else "").strip()
                    pvv = str(pred_list[0] if pred_list else "").strip()
                    dvv = str(desc_list[0] if desc_list else "").strip()
                except Exception:
                    continue
                if not a or not c:
                    continue
                edges.append({
                    "source": a, "target": c, "predicate": pvv, "description": dvv,
                })
            return {"nodes": nodes, "edges": edges}

    def reimport_from_preloaded(self, max_triples: int = 3000) -> Dict[str, Any]:
        """
        在单持锁段内先短暂停顿、再清空业务 entity 顶点、再按预置 TSV 或 JSON 灌入，
        避免「先 clear 再长耗时 load」时与其它 Gremlin 读/写事务在 JE 层交叉死锁。
        """
        openkg = os.path.join("data", "openkg_triples.tsv")
        pjson = os.path.join("data", "preloaded_knowledge_graph.json")
        with self._janus_lock:
            self._entity_docs.clear()
            self._doc_entities.clear()
            if os.path.exists(openkg):
                # max_triples <= 0 表示全量导入（串行慢速，适合一次性初始化）
                triple_limit = None if max_triples <= 0 else max_triples
                triples: List[Tuple[str, str, str]] = []
                entities: Set[str] = set()
                with open(openkg, "r", encoding="utf-8") as f:
                    for line in f:
                        if triple_limit is not None and len(triples) >= triple_limit:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split("\t")
                        if len(parts) < 3:
                            continue
                        s, p, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
                        if not (s and p and o):
                            continue
                        triples.append((s, p, o))
                        entities.add(s)
                        entities.add(o)

                # 去重后再写入，避免重复边在初始化阶段放大写冲突
                triples = list(dict.fromkeys(triples))
                sorted_entities = sorted(entities)
                entity_doc_count: Dict[str, int] = {}
                for s, _p, o in triples:
                    entity_doc_count[s] = int(entity_doc_count.get(s, 0)) + 1
                    entity_doc_count[o] = int(entity_doc_count.get(o, 0)) + 1

                ckpt = self._load_import_checkpoint(IMPORT_CHECKPOINT_FILE) if IMPORT_RESUME_ENABLED else None
                can_resume = bool(
                    ckpt
                    and ckpt.get("source") == openkg
                    and int(ckpt.get("total_entities", -1)) == len(sorted_entities)
                    and int(ckpt.get("total_edges", -1)) == len(triples)
                    and str(ckpt.get("status", "")) in {"entities", "edges"}
                )
                entity_start = int(ckpt.get("entity_index", 0) or 0) if can_resume else 0
                edge_start = int(ckpt.get("edge_index", 0) or 0) if can_resume else 0

                if not can_resume:
                    time.sleep(0.25)
                    try:
                        self._script_submit("g.V().hasLabel(lb).drop().iterate()", {"lb": VERTEX_LABEL})
                    except Exception as e:
                        logger.warning("reimport 清空 entity 图: %s", e)
                    self._remove_import_checkpoint(IMPORT_CHECKPOINT_FILE)
                else:
                    logger.warning(
                        "检测到断点续导，继续执行: entity_index=%s edge_index=%s",
                        entity_start,
                        edge_start,
                    )
                self._print_cli_progress("entities", entity_start, len(sorted_entities))

                for idx in range(entity_start, len(sorted_entities)):
                    ent = sorted_entities[idx]
                    self._script_submit(
                        "g.V().hasLabel('entity').has('name', nm).fold().coalesce(unfold(), addV('entity').property('name', nm).property('entity_type', tv).property('description', dvv).property('doc_count', c1).property('created_at', cre)).iterate()",
                        {
                            "nm": ent,
                            "tv": "未分类",
                            "dvv": "",
                            "c1": int(entity_doc_count.get(ent, 1)),
                            "cre": datetime.now().isoformat(),
                        },
                    )
                    cidx = idx + 1
                    if cidx == len(sorted_entities) or cidx % max(1, IMPORT_PROGRESS_BAR_EVERY) == 0:
                        self._print_cli_progress("entities", cidx, len(sorted_entities))
                    if cidx % max(1, IMPORT_PROGRESS_EVERY) == 0:
                        self._save_import_checkpoint(
                            IMPORT_CHECKPOINT_FILE,
                            {
                                "source": openkg,
                                "status": "entities",
                                "entity_index": cidx,
                                "edge_index": edge_start,
                                "total_entities": len(sorted_entities),
                                "total_edges": len(triples),
                                "updated_at": datetime.now().isoformat(),
                            },
                        )
                        logger.warning("导入进度[entities]: %s/%s", cidx, len(sorted_entities))
                    if cidx % max(1, IMPORT_SLEEP_EVERY) == 0:
                        time.sleep(max(0.0, IMPORT_SLEEP_MS / 1000.0))

                self._print_cli_progress("edges", edge_start, len(triples))
                for idx in range(edge_start, len(triples)):
                    s, p, o = triples[idx]
                    self._script_submit(
                        "g.V().hasLabel('entity').has('name', s).as('a').V().hasLabel('entity').has('name', o).as('b').addE('relation').from('a').to('b').property('predicate', pvv).property('description', dvv).property('doc_id', didv).property('created_at', cret).iterate()",
                        {
                            "s": s,
                            "o": o,
                            "pvv": p,
                            "dvv": "",
                            "didv": "",
                            "cret": datetime.now().isoformat(),
                        },
                    )
                    cidx = idx + 1
                    if cidx == len(triples) or cidx % max(1, IMPORT_PROGRESS_BAR_EVERY) == 0:
                        self._print_cli_progress("edges", cidx, len(triples))
                    if cidx % max(1, IMPORT_PROGRESS_EVERY) == 0:
                        self._save_import_checkpoint(
                            IMPORT_CHECKPOINT_FILE,
                            {
                                "source": openkg,
                                "status": "edges",
                                "entity_index": len(sorted_entities),
                                "edge_index": cidx,
                                "total_entities": len(sorted_entities),
                                "total_edges": len(triples),
                                "updated_at": datetime.now().isoformat(),
                            },
                        )
                        logger.warning("导入进度[edges]: %s/%s", cidx, len(triples))
                    if cidx % max(1, IMPORT_SLEEP_EVERY) == 0:
                        time.sleep(max(0.0, IMPORT_SLEEP_MS / 1000.0))

                self._remove_import_checkpoint(IMPORT_CHECKPOINT_FILE)
                st = self.get_stats()
                if len(triples) > 0:
                    return {
                        "success": True,
                        "message": f"已从 {openkg} 重新加载",
                        "entity_count": st["entity_count"],
                        "relation_count": st["relation_count"],
                        "imported_edges": len(triples),
                    }
            if os.path.exists(pjson):
                with open(pjson, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "entities" in data and "relations" in data:
                    for ent in data.get("entities", []):
                        self.add_entity(
                            ent.get("name", ""),
                            ent.get("type", "未分类"),
                            ent.get("description", ""),
                            None,
                        )
                        for did in ent.get("documents", []):
                            self._entity_docs.setdefault(ent.get("name", ""), set()).add(did)
                            self._doc_entities.setdefault(did, set()).add(ent.get("name", ""))
                    for rel in data.get("relations", []):
                        self.add_relation(
                            rel.get("subject", ""),
                            rel.get("predicate", ""),
                            rel.get("object", ""),
                            rel.get("description", ""),
                            rel.get("doc_id"),
                        )
                elif isinstance(data, dict) and "triples" in data:
                    for t in data.get("triples", []):
                        s = (t.get("subject") or "").strip()
                        p = (t.get("predicate") or "").strip()
                        o = (t.get("object") or "").strip()
                        if s:
                            self.add_entity(s, "未分类")
                        if o:
                            self.add_entity(o, "未分类")
                        if s and p and o:
                            self.add_relation(s, p, o)
                else:
                    return {
                        "success": False,
                        "error": "预置 JSON 结构不支持",
                    }
                st2 = self.get_stats()
                ec2 = st2.get("entity_count", 0) or 0
                if ec2 > 0:
                    return {
                        "success": True,
                        "message": f"已从 {pjson} 重新加载",
                        "entity_count": ec2,
                        "relation_count": st2.get("relation_count", 0) or 0,
                    }
        return {
            "success": False,
            "error": "未找到或未能导入预置数据（请检查 data/openkg_triples.tsv 与 preloaded_knowledge_graph.json）",
        }


def is_janusgraph_available() -> bool:
    return _GREMLIN_AVAILABLE


class JanusGraphAdapter:
    """
    将 JanusGraphBackend 适配为 KGRetrievalService 可用的接口：
    提供 .graph 的 NetworkX 风格用法（has_node, nodes, number_of_nodes, edges），
    以及与 KnowledgeGraph 相同的方法委托。
    """

    def __init__(self, gremlin_url: Optional[str] = None):
        self._backend = JanusGraphBackend(gremlin_url)
        self.graph = _GraphFacade(self._backend)

    def add_entity(self, entity_name: str, entity_type: str, description: str = "", doc_id: Optional[str] = None):
        self._backend.add_entity(entity_name, entity_type, description, doc_id)

    def add_relation(self, subject: str, predicate: str, object_entity: str, description: str = "", doc_id: Optional[str] = None):
        self._backend.add_relation(subject, predicate, object_entity, description, doc_id)

    def remove_entity(self, entity_name: str) -> bool:
        return self._backend.remove_entity(entity_name)

    def remove_relation(self, subject: str, predicate: str, object_entity: str) -> bool:
        return self._backend.remove_relation(subject, predicate, object_entity)

    def get_entity_relations(self, entity: str) -> Dict[str, List[Dict[str, Any]]]:
        return self._backend.get_entity_relations(entity)

    def get_related_entities(self, entity: str, max_distance: int = 2) -> List[Dict[str, Any]]:
        return self._backend.get_related_entities(entity, max_distance)

    def get_entity_documents(self, entity: str) -> List[str]:
        return self._backend.get_entity_documents(entity)

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self._backend.search_entities(query, limit)

    def get_stats(self) -> Dict[str, Any]:
        return self._backend.get_stats()

    def clear_graph(self):
        self._backend.clear_graph()

    def export_graph_data(self) -> Dict[str, Any]:
        return self._backend.export_graph_data()

    def load_from_openkg_triples(self, filepath: str, max_triples: int = 5000) -> bool:
        return self._backend.load_from_openkg_triples(filepath, max_triples)

    def load_from_json_file(self, filepath: str) -> bool:
        return self._backend.load_from_json_file(filepath)

    def reimport_from_preloaded(self, max_triples: int = 3000) -> Dict[str, Any]:
        return self._backend.reimport_from_preloaded(max_triples)

    def save_graph(self, filepath: Optional[str] = None):
        self._backend.save_graph(filepath)

    def upsert_ontology_schema(self, schema_dict: Dict[str, Any]) -> bool:
        return self._backend.upsert_ontology_schema(schema_dict)

    def fetch_ontology_schema(self) -> Optional[Dict[str, Any]]:
        return self._backend.fetch_ontology_schema()

    @property
    def janus_backend(self) -> "JanusGraphBackend":
        return self._backend

    def upsert_ontology_service(self, **kwargs) -> bool:
        return self._backend.upsert_ontology_service(**kwargs)

    def heartbeat_ontology_service(self, service_id: str) -> bool:
        return self._backend.heartbeat_ontology_service(service_id)

    def touch_index_refresh_ontology_service(self, service_id: str) -> bool:
        return self._backend.touch_index_refresh_ontology_service(service_id)

    def set_ontology_service_status(self, service_id: str, status: str) -> bool:
        return self._backend.set_ontology_service_status(service_id, status)

    def get_ontology_service(self, service_id: str) -> Optional[Dict[str, Any]]:
        return self._backend.get_ontology_service(service_id)

    def list_ontology_services(self, ontology_domain: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._backend.list_ontology_services(ontology_domain)

    def is_ontology_service_available(self, service_id: str) -> bool:
        return self._backend.is_ontology_service_available(service_id)


class _GraphFacade:
    """模拟 NetworkX 的 .graph 接口，供 KGRetrievalService 中 has_node / nodes / edges 调用。"""

    def __init__(self, backend: JanusGraphBackend):
        self._backend = backend

    def has_node(self, name: str) -> bool:
        with self._backend._janus_lock:
            return self._backend._has_entity(name)

    def number_of_nodes(self) -> int:
        with self._backend._janus_lock:
            return self._backend.get_stats().get("entity_count", 0)

    def number_of_edges(self) -> int:
        with self._backend._janus_lock:
            return self._backend.get_stats().get("relation_count", 0)

    @property
    def nodes(self):
        return _NodesFacade(self._backend)

    def edges(self, data: bool = False):
        with self._backend._janus_lock:
            viz = self._backend.get_visualization_data()
        if not data:
            return [(e["source"], e["target"]) for e in viz["edges"]]
        return [(e["source"], e["target"], {"predicate": e.get("predicate", ""), "description": e.get("description", "")}) for e in viz["edges"]]


class _NodesFacade:
    def __init__(self, backend: JanusGraphBackend):
        self._backend = backend

    def __getitem__(self, name: str) -> Dict[str, Any]:
        with self._backend._janus_lock:
            if not self._backend._has_entity(name):
                raise KeyError(name)
            v = self._backend._v_by_name(name).next()
            props = self._backend._get_vertex_props(v)
            return {
                "entity_type": props.get("entity_type", "未分类"),
                "description": props.get("description", ""),
                "doc_count": props.get("doc_count", 0),
                "created_at": props.get("created_at", ""),
            }

    def __iter__(self):
        with self._backend._janus_lock:
            try:
                for v in self._backend._g.V().has_label(VERTEX_LABEL).to_list():
                    name = self._backend._g.V(v).values(VERTEX_PROP_NAME).next()
                    yield name
            except Exception:
                pass

    def __call__(self):
        return list(self)
