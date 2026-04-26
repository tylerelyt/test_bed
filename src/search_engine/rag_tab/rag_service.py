#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG服务模块
基于现有的倒排索引和TF-IDF实现检索增强生成
"""

import json
import re
import os
from collections import deque
import requests
from typing import List, Dict, Tuple, Optional, Any, Generator
from datetime import datetime

# ==================== LLM 调用 ====================
def call_llm(messages, model="qwen-max"):
    """调用 LLM"""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM调用失败: {str(e)}"

class RAGService:
    """RAG服务：基于倒排索引的检索增强生成"""
    
    def __init__(self, index_service, ollama_url: str = "http://localhost:11434"):
        """
        初始化RAG服务
        
        Args:
            index_service: 索引服务实例
            ollama_url: Ollama服务URL (保留兼容性)
        """
        self.index_service = index_service
        self.ollama_url = ollama_url
        self.default_model = "qwen-max"  # 改为DashScope模型
        
    def check_ollama_connection(self) -> Tuple[bool, str]:
        """检查Ollama连接状态 (保留兼容性)"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                return True, f"✅ Ollama连接成功！\n可用模型: {', '.join(model_names)}"
            else:
                return False, f"❌ Ollama连接失败，状态码: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"❌ Ollama连接失败: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        # 返回DashScope可用模型
        return ["qwen-max", "qwen-plus", "qwen-turbo", "qwen2.5-72b-instruct"]
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        使用倒排索引检索相关文档
        
        Args:
            query: 查询字符串
            top_k: 返回top_k个文档
            
        Returns:
            List[Tuple[str, float, str]]: (doc_id, score, content)
        """
        try:
            # 使用现有的索引服务进行检索
            results = self.index_service.search(query, top_k)
            print(f"📖 检索到 {len(results)} 个相关文档")
            return results
        except Exception as e:
            print(f"❌ 文档检索失败: {e}")
            return []
    
    def generate_answer(self, query: str, context: str, model: Optional[str] = None) -> str:
        """
        使用DashScope生成回答
        
        Args:
            query: 用户查询
            context: 检索到的上下文
            model: 使用的模型名称
            
        Returns:
            str: 生成的回答
        """
        if model is None:
            model = self.default_model
            
        # 构建提示词
        system_prompt = """你是一个专业的AI助手，请基于提供的上下文信息回答用户问题。如果上下文中没有相关信息，请说明无法根据提供的信息回答。请用中文回答。"""
        
        user_prompt = f"""上下文信息：
{context}

用户问题：{query}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            return call_llm(messages, model)
        except Exception as e:
            return f"❌ 调用LLM失败: {str(e)}"
    
    def generate_answer_with_prompt(self, prompt: str, model: Optional[str] = None) -> str:
        """
        直接使用提示词生成回答
        
        Args:
            prompt: 完整的提示词
            model: 使用的模型名称
            
        Returns:
            str: 生成的回答
        """
        if model is None:
            model = self.default_model
            
        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            return call_llm(messages, model)
        except Exception as e:
            return f"❌ 调用LLM失败: {str(e)}"

    def _walk_knowledge_graph(
        self,
        query: str,
        max_depth: int = 2,
        expand_per_hop: int = 4,
        max_seed_entities: int = 3,
        max_facts: int = 48,
    ) -> Dict[str, Any]:
        """基于实体匹配结果执行有界图谱游走，产出结构化事实。"""
        clean_query = (query or "").strip()
        seeds = self.index_service.search_entities(clean_query, limit=max_seed_entities) if clean_query else []
        seed_entities = [str(item.get("entity", "")).strip() for item in seeds if str(item.get("entity", "")).strip()]

        # 兜底：如果整句未命中，尝试按空白切分关键词补种子。
        if not seed_entities and clean_query:
            for token in [part.strip() for part in clean_query.split() if part.strip()]:
                token_hits = self.index_service.search_entities(token, limit=1)
                for hit in token_hits:
                    name = str(hit.get("entity", "")).strip()
                    if name and name not in seed_entities:
                        seed_entities.append(name)
                if len(seed_entities) >= max_seed_entities:
                    break

        queue: deque[Tuple[str, int]] = deque((name, 0) for name in seed_entities)
        visited = set(seed_entities)
        walk_facts: List[Dict[str, Any]] = []
        seen_edges = set()

        while queue and len(walk_facts) < max_facts:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            relation_data = self.index_service.query_entity_relations(current) or {}
            relations = relation_data.get("relations", {}) if isinstance(relation_data, dict) else {}
            outgoing = relations.get("outgoing", []) if isinstance(relations, dict) else []
            incoming = relations.get("incoming", []) if isinstance(relations, dict) else []

            merged_edges: List[Tuple[str, str, str]] = []
            for item in outgoing:
                target = str(item.get("target", "")).strip()
                predicate = str(item.get("predicate", "")).strip()
                if target and predicate:
                    merged_edges.append((current, predicate, target))
            for item in incoming:
                source = str(item.get("source", "")).strip()
                predicate = str(item.get("predicate", "")).strip()
                if source and predicate:
                    merged_edges.append((source, predicate, current))

            local_added = 0
            for source, predicate, target in merged_edges:
                edge_key = (source, predicate, target)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                walk_facts.append(
                    {
                        "depth": depth + 1,
                        "source": source,
                        "predicate": predicate,
                        "target": target,
                    }
                )
                local_added += 1
                if len(walk_facts) >= max_facts or local_added >= expand_per_hop:
                    break

            if depth + 1 < max_depth:
                for source, _, target in merged_edges[: expand_per_hop * 2]:
                    for nb in (source, target):
                        if nb and nb not in visited:
                            visited.add(nb)
                            queue.append((nb, depth + 1))

        return {
            "seed_entities": seed_entities,
            "visited_entities": sorted(visited),
            "facts": walk_facts,
            "max_depth": max_depth,
            "expand_per_hop": expand_per_hop,
        }

    def _build_oag_prompt(self, query: str, walk_result: Dict[str, Any]) -> Tuple[str, str]:
        """将图谱游走结果转为模型可消费上下文与提示词。"""
        seeds = walk_result.get("seed_entities", []) or []
        facts = walk_result.get("facts", []) or []
        visited_entities = walk_result.get("visited_entities", []) or []

        if facts:
            lines = [
                f"[第{int(item.get('depth', 0) or 0)}跳] "
                f"{item.get('source', '')} --[{item.get('predicate', '')}]--> {item.get('target', '')}"
                for item in facts
            ]
            fact_block = "\n".join(lines)
        else:
            fact_block = "未检索到有效图谱关系。"

        context = (
            f"种子实体: {', '.join(seeds) if seeds else '无'}\n"
            f"游走覆盖实体数: {len(visited_entities)}\n"
            f"图谱事实:\n{fact_block}"
        )
        prompt = f"""你是一个企业知识分析助手。请基于给定的知识图谱游走事实回答用户问题。

要求：
1. 只能基于给定事实作答，若事实不足请明确说明不足点。
2. 先给出结论，再给出依据（引用关键关系）。
3. 输出中文，避免臆造不存在的实体关系。

用户问题：
{query}

图谱游走上下文：
{context}
"""
        return context, prompt

    @staticmethod
    def _format_agent_step(step_id: int, thought: str, action: str, action_input: str, observation: str, fact_count: int) -> Dict[str, Any]:
        """统一 ReAct 步骤结构，便于 UI 稳定渲染。"""
        return {
            "step": step_id,
            "thought": thought.strip(),
            "action": action.strip(),
            "action_input": action_input.strip(),
            "observation": observation.strip(),
            "facts_collected": int(fact_count),
        }

    @staticmethod
    def _clean_action_argument(raw_arg: str) -> str:
        """清洗 Action 参数，兼容带/不带引号两种格式。"""
        value = str(raw_arg or "").strip()
        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1].strip()
        return value

    def _oag_react_iter_snapshot(
        self,
        steps: List[Dict[str, Any]],
        walk_facts: List[Dict[str, Any]],
        discovered_entities: Dict[str, int],
        scratchpad: List[str],
        final_answer: str,
        react_finished: bool,
    ) -> Dict[str, Any]:
        return {
            "steps": list(steps),
            "graph_walk": list(walk_facts),
            "seed_entities": sorted(discovered_entities.keys()),
            "react_trace": "\n\n".join(scratchpad),
            "final_answer": final_answer,
            "react_finished": react_finished,
        }

    def _iter_react_oag_reasoning(
        self,
        query: str,
        model: Optional[str],
        max_depth: int,
        expand_per_hop: int,
        max_steps: int = 6,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        ReAct OAG，每完成一步即 yield 一帧，供界面上流式展示。
        - Thought -> Action -> Observation；Action 经 IndexService 访问 JanusGraph。
        """
        use_model = model or self.default_model
        max_steps = max(2, min(int(max_steps or 6), 10))

        search_pattern = re.compile(r"Action:\s*SEARCH_ENTITIES\(([\s\S]*?)\)", re.IGNORECASE)
        expand_pattern = re.compile(r"Action:\s*EXPAND_ENTITY\(([\s\S]*?)\)", re.IGNORECASE)
        finish_pattern = re.compile(r"Action:\s*FINISH\(([\s\S]*?)\)", re.IGNORECASE)
        thought_pattern = re.compile(r"Thought:\s*([\s\S]*?)(?:\nAction:|$)", re.IGNORECASE)

        discovered_entities: Dict[str, int] = {}
        expanded_entities = set()
        walk_facts: List[Dict[str, Any]] = []
        seen_edges = set()
        steps: List[Dict[str, Any]] = []
        scratchpad: List[str] = []
        final_answer = ""

        # 初始化种子实体，降低首轮无动作风险。
        for seed_hit in self.index_service.search_entities((query or "").strip(), limit=3):
            entity_name = str(seed_hit.get("entity", "")).strip()
            if entity_name:
                discovered_entities.setdefault(entity_name, 0)

        for step in range(1, max_steps + 1):
            known_facts_lines = [
                f"{item.get('source', '')} --[{item.get('predicate', '')}]--> {item.get('target', '')}"
                for item in walk_facts[-20:]
            ]
            prompt = f"""你是一个图谱推理智能体。目标：基于 JanusGraph 多轮查询回答问题。

用户问题：
{query}

当前可用工具（只能二选一，或结束）：
1) SEARCH_ENTITIES("关键词")：返回候选实体名列表
2) EXPAND_ENTITY("实体名")：返回该实体一跳关系（出边+入边）
3) FINISH("最终答案")

规则：
- 每轮必须输出 Thought 与 Action 两行。
- 若事实不足，继续调用工具，不要臆造关系。
- 优先扩展与问题最相关、且尚未展开的实体。
- 若已可回答，使用 FINISH 给出结论+依据。
- 输出格式严格如下：
Thought: <一句简短思考>
Action: SEARCH_ENTITIES("...") 或 Action: EXPAND_ENTITY("...") 或 Action: FINISH("...")

当前已发现实体（深度）：
{json.dumps(discovered_entities, ensure_ascii=False)}

当前已采集事实（最近）：
{chr(10).join(known_facts_lines) if known_facts_lines else "暂无"}

历史轨迹：
{chr(10).join(scratchpad) if scratchpad else "暂无"}
"""
            llm_text = self.generate_answer_with_prompt(prompt, use_model).strip()
            thought_match = thought_pattern.search(llm_text)
            thought = thought_match.group(1).strip() if thought_match else "继续检索相关事实"

            finish_match = finish_pattern.search(llm_text)
            if finish_match:
                final_answer = self._clean_action_argument(finish_match.group(1))
                observation = "智能体认为事实已足够，结束推理。"
                step_item = self._format_agent_step(step, thought, "FINISH", final_answer, observation, len(walk_facts))
                steps.append(step_item)
                scratchpad.append(
                    f"Step {step}\nThought: {thought}\nAction: FINISH(\"{final_answer}\")\nObservation: {observation}"
                )
                yield self._oag_react_iter_snapshot(
                    steps, walk_facts, discovered_entities, scratchpad, final_answer, True
                )
                break

            search_match = search_pattern.search(llm_text)
            if search_match:
                keyword = self._clean_action_argument(search_match.group(1))
                matches = self.index_service.search_entities(keyword, limit=max(2, min(expand_per_hop, 6)))
                found_entities = []
                for hit in matches:
                    name = str(hit.get("entity", "")).strip()
                    if not name:
                        continue
                    found_entities.append(name)
                    if name not in discovered_entities:
                        discovered_entities[name] = 0
                observation = f"匹配实体: {', '.join(found_entities) if found_entities else '无'}"
                step_item = self._format_agent_step(step, thought, "SEARCH_ENTITIES", keyword, observation, len(walk_facts))
                steps.append(step_item)
                scratchpad.append(
                    f"Step {step}\nThought: {thought}\nAction: SEARCH_ENTITIES(\"{keyword}\")\nObservation: {observation}"
                )
                yield self._oag_react_iter_snapshot(
                    steps, walk_facts, discovered_entities, scratchpad, final_answer, False
                )
                continue

            expand_match = expand_pattern.search(llm_text)
            if expand_match:
                entity = self._clean_action_argument(expand_match.group(1))
                current_depth = int(discovered_entities.get(entity, 0))
                if not entity:
                    observation = "实体名为空，跳过。"
                    step_item = self._format_agent_step(step, thought, "EXPAND_ENTITY", entity, observation, len(walk_facts))
                    steps.append(step_item)
                    scratchpad.append(
                        f"Step {step}\nThought: {thought}\nAction: EXPAND_ENTITY(\"{entity}\")\nObservation: {observation}"
                    )
                    yield self._oag_react_iter_snapshot(
                        steps, walk_facts, discovered_entities, scratchpad, final_answer, False
                    )
                    continue
                if current_depth >= max_depth:
                    observation = f"实体 {entity} 已到最大深度 {max_depth}，跳过扩展。"
                    step_item = self._format_agent_step(step, thought, "EXPAND_ENTITY", entity, observation, len(walk_facts))
                    steps.append(step_item)
                    scratchpad.append(
                        f"Step {step}\nThought: {thought}\nAction: EXPAND_ENTITY(\"{entity}\")\nObservation: {observation}"
                    )
                    yield self._oag_react_iter_snapshot(
                        steps, walk_facts, discovered_entities, scratchpad, final_answer, False
                    )
                    continue
                if entity in expanded_entities:
                    observation = f"实体 {entity} 已扩展过，跳过重复扩展。"
                    step_item = self._format_agent_step(step, thought, "EXPAND_ENTITY", entity, observation, len(walk_facts))
                    steps.append(step_item)
                    scratchpad.append(
                        f"Step {step}\nThought: {thought}\nAction: EXPAND_ENTITY(\"{entity}\")\nObservation: {observation}"
                    )
                    yield self._oag_react_iter_snapshot(
                        steps, walk_facts, discovered_entities, scratchpad, final_answer, False
                    )
                    continue

                relation_data = self.index_service.query_entity_relations(entity) or {}
                rel_map = relation_data.get("relations", {}) if isinstance(relation_data, dict) else {}
                outgoing = rel_map.get("outgoing", []) if isinstance(rel_map, dict) else []
                incoming = rel_map.get("incoming", []) if isinstance(rel_map, dict) else []
                merged_edges: List[Tuple[str, str, str]] = []
                for item in outgoing:
                    target = str(item.get("target", "")).strip()
                    predicate = str(item.get("predicate", "")).strip()
                    if target and predicate:
                        merged_edges.append((entity, predicate, target))
                for item in incoming:
                    source = str(item.get("source", "")).strip()
                    predicate = str(item.get("predicate", "")).strip()
                    if source and predicate:
                        merged_edges.append((source, predicate, entity))

                added_count = 0
                for source, predicate, target in merged_edges:
                    edge_key = (source, predicate, target)
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    walk_facts.append(
                        {
                            "depth": current_depth + 1,
                            "source": source,
                            "predicate": predicate,
                            "target": target,
                        }
                    )
                    added_count += 1
                    if added_count >= expand_per_hop:
                        break
                expanded_entities.add(entity)

                # 新邻居登记为下一层候选实体
                for source, _, target in merged_edges[: max(1, expand_per_hop * 2)]:
                    for nb in (source, target):
                        if not nb:
                            continue
                        if nb not in discovered_entities:
                            discovered_entities[nb] = min(current_depth + 1, max_depth)
                        else:
                            discovered_entities[nb] = min(discovered_entities[nb], current_depth + 1)

                observation = f"扩展实体 {entity}，新增事实 {added_count} 条。"
                step_item = self._format_agent_step(step, thought, "EXPAND_ENTITY", entity, observation, len(walk_facts))
                steps.append(step_item)
                scratchpad.append(
                    f"Step {step}\nThought: {thought}\nAction: EXPAND_ENTITY(\"{entity}\")\nObservation: {observation}"
                )
                yield self._oag_react_iter_snapshot(
                    steps, walk_facts, discovered_entities, scratchpad, final_answer, False
                )
                continue

            # 未解析有效动作，自动回退为一次 SEARCH。
            fallback_keyword = (query or "").strip().split(" ")[0] if (query or "").strip() else ""
            fallback_matches = self.index_service.search_entities(fallback_keyword, limit=3) if fallback_keyword else []
            found_entities = []
            for hit in fallback_matches:
                name = str(hit.get("entity", "")).strip()
                if name:
                    found_entities.append(name)
                    discovered_entities.setdefault(name, 0)
            observation = "动作解析失败，系统回退执行 SEARCH_ENTITIES。"
            step_item = self._format_agent_step(step, thought, "SEARCH_ENTITIES", fallback_keyword, observation, len(walk_facts))
            steps.append(step_item)
            scratchpad.append(
                f"Step {step}\nThought: {thought}\nAction: SEARCH_ENTITIES(\"{fallback_keyword}\")\nObservation: {observation}; found={found_entities}"
            )
            yield self._oag_react_iter_snapshot(
                steps, walk_facts, discovered_entities, scratchpad, final_answer, False
            )

        if not walk_facts:
            # ReAct 未采集到事实时，回退到保守 BFS，保证可用性。
            fallback_walk = self._walk_knowledge_graph(
                query=query,
                max_depth=max_depth,
                expand_per_hop=expand_per_hop,
            )
            walk_facts = list(fallback_walk.get("facts", []))
            for ent in fallback_walk.get("seed_entities", []):
                discovered_entities.setdefault(str(ent), 0)
            yield self._oag_react_iter_snapshot(
                steps, walk_facts, discovered_entities, scratchpad, final_answer, False
            )

    def _react_oag_reasoning(
        self,
        query: str,
        model: Optional[str],
        max_depth: int,
        expand_per_hop: int,
        max_steps: int = 6,
    ) -> Dict[str, Any]:
        """消费迭代器，返回与历史接口一致的结果（最后一帧，含 BFS 兜底）。"""
        last: Optional[Dict[str, Any]] = None
        for snap in self._iter_react_oag_reasoning(
            query=query,
            model=model,
            max_depth=max_depth,
            expand_per_hop=expand_per_hop,
            max_steps=max_steps,
        ):
            last = snap
        if not last:
            return {
                "steps": [],
                "graph_walk": [],
                "seed_entities": [],
                "react_trace": "",
                "final_answer": "",
            }
        return {
            "steps": list(last.get("steps", [])),
            "graph_walk": list(last.get("graph_walk", [])),
            "seed_entities": list(last.get("seed_entities", [])),
            "react_trace": str(last.get("react_trace", "")),
            "final_answer": str(last.get("final_answer", "")),
        }

    def oag_query(
        self,
        query: str,
        max_depth: int = 2,
        expand_per_hop: int = 4,
        model: Optional[str] = None,
        generate_answer: bool = True,
        use_react_agent: bool = True,
        max_steps: int = 6,
    ) -> Dict[str, Any]:
        """执行 OAG 查询：先图谱游走，再基于事实生成答案。"""
        start_time = datetime.now()
        max_depth = max(1, min(int(max_depth or 2), 4))
        expand_per_hop = max(1, min(int(expand_per_hop or 4), 10))

        if use_react_agent:
            react_result = self._react_oag_reasoning(
                query=query,
                model=model,
                max_depth=max_depth,
                expand_per_hop=expand_per_hop,
                max_steps=max_steps,
            )
            walk_result = {
                "seed_entities": react_result.get("seed_entities", []),
                "visited_entities": react_result.get("seed_entities", []),
                "facts": react_result.get("graph_walk", []),
                "max_depth": max_depth,
                "expand_per_hop": expand_per_hop,
            }
            agent_steps = react_result.get("steps", [])
            react_trace = react_result.get("react_trace", "")
            final_answer = react_result.get("final_answer", "")
        else:
            walk_result = self._walk_knowledge_graph(
                query=query,
                max_depth=max_depth,
                expand_per_hop=expand_per_hop,
            )
            agent_steps = []
            react_trace = ""
            final_answer = ""

        context, prompt = self._build_oag_prompt(query, walk_result)

        answer = ""
        if generate_answer:
            answer = final_answer or self.generate_answer_with_prompt(prompt, model)

        processing_time = (datetime.now() - start_time).total_seconds()
        return {
            "query": query,
            "answer": answer,
            "context": context,
            "prompt_sent": prompt,
            "graph_walk": walk_result.get("facts", []),
            "seed_entities": walk_result.get("seed_entities", []),
            "visited_entities": walk_result.get("visited_entities", []),
            "agent_steps": agent_steps,
            "react_trace": react_trace,
            "processing_time": processing_time,
            "model_used": model or self.default_model,
            "mode": "oag",
        }

    def iter_oag_query(
        self,
        query: str,
        max_depth: int = 2,
        expand_per_hop: int = 4,
        model: Optional[str] = None,
        generate_answer: bool = True,
        use_react_agent: bool = True,
        max_steps: int = 6,
    ) -> Generator[Dict[str, Any], None, None]:
        """与 oag_query 语义一致；在 ReAct 每完成一步即 yield 一帧，便于 Gradio 实时刷新。"""
        if not use_react_agent:
            yield self.oag_query(
                query=query,
                max_depth=max_depth,
                expand_per_hop=expand_per_hop,
                model=model,
                generate_answer=generate_answer,
                use_react_agent=False,
                max_steps=max_steps,
            )
            return
        t0 = datetime.now()
        max_d = max(1, min(int(max_depth or 2), 4))
        ex = max(1, min(int(expand_per_hop or 4), 10))
        use_m = model or self.default_model
        last: Optional[Dict[str, Any]] = None
        for react in self._iter_react_oag_reasoning(
            query=query,
            model=model,
            max_depth=max_d,
            expand_per_hop=ex,
            max_steps=max_steps,
        ):
            last = react
            walk_result = {
                "seed_entities": react["seed_entities"],
                "visited_entities": react["seed_entities"],
                "facts": react["graph_walk"],
                "max_depth": max_d,
                "expand_per_hop": ex,
            }
            context, prompt = self._build_oag_prompt(query, walk_result)
            n = len(react.get("steps", []))
            last_act = "—"
            if react.get("steps"):
                last_act = str(react["steps"][-1].get("action", "")) or "—"
            if react.get("react_finished") and (react.get("final_answer") or "").strip():
                status_ans = (react.get("final_answer") or "").strip()
            else:
                status_ans = f"OAG 推理中… 第 {n} 步已结束（{last_act}）"
            yield {
                "query": query,
                "answer": status_ans,
                "context": context,
                "prompt_sent": prompt,
                "graph_walk": walk_result.get("facts", []),
                "seed_entities": walk_result.get("seed_entities", []),
                "visited_entities": walk_result.get("visited_entities", []),
                "agent_steps": react.get("steps", []),
                "react_trace": react.get("react_trace", ""),
                "processing_time": (datetime.now() - t0).total_seconds(),
                "model_used": use_m,
                "mode": "oag",
                "stream_phase": "react",
            }
        if last is None:
            return
        walk_result = {
            "seed_entities": last["seed_entities"],
            "visited_entities": last["seed_entities"],
            "facts": last["graph_walk"],
            "max_depth": max_d,
            "expand_per_hop": ex,
        }
        context, prompt = self._build_oag_prompt(query, walk_result)
        final_ans = (last.get("final_answer") or "").strip()
        if generate_answer:
            answer = final_ans or self.generate_answer_with_prompt(prompt, model)
        else:
            answer = final_ans or "（ReAct 已完成，下方完整 Prompt 将用于本地模型生成）"
        processing_time = (datetime.now() - t0).total_seconds()
        yield {
            "query": query,
            "answer": answer,
            "context": context,
            "prompt_sent": prompt,
            "graph_walk": walk_result.get("facts", []),
            "seed_entities": walk_result.get("seed_entities", []),
            "visited_entities": walk_result.get("visited_entities", []),
            "agent_steps": last.get("steps", []),
            "react_trace": last.get("react_trace", ""),
            "processing_time": processing_time,
            "model_used": use_m,
            "mode": "oag",
            "stream_phase": "final",
        }
    
    def _react_reasoning(self, query: str, model: Optional[str], retrieval_enabled: bool, top_k: int = 5, max_steps: int = 5) -> Tuple[str, str]:
        """
        ReAct风格多步推理：Thought -> Action(SEARCH/FINISH) -> Observation，循环直到FINISH或步数上限。
        返回 (final_answer, trace_text)
        """
        if model is None:
            model = self.default_model
        
        trace_lines: List[str] = []
        observations: List[str] = []

        tool_desc = (
            "你可以使用一个工具：SEARCH(\"查询词\")，它会返回与查询词最相关的文档片段列表。"
        )
        format_instructions = (
            "每轮请严格输出以下格式中的一行Action，便于解析：\n"
            "Thought: <你的简短思考>\n"
            "Action: SEARCH(\"<查询词>\") 或 Action: FINISH(\"<最终答案>\")\n"
            "不要输出其他多余内容。"
        )

        search_pattern = re.compile(r"Action:\s*SEARCH\(\"([\s\S]*?)\"\)")
        finish_pattern = re.compile(r"Action:\s*FINISH\(\"([\s\S]*?)\"\)")

        scratchpad = ""
        for step in range(1, max_steps + 1):
            prompt = (
                f"你是一个会逐步思考并合理使用工具的助手。\n"
                f"用户问题：{query}\n\n"
                f"工具说明：{tool_desc}\n"
                f"注意：{'当前禁止使用SEARCH工具。' if not retrieval_enabled else '可以使用SEARCH工具。'}\n\n"
                f"历史推理：\n{scratchpad}\n\n"
                f"请开始第{step}步。\n{format_instructions}"
            )
            try:
                resp = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=60
                )
                if resp.status_code != 200:
                    trace_lines.append(f"系统: 模型调用失败，状态码 {resp.status_code}")
                    break
                text = resp.json().get("response", "").strip()
            except requests.exceptions.RequestException as e:
                trace_lines.append(f"系统: 模型调用异常 {str(e)}")
                break

            # 记录模型输出
            trace_lines.append(f"Step {step} 模型输出:\n{text}")

            # 解析动作
            finish_match = finish_pattern.search(text)
            if finish_match:
                final_answer = finish_match.group(1)
                trace_lines.append("Action: FINISH")
                return final_answer, "\n\n".join(trace_lines)

            search_match = search_pattern.search(text)
            if search_match:
                search_query = search_match.group(1).strip()
                if retrieval_enabled:
                    # 执行检索
                    docs = self.retrieve_documents(search_query, top_k=top_k)
                    if not docs:
                        observation = "未检索到相关文档。"
                    else:
                        # 只取前3条，避免上下文过长
                        obs_parts = []
                        for i, (doc_id, score, content) in enumerate(docs[:3], 1):
                            snippet = content[:400]
                            obs_parts.append(f"[{i}] id={doc_id} score={score:.4f} snippet={snippet}")
                        observation = "\n".join(obs_parts)
                    observations.append(observation)
                    trace_lines.append(f"Observation:\n{observation}")
                    scratchpad += f"Thought/Action(SEARCH): {search_query}\nObservation: {observation}\n\n"
                    continue
                else:
                    observation = "SEARCH工具被禁用。请直接FINISH。"
                    observations.append(observation)
                    trace_lines.append(f"Observation:\n{observation}")
                    scratchpad += f"Action(SEARCH被拒): {search_query}\nObservation: {observation}\n\n"
                    continue

            # 若无法解析动作，提示并继续下一步
            notice = "未解析到有效的Action，请按格式输出。"
            trace_lines.append(f"系统: {notice}")
            scratchpad += f"系统提示: {notice}\n\n"

        # 未显式FINISH时，尝试让模型基于观察做最终总结
        summary_context = "\n\n".join(observations[-3:]) if observations else ""
        final_prompt = (
            f"请基于以下观察与你已有的推理，给出问题的最终中文答案。若观察为空，请直接根据常识作答。\n\n"
            f"问题：{query}\n\n"
            f"观察：\n{summary_context}\n\n"
            f"请直接输出答案，不要再输出思维过程。"
        )
        try:
            final_resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": model, "prompt": final_prompt, "stream": False},
                timeout=60
            )
            if final_resp.status_code != 200:
                answer = f"❌ 多步推理总结失败，状态码: {final_resp.status_code}"
            else:
                answer = final_resp.json().get("response", "生成回答失败")
        except requests.exceptions.RequestException as e:
            answer = f"❌ 调用Ollama失败: {str(e)}"
        trace_lines.append("系统: 未检测到FINISH，已进行自动总结。")
        return answer, "\n\n".join(trace_lines)

    def rag_query(self, query: str, top_k: int = 5, model: Optional[str] = None, retrieval_enabled: bool = True, multi_step: bool = False) -> Dict[str, Any]:
        """
        执行RAG查询
        
        Args:
            query: 用户查询
            top_k: 检索文档数量
            model: 使用的模型
            retrieval_enabled: 是否开启检索增强
            multi_step: 是否开启多步推理
            
        Returns:
            Dict: 包含检索结果和生成答案的字典
        """
        start_time = datetime.now()
        
        # 如果关闭检索与多步推理，则直接问 LLM（无上下文直连）
        if not retrieval_enabled and not multi_step:
            direct_prompt = f"请用中文回答用户问题：\n\n问题：{query}"
            answer = self.generate_answer_with_prompt(direct_prompt, model)
            return {
                "query": query,
                "retrieved_docs": [],
                "context": "",
                "answer": answer,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "model_used": model or self.default_model,
                "prompt_sent": direct_prompt
            }

        # 1) 若开启检索，先检索并构建上下文；否则上下文为空
        retrieved_docs = []
        context = ""
        if retrieval_enabled:
            retrieved_docs = self.retrieve_documents(query, top_k)
            # 即使未检索到文档，也继续，让模型直接回答或多步推理
            if retrieved_docs:
                context_parts = []
                for i, (doc_id, score, content) in enumerate(retrieved_docs, 1):
                    context_parts.append(f"文档{i} (ID: {doc_id}, 相关度: {score:.4f}):\n{content}")
                context = "\n\n".join(context_parts)

        # 2) 生成回答：多步推理优先，否则普通单步回答
        if multi_step:
            answer, trace_text = self._react_reasoning(
                query=query,
                model=model,
                retrieval_enabled=retrieval_enabled,
                top_k=top_k
            )
            prompt_used = trace_text  # 将完整推理轨迹回显
        else:
            # 构建标准提示
            prompt = f"""基于以下上下文信息，回答用户的问题。如果上下文中没有相关信息，请说明无法根据提供的信息回答。
            
上下文信息：
{context}
            
用户问题：{query}
            
请用中文回答："""
            answer = self.generate_answer_with_prompt(prompt, model)
            prompt_used = prompt
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "context": context,
            "answer": answer,
            "processing_time": processing_time,
            "model_used": model or self.default_model,
            "prompt_sent": prompt_used if prompt_used is not None else "多步推理（内部多提示）"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取RAG服务统计信息"""
        index_stats = self.index_service.get_stats()
        ollama_connected, ollama_status = self.check_ollama_connection()
        
        return {
            "ollama_connected": ollama_connected,
            "ollama_status": ollama_status,
            "ollama_url": self.ollama_url,
            "available_models": self.get_available_models(),
            "index_stats": index_stats
        } 