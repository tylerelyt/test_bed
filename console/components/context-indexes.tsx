"use client";
import { useEffect, useState } from "react";
import { platformRequest as api } from "../lib/platform-api";
import { WorkspaceLink, useWorkspace, useDetailName } from "../lib/workspace-navigation";
type Evidence = { documents?: { id: string; name: string; version: number; source: string; content: string }[]; answer?: string; prompt_sent?: string };
type Resource = { id: string; name: string; kind: string; version: number };
type Index = {
  id: string;
  name: string;
  digest: string;
  manifest: { sources: { id: string; name: string; version: number }[] };
};
export default function ContextIndexes({
  onPublished,
  initialMode = "search",
}: {
  onPublished?: () => Promise<void>;
  initialMode?: "sources" | "search";
}) {
  const {route,navigate} = useWorkspace();
  const [mode, setMode] = useState<"sources" | "search" | "build">(initialMode), [sourceType, setSourceType] = useState("documents"), [loaded, setLoaded] = useState(false);
  const [resources, setResources] = useState<Resource[]>([]),
    [indexes, setIndexes] = useState<Index[]>([]),
    [selected, setSelected] = useState<string[]>([]),
    [name, setName] = useState(""),
    [indexId, setIndexId] = useState(""),
    [query, setQuery] = useState(""),
    [result, setResult] = useState<Evidence | null>(null),
    [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [busy, setBusy] = useState(false);
  const [publishedResource,setPublishedResource] = useState("");
  const [sources, setSources] = useState<
      { id: string; chars: number; preview: string }[]
    >([]),
    [sourceId, setSourceId] = useState(""),
    [sourceName, setSourceName] = useState("");
  const [models, setModels] = useState<{ id: string; name: string }[]>([]),
    [modelId, setModelId] = useState("");
  const [entity, setEntity] = useState(""),
    [graphName, setGraphName] = useState("");
  async function refresh() {
    const [context, items, overview] = await Promise.all([
      api("context"),
      api("context-supply/indexes"),
      api("overview"),
    ]);
    setResources(
      context.resources.filter((r: Resource) => r.kind === "knowledge"),
    );
    setIndexes(items);
    setModels(overview.models);
    setLoaded(true);
  }
  useEffect(() => {
    void refresh().catch((e) => setError(e.message));
  }, []);
  useEffect(() => { if (route.area === -2 && route.view === "indexes") { setIndexId(route.record || "");setMode("search");setResult(null); } }, [route.area,route.view,route.record]);
  useDetailName(route.view === "indexes" && indexId ? indexes.find((item) => item.id === indexId)?.name : undefined);
  async function act(work: () => Promise<void>) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await work();
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section data-verify-unit="ContextIndexes" data-verify-state={loaded ? "ready" : "loading"}>
      <div className="collection-heading"><div><h1>{mode === "sources" ? "接入业务资料" : mode === "build" ? "构建资料索引" : indexId ? indexes.find((item) => item.id === indexId)?.name || "索引详情" : "检索索引"}</h1><p>{mode === "sources" ? "从已有文档或图谱服务保存一份有来源的资料快照。" : "索引固定资料版本，供岗位 Agent 按问题检索。"}</p></div>
        {mode === "sources" ? <WorkspaceLink area={-2} view="resources" className="text-button">返回资料库</WorkspaceLink> : (mode === "build" || indexId ? <button className="text-button" onClick={() => { setMode("search");setIndexId("");setResult(null);if(route.view === "indexes") navigate(-2,"indexes"); }}>返回索引列表</button> : <button className="primary" onClick={() => setMode("build")}>新建索引</button>)}
      </div>
      {error && <p role="alert">{error}</p>}
      {notice && <p role="status">{notice}</p>}
      {publishedResource && <div className="job-outcome"><p>{mode === "sources" ? "资料快照已保存，可继续构建检索索引或加入供应配置。" : "检索引用已保存到资料库，加入供应配置后即可为岗位使用。"}</p><div className="agent-actions"><WorkspaceLink className="resource-link" area={-2} view="resources" record={publishedResource}>查看已生成资源</WorkspaceLink><WorkspaceLink className="secondary" area={-2} view={mode === "sources" ? "indexes" : "configuration"}>{mode === "sources" ? "前往检索索引" : "前往供应配置"}</WorkspaceLink></div></div>}
      {mode === "sources" && <label className="source-selector">资料来源<select aria-label="资料来源" value={sourceType} onChange={(e) => setSourceType(e.target.value)}><option value="documents">文档索引服务</option><option value="graph">知识图谱服务</option></select></label>}
      {mode === "search" && !indexId && (!indexes.length ? <div className="collection-empty"><h3>尚无检索索引</h3><p>先准备企业资料，再选择具体版本构建索引。</p><WorkspaceLink area={-2} view="resources" className="resource-link">查看资料库</WorkspaceLink></div> : <div className="resource-table-scroll"><table className="resource-table"><thead><tr><th>索引名称</th><th>资料数量</th><th>版本摘要</th></tr></thead><tbody>{indexes.map((item) => <tr key={item.id}><td><WorkspaceLink className="resource-link" area={-2} view="indexes" record={item.id}>{item.name}</WorkspaceLink></td><td>{item.manifest.sources.length}</td><td><code>{item.digest.slice(0, 12)}</code></td></tr>)}</tbody></table></div>)}
      <div className="editor-page">
        {mode === "sources" && sourceType === "documents" &&
        <section className="agent-card">
          <h3>从原文档服务选取资料</h3>
          <button
            className="secondary"
            disabled={busy}
            onClick={() =>
              void act(async () =>
                setSources(await api("context-supply/sources/legacy-index")),
              )
            }
          >
            读取原文档目录
          </button>
          <label>
            原文档
            <select
              aria-label="原文档"
              value={sourceId}
              onChange={(e) => setSourceId(e.target.value)}
            >
              <option value="">请选择原文档</option>
              {sources.map((x) => (
                <option key={x.id} value={x.id}>
                  {x.id} · {x.chars} 字符 · {x.preview.slice(0, 30)}
                </option>
              ))}
            </select>
          </label>
          <label>
            资料快照名称
            <input
              aria-label="资料快照名称"
              value={sourceName}
              maxLength={120}
              onChange={(e) => setSourceName(e.target.value)}
            />
          </label>
          <button
            className="primary"
            disabled={busy || !sourceId || !sourceName.trim()}
            onClick={() =>
              void act(async () => {
                const saved = await api("context-supply/sources/legacy-index", {
                  document_id: sourceId,
                  name: sourceName,
                });
                setPublishedResource(saved.id);
                await onPublished?.();
                setNotice("已保存原文资料快照，来源和摘要已记录。");
              })
            }
          >
            保存文档快照
          </button>
        </section>
        }
        {mode === "build" &&
        <section className="agent-card">
          <h3>用资料版本构建索引</h3>
          <button
            className="text-button"
            disabled={busy}
            onClick={() => void refresh().catch((e) => setError(e.message))}
          >
            刷新可选资料
          </button>
          <label>
            索引名称
            <input
              aria-label="索引名称"
              value={name}
              maxLength={120}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <fieldset>
            <legend>索引资料版本</legend>
            {resources.map((r) => (
              <label key={r.id} className="native-check">
                <input
                  type="checkbox"
                  checked={selected.includes(r.id)}
                  onChange={(e) =>
                    setSelected((old) =>
                      e.target.checked
                        ? [...old, r.id]
                        : old.filter((x) => x !== r.id),
                    )
                  }
                />
                {r.name} · v{r.version}
              </label>
            ))}
          </fieldset>
          <button
            className="primary"
            disabled={busy || !name.trim() || !selected.length}
            onClick={() =>
              void act(async () => {
                const item = await api("context-supply/indexes", {
                  name,
                  resources: selected.map((id) => ({
                    id,
                    version: resources.find((r) => r.id === id)!.version,
                  })),
                });
                setIndexId(item.id);
                setMode("search");
                if(route.view === "indexes") navigate(-2,"indexes",item.id);
                setNotice("索引已构建，原资料版本已固定。");
              })
            }
          >
            构建资料索引
          </button>
        </section>
        }
      </div>
      {mode === "sources" && sourceType === "graph" &&
      <section className="agent-card editor-page">
        <h3>从图谱服务保存事实快照</h3>
        <p className="run-help">
          只读取已有实体关系，不重载或写入图谱。快照记录主体、关系、客体及文档来源；图谱更新后需另存新快照。
        </p>
        <label>
          图谱实体
          <input
            aria-label="图谱实体"
            value={entity}
            onChange={(e) => setEntity(e.target.value)}
            maxLength={200}
          />
        </label>
        <label>
          图谱快照名称
          <input
            aria-label="图谱快照名称"
            value={graphName}
            onChange={(e) => setGraphName(e.target.value)}
            maxLength={120}
          />
        </label>
        <button
          className="secondary"
          disabled={busy || !entity.trim() || !graphName.trim()}
          onClick={() =>
            void act(async () => {
              const saved = await api("context-supply/sources/legacy-graph", {
                entity,
                name: graphName,
              });
              setPublishedResource(saved.id);
              await onPublished?.();
              setNotice("已保存图谱事实快照，可与其他资料组合索引。");
            })
          }
        >
          保存图谱事实快照
        </button>
      </section>
      }
      {mode === "search" && indexId && <section className="agent-card">
        <h3>检索检查与岗位引用</h3>
        <p className="run-help">资料版本：{indexes.find((item) => item.id === indexId)?.manifest.sources.map((source) => source.name + " · v" + source.version).join("、")}</p>
        <label>
          检索索引
          <select
            aria-label="检索索引"
            value={indexId}
            onChange={(e) => {
              setIndexId(e.target.value);
              setResult(null);
            }}
          >
            <option value="">请选择索引</option>
            {indexes.map((x) => (
              <option key={x.id} value={x.id}>
                {x.name} · {x.digest.slice(0, 12)}
              </option>
            ))}
          </select>
        </label>
        <label>
          检索问题
          <input
            aria-label="检索问题"
            value={query}
            maxLength={1000}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <div className="agent-actions">
          <button
            className="secondary"
            disabled={busy || !indexId || !query.trim()}
            onClick={() =>
              void act(async () =>
                setResult(
                  await api("context-supply/indexes/" + indexId + "/search", {
                    query,
                  }),
                ),
              )
            }
          >
            检查检索结果
          </button>
          <button
            className="primary"
            disabled={busy || !indexId}
            onClick={() =>
              void act(async () => {
                const saved = await api("context-supply/indexes/" + indexId + "/reference", {
                  name:
                    (indexes.find((x) => x.id === indexId)?.name || "资料") +
                    " · 岗位检索",
                });
                setPublishedResource(saved.id);
                await onPublished?.();
                setNotice(
                  "已生成检索资源，请将它加入供应配置后为岗位创建新版本。",
                );
              })
            }
          >
            生成岗位检索引用
          </button>
        </div>
        <details className="advanced-options"><summary>用模型验证回答</summary><label>
          验证使用的模型服务
          <select
            aria-label="验证使用的模型服务"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
          >
            <option value="">请选择模型服务</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </label>
        <p className="run-help">
          验证会调用选定模型，并返回实际检索资料及提示词；结果不会自动进入供应配置。
        </p>
        <button
          className="secondary"
          disabled={busy || !indexId || !modelId || !query.trim()}
          onClick={() =>
            void act(async () =>
              setResult(
                await api("context-supply/validate", {
                  index_id: indexId,
                  model_id: modelId,
                  query,
                }),
              ),
            )
          }
        >
          验证资料与模型组合
        </button>
        </details>
        {result != null && (
          <div className="context-preview evidence-results">
            {result.answer && <section><h4>回答</h4><p>{result.answer}</p></section>}
            <h4>检索资料 · {result.documents?.length || 0} 条</h4>
            {!result.documents?.length && <p>未检索到匹配资料，请调整问题或索引内容。</p>}
            {result.documents?.map((document) => <article className="evidence-document" key={document.id}><strong>{document.name} · v{document.version}</strong><p className="run-help">{document.source}</p><p>{document.content.slice(0, 600)}</p>{document.content.length > 600 && <details><summary>查看完整资料</summary><pre>{document.content}</pre></details>}</article>)}
            {result.prompt_sent && <details><summary>实际提示词</summary><pre>{result.prompt_sent}</pre></details>}
            <details><summary>原始响应</summary><pre>{JSON.stringify(result, null, 2)}</pre></details>
          </div>
        )}
      </section>
      }
    </section>
  );
}
