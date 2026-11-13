"use client";
import { useEffect, useState } from "react";
import { platformRequest as api } from "../lib/platform-api";
type Resource = { id: string; name: string; kind: string; version: number };
type Index = {
  id: string;
  name: string;
  digest: string;
  manifest: { sources: { id: string; name: string; version: number }[] };
};
export default function ContextIndexes({
  onPublished,
}: {
  onPublished?: () => Promise<void>;
}) {
  const [resources, setResources] = useState<Resource[]>([]),
    [indexes, setIndexes] = useState<Index[]>([]),
    [selected, setSelected] = useState<string[]>([]),
    [name, setName] = useState(""),
    [indexId, setIndexId] = useState(""),
    [query, setQuery] = useState(""),
    [result, setResult] = useState<unknown>(null),
    [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [busy, setBusy] = useState(false);
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
  }
  useEffect(() => {
    void refresh().catch((e) => setError(e.message));
  }, []);
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
    <details className="run-editor">
      <summary>资料来源与检索索引</summary>
      <p className="run-help">
        直接读取原文档索引服务的资料，或选用已登记资料构建独立索引。索引固定原文版本；岗位只引用明确发布的索引。
      </p>
      {error && <p role="alert">{error}</p>}
      {notice && <p role="status">{notice}</p>}
      <div className="agent-grid">
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
                await api("context-supply/sources/legacy-index", {
                  document_id: sourceId,
                  name: sourceName,
                });
                await onPublished?.();
                setNotice("已保存原文资料快照，来源和摘要已记录。");
              })
            }
          >
            保存文档快照
          </button>
        </section>
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
                setNotice("索引已构建，原资料版本已固定。");
              })
            }
          >
            构建资料索引
          </button>
        </section>
      </div>
      <section className="agent-card">
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
              await api("context-supply/sources/legacy-graph", {
                entity,
                name: graphName,
              });
              await onPublished?.();
              setNotice("已保存图谱事实快照，可与其他资料组合索引。");
            })
          }
        >
          保存图谱事实快照
        </button>
      </section>
      <section className="agent-card">
        <h3>检索检查与岗位引用</h3>
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
                await api("context-supply/indexes/" + indexId + "/reference", {
                  name:
                    (indexes.find((x) => x.id === indexId)?.name || "资料") +
                    " · 岗位检索",
                });
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
        <label>
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
        {result != null && (
          <pre className="context-preview">
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </section>
    </details>
  );
}
