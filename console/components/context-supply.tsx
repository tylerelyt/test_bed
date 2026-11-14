"use client";
import { useEffect, useState } from "react";
import { downloadJSON } from "../lib/workbench-api";
import { platformRequest as request } from "../lib/platform-api";
import ContextImport from "./context-import";
import ContextIndexes from "./context-indexes";
import { useWorkspace, WorkspaceLink, useDetailName, workspaceHref } from "../lib/workspace-navigation";
type Resource = {
  id: string;
  name: string;
  kind: string;
  permission: string;
  version: number;
  source: string;
  tags: string[];
  content?: string;
};
type Ref = {
  id: string;
  version?: number;
  permission?: string;
  required?: boolean;
};
type Bundle = {
  id: string;
  name: string;
  version: number;
  max_chars: number;
  resources: Ref[];
};
type Assembled = {
  name: string;
  bundle_id: string;
  version: number;
  used_chars: number;
  max_chars: number;
  resources: (Resource & { resolved_version: number; chars: number })[];
  omitted: { id: string; reason: string }[];
  instructions: string;
};
const kinds: Record<string, string> = {
  knowledge: "资料",
  instructions: "操作说明",
  tool: "工具定义",
  experience: "已审阅经验",
  notebook: "Agent 笔记",
  role: "角色职责",
  retrieval: "检索索引",
};
function ContextEditor({ legacy, title, open, children }: { legacy: boolean; title: string; open?: boolean; children: React.ReactNode }) {
  return legacy ? <details className="run-editor" open={open}><summary>{title}</summary>{children}</details> : <section className="agent-card editor-page"><h2>{title}</h2>{children}</section>;
}
export default function ContextSupply({ view = "all" }: { view?: "all" | "resources" | "bundles" }) {
  const { navigate, route } = useWorkspace();
  const [resourceEditing, setResourceEditing] = useState(false), [bundleEditing, setBundleEditing] = useState(false), [importing, setImporting] = useState(false), [filter, setFilter] = useState("");
  const [resources, setResources] = useState<Resource[]>([]),
    [bundleResources, setBundleResources] = useState<Resource[]>([]),
    [bundles, setBundles] = useState<Bundle[]>([]),
    [bundleId, setBundleId] = useState(""),
    [bundle, setBundle] = useState<Bundle | null>(null),
    [preview, setPreview] = useState<Assembled | null>(null);
  const [resourceId, setResourceId] = useState(""),
    [resourceVersion, setResourceVersion] = useState<number | undefined>(),
    [name, setName] = useState(""),
    [kind, setKind] = useState("knowledge"),
    [permission, setPermission] = useState("read_only"),
    [content, setContent] = useState(""),
    [source, setSource] = useState(""),
    [tags, setTags] = useState("");
  const [bundleName, setBundleName] = useState(""),
    [refs, setRefs] = useState<Ref[]>([]),
    [budget, setBudget] = useState(12000),
    [editingBundle, setEditingBundle] = useState<Bundle | null>(null),
    [query, setQuery] = useState(""),
    [error, setError] = useState(""),
    [message, setMessage] = useState(""),
    [loaded, setLoaded] = useState(false),
    [pending, setPending] = useState(false);
  async function refresh() {
    const data = await request("context");
    setResources(data.resources);
    setBundles(data.bundles);
    setLoaded(true);
  }
  useEffect(() => {
    void refresh().catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    if (view === "all" || !loaded) return;
    setResourceEditing(false);setBundleEditing(false);
    if (!route.record) { setResourceId("");setBundle(null);return; }
    setPending(true);
    void (view === "resources" ? selectResource(route.record!) : selectBundle(route.record!))
      .catch((e) => setError(e.message)).finally(() => setPending(false));
  }, [view, route.record, loaded]);
  async function act(work: () => Promise<void>) {
    setPending(true);
    setError("");
    setMessage("");
    try {
      await work();
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  async function selectResource(id: string, version?: number) {
    if (!id) {
      setResourceId("");
      setResourceVersion(undefined);
      setName("");
      setContent("");
      setSource("");
      setTags("");
      setKind("knowledge");setPermission("read_only");
      return;
    }
    const item: Resource = await request(
      `context/resources/${id}${version ? `?version=${version}` : ""}`,
    );
    setResourceId(id);
    setResourceVersion(item.version);
    setName(item.name);
    setKind(item.kind);
    setPermission(item.permission);
    setContent(item.content || "");
    setSource(item.source);
    setTags(item.tags.join("、"));
  }
  async function selectBundle(id: string) {
    setBundleId(id);
    setPreview(null);
    setBundleResources([]);
    if (!id) { setBundle(null);return; }
    const item: Bundle = await request("context/bundles/" + id);
    setBundle(item);
    if (view !== "all") setBundleResources(await Promise.all(item.resources.map((ref) => request("context/resources/" + ref.id + (ref.version ? "?version=" + ref.version : "")))));
  }
  const historical =
    resourceId &&
    resources.find((item) => item.id === resourceId)?.version !==
      resourceVersion;
  useDetailName(view === "resources" && resourceId ? name : view === "bundles" ? bundle?.name : undefined);
  return (
    <div
      className="context-supply"
      data-verify-unit="ContextSupply"
      data-verify-state={!loaded ? "loading" : pending ? "pending" : "ready"}
    >
      {view === "all" && <ContextIndexes onPublished={refresh} />}
      {view !== "all" && <div className="collection-heading"><div><h1>{resourceEditing ? resourceId ? "编辑「" + name + "」" : "新建上下文资源" : bundleEditing ? editingBundle ? "编辑「" + editingBundle.name + "」" : "新建供应配置" : importing ? "导入资料文件" : resourceId ? name : bundle?.name || (view === "resources" ? "资料库" : "供应配置")}</h1><p>{view === "resources" ? "维护企业资料、规则和可复用经验，保留每次修改的版本。" : "选择资源版本，组合成岗位 Agent 使用的上下文。"}</p></div>
        <div className="agent-actions">{resourceEditing || bundleEditing || importing || resourceId || bundle ? <button className="text-button" onClick={() => { if (route.record && (resourceEditing || bundleEditing)) { setResourceEditing(false);setBundleEditing(false);return; }setResourceEditing(false);setBundleEditing(false);setImporting(false);setResourceId("");setBundle(null);if (route.record) navigate(-2, view === "resources" ? "resources" : "configuration"); }}>{route.record && (resourceEditing || bundleEditing) ? "返回详情" : "返回列表"}</button>
        : view === "resources" ? <><WorkspaceLink className="secondary" area={-2} view="sources">接入业务资料</WorkspaceLink><button className="secondary" onClick={() => setImporting(true)}>导入文件</button><button className="primary" onClick={() => { void selectResource("");setResourceEditing(true); }}>新建资源</button></>
        : <button className="primary" onClick={() => { setEditingBundle(null);setBundleName("");setRefs([]);setBudget(12000);setBundleEditing(true); }}>新建供应配置</button>}</div>
      </div>}
      {view !== "all" && !resourceEditing && !bundleEditing && !importing && !resourceId && !bundle && <div className="collection-page">
        <input className="collection-search" aria-label="搜索上下文资源" placeholder="按名称或标签搜索" value={filter} onChange={(e) => setFilter(e.target.value)} />
        {view === "resources" ? (!resources.length ? <div className="collection-empty"><h3>尚无上下文资源</h3><p>导入企业资料或新建规则，再将它们加入供应配置。</p></div> : <div className="resource-table-scroll"><table className="resource-table"><thead><tr><th>资源名称</th><th>类型</th><th>最新版本</th><th>来源</th></tr></thead><tbody>{resources.filter((item) => (item.name + item.tags.join(" ")).includes(filter)).map((item) => <tr key={item.id}><td><WorkspaceLink className="resource-link" area={-2} view="resources" record={item.id}>{item.name}</WorkspaceLink></td><td>{kinds[item.kind]}</td><td>v{item.version}</td><td>{item.source || "未填写"}</td></tr>)}</tbody></table></div>)
        : (!bundles.length ? <div className="collection-empty"><h3>尚无供应配置</h3><p>创建一份配置，明确岗位使用哪些资料、规则和工具定义。</p></div> : <div className="resource-table-scroll"><table className="resource-table"><thead><tr><th>配置名称</th><th>版本</th><th>资源数量</th><th>字符预算</th></tr></thead><tbody>{bundles.filter((item) => item.name.includes(filter)).map((item) => <tr key={item.id}><td><WorkspaceLink className="resource-link" area={-2} view="configuration" record={item.id}>{item.name}</WorkspaceLink></td><td>v{item.version}</td><td>{item.resources.length}</td><td>{item.max_chars}</td></tr>)}</tbody></table></div>)}
      </div>}
      {view === "resources" && resourceId && !resourceEditing && <div className="agent-card"><div className="collection-heading"><h2>资源内容</h2><button className="primary" onClick={() => setResourceEditing(true)}>编辑新版本</button></div><dl className="environment-list"><dt>类型</dt><dd>{kinds[kind]}</dd><dt>版本</dt><dd>v{resourceVersion}</dd><dt>来源</dt><dd>{source || "未填写"}</dd><dt>标签</dt><dd>{tags || "无"}</dd></dl><pre className="context-preview">{content}</pre><div className="editor-actions"><WorkspaceLink area={-2} view="indexes" className="secondary">前往构建索引</WorkspaceLink><WorkspaceLink area={-2} view="configuration" className="secondary">前往供应配置</WorkspaceLink></div></div>}
      {view === "resources" && importing && <div className="editor-page"><ContextImport showJobs={false} onSubmitted={(id) => navigate(-4, "jobs", id, {returnTo:route.returnTo || workspaceHref(-2,"resources"),returnLabel:route.returnLabel || "返回资料库"})} /></div>}
      {view === "all" && <>
      <p className="run-help">
        为 Agent 组合资料、操作说明、工具定义和经验。MCP 目录只返回元数据，Agent
        按需读取；配置固定资源版本，笔记写入单独授权。资料、MCP 与岗位发布
        共用同一份上下文记录，更新后可为岗位生成新的候选版本。
      </p>
      <button
        className="secondary"
        disabled={pending}
        onClick={() => void act(async () => {})}
      >
        刷新资源库
      </button>
      <details className="run-editor">
        <summary>从文件导入资料</summary>
        <ContextImport />
      </details>
      </>}
      {error && (
        <p role="alert" className="inline-error">
          {error}
        </p>
      )}
      {message && (
        <p role="status" className="run-message">
          {message}
        </p>
      )}
      {view === "all" && <div className="native-field">
        <label htmlFor="context-bundle">供应配置</label>
        <select
          id="context-bundle"
          value={bundleId}
          onChange={(e) => void act(() => selectBundle(e.target.value))}
        >
          <option value="">选择一份配置</option>
          {bundles.map((item) => (
            <option key={item.id} value={item.id}>
              {item.name} · v{item.version}
            </option>
          ))}
        </select>
      </div>
      }
      {bundle && (view === "all" || !bundleEditing) && (
        <div className="run-detail">
          <div className="run-meta">
            <strong>
              {view === "all" ? bundle.name + " · " : "配置版本 "}v{bundle.version}
            </strong>
            <span>
              {bundle.resources.length} 个资源 ·{" "}
              {bundle.max_chars.toLocaleString("zh-CN")} 字符预算
            </span>
          </div>
          {view !== "all" && <section className="bundle-members"><h2>包含的资源</h2><div className="resource-table-scroll"><table className="resource-table"><thead><tr><th>资源名称</th><th>类型</th><th>固定版本</th><th>使用规则</th></tr></thead><tbody>{bundle.resources.map((ref) => { const item = bundleResources.find((resource) => resource.id === ref.id);return <tr key={ref.id}><td>{item?.name || (pending ? "正在载入…" : "未能读取资源")}</td><td>{item ? kinds[item.kind] : "—"}</td><td>v{ref.version || item?.version || "—"}</td><td>{ref.required ? "必选" : "按需"} · {ref.permission === "read_write" ? "允许更新笔记" : "只读"}</td></tr>; })}</tbody></table></div></section>}
          <details className="advanced-options" open={view === "all" || undefined}><summary>MCP 接入信息</summary><p className="context-uri">
            MCP 资源：
            <code>
              context://bundles/{bundle.id}/{bundle.version}
            </code>
          </p>
          <p className="run-help">
            Agent 可用 discover_context 查找资源，再用 load_context
            按需装配。只有授权的笔记可以通过 write_context_notebook 更新。
          </p>
          </details>
          <div className="native-field">
            <label htmlFor="context-query">
              按名称、类型或标签筛选装配资源（可选）
            </label>
            <input
              id="context-query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="必选资源始终保留"
            />
          </div>
          <div className="run-actions">
            <button
              className="secondary"
              disabled={pending}
              onClick={() =>
                void act(async () =>
                  setPreview(
                    await request("context", {
                      action: "assemble",
                      bundle_id: bundle.id,
                      version: bundle.version,
                      query,
                    }),
                  ),
                )
              }
            >
              检查装配结果
            </button>
            <button
              className="text-button"
              onClick={() =>
                downloadJSON("上下文配置-" + bundle.name + ".json", bundle)
              }
            >
              导出配置
            </button>
            <button
              className="text-button"
              onClick={() => {
                setEditingBundle(bundle);
                setBundleEditing(true);
                setBundleName(bundle.name);
                setRefs(bundle.resources);
                setBudget(bundle.max_chars);
                setMessage("已载入配置，可在下方修改并保存新版本。");
              }}
            >
              编辑新版本
            </button>
          </div>
          {preview && (
            <div className="context-preview">
              <h4>
                实际装配 · {preview.used_chars} / {preview.max_chars} 字符
              </h4>
              <p className="run-help">{preview.instructions}</p>
              {preview.resources.map((item) => (
                <details key={item.id}>
                  <summary>
                    {kinds[item.kind]} · {item.name} · v{item.resolved_version}
                  </summary>
                  <p className="run-help">
                    来源：{item.source || "未填写"} ·{" "}
                    {item.permission === "read_write" ? "允许笔记写入" : "只读"}
                  </p>
                  <pre>{item.content}</pre>
                </details>
              ))}
              {preview.omitted.map((item) => (
                <p key={item.id} className="run-help">
                  {resources.find((resource) => resource.id === item.id)
                    ?.name || "资源"}
                  ：{item.reason}
                </p>
              ))}
              <button
                className="text-button"
                onClick={() => downloadJSON("上下文装配证据.json", preview)}
              >
                导出装配证据
              </button>
            </div>
          )}
        </div>
      )}
      {(view === "all" || bundleEditing) && <ContextEditor legacy={view === "all"} open={bundleEditing || editingBundle != null} title={editingBundle ? "编辑供应配置" : "组合供应配置"}>
        <div className="native-field">
          <label htmlFor="bundle-name">配置名称</label>
          <input
            id="bundle-name"
            value={bundleName}
            maxLength={120}
            onChange={(e) => setBundleName(e.target.value)}
          />
          <label htmlFor="context-budget">字符预算</label>
          <input
            id="context-budget"
            type="number"
            min={100}
            max={200000}
            value={budget}
            onChange={(e) => setBudget(Number(e.target.value))}
          />
          <fieldset>
            <legend>选择资源（按勾选顺序装配）</legend>
            {resources.length ? (
              resources.map((item) => {
                const ref = refs.find((ref) => ref.id === item.id);
                return (
                  <div className="resource-choice" key={item.id}>
                    <label className="native-check">
                      <input
                        type="checkbox"
                        checked={Boolean(ref)}
                        onChange={(e) =>
                          setRefs((old) =>
                            e.target.checked
                              ? [
                                  ...old,
                                  {
                                    id: item.id,
                                    version: item.version,
                                    permission: "read_only",
                                    required: false,
                                  },
                                ]
                              : old.filter((ref) => ref.id !== item.id),
                          )
                        }
                      />
                      {item.name}
                      <small>
                        {kinds[item.kind]} · v{ref?.version || item.version}
                      </small>
                    </label>
                    {ref && (
                      <div className="resource-options">
                        <label className="native-check">
                          <input
                            type="checkbox"
                            checked={Boolean(ref.required)}
                            onChange={(e) =>
                              setRefs((old) =>
                                old.map((ref) =>
                                  ref.id === item.id
                                    ? { ...ref, required: e.target.checked }
                                    : ref,
                                ),
                              )
                            }
                          />
                          必选
                        </label>
                        {item.permission === "read_write" && (
                          <label className="native-check">
                            <input
                              type="checkbox"
                              checked={ref.permission === "read_write"}
                              onChange={(e) =>
                                setRefs((old) =>
                                  old.map((ref) =>
                                    ref.id === item.id
                                      ? {
                                          ...ref,
                                          permission: e.target.checked
                                            ? "read_write"
                                            : "read_only",
                                        }
                                      : ref,
                                  ),
                                )
                              }
                            />
                            允许 Agent 更新笔记
                          </label>
                        )}
                        <button
                          className="text-button"
                          onClick={() =>
                            setRefs((old) =>
                              old.map((ref) =>
                                ref.id === item.id
                                  ? { ...ref, version: item.version }
                                  : ref,
                              ),
                            )
                          }
                        >
                          使用最新资源版本
                        </button>
                      </div>
                    )}
                  </div>
                );
              })
            ) : (
              <p className="run-help">
                请先在资料库创建资源，或将已审阅经验加入资源库。
              </p>
            )}
          </fieldset>
          <button
            className="secondary"
            disabled={pending || !bundleName.trim() || !refs.length}
            onClick={() =>
              void act(async () => {
                const saved = await request("context", {
                  action: "bundle",
                  name: bundleName,
                  resources: refs,
                  max_chars: budget,
                  bundle_id: editingBundle?.id,
                  expected_version: editingBundle?.version,
                });
                setEditingBundle(null);
                setRefs([]);
                setBundleName("");
                setBundleId(saved.id);
                setBundle(saved);
                setBundleEditing(false);
                setMessage("供应配置已保存，原版本仍可读取。");
                if (view !== "all") navigate(-2, "configuration", saved.id);
              })
            }
          >
            保存供应配置{editingBundle ? "新版本" : ""}
          </button>
          {editingBundle && (
            <button
              className="text-button"
              onClick={() => {
                setEditingBundle(null);
                setBundleName("");
                setRefs([]);
              }}
            >
              取消编辑
            </button>
          )}
        </div>
      </ContextEditor>
      }
      {(view === "all" || resourceEditing) && <ContextEditor legacy={view === "all"} title={view === "all" ? "维护上下文资源" : resourceId ? "编辑上下文资源" : "新建上下文资源"}>
        <div className="native-field">
          {view === "all" && <><label htmlFor="context-resource">编辑资源或创建新资源</label>
          <select
            id="context-resource"
            value={resourceId}
            onChange={(e) => void act(() => selectResource(e.target.value))}
          >
            <option value="">创建新资源</option>
            {resources.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name} · {kinds[item.kind]} · v{item.version}
              </option>
            ))}
          </select>
          </>}
          {resourceId && (
            <div className="run-actions">
              <span>
                当前读取 v{resourceVersion}
                {historical ? "（历史版本，只读）" : ""}
              </span>
              <button
                className="text-button"
                disabled={(resourceVersion || 1) <= 1}
                onClick={() =>
                  void act(() =>
                    selectResource(resourceId, (resourceVersion || 1) - 1),
                  )
                }
              >
                读取上一版本
              </button>
              <button
                className="text-button"
                onClick={() => void act(() => selectResource(resourceId))}
              >
                读取最新版本
              </button>
            </div>
          )}
          <div className="run-grid">
            <div className="native-field">
              <label htmlFor="resource-name">资源名称</label>
              <input
                id="resource-name"
                value={name}
                maxLength={120}
                onChange={(e) => setName(e.target.value)}
              />
            </div>
            <div className="native-field">
              <label htmlFor="resource-kind">资源类型</label>
              <select
                id="resource-kind"
                value={kind}
                disabled={Boolean(resourceId)}
                onChange={(e) => {
                  setKind(e.target.value);
                  setPermission("read_only");
                }}
              >
                {Object.entries(kinds).filter(([key]) => key !== "retrieval" || Boolean(resourceId)).map(([key, name]) => (
                  <option key={key} value={key}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <label htmlFor="resource-source">来源与适用范围</label>
          <input
            id="resource-source"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="文档地址、维护团队或适用任务"
          />
          <label htmlFor="resource-tags">标签</label>
          <input
            id="resource-tags"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="用顿号或逗号分隔"
          />
          <label htmlFor="resource-content">
            {kind === "tool"
              ? "工具定义 JSON（name、description、input_schema）"
              : "内容"}
          </label>
          <textarea
            id="resource-content"
            rows={7}
            value={content}
            onChange={(e) => setContent(e.target.value)}
            readOnly={Boolean(historical)}
          />
          {kind === "notebook" && (
            <label className="native-check">
              <input
                type="checkbox"
                disabled={Boolean(resourceId)}
                checked={permission === "read_write"}
                onChange={(e) =>
                  setPermission(e.target.checked ? "read_write" : "read_only")
                }
              />
              允许在配置中授权 Agent 写入此笔记
            </label>
          )}
          <button
            className="secondary"
            disabled={pending || Boolean(historical) || !name.trim()}
            onClick={() =>
              void act(async () => {
                const saved = await request("context", {
                  action: "resource",
                  resource_id: resourceId || undefined,
                  expected_version: resourceVersion,
                  name,
                  kind,
                  permission,
                  content,
                  source,
                  tags: tags
                    .split(/[、,，]/)
                    .map((tag) => tag.trim())
                    .filter(Boolean),
                });
                setResourceId(saved.id);
                setResourceVersion(saved.version);
                setResourceEditing(false);
                setMessage("资源版本已保存。配置中的只读资源仍固定在原版本。");
                if (view !== "all") navigate(-2, "resources", saved.id);
              })
            }
          >
            保存资源{resourceId ? "新版本" : ""}
          </button>
        </div>
      </ContextEditor>
      }
    </div>
  );
}
