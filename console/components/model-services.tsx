"use client";
import { useEffect, useState } from "react";
import { platformRequest as api } from "../lib/platform-api";
import { WorkspaceLink, useWorkspace } from "../lib/workspace-navigation";
type Model = { id: string; name: string; model: string; base_url: string; has_credential: boolean; context_window: number; max_tokens: number };

export default function ModelServices() {
  const { route, navigate } = useWorkspace();
  const [models, setModels] = useState<Model[]>([]), [loaded, setLoaded] = useState(false), [editor, setEditor] = useState(false);
  const [selected, setSelected] = useState<Model | null>(null), [query, setQuery] = useState("");
  const [name, setName] = useState(""), [code, setCode] = useState(""), [url, setUrl] = useState(""), [key, setKey] = useState("");
  const [windowSize, setWindowSize] = useState(32000), [maxTokens, setMaxTokens] = useState(2048);
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [notice, setNotice] = useState("");
  async function refresh() { const data = await api("overview");setModels(data.models);setLoaded(true); }
  useEffect(() => { void refresh().catch((e) => setError(e.message)); }, []);
  useEffect(() => { setSelected(models.find((model) => model.id === route.record) || null); }, [route.record, models]);
  const rows = models.filter((model) => (model.name + " " + model.model).includes(query));
  return <section data-verify-unit="ModelServices" data-verify-state={loaded ? "ready" : "loading"}>
    {error && <p role="alert" className="inline-error">{error}</p>}
    {notice && <p role="status" className="run-message">{notice}</p>}
    {editor ? <form className="editor-page agent-card" onSubmit={async (event) => {
      event.preventDefault();setBusy(true);setError("");
      try {
        await api("models", { name, model: code, base_url: url, api_key: key, context_window: windowSize, max_tokens: maxTokens });
        setKey("");await refresh();setEditor(false);setNotice("模型服务配置已保存，请通过岗位试运行核对实际能力。");
      } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
    }}>
      <div className="collection-heading"><h2>接入模型服务</h2><button type="button" className="text-button" disabled={busy} onClick={() => setEditor(false)}>返回模型服务</button></div>
      <p className="run-help">填写已运行的模型服务地址，供岗位 Agent 选用。</p>
      <label>服务名称<input value={name} onChange={(e) => setName(e.target.value)} required maxLength={120} /></label>
      <label>服务地址<input value={url} onChange={(e) => setUrl(e.target.value)} required type="url" placeholder="http://127.0.0.1:8000/v1" /></label>
      <label>模型标识<input value={code} onChange={(e) => setCode(e.target.value)} required placeholder="服务实际提供的模型名称" /></label>
      <label>服务密钥（可选）<input value={key} onChange={(e) => setKey(e.target.value)} type="password" autoComplete="off" /></label>
      <details className="advanced-options"><summary>调用参数</summary><div className="agent-grid">
        <label>上下文长度<input type="number" min={1024} max={2000000} value={windowSize} onChange={(e) => setWindowSize(Number(e.target.value))} /></label>
        <label>输出上限<input type="number" min={1} max={32000} value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))} /></label>
      </div></details>
      <div className="editor-actions"><button type="button" className="secondary" disabled={busy} onClick={() => setEditor(false)}>取消</button><button className="primary" disabled={busy}>保存模型服务</button></div>
    </form> : selected ? <div className="agent-card">
      <div className="collection-heading"><h2>{selected.name}</h2><button className="text-button" onClick={() => navigate(-3, "services")}>返回模型服务</button></div>
      <dl className="environment-list"><dt>模型标识</dt><dd>{selected.model}</dd><dt>服务地址</dt><dd>{selected.base_url}</dd><dt>上下文长度</dt><dd>{selected.context_window}</dd><dt>输出上限</dt><dd>{selected.max_tokens}</dd><dt>调用凭证</dt><dd>{selected.has_credential ? "已保存" : "无需密钥"}</dd></dl>
      <WorkspaceLink area={-1} view="runs" className="primary">前往岗位配置</WorkspaceLink>
    </div> : <div className="collection-page">
      <div className="collection-heading"><div><h2>已接入服务 <span className="collection-count">{models.length}</span></h2><p>管理岗位 Agent 可以调用的模型服务。</p></div><button className="primary" onClick={() => { setName("");setCode("");setUrl("");setKey("");setEditor(true);setNotice("");setError(""); }}>接入模型服务</button></div>
      {models.length > 0 && <input className="collection-search" aria-label="搜索模型服务" placeholder="搜索服务名称或模型标识" value={query} onChange={(e) => setQuery(e.target.value)} />}
      {!models.length ? <div className="collection-empty"><h3>尚未接入模型服务</h3><p>接入一个已有服务后，即可为岗位 Agent 选择模型。</p></div> : <div className="resource-table-scroll"><table className="resource-table"><thead><tr><th>服务名称</th><th>模型标识</th><th>服务地址</th><th>凭证</th></tr></thead><tbody>
        {rows.map((model) => <tr key={model.id}><td><WorkspaceLink className="resource-link" area={-3} view="services" record={model.id}>{model.name}</WorkspaceLink></td><td>{model.model}</td><td>{model.base_url}</td><td>{model.has_credential ? "已配置" : "无需密钥"}</td></tr>)}
      </tbody></table>{!rows.length && <p className="collection-empty">没有匹配的服务。</p>}</div>}
    </div>}
  </section>;
}
