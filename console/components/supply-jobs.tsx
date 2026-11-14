"use client";
import { useEffect, useState } from "react";
import { platformRequest as api } from "../lib/platform-api";
import { useWorkspace, WorkspaceLink } from "../lib/workspace-navigation";

type Job = { id: string; name: string; kind: string; status: string; error: string; result: Record<string, unknown>; log?: string; created_at: number; finished_at?: number };
const states: Record<string, string> = { queued: "排队中", running: "执行中", serving: "服务运行中", cancelling: "正在停止", cancelled: "已取消", completed: "已完成", failed: "失败", interrupted: "已中断，请核对" };
const kinds: Record<string, string> = { import: "资料导入", train: "模型训练", merge: "合并导出", deploy: "模型部署" };
const active = ["queued", "running", "serving", "cancelling"];
const date = (value?: number) => value ? new Date(value * 1000).toLocaleString("zh-CN") : "—";

export default function SupplyJobs({ kind }: { kind?: string }) {
  const { route, navigate } = useWorkspace();
  const [jobs, setJobs] = useState<Job[]>([]), [loaded, setLoaded] = useState(false), [selected, setSelected] = useState<Job | null>(null);
  const [query, setQuery] = useState(""), [type, setType] = useState(""), [status, setStatus] = useState(""), [error, setError] = useState("");
  const selectedId = route.area === -4 ? route.record : undefined;
  useEffect(() => {
    let live = true;
    async function refresh() {
      try {
        const data: Job[] = await api("jobs");
        if (live) { setJobs(data.filter((job) => !kind || (kind === "model" ? ["train", "merge", "deploy"].includes(job.kind) : job.kind === kind)));setLoaded(true); }
      } catch (e) { if (live) setError((e as Error).message); }
    }
    void refresh();const timer = setInterval(() => void refresh(), 3000);
    return () => { live = false;clearInterval(timer); };
  }, [kind]);
  useEffect(() => {
    setSelected(null);setError("");
    if (!selectedId) return;
    let live = true;
    void api("jobs/" + selectedId).then((job) => { if (live) setSelected(job); }).catch((e) => { if (live) setError(e.message); });
    return () => { live = false; };
  }, [selectedId]);
  useEffect(() => {
    if (!selected || !active.includes(selected.status)) return;
    let live = true;
    const timer = setInterval(() => void api("jobs/" + selected.id).then((job) => { if (live) setSelected(job); }).catch((e) => { if (live) setError(e.message); }), 1000);
    return () => { live = false;clearInterval(timer); };
  }, [selected?.id, selected?.status]);
  const rows = jobs.filter((job) => job.name.includes(query) && (!type || job.kind === type) && (!status || job.status === status));
  return <section data-verify-unit="SupplyJobs" data-verify-state={loaded ? "ready" : "loading"} aria-label="供应作业">
    {error && <p role="alert" className="inline-error">{error}</p>}
    {selectedId ? <>
      <div className="collection-heading"><div><h2>{selected?.name || "作业详情"}</h2><p>{selected ? kinds[selected.kind] + " · " + states[selected.status] : "正在载入…"}</p></div><button className="text-button" onClick={() => navigate(-4)}>返回作业列表</button></div>
      {selected && <div className="agent-card" data-supply-job-state={selected.status}>
        <dl className="environment-list"><dt>状态</dt><dd><span className={"status-badge status-" + selected.status}>{states[selected.status]}</span></dd><dt>提交时间</dt><dd>{date(selected.created_at)}</dd><dt>结束时间</dt><dd>{date(selected.finished_at)}</dd><dt>作业编号</dt><dd><code>{selected.id}</code></dd></dl>
        {selected.error && <p role="alert" className="inline-error">{selected.error}</p>}
        {selected.status === "completed" && <div className="job-outcome"><h3>执行结果</h3>
          {selected.result.resource_id != null && <><p>已生成上下文资源 · v{String(selected.result.version)} · {String(selected.result.chars)} 字符</p><WorkspaceLink area={-2} view="resources" record={String(selected.result.resource_id)} className="resource-link">查看生成的资料</WorkspaceLink></>}
          {selected.result.artifact_id != null && <><p>模型产物已登记。</p><WorkspaceLink area={-3} view="artifacts" record={String(selected.result.artifact_id)} className="resource-link">查看模型产物</WorkspaceLink></>}
        </div>}
        {selected.status === "serving" && <WorkspaceLink area={-3} view="services" className="resource-link">查看已登记的模型服务</WorkspaceLink>}
        {active.includes(selected.status) && <div className="agent-actions"><button className="secondary" disabled={selected.status === "cancelling"} onClick={() => void api("jobs/" + selected.id + "/cancel", {}).then(setSelected).catch((e) => setError(e.message))}>{selected.status === "serving" ? "停止模型服务" : "取消作业"}</button></div>}
        <details className="advanced-options" open={selected.status === "failed"}><summary>执行日志</summary><pre className="context-preview">{selected.log || "该作业没有进程日志。"}</pre></details>
        <details className="advanced-options"><summary>完整结果</summary><pre className="context-preview">{JSON.stringify(selected.result, null, 2)}</pre></details>
      </div>}
    </> : <>
      <div className="collection-heading"><div><h2>供应作业 <span className="collection-count">{jobs.length}</span></h2><p>跟踪已提交的资料导入、训练、合并与部署。</p></div></div>
      {jobs.length > 0 && <div className="resource-filters"><input aria-label="搜索作业" placeholder="按作业名称搜索" value={query} onChange={(e) => setQuery(e.target.value)} /><select aria-label="作业类型" value={type} onChange={(e) => setType(e.target.value)}><option value="">全部类型</option>{Object.entries(kinds).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select><select aria-label="作业状态" value={status} onChange={(e) => setStatus(e.target.value)}><option value="">全部状态</option>{Object.entries(states).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></div>}
      {loaded && !jobs.length ? <div className="collection-empty"><h3>暂无供应作业</h3><p>导入资料或提交模型任务后，在这里查看执行进度。</p><div className="agent-actions"><WorkspaceLink area={-2} view="resources" className="secondary">前往资料库</WorkspaceLink><WorkspaceLink area={-3} view="artifacts" className="secondary">前往模型产物</WorkspaceLink></div></div>
      : <div className="resource-table-scroll"><table className="resource-table"><thead><tr><th>作业名称</th><th>类型</th><th>状态</th><th>提交时间</th></tr></thead><tbody>{rows.map((job) => <tr key={job.id} data-supply-job-state={job.status}><td><WorkspaceLink area={-4} record={job.id} className="resource-link">{job.name}</WorkspaceLink></td><td>{kinds[job.kind]}</td><td><span className={"status-badge status-" + job.status}>{states[job.status]}</span></td><td>{date(job.created_at)}</td></tr>)}</tbody></table>{loaded && !rows.length && <p className="collection-empty">没有匹配的作业。</p>}</div>}
    </>}
  </section>;
}
