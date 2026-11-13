"use client";
import { useEffect, useState } from "react";
import { platformRequest as api } from "../lib/platform-api";

type Job = {
  id: string;
  name: string;
  kind: string;
  status: string;
  error: string;
  result: Record<string, unknown>;
  log?: string;
  created_at: number;
};
const states: Record<string, string> = {
  queued: "排队中",
  running: "执行中",
  serving: "服务运行中",
  cancelling: "正在停止",
  cancelled: "已取消",
  completed: "已完成",
  failed: "失败",
  interrupted: "已中断，请核对",
};
const kinds: Record<string, string> = {
  import: "资料导入",
  train: "模型训练",
  merge: "合并导出",
  deploy: "模型部署",
};
export default function SupplyJobs({ kind }: { kind?: string }) {
  const [jobs, setJobs] = useState<Job[]>([]),
    [loaded, setLoaded] = useState(false),
    [selected, setSelected] = useState<Job | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    async function refresh() {
      try {
        const data = await api("jobs");
        if (live) {
          setJobs(data.filter((x: Job) => !kind || (kind === "model" ? ["train", "merge", "deploy"].includes(x.kind) : x.kind === kind)));
          setLoaded(true);
        }
      } catch (e) {
        if (live) setError((e as Error).message);
      }
    }
    void refresh();
    const timer = setInterval(() => void refresh(), 2000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [kind]);
  useEffect(() => {
    if (!selected) return;
    let live = true;
    const id = selected.id;
    const timer = setInterval(
      () =>
        void api("jobs/" + id)
          .then((x) => {
            if (live) setSelected(x);
          })
          .catch((e) => {
            if (live) setError(e.message);
          }),
      1500,
    );
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [selected?.id]);
  async function inspect(id: string) {
    try {
      setError("");
      setSelected(await api("jobs/" + id));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <section
      className="agent-card"
      aria-label="供应作业"
      data-verify-unit="SupplyJobs"
      data-verify-state={loaded ? "ready" : "loading"}
    >
      <h2>{kind === "import" ? "资料导入作业" : kind === "model" ? "模型作业" : "供应作业"}</h2>
      <p className="run-help">
        查看已提交的资料导入、模型训练、合并和部署作业。岗位任务在“岗位 Agent”查看，检索和验证记录在对应能力中查看。
      </p>
      {error && (
        <p role="alert" className="inline-error">
          {error}
        </p>
      )}
      {loaded && !jobs.length && <p>暂无供应作业。请先在上下文供应提交资料导入，或在模型供应提交模型作业。</p>}
      {jobs.map((job) => (
        <div
          className="agent-row"
          key={job.id}
          data-supply-job-state={job.status}
        >
          <div>
            <strong>{job.name}</strong>
            <p>
              {kinds[job.kind]} ·{" "}
              {states[job.status] || job.status}
            </p>
            <small>提交于 {new Date(job.created_at * 1000).toLocaleString("zh-CN")}</small>
            {job.error && <p>{job.error}</p>}
          </div>
          <button className="secondary" onClick={() => void inspect(job.id)}>
            查看详情与日志
          </button>
        </div>
      ))}
      {selected && (
        <div className="context-preview">
          <h3>
            {selected.name} · {states[selected.status]}
          </h3>
          <p>作业编号：{selected.id}</p>
          {selected.error && <p role="alert">{selected.error}</p>}
          <pre>{JSON.stringify(selected.result, null, 2)}</pre>
          <h4>原生进程日志（最近 32 KB）</h4>
          <pre>{selected.log || "尚无原生进程日志；资料导入结果见上方。"}</pre>
          {["queued", "running", "serving", "cancelling"].includes(
            selected.status,
          ) && (
            <button
              className="secondary"
              disabled={selected.status === "cancelling"}
              onClick={() =>
                void api("jobs/" + selected.id + "/cancel", {})
                  .then(setSelected)
                  .catch((e) => setError(e.message))
              }
            >
              {selected.status === "serving" ? "停止模型服务" : "取消作业"}
            </button>
          )}
        </div>
      )}
    </section>
  );
}
