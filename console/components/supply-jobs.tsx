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
  module?: string;
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
  stopped: "已停止",
  awaiting_review: "待人工放行",
  rejected: "未放行",
};
const kinds: Record<string, string> = {
  import: "资料导入",
  train: "模型训练",
  merge: "合并导出",
  deploy: "模型部署",
  business: "业务操作",
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
      <h2>供应作业</h2>
      <p className="run-help">
        资料导入、训练和模型部署共用本机作业队列；岗位 Agent
        由独立的执行队列处理。检索、图谱、多模态、训练与评测运行也在这里统一查看。
      </p>
      {error && (
        <p role="alert" className="inline-error">
          {error}
        </p>
      )}
      {!jobs.length && <p>暂无作业。</p>}
      {jobs.map((job) => (
        <div
          className="agent-row"
          key={job.id}
          data-supply-job-state={job.status}
        >
          <div>
            <strong>{job.name}</strong>
            <p>
              {job.module || kinds[job.kind]} ·{" "}
              {states[job.status] || job.status}
            </p>
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
