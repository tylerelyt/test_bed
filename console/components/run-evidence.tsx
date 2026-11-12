"use client";
import { useEffect, useState } from "react";
import { NativeRun, request, verdictStatus } from "../lib/workbench-api";
type Event = {
  seq: number;
  type: string;
  fn_id?: number;
  at?: number;
  error?: string;
  updates?: Record<string, unknown>;
};
export default function RunEvidence({
  run,
  actions,
}: {
  run: NativeRun;
  actions: Record<string, string>;
}) {
  const [events, setEvents] = useState<Event[]>([]),
    [more, setMore] = useState(false),
    [error, setError] = useState("");
  useEffect(() => {
    let cancelled = false;
    const load = () =>
      request(`runs/${run.id}/trace`)
        .then((data) => {
          if (!cancelled) {
            setEvents(data.events);
            setMore(data.has_more);
          }
        })
        .catch((e) => {
          if (!cancelled) setError(e.message);
        });
    void load();
    const active = ["queued", "running", "awaiting_review"].includes(
      run.status,
    );
    const timer = active ? setInterval(() => void load(), 3000) : null;
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [run.id, run.status]);
  const kinds: Record<string, string> = {
    started: "开始操作",
    update: "更新结果",
    completed: "操作返回",
    failed: "操作失败",
  };
  return (
    <div
      className="run-evidence"
      data-verify-unit="RunEvidence"
      data-verify-run={run.id}
      data-verify-state={run.status}
    >
      {run.evaluation && (
        <div className="evaluation-verdict">
          <strong>
            {verdictStatus[run.evaluation.verdict] || run.evaluation.verdict} ·
            任务 v{run.evaluation.task_version || 1}
          </strong>
          <ul>
            {run.evaluation.checks.map((check) => (
              <li key={check.rule}>
                {check.label}：{verdictStatus[check.verdict]}
                {check.score != null && `（${check.score}）`}
                {check.reason && ` · ${check.reason}`}
              </li>
            ))}
          </ul>
        </div>
      )}
      {run.context && (
        <details>
          <summary>本次附加上下文 · 配置 v{run.context.bundle_version}</summary>
          <p className="run-help">
            已加入业务函数的文本输入。实际提示词和输出可在原模块与运行证据中核对。
          </p>
          <ul>
            {run.context.resources.map((resource) => (
              <li key={resource.id}>
                {resource.name} · v{resource.version}
              </li>
            ))}
          </ul>
          <p className="run-help">
            实际内容 {run.context.used_chars} 字符 · 内容摘要{" "}
            {run.context.content_sha256.slice(0, 16)}…
          </p>
        </details>
      )}
      {run.code && (
        <p className="run-help">
          执行源码：{run.code.git_commit?.slice(0, 7) || "未记录 Git 版本"} ·
          内容摘要 {run.code.source_sha256.slice(0, 12)}
        </p>
      )}
      {run.acceptance && (
        <p className="run-help">
          人工验收：{run.acceptance.passed ? "通过" : "未通过"} ·{" "}
          {run.acceptance.reason}
        </p>
      )}
      {!!run.usage?.length && (
        <details>
          <summary>实际模型用量</summary>
          <div className="native-table-scroll">
            <table>
              <thead>
                <tr>
                  <th>模型</th>
                  <th>输入 token</th>
                  <th>输出 token</th>
                  <th>按配置价格估算的成本</th>
                </tr>
              </thead>
              <tbody>
                {run.usage.map((call, index) => (
                  <tr key={index}>
                    <td>{call.model}</td>
                    <td>{call.input_tokens}</td>
                    <td>{call.output_tokens}</td>
                    <td>
                      {call.cost == null
                        ? "价格未配置"
                        : `${call.cost.toFixed(6)} ${call.currency || ""}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
      {!!run.artifacts?.length && (
        <div className="artifact-list">
          <h4>归档产物</h4>
          {run.artifacts.map((file) => (
            <div key={file.id}>
              <a href={file.url} download={file.name}>
                {file.name}
              </a>
              <span>{(file.size / 1024).toFixed(1)} KB</span>
              <code title={file.sha256}>
                SHA-256 {file.sha256.slice(0, 16)}…
              </code>
            </div>
          ))}
        </div>
      )}
      <details>
        <summary>
          操作时间线 · {events.length}
          {more ? "+" : ""} 个事件
        </summary>
        {error && (
          <p role="alert" className="inline-error">
            {error}
          </p>
        )}
        <ol className="run-timeline">
          {events.map((event) => (
            <li key={event.seq}>
              <time>
                {event.at
                  ? new Date(event.at * 1000).toLocaleTimeString("zh-CN")
                  : "—"}
              </time>
              <strong>{kinds[event.type] || event.type}</strong>
              <span>
                {event.fn_id != null
                  ? actions[event.fn_id] || `操作 ${event.fn_id}`
                  : ""}
              </span>
              {event.error && <p>{event.error}</p>}
              {event.updates && (
                <small>更新 {Object.keys(event.updates).length} 项结果</small>
              )}
            </li>
          ))}
        </ol>
        {more && (
          <button
            className="text-button"
            onClick={() =>
              void request(
                `runs/${run.id}/trace?after=${events.at(-1)?.seq || 0}`,
              )
                .then((data) => {
                  setEvents((old) => [...old, ...data.events]);
                  setMore(data.has_more);
                })
                .catch((e) => setError(e.message))
            }
          >
            加载后续事件
          </button>
        )}
      </details>
    </div>
  );
}
