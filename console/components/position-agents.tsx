"use client";
import { useEffect, useRef, useState } from "react";
import NativeWorkbench from "./native-workbench";
import { downloadJSON } from "../lib/workbench-api";
import { useWorkspace, WorkspaceLink } from "../lib/workspace-navigation";

type Position = { id: string; name: string; duties: string };
type Agent = {
  id: string;
  name: string;
  position_id: string;
  current_release: string | null;
};
type Model = {
  id: string;
  name: string;
  model: string;
  base_url: string;
  has_credential: boolean;
};
type Bundle = { id: string; name: string; version: number };
type Release = {
  id: string;
  agent_id: string;
  version: number;
  status: string;
  digest: string;
  manifest: {
    harness: { kind: string; version: string };
    model: Model;
    context: { name: string; version: number };
    tools: string[];
  };
};
type Event = { seq: number; type: string; toolName?: string };
type Run = {
  id: string;
  agent_id: string;
  release_id: string;
  task: string;
  status: string;
  result: string;
  error: string;
  native_session: string;
  events?: Event[];
};
type Overview = {
  positions: Position[];
  agents: Agent[];
  models: Model[];
  releases: Release[];
  runs: Run[];
  harnesses: { installed: boolean }[];
  storage: string;
};
const empty: Overview = {
  positions: [],
  agents: [],
  models: [],
  releases: [],
  runs: [],
  harnesses: [],
  storage: "",
};
const statuses: Record<string, string> = {
  queued: "排队中",
  running: "执行中",
  cancelling: "正在停止",
  cancelled: "已取消",
  completed: "已完成",
  failed: "执行失败",
  interrupted: "已中断，需核对",
  candidate: "待试运行",
  published: "已采用",
};
async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    "/api/platform/" + path,
    body === undefined
      ? { cache: "no-store" }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const result = await response.json();
  if (!response.ok) throw Error(result.error || "操作未完成");
  return result;
}

export default function PositionAgents({
  initialView = "agents",
  hideTabs = false,
}: {
  initialView?: string;
  hideTabs?: boolean;
}) {
  const { route, navigate } = useWorkspace();
  const [editor, setEditor] = useState<"position" | "agent" | null>(null), [runStep, setRunStep] = useState<"configure" | "try" | "review">("configure");
  const [data, setData] = useState<Overview>(empty),
    [bundles, setBundles] = useState<Bundle[]>([]),
    [view, setView] = useState(initialView);
  useEffect(() => setView(initialView), [initialView]);
  useEffect(() => { if (route.area === -1 && route.record) setAgentId(route.record); }, [route.record, route.area]);
  const [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [busy, setBusy] = useState(false),
    [loading, setLoading] = useState(true);
  const [positionName, setPositionName] = useState(""),
    [duties, setDuties] = useState(""),
    [positionId, setPositionId] = useState(""),
    [agentName, setAgentName] = useState(""),
    [agentId, setAgentId] = useState("");
  const [modelId, setModelId] = useState("");
  const [bundleId, setBundleId] = useState(""),
    [releaseId, setReleaseId] = useState("");
  const [task, setTask] = useState(""),
    [runId, setRunId] = useState(""),
    [run, setRun] = useState<Run | null>(null);
  const requestId = useRef<{ key: string; id: string } | null>(null);
  async function refresh() {
    const [next, context] = await Promise.all([
      api<Overview>("overview"),
      fetch("/api/platform/context", { cache: "no-store" }).then(async (r) => {
        const j = await r.json();
        if (!r.ok) throw Error(j.error);
        return j as { bundles: Bundle[] };
      }),
    ]);
    setData(next);
    setBundles(context.bundles);
  }
  useEffect(() => {
    let live = true;
    refresh()
      .catch((e) => {
        if (live) setError(e.message);
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
  }, []);
  useEffect(() => {
    if (!runId) {
      setRun(null);
      return;
    }
    let live = true,
      timer: ReturnType<typeof setTimeout>;
    let cursor = 0;
    const collected = new Map<number, Event>();
    const poll = async () => {
      try {
        const next = await api<Run>("runs/" + runId + "?after=" + cursor);
        if (!live) return;
        for (const event of next.events || []) {
          collected.set(event.seq, event);
          cursor = Math.max(cursor, event.seq);
        }
        setRun({ ...next, events: [...collected.values()] });
        if (
          ["queued", "running", "cancelling"].includes(next.status) ||
          (next.events?.length || 0) === 100
        )
          timer = setTimeout(poll, 800);
        else await refresh();
      } catch (e) {
        if (live) setError((e as Error).message);
      }
    };
    void poll();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [runId]);
  async function act(fn: () => Promise<void>) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await fn();
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const selectedAgent = data.agents.find(
    (a) => a.id === agentId,
  );
  const releases = data.releases.filter(
    (r) => r.agent_id === selectedAgent?.id,
  );
  const selectedRelease = releases.find(
    (r) => r.id === (releaseId || releases[0]?.id),
  );
  function chooseAgent(id: string) {
    setAgentId(id);
    setReleaseId("");
    setRunId("");
    setNotice("");
    setError("");
    setRunStep(data.releases.some((release) => release.agent_id === id) ? "try" : "configure");
  }
  const formSubmit = (fn: () => Promise<void>) => (e: React.FormEvent) => {
    e.preventDefault();
    void act(fn);
  };
  return (
    <section
      className="position-agents"
      aria-label="岗位 Agent 管理"
      data-verify-unit="PositionAgents"
      data-verify-state={
        loading ? "loading" : data.harnesses.length ? "ready" : "unavailable"
      }
    >
      <p className="run-help">
        为岗位配置 Agent，预先选好上下文和模型，试运行后采用对应版本。
      </p>
      {!hideTabs && (
        <div
          className="history-tabs"
          role="tablist"
          aria-label="岗位 Agent 工作区"
        >
          {[
            ["agents", "岗位与 Agent"],
            ["runs", "发布与运行"],
            ["desktop", "桌面 Agent"],
          ].map(([id, label]) => (
            <button
              role="tab"
              key={id}
              aria-selected={view === id}
              onClick={() => {
                setView(id);
                setNotice("");
                setError("");
                if (id === "runs")
                  void refresh().catch((e) => setError(e.message));
              }}
            >
              {label}
            </button>
          ))}
        </div>
      )}
      {error && (
        <p role="alert" className="inline-error">
          {error}
        </p>
      )}
      {notice && (
        <p role="status" className="agent-notice">
          {notice}
        </p>
      )}
      {loading && <p>正在载入岗位配置…</p>}
      {view === "desktop" && <NativeWorkbench moduleIndex={5} tabLabels={["GUI-Agent"]} description="配置桌面执行环境与视觉模型，验证 GUI Agent 的桌面任务。" />}
      {view === "agents" && (
        <div>
          <div className="collection-heading"><div><h2>岗位与 Agent <span className="collection-count">{data.positions.length}</span></h2><p>为每个岗位设置 Agent，再选择它使用的模型和上下文。</p></div>{editor ? <button className="text-button" disabled={busy} onClick={() => setEditor(null)}>返回岗位列表</button> : <div className="agent-actions">{data.positions.length > 0 && <button className="secondary" onClick={() => { setPositionId("");setAgentName("");setEditor("agent"); }}>新增 Agent</button>}<button className="primary" onClick={() => setEditor("position")}>新建岗位</button></div>}</div>
          {editor === "position" &&
          <form
            className="agent-card editor-page"
            onSubmit={formSubmit(async () => {
              const item = await api<Position>("positions", {
                name: positionName,
                duties,
              });
              setPositionId(item.id);
              setPositionName("");
              setDuties("");
              setEditor("agent");
              setNotice("岗位已创建");
            })}
          >
            <h2>创建岗位</h2>
            <label>
              岗位名称
              <input
                value={positionName}
                onChange={(e) => setPositionName(e.target.value)}
                required
                maxLength={120}
              />
            </label>
            <label>
              岗位职责
              <textarea
                value={duties}
                onChange={(e) => setDuties(e.target.value)}
                required
                rows={5}
                maxLength={12000}
              />
            </label>
            <button className="primary" disabled={busy}>
              保存岗位
            </button>
          </form>
          }
          {editor === "agent" &&
          <form
            className="agent-card editor-page"
            onSubmit={formSubmit(async () => {
              const item = await api<Agent>("agents", {
                position_id: positionId,
                name: agentName,
              });
              chooseAgent(item.id);
              setAgentName("");
              setEditor(null);navigate(-1, "runs", item.id);
              setNotice("岗位 Agent 已创建，可继续配置上下文和模型");
            })}
          >
            <h2>设置岗位 Agent</h2>
            <label>
              所属岗位
              <select
                aria-label="所属岗位"
                value={positionId}
                onChange={(e) => setPositionId(e.target.value)}
                required
              >
                <option value="">请选择岗位</option>
                {data.positions.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Agent 名称
              <input
                value={agentName}
                onChange={(e) => setAgentName(e.target.value)}
                required
                maxLength={120}
              />
            </label>
            <button
              className="primary"
              disabled={busy || !positionId}
            >
              保存 Agent
            </button>
            <p className="run-help">
              首版使用 Pi 1.5.2，执行由原生 Harness 完成。
            </p>
          </form>
          }
          {!editor && (!data.positions.length ? <div className="collection-empty"><h3>尚未配置岗位</h3><p>先创建岗位，填写职责，再为它设置 Agent。</p></div> : <div className="resource-table-scroll"><table className="resource-table"><thead><tr><th>岗位</th><th>Agent</th><th>采用状态</th><th>操作</th></tr></thead><tbody>{data.positions.flatMap((position) => {
            const agents = data.agents.filter((agent) => agent.position_id === position.id);
            return agents.length ? agents.map((agent) => <tr key={agent.id}><td>{position.name}<small className="table-secondary">{position.duties.slice(0, 70)}</small></td><td>{agent.name}</td><td>{agent.current_release ? "已有采用版本" : "尚未采用版本"}</td><td><button className="resource-link" onClick={() => { chooseAgent(agent.id);navigate(-1, "runs", agent.id); }}>配置与试运行</button></td></tr>) : [<tr key={position.id}><td>{position.name}</td><td>尚未设置</td><td>—</td><td><button className="resource-link" onClick={() => { setPositionId(position.id);setEditor("agent"); }}>设置 Agent</button></td></tr>];
          })}</tbody></table></div>)}
        </div>
      )}
      {view === "runs" && (
        <>
          <div className="agent-card">
            <label>
              当前岗位 Agent
              <select
                aria-label="当前岗位 Agent"
                value={selectedAgent?.id || ""}
                onChange={(e) => { chooseAgent(e.target.value);navigate(-1, "runs", e.target.value || undefined); }}
              >
                <option value="">请选择 Agent</option>
                {data.agents.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.name}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {selectedAgent && (
            <div className="agent-flow">
              <ol className="flow-steps" aria-label="Agent 发布流程">{([["configure", "1 配置版本"], ["try", "2 试运行"], ["review", "3 结果与采用"]] as const).map(([step, title]) => <li key={step}><button aria-current={runStep === step ? "step" : undefined} disabled={step === "try" ? !selectedRelease : step === "review" ? !runId && !data.runs.some((item) => item.agent_id === selectedAgent.id) : false} onClick={() => { if (step === "review" && !runId) { const recent = data.runs.find((item) => item.agent_id === selectedAgent.id);if (recent) { setRunId(recent.id);setReleaseId(recent.release_id); } }setRunStep(step); }}>{title}</button></li>)}</ol>
              {runStep === "configure" &&
              <form
                className="agent-card editor-page"
                onSubmit={formSubmit(async () => {
                  const item = await api<Release>("releases", {
                    agent_id: selectedAgent.id,
                    model_id: modelId,
                    bundle_id: bundleId,
                  });
                  setReleaseId(item.id);
                  setRunId("");
                  setRunStep("try");
                  setNotice("候选版本已生成，岗位职责、上下文和模型配置已固定");
                })}
              >
                <h2>组合岗位 Agent 版本</h2>
                <p className="run-help">
                  Harness：Pi 1.5.2 · 工具按上下文配置装配
                </p>
                <label>
                  使用的模型服务
                  <select
                    aria-label="使用的模型服务"
                    value={modelId}
                    onChange={(e) => setModelId(e.target.value)}
                    required
                  >
                    <option value="">请选择模型服务</option>
                    {data.models.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name} · {m.model}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  预配的上下文
                  <select
                    aria-label="预配的上下文"
                    value={bundleId}
                    onChange={(e) => setBundleId(e.target.value)}
                    required
                  >
                    <option value="">请选择上下文配置</option>
                    {bundles.map((b) => (
                      <option key={b.id} value={b.id}>
                        {b.name} · v{b.version}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  className="primary"
                  disabled={busy || !data.models.length || !bundles.length}
                >
                  生成候选版本
                </button>
                <p className="run-help">
                  <WorkspaceLink area={-2} view="configuration">维护上下文配置</WorkspaceLink> · <WorkspaceLink area={-3} view="services">管理模型服务</WorkspaceLink>
                </p>
              </form>
              }
              {runStep === "try" && <div className="agent-card editor-page">
                <h2>试运行</h2>
                <label>
                  Agent 版本
                  <select
                    aria-label="Agent 版本"
                    value={selectedRelease?.id || ""}
                    onChange={(e) => {
                      setReleaseId(e.target.value);
                      setRunId("");
                    }}
                  >
                    <option value="">请选择版本</option>
                    {releases.map((r) => (
                      <option key={r.id} value={r.id}>
                        v{r.version} · {statuses[r.status] || r.status}
                      </option>
                    ))}
                  </select>
                </label>
                {selectedRelease && (
                  <p className="run-help">
                    {selectedRelease.manifest.context.name} · v
                    {selectedRelease.manifest.context.version}
                    <br />
                    {selectedRelease.manifest.model.name} ·{" "}
                    {selectedRelease.manifest.model.model}
                    <br />
                    配置摘要：{selectedRelease.digest.slice(0, 12)}
                  </p>
                )}
                <label>
                  岗位任务
                  <textarea
                    value={task}
                    onChange={(e) => setTask(e.target.value)}
                    rows={5}
                    maxLength={12000}
                    placeholder="输入希望这个岗位 Agent 完成的任务"
                  />
                </label>
                <div className="agent-actions">
                  <button
                    className="primary"
                    disabled={
                      busy ||
                      !selectedRelease ||
                      !task.trim() ||
                      !data.harnesses[0]?.installed
                    }
                    onClick={() =>
                      void act(async () => {
                        const key = selectedRelease!.id + task;
                        if (requestId.current?.key !== key)
                          requestId.current = {
                            key,
                            id: crypto.randomUUID().replaceAll("-", ""),
                          };
                        const item = await api<Run>("runs", {
                          agent_id: selectedAgent.id,
                          release_id: selectedRelease!.id,
                          task,
                          request_id: requestId.current.id,
                        });
                        setRunId(item.id);
                        setRunStep("review");
                        requestId.current = null;
                        setNotice("任务已提交，将使用所选版本执行");
                      })
                    }
                  >
                    运行任务
                  </button>
                  {selectedRelease && (
                    <button
                      className="secondary"
                      onClick={() =>
                        downloadJSON(
                          "岗位Agent-v" + selectedRelease.version + ".json",
                          selectedRelease,
                        )
                      }
                    >
                      导出发布包
                    </button>
                  )}
                </div>
                {!data.harnesses[0]?.installed && (
                  <p role="alert" className="inline-error">
                    Pi 运行环境尚未安装，请按项目启动说明安装后再运行。
                  </p>
                )}
              </div>
              }
              {runStep === "review" && <div className="agent-card">
                <h2>运行记录</h2>
                <label>
                  选择运行
                  <select
                    aria-label="选择运行"
                    value={runId}
                    onChange={(e) => {
                      setRunId(e.target.value);
                      const item = data.runs.find(
                        (r) => r.id === e.target.value,
                      );
                      if (item) setReleaseId(item.release_id);
                    }}
                  >
                    <option value="">请选择运行</option>
                    {data.runs
                      .filter((r) => r.agent_id === selectedAgent.id)
                      .map((r) => (
                        <option key={r.id} value={r.id}>
                          {statuses[r.status] || r.status} ·{" "}
                          {r.task.slice(0, 45)}
                        </option>
                      ))}
                  </select>
                </label>
                {run && (
                  <div
                    className="agent-result"
                    data-agent-run-state={run.status}
                  >
                    <p>
                      <strong>{statuses[run.status] || run.status}</strong>
                      {run.native_session && " · 原生会话已保存"}
                    </p>
                    <p className="run-help">
                      {run.events
                        ?.filter((e) => e.type === "tool_execution_start")
                        .map((e) =>
                          e.toolName === "read_enterprise_context"
                            ? "已调用岗位上下文读取工具"
                            : e.toolName,
                        )
                        .join(" · ")}
                    </p>
                    {run.error && (
                      <p role="alert" className="inline-error">
                        {run.error}
                      </p>
                    )}
                    {run.result && <pre>{run.result}</pre>}
                    <div className="agent-actions">
                      {["queued", "running", "cancelling"].includes(
                        run.status,
                      ) && (
                        <button
                          className="secondary"
                          disabled={busy || run.status === "cancelling"}
                          onClick={() =>
                            void act(async () => {
                              await api("runs/" + run.id + "/cancel", {});
                              setRun(await api<Run>("runs/" + run.id));
                            })
                          }
                        >
                          停止本次运行
                        </button>
                      )}
                      {run.status === "completed" && (
                        <>
                          <a
                            className="secondary"
                            href={"/api/platform/runs/" + run.id + "/result"}
                            download
                          >
                            下载任务结果
                          </a>
                          <button
                            className="primary"
                            disabled={
                              busy ||
                              selectedRelease?.id !== run.release_id ||
                              selectedRelease.status === "published"
                            }
                            onClick={() =>
                              void act(async () => {
                                await api(
                                  "releases/" + run.release_id + "/publish",
                                  { run_id: run.id },
                                );
                                setNotice(
                                  "已采用该 Agent 版本，后续可按此配置运行",
                                );
                              })
                            }
                          >
                            核对结果后采用此版本
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                )}
              </div>
              }
            </div>
          )}
        </>
      )}
    </section>
  );
}
