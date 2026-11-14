"use client";
import { useEffect, useRef, useState } from "react";
import NativeWorkbench from "./native-workbench";
import { downloadJSON } from "../lib/workbench-api";
import {
  useWorkspace,
  WorkspaceLink,
  useDetailName,
} from "../lib/workspace-navigation";

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
}: {
  initialView?: string;
  hideTabs?: boolean;
}) {
  const { route, navigate } = useWorkspace();
  const agentId = initialView === "desktop" ? "" : route.record || "";
  const tab = ["configuration", "new-version", "try", "history"].includes(
      route.tab || "",
    )
      ? route.tab!
      : "configuration",
    runId = route.run || "";
  const [editor, setEditor] = useState<"position" | "agent" | null>(null);
  const [data, setData] = useState<Overview>(empty),
    [bundles, setBundles] = useState<Bundle[]>([]);
  const [error, setError] = useState(""),
    [notice, setNoticeState] = useState({ text: "", agent: "" }),
    [busy, setBusy] = useState(false),
    [loading, setLoading] = useState(true);
  const [positionName, setPositionName] = useState(""),
    [duties, setDuties] = useState(""),
    [positionId, setPositionId] = useState(""),
    [agentName, setAgentName] = useState("");
  const [modelId, setModelId] = useState(""),
    [bundleId, setBundleId] = useState("");
  const [task, setTask] = useState(""),
    [run, setRun] = useState<Run | null>(null);
  const requestId = useRef<{ key: string; id: string } | null>(null);
  function setNotice(text: string, agent = agentId) {
    setNoticeState({ text, agent });
  }
  useEffect(() => {
    let saved: { modelId?: string; bundleId?: string } = {};
    try {
      saved = JSON.parse(
        sessionStorage.getItem("agent-configuration-" + agentId) || "{}",
      );
    } catch {}
    setModelId(saved.modelId || "");
    setBundleId(saved.bundleId || "");
    setTask("");
    setError("");
  }, [agentId]);
  function chooseDependency(field: "modelId" | "bundleId", id: string) {
    const draft = { modelId, bundleId, [field]: id };
    setModelId(draft.modelId);
    setBundleId(draft.bundleId);
    sessionStorage.setItem(
      "agent-configuration-" + agentId,
      JSON.stringify(draft),
    );
  }
  async function refresh() {
    const [next, context] = await Promise.all([
      api<Overview>("overview"),
      api<{ bundles: Bundle[] }>("context"),
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
    setRun(null);
    if (!runId || !agentId) return;
    let live = true,
      timer: ReturnType<typeof setTimeout>;
    let cursor = 0;
    const collected = new Map<number, Event>();
    const poll = async () => {
      try {
        const next = await api<Run>("runs/" + runId + "?after=" + cursor);
        if (!live) return;
        if (next.agent_id !== agentId)
          throw Error("该运行记录不属于当前 Agent。");
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
  }, [runId, agentId]);
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
  const selectedAgent = data.agents.find((a) => a.id === agentId);
  const position = data.positions.find(
    (item) => item.id === selectedAgent?.position_id,
  );
  const releases = data.releases.filter((r) => r.agent_id === agentId);
  const runs = data.runs.filter((item) => item.agent_id === agentId);
  const selectedRelease = releases.find(
    (r) =>
      r.id ===
      (route.release ||
        runs.find((item) => item.id === runId)?.release_id ||
        selectedAgent?.current_release ||
        releases[0]?.id),
  );
  const configure = tab === "configuration" || tab === "new-version";
  useDetailName(selectedAgent?.name);
  function detail(
    nextTab: string,
    release = selectedRelease?.id,
    run?: string,
  ) {
    navigate(-1, "agents", agentId, { tab: nextTab, release, run });
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
      {error && (
        <p role="alert" className="inline-error">
          {error}
        </p>
      )}
      {notice.text && notice.agent === agentId && (
        <p role="status" className="agent-notice">
          {notice.text}
        </p>
      )}
      {loading && <p>正在载入岗位配置…</p>}
      {initialView === "desktop" ? (
        <NativeWorkbench
          moduleIndex={5}
          tabLabels={["GUI-Agent"]}
          description="配置桌面执行环境与视觉模型，验证 GUI Agent 的桌面任务。"
        />
      ) : !agentId ? (
        <>
          <div className="collection-heading">
            <div>
              <h1>
                {editor === "position"
                  ? "新建岗位"
                  : editor === "agent"
                    ? "新增 Agent"
                    : "岗位与 Agent"}
              </h1>
              <p>按岗位管理 Agent，进入详情配置模型、上下文和运行版本。</p>
            </div>
            {editor ? (
              <button
                className="text-button"
                disabled={busy}
                onClick={() => setEditor(null)}
              >
                返回岗位列表
              </button>
            ) : (
              <div className="agent-actions">
                {data.positions.length > 0 && (
                  <button
                    className="secondary"
                    onClick={() => {
                      setPositionId("");
                      setAgentName("");
                      setEditor("agent");
                    }}
                  >
                    新增 Agent
                  </button>
                )}
                <button
                  className="primary"
                  onClick={() => setEditor("position")}
                >
                  新建岗位
                </button>
              </div>
            )}
          </div>
          {editor === "position" && (
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
          )}
          {editor === "agent" && (
            <form
              className="agent-card editor-page"
              onSubmit={formSubmit(async () => {
                const item = await api<Agent>("agents", {
                  position_id: positionId,
                  name: agentName,
                });
                setAgentName("");
                setEditor(null);
                navigate(-1, "agents", item.id);
                setNotice("岗位 Agent 已创建，可继续配置上下文和模型", item.id);
              })}
            >
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
              <button className="primary" disabled={busy || !positionId}>
                保存 Agent
              </button>
            </form>
          )}
          {!editor &&
            !loading &&
            (!data.positions.length ? (
              <div className="collection-empty">
                <h3>尚未配置岗位</h3>
                <p>创建岗位并填写职责，再为它设置 Agent。</p>
              </div>
            ) : (
              <div className="resource-table-scroll">
                <table className="resource-table">
                  <thead>
                    <tr>
                      <th>岗位</th>
                      <th>Agent</th>
                      <th>采用状态</th>
                      <th>操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.positions.flatMap((p) => {
                      const agents = data.agents.filter(
                        (agent) => agent.position_id === p.id,
                      );
                      return agents.length
                        ? agents.map((agent) => (
                            <tr key={agent.id}>
                              <td>
                                {p.name}
                                <small className="table-secondary">
                                  {p.duties.slice(0, 70)}
                                </small>
                              </td>
                              <td>
                                <WorkspaceLink
                                  className="resource-link"
                                  area={-1}
                                  view="agents"
                                  record={agent.id}
                                >
                                  {agent.name}
                                </WorkspaceLink>
                              </td>
                              <td>
                                {agent.current_release
                                  ? "已有采用版本"
                                  : "尚未采用版本"}
                              </td>
                              <td>
                                <WorkspaceLink
                                  className="resource-link"
                                  area={-1}
                                  view="agents"
                                  record={agent.id}
                                >
                                  查看 Agent
                                </WorkspaceLink>
                              </td>
                            </tr>
                          ))
                        : [
                            <tr key={p.id}>
                              <td>{p.name}</td>
                              <td>尚未设置</td>
                              <td>—</td>
                              <td>
                                <button
                                  className="resource-link"
                                  onClick={() => {
                                    setPositionId(p.id);
                                    setEditor("agent");
                                  }}
                                >
                                  设置 Agent
                                </button>
                              </td>
                            </tr>,
                          ];
                    })}
                  </tbody>
                </table>
              </div>
            ))}
        </>
      ) : selectedAgent ? (
        <>
          <div className="collection-heading">
            <div>
              <h1>{selectedAgent.name}</h1>
              <p>
                {position?.name} ·{" "}
                {selectedAgent.current_release
                  ? "已有采用版本"
                  : "尚未采用版本"}
              </p>
            </div>
            <WorkspaceLink area={-1} view="agents" className="text-button">
              返回岗位与 Agent
            </WorkspaceLink>
          </div>
          <p className="agent-duty">{position?.duties}</p>
          <div className="object-tabs" role="tablist" aria-label="Agent 详情">
            {[
              ["configuration", "配置与版本"],
              ["try", "试运行"],
              ["history", "运行记录"],
            ].map(([id, label]) => (
              <WorkspaceLink
                key={id}
                area={-1}
                view="agents"
                record={agentId}
                options={{ tab: id, release: selectedRelease?.id }}
                role="tab"
                aria-selected={id === "configuration" ? configure : tab === id}
              >
                {label}
              </WorkspaceLink>
            ))}
          </div>
          {configure && (
            <>
              {tab !== "new-version" && releases.length > 0 ? (
                <>
                  <div className="collection-heading">
                    <h2>
                      配置版本{" "}
                      <span className="collection-count">
                        {releases.length}
                      </span>
                    </h2>
                    <button
                      className="primary"
                      onClick={() => detail("new-version")}
                    >
                      新建配置版本
                    </button>
                  </div>
                  <div className="resource-table-scroll">
                    <table className="resource-table">
                      <thead>
                        <tr>
                          <th>版本</th>
                          <th>模型</th>
                          <th>上下文配置</th>
                          <th>状态</th>
                          <th>操作</th>
                        </tr>
                      </thead>
                      <tbody>
                        {releases.map((release) => (
                          <tr key={release.id}>
                            <td>v{release.version}</td>
                            <td>
                              {release.manifest.model.name}
                              <small className="table-secondary">
                                {release.manifest.harness.kind} ·{" "}
                                {release.manifest.harness.version}
                              </small>
                            </td>
                            <td>
                              {release.manifest.context.name} · v
                              {release.manifest.context.version}
                            </td>
                            <td>
                              {selectedAgent.current_release === release.id
                                ? "当前采用"
                                : statuses[release.status] || release.status}
                            </td>
                            <td>
                              <WorkspaceLink
                                area={-1}
                                view="agents"
                                record={agentId}
                                options={{ tab: "try", release: release.id }}
                                className="resource-link"
                              >
                                试运行此版本
                              </WorkspaceLink>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <form
                  className="agent-card editor-page"
                  onSubmit={formSubmit(async () => {
                    const item = await api<Release>("releases", {
                      agent_id: agentId,
                      model_id: modelId,
                      bundle_id: bundleId,
                    });
                    detail("try", item.id);
                    setNotice(
                      "候选版本已生成，岗位职责、上下文和模型配置已固定",
                    );
                  })}
                >
                  <h2>{releases.length ? "新建配置版本" : "配置第一个版本"}</h2>
                  <p className="run-help">
                    Harness：Pi
                    1.5.2。选择模型与上下文，保存为固定版本后试运行。
                  </p>
                  <label>
                    使用的模型服务
                    <select
                      aria-label="使用的模型服务"
                      value={modelId}
                      onChange={(e) =>
                        chooseDependency("modelId", e.target.value)
                      }
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
                  <WorkspaceLink
                    area={-3}
                    view="services"
                    related={"返回「" + selectedAgent.name + "」配置"}
                    className="resource-link"
                  >
                    接入或查看模型服务
                  </WorkspaceLink>
                  <label>
                    预配的上下文
                    <select
                      aria-label="预配的上下文"
                      value={bundleId}
                      onChange={(e) =>
                        chooseDependency("bundleId", e.target.value)
                      }
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
                  <WorkspaceLink
                    area={-2}
                    view="configuration"
                    related={"返回「" + selectedAgent.name + "」配置"}
                    className="resource-link"
                  >
                    创建或查看上下文配置
                  </WorkspaceLink>
                  <div className="editor-actions">
                    {releases.length > 0 && (
                      <button
                        type="button"
                        className="secondary"
                        disabled={busy}
                        onClick={() => detail("configuration")}
                      >
                        取消
                      </button>
                    )}
                    <button
                      className="primary"
                      disabled={busy || !modelId || !bundleId}
                    >
                      生成候选版本
                    </button>
                  </div>
                </form>
              )}
            </>
          )}
          {tab === "try" &&
            (selectedRelease ? (
              <div className="agent-card editor-page">
                <h2>试运行岗位任务</h2>
                <label>
                  Agent 版本
                  <select
                    aria-label="Agent 版本"
                    value={selectedRelease.id}
                    onChange={(e) => detail("try", e.target.value)}
                  >
                    {releases.map((r) => (
                      <option key={r.id} value={r.id}>
                        v{r.version} · {statuses[r.status] || r.status}
                      </option>
                    ))}
                  </select>
                </label>
                <p className="run-help">
                  {selectedRelease.manifest.context.name} · v
                  {selectedRelease.manifest.context.version}
                  <br />
                  {selectedRelease.manifest.model.name} ·{" "}
                  {selectedRelease.manifest.model.model}
                  <br />
                  配置摘要：{selectedRelease.digest.slice(0, 12)}
                </p>
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
                      busy || !task.trim() || !data.harnesses[0]?.installed
                    }
                    onClick={() =>
                      void act(async () => {
                        const key = selectedRelease.id + task;
                        if (requestId.current?.key !== key)
                          requestId.current = {
                            key,
                            id: crypto.randomUUID().replaceAll("-", ""),
                          };
                        const item = await api<Run>("runs", {
                          agent_id: agentId,
                          release_id: selectedRelease.id,
                          task,
                          request_id: requestId.current.id,
                        });
                        detail("history", selectedRelease.id, item.id);
                        requestId.current = null;
                        setNotice("任务已提交，将使用所选版本执行");
                      })
                    }
                  >
                    运行任务
                  </button>
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
                </div>
                {!data.harnesses[0]?.installed && (
                  <p role="alert" className="inline-error">
                    Pi 运行环境尚未安装，请按项目启动说明安装后再运行。
                  </p>
                )}
              </div>
            ) : (
              <div className="collection-empty">
                <h3>尚无可运行版本</h3>
                <p>先为这个 Agent 选择模型和上下文并保存版本。</p>
                <button
                  className="secondary"
                  onClick={() => detail("configuration")}
                >
                  前往配置
                </button>
              </div>
            ))}
          {tab === "history" &&
            (runId ? (
              <div className="agent-card">
                <div className="collection-heading">
                  <h2>
                    运行结果
                    {selectedRelease ? " · v" + selectedRelease.version : ""}
                  </h2>
                  <button
                    className="text-button"
                    onClick={() => detail("history")}
                  >
                    返回运行记录
                  </button>
                </div>
                {run ? (
                  <div
                    className="agent-result"
                    data-agent-run-state={run.status}
                  >
                    <p>
                      <strong>{statuses[run.status] || run.status}</strong>
                      {run.native_session && " · 原生会话已保存"}
                    </p>
                    <p>{run.task}</p>
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
                ) : (
                  <p>正在载入运行结果…</p>
                )}
              </div>
            ) : !runs.length ? (
              <div className="collection-empty">
                <h3>尚无运行记录</h3>
                <p>试运行这个 Agent 后，可在这里查看结果和采用版本。</p>
                <button className="secondary" onClick={() => detail("try")}>
                  前往试运行
                </button>
              </div>
            ) : (
              <div className="resource-table-scroll">
                <table className="resource-table">
                  <thead>
                    <tr>
                      <th>岗位任务</th>
                      <th>版本</th>
                      <th>状态</th>
                      <th>操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((item) => (
                      <tr key={item.id}>
                        <td>{item.task}</td>
                        <td>
                          v
                          {releases.find(
                            (release) => release.id === item.release_id,
                          )?.version || "—"}
                        </td>
                        <td>{statuses[item.status] || item.status}</td>
                        <td>
                          <WorkspaceLink
                            className="resource-link"
                            area={-1}
                            view="agents"
                            record={agentId}
                            options={{
                              tab: "history",
                              release: item.release_id,
                              run: item.id,
                            }}
                          >
                            查看结果
                          </WorkspaceLink>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
        </>
      ) : (
        !loading && (
          <div className="collection-empty">
            <h1>未找到这个 Agent</h1>
            <p>请返回列表核对岗位与 Agent。</p>
            <WorkspaceLink className="secondary" area={-1} view="agents">
              返回岗位与 Agent
            </WorkspaceLink>
          </div>
        )
      )}
    </section>
  );
}
