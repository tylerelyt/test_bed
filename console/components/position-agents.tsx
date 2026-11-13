"use client";
import { useEffect, useRef, useState } from "react";
import ContextSupply from "./context-supply";
import ModelSupply from "./model-supply";
import { downloadJSON } from "../lib/workbench-api";

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
  const [data, setData] = useState<Overview>(empty),
    [bundles, setBundles] = useState<Bundle[]>([]),
    [view, setView] = useState(initialView);
  const [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [busy, setBusy] = useState(false),
    [loading, setLoading] = useState(true);
  const [positionName, setPositionName] = useState(""),
    [duties, setDuties] = useState(""),
    [positionId, setPositionId] = useState(""),
    [agentName, setAgentName] = useState(""),
    [agentId, setAgentId] = useState("");
  const [modelName, setModelName] = useState(""),
    [modelId, setModelId] = useState(""),
    [modelCode, setModelCode] = useState(""),
    [baseUrl, setBaseUrl] = useState(""),
    [apiKey, setApiKey] = useState("");
  const [windowSize, setWindowSize] = useState(32000),
    [maxTokens, setMaxTokens] = useState(2048),
    [bundleId, setBundleId] = useState(""),
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
    (a) => a.id === (agentId || data.agents[0]?.id),
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
            ["context", "上下文配置"],
            ["models", "模型服务"],
            ["runs", "发布与运行"],
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
      {view === "context" && (
        <div className="agent-card">
          <ContextSupply />
        </div>
      )}
      {view === "agents" && (
        <div className="agent-grid">
          <form
            className="agent-card"
            onSubmit={formSubmit(async () => {
              const item = await api<Position>("positions", {
                name: positionName,
                duties,
              });
              setPositionId(item.id);
              setPositionName("");
              setDuties("");
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
          <form
            className="agent-card"
            onSubmit={formSubmit(async () => {
              const item = await api<Agent>("agents", {
                position_id: positionId || data.positions[0]?.id,
                name: agentName,
              });
              chooseAgent(item.id);
              setAgentName("");
              setNotice("岗位 Agent 已创建，可继续配置上下文和模型");
            })}
          >
            <h2>设置岗位 Agent</h2>
            <label>
              所属岗位
              <select
                aria-label="所属岗位"
                value={positionId || data.positions[0]?.id || ""}
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
              disabled={busy || !data.positions.length}
            >
              保存 Agent
            </button>
            <p className="run-help">
              首版使用 Pi 1.5.2，执行由原生 Harness 完成。
            </p>
          </form>
          <div className="agent-card agent-wide">
            <h2>已设置的 Agent</h2>
            {!data.agents.length && (
              <p className="run-help">先创建一个岗位，再为它设置 Agent。</p>
            )}
            {data.agents.map((a) => (
              <div className="agent-row" key={a.id}>
                <div>
                  <strong>{a.name}</strong>
                  <p>
                    {data.positions.find((p) => p.id === a.position_id)?.name} ·{" "}
                    {a.current_release ? "已有采用版本" : "尚未采用版本"}
                  </p>
                </div>
                <button
                  className="secondary"
                  onClick={() => {
                    chooseAgent(a.id);
                    setView("runs");
                  }}
                >
                  配置与试运行
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
      {view === "models" && <ModelSupply />}
      {view === "models" && (
        <div className="agent-grid">
          <form
            className="agent-card"
            onSubmit={formSubmit(async () => {
              const item = await api<Model>("models", {
                name: modelName,
                model: modelCode,
                base_url: baseUrl,
                api_key: apiKey,
                context_window: windowSize,
                max_tokens: maxTokens,
              });
              setModelId(item.id);
              setApiKey("");
              setNotice("模型服务配置已保存；完成真实试运行后再核对可用性");
            })}
          >
            <h2>登记模型服务</h2>
            <p className="run-help">
              接入 OpenAI Chat Completions 兼容服务，例如已有的 vLLM 或 Ollama
              服务。
            </p>
            <label>
              服务名称
              <input
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                required
                maxLength={120}
              />
            </label>
            <label>
              服务地址
              <input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                required
                type="url"
                placeholder="http://127.0.0.1:8000/v1"
              />
            </label>
            <label>
              模型标识
              <input
                value={modelCode}
                onChange={(e) => setModelCode(e.target.value)}
                required
                placeholder="服务实际提供的模型名称"
              />
            </label>
            <label>
              服务密钥（可选）
              <input
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                type="password"
                autoComplete="off"
              />
            </label>
            <div className="agent-grid">
              <label>
                上下文长度
                <input
                  type="number"
                  min={1024}
                  max={2000000}
                  value={windowSize}
                  onChange={(e) => setWindowSize(Number(e.target.value))}
                />
              </label>
              <label>
                输出上限
                <input
                  type="number"
                  min={1}
                  max={32000}
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                />
              </label>
            </div>
            <button className="primary" disabled={busy}>
              保存模型服务
            </button>
          </form>
          <div className="agent-card">
            <h2>已登记的模型服务</h2>
            {data.models.length === 0 && (
              <p className="run-help">
                登记已有的模型服务后，可为岗位 Agent 选择。
              </p>
            )}
            {data.models.map((m) => (
              <div className="agent-row" key={m.id}>
                <div>
                  <strong>{m.name}</strong>
                  <p>{m.model}</p>
                  <small>{m.base_url}</small>
                  <p>
                    {m.has_credential ? "密钥已保存，不回显" : "无需服务密钥"}
                  </p>
                </div>
              </div>
            ))}
          </div>
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
                onChange={(e) => chooseAgent(e.target.value)}
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
            <div className="agent-grid">
              <form
                className="agent-card"
                onSubmit={formSubmit(async () => {
                  const item = await api<Release>("releases", {
                    agent_id: selectedAgent.id,
                    model_id: modelId || data.models[0]?.id,
                    bundle_id: bundleId || bundles[0]?.id,
                  });
                  setReleaseId(item.id);
                  setRunId("");
                  setNotice("候选版本已生成，岗位职责、上下文和模型配置已固定");
                })}
              >
                <h2>组合岗位 Agent 版本</h2>
                <p className="run-help">
                  Harness：Pi 1.5.2 · 工具：只读预配上下文
                </p>
                <label>
                  使用的模型服务
                  <select
                    aria-label="使用的模型服务"
                    value={modelId || data.models[0]?.id || ""}
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
                    value={bundleId || bundles[0]?.id || ""}
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
                  资料与规则在“上下文配置”维护，生成版本时固定当时的配置。
                </p>
              </form>
              <div className="agent-card">
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
              <div className="agent-card agent-wide">
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
            </div>
          )}
        </>
      )}
    </section>
  );
}
