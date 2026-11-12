"use client";
import { useEffect, useRef, useState } from "react";
import { displayLabel } from "../lib/presentation";
import {
  SavedTask,
  NativeRun,
  Rule,
  ContextBinding,
  downloadJSON,
  request,
  runStatus,
  verdictStatus,
} from "../lib/workbench-api";
import TaskRules from "./task-rules";
import RunEvidence from "./run-evidence";
import EvaluationSuites from "./evaluation-suites";
import ContextSupply from "./context-supply";
import ExperienceConsolidation from "./experience-consolidation";
import TaskContext from "./task-context";
export type { SavedTask } from "../lib/workbench-api";
export { request } from "../lib/workbench-api";
type Note = {
  id: string;
  title: string;
  body: string;
  source_run: string;
  version?: number;
  status?: string;
  source_runs?: string[];
};
type Comparison = {
  duration_seconds: (number | null)[];
  inputs: { component: string; before: unknown; after: unknown }[];
  outputs: { component: string; before: unknown; after: unknown }[];
  notice: string;
};
type Catalog = {
  runs: NativeRun[];
  tasks: SavedTask[];
  notes: Note[];
  suites: Parameters<typeof EvaluationSuites>[0]["suites"];
  batches: Parameters<typeof EvaluationSuites>[0]["batches"];
  schedules: Parameters<typeof EvaluationSuites>[0]["schedules"];
};
type View = "runs" | "tasks" | "evaluations" | "notes" | "context";
const show = (value: unknown) => JSON.stringify(value ?? null, null, 2);
const empty: Catalog = {
  runs: [],
  tasks: [],
  notes: [],
  suites: [],
  batches: [],
  schedules: [],
};

export default function RunHistory({
  module,
  revision,
  labels,
  actions,
  onRestore,
  onReconnect,
  activeTask,onExitPreset,onPresetPending,
}: {
  module: string;
  revision: number;
  activeTask:SavedTask|null;onExitPreset:()=>void;onPresetPending:(pending:boolean)=>void;
  labels: Record<string, string>;
  actions: Record<string, string>;
  onRestore: (
    task: SavedTask,
    values: Record<string, unknown>,
    missing: string[],
  ) => void;
  onReconnect: (run: NativeRun) => void;
}) {
  const [open, setOpen] = useState(false),
    [view, setView] = useState<View>("runs"),
    [data, setData] = useState<Catalog>(empty);
  const [selected, setSelected] = useState(""),
    [run, setRun] = useState<NativeRun | null>(null),
    [taskId, setTaskId] = useState(""),
    [filter, setFilter] = useState(""),
    [comparison, setComparison] = useState<Comparison | null>(null);
  const [name, setName] = useState(""),
    [expectation, setExpectation] = useState(""),
    [newRules, setNewRules] = useState<Rule[]>([]),
    [editRules, setEditRules] = useState<Rule[]>([]),
    [reviewRequired, setReviewRequired] = useState(false);
  const [title, setTitle] = useState(""),
    [body, setBody] = useState(""),
    [acceptance, setAcceptance] = useState(true),
    [reason, setReason] = useState("");
  const [message, setMessage] = useState(""),
    [error, setError] = useState(""),
    [pending, setPending] = useState(false);
  const [secretEnv, setSecretEnv] = useState<Record<string, string>>({});
  const [caseKind, setCaseKind] = useState("unspecified");
  const [contextBinding, setContextBinding] = useState<ContextBinding | null>(
    null,
  );
  const currentTask = data.tasks.find((task) => task.id === taskId);
  const selectedRef = useRef("");
  async function refresh() {
    const catalog = await request("runs?module=" + encodeURIComponent(module));
    setData(catalog);
  }
  async function act(work: () => Promise<void>) {
    setPending(true);
    setError("");
    setMessage("");
    try {
      await work();
      await refresh();
      const id = selectedRef.current;
      if (id) {
        const latest = await request("runs/" + id);
        if (selectedRef.current === id) setRun(latest);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  useEffect(() => {
    selectedRef.current = "";
    setData(empty);
    setSelected("");
    setRun(null);
    setTaskId("");
    setComparison(null);
    setMessage("");
    setError("");
    setView("runs");
  }, [module]);
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const load = () =>
      request("runs?module=" + encodeURIComponent(module))
        .then((value) => {
          if (!cancelled) setData(value);
        })
        .catch((e) => {
          if (!cancelled) setError(e.message);
        });
    void load();
    const timer = setInterval(() => void load(), 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [open, module, revision]);
  useEffect(() => {
    if (!open || !selected) return;
    let cancelled = false;
    const load = () =>
      request("runs/" + selected)
        .then((value) => {
          if (!cancelled) setRun(value);
        })
        .catch((e) => {
          if (!cancelled) setError(e.message);
        });
    void load();
    const timer = setInterval(() => void load(), 3000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [open, selected, revision]);
  useEffect(() => {
    setCaseKind(currentTask?.case_kind || "unspecified");
    setContextBinding(currentTask?.context_binding || null);
    setSecretEnv(currentTask?.secret_env || {});
    setEditRules(currentTask?.rules || []);
    setReviewRequired(currentTask?.requires_review || false);
  }, [currentTask?.id, currentTask?.version]);
  useEffect(()=>{if(activeTask?.module===module)setTaskId(activeTask.id)},[activeTask?.id,activeTask?.version,module]);
  async function selectRun(id: string) {
    selectedRef.current = id;
    setSelected(id);
    setComparison(null);
    setView("runs");
    setRun(id ? await request("runs/" + id) : null);
  }
  async function restoreSelectedTask() {
    onPresetPending(true);
    try {
      await act(async () => {
        const result = await request("tasks", { action: "restore", task_id: taskId });
        onRestore(result.task, result.values, result.protected || result.missing);
        setMessage(result.missing.length
          ? "参数已恢复；需补填密钥或文件。核对业务状态后点击原操作按钮。"
          : "参数已恢复，核对后点击原操作按钮执行。");
      });
    } finally {
      onPresetPending(false);
    }
  }
  function changes(items: Comparison["inputs"]) {
    return items.length ? (
      <div className="run-diff">
        {items.map((item) => (
          <div key={item.component}>
            <strong>{labels[item.component] || "结果"}</strong>
            <div>
              <pre aria-label="基线值">{show(item.before)}</pre>
              <pre aria-label="当前值">{show(item.after)}</pre>
            </div>
          </div>
        ))}
      </div>
    ) : (
      <p className="run-help">无差异。</p>
    );
  }
  async function exportEvidence() {
    if (!run) return;
    let after = 0,
      hasMore = true;
    const events: unknown[] = [];
    while (hasMore && events.length < 10000) {
      const page = await request(`runs/${run.id}/trace?after=${after}`);
      events.push(...page.events);
      hasMore = page.has_more;
      after = page.events.at(-1)?.seq || after;
    }
    downloadJSON("运行证据-" + run.id + ".json", {
      format_version: 1,
      run,
      comparison,
      events,
      truncated: hasMore,
    });
  }
  const views: [View, string][] = [
    ["runs", "运行证据"],
    ["tasks", "固定任务"],
    ["evaluations", "批量评测"],
    ["notes", "经验"],
  ];
  if (module.includes("上下文")) views.push(["context", "供应配置"]);
  const runs = data.runs.filter((record) =>
    (displayLabel(record.action) + " " + runStatus[record.status]).includes(
      filter,
    ),
  );
  return (
    <details
      className="run-history"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
      data-verify-unit="RunHistory"
      data-verify-open={open}
      data-verify-view={view}
    >
      <summary>
        任务与运行记录 <span>上下文配置 · 执行证据 · 任务验证</span>
      </summary>
      {open && (
        <div className="run-history-body">
          <div
            className="history-tabs"
            role="tablist"
            aria-label="任务与运行管理"
          >
            {views.map(([id, label]) => (
              <button
                role="tab"
                id={`history-${id}`}
                key={id}
                aria-selected={view === id}
                onClick={() => setView(id)}
              >
                {label}
                {id === "notes" && data.notes.length > 0 && (
                  <small>{data.notes.length}</small>
                )}
              </button>
            ))}
          </div>
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
          <div role="tabpanel" aria-labelledby={`history-${view}`}>
            {view === "context" ? (
              <ContextSupply />
            ) : view === "evaluations" ? (
              <EvaluationSuites
                module={module}
                suites={data.suites}
                batches={data.batches}
                schedules={data.schedules}
                labels={labels}
                onRefresh={refresh}
                onRun={(id) => void act(() => selectRun(id))}
              />
            ) : (
              <>
                {view !== "notes" && (
                  <div className="native-field">
                    <label htmlFor="saved-task">固定任务</label>
                    <select
                      id="saved-task"
                      value={taskId}
                      onChange={(event) => {
                        onExitPreset();setTaskId(event.target.value);
                        setComparison(null);
                      }}
                    >
                      <option value="">选择本模块的固定任务</option>
                      {data.tasks.map((task) => (
                        <option key={task.id} value={task.id}>
                          {task.name} · v{task.version || 1}
                        </option>
                      ))}
                    </select>
                    {currentTask && (
                      <>
                        <p className="run-help">
                          验收要点：{currentTask.expectation}
                        </p>
                        <button
                          className="secondary"
                          disabled={pending}
                          onClick={() => void restoreSelectedTask()}
                        >
                          恢复任务参数
                        </button>
                      </>
                    )}
                  </div>
                )}
                {view === "tasks" && (
                  <>
                    {currentTask ? (
                      <div className="run-detail">
                        <h4>{currentTask.name} · 规则与执行方式</h4>
                        <div className="native-field">
                          <label htmlFor="case-kind">题目类型标注</label>
                          <select
                            id="case-kind"
                            value={caseKind}
                            onChange={(event) =>
                              setCaseKind(event.target.value)
                            }
                          >
                            <option value="unspecified">未标注</option>
                            <option value="positive">正常任务</option>
                            <option value="negative">反例</option>
                            <option value="edge">边界任务</option>
                          </select>
                        </div>
                        <TaskRules
                          key={`${currentTask.id}-${currentTask.version}`}
                          rules={editRules}
                          onChange={setEditRules}
                          outputs={(currentTask.output_ids || []).map(String)}
                          labels={labels}
                        />
                        <TaskContext
                          binding={contextBinding}
                          onChange={setContextBinding}
                          inputs={currentTask.context_input_ids || []}
                          labels={labels}
                        />
                        <details>
                          <summary>服务端密钥引用（可选）</summary>
                          <p className="run-help">
                            只填写环境变量名，执行时由服务端解析。界面不会读取密钥值。
                          </p>
                          {Object.entries(currentTask.inputs)
                            .filter(([, value]) => value === "【已隐藏】")
                            .map(([id]) => (
                              <div className="native-field" key={id}>
                                <label htmlFor={"secret-ref-" + id}>
                                  {labels[id] || "密钥输入"}对应的环境变量
                                </label>
                                <input
                                  id={"secret-ref-" + id}
                                  value={secretEnv[id] || ""}
                                  placeholder="DASHSCOPE_API_KEY"
                                  autoComplete="off"
                                  onChange={(event) =>
                                    setSecretEnv((old) => {
                                      const next = { ...old };
                                      if (event.target.value.trim())
                                        next[id] = event.target.value.trim();
                                      else delete next[id];
                                      return next;
                                    })
                                  }
                                />
                              </div>
                            ))}
                        </details>
                        <label className="native-check">
                          <input
                            type="checkbox"
                            checked={reviewRequired}
                            onChange={(event) =>
                              setReviewRequired(event.target.checked)
                            }
                          />
                          执行前需要人工放行
                        </label>
                        <button
                          className="secondary"
                          disabled={pending}
                          onClick={() =>
                            void act(async () => {
                              await request("tasks", {
                                action: "rules",
                                task_id: taskId,
                                rules: editRules,
                                requires_review: reviewRequired,
                                expected_version: currentTask.version || 1,
                                secret_env: secretEnv,
                                context_binding: contextBinding || {},
                                case_kind: caseKind,
                              });
                              onExitPreset();setMessage(
                                "任务规则已保存为新版本。相关任务集需要重新建立，基线需要重新核对。",
                              );
                            })
                          }
                        >
                          保存任务新版本
                        </button>
                        <p className="run-help">
                          密钥由服务端提供，临时文件和内部会话状态不随预设恢复。规则与人工验收分别记录，原业务操作流程保持现有方式。
                        </p>
                      </div>
                    ) : (
                      <p className="run-help">
                        从“运行证据”中选择一次实际运行并保存固定任务，再配置规则、任务集和周期验证。
                      </p>
                    )}
                  </>
                )}
                {view === "runs" && (
                  <>
                    <div className="run-grid">
                      <div className="native-field">
                        <label htmlFor="run-filter">筛选运行</label>
                        <input
                          id="run-filter"
                          placeholder="按操作或状态筛选"
                          value={filter}
                          onChange={(event) => setFilter(event.target.value)}
                        />
                      </div>
                      <div className="native-field">
                        <label htmlFor="recorded-run">
                          运行记录（最近 100 条）
                        </label>
                        <select
                          id="recorded-run"
                          value={selected}
                          onChange={(event) =>
                            void act(() => selectRun(event.target.value))
                          }
                        >
                          <option value="">选择一次运行</option>
                          {selected &&
                            run &&
                            !runs.some((record) => record.id === selected) && (
                              <option value={selected}>
                                任务集中的运行 · {displayLabel(run.action)}
                              </option>
                            )}
                          {runs.map((record) => (
                            <option key={record.id} value={record.id}>
                              {new Date(
                                record.started_at * 1000,
                              ).toLocaleString("zh-CN")}{" "}
                              · {displayLabel(record.action)} ·{" "}
                              {runStatus[record.status]}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                    {run && (
                      <div
                        className="run-detail"
                        data-verify-state={run.status}
                      >
                        <div className="run-meta">
                          <strong>
                            {displayLabel(run.action)}{" "}
                            <span
                              className={`status-badge status-${run.status}`}
                            >
                              {runStatus[run.status]}
                            </span>
                          </strong>
                          <span>脱敏参数版本 {run.config_version}</span>
                        </div>
                        {run.error && (
                          <p role="alert" className="inline-error">
                            {run.error}
                          </p>
                        )}
                        <div className="run-actions">
                          {["queued", "running", "awaiting_review"].includes(
                            run.status,
                          ) && (
                            <button
                              className="secondary"
                              onClick={() => onReconnect(run)}
                            >
                              连接运行输出
                            </button>
                          )}
                          {run.status === "awaiting_review" && (
                            <>
                              <button
                                className="primary"
                                onClick={() =>
                                  void act(async () => {
                                    await request("runs/control", {
                                      action: "approve",
                                      run_id: run.id,
                                    });
                                    setMessage("已放行，操作进入执行队列。");
                                  })
                                }
                              >
                                放行本次操作
                              </button>
                              <button
                                className="text-button"
                                onClick={() =>
                                  void act(async () => {
                                    await request("runs/control", {
                                      action: "reject",
                                      run_id: run.id,
                                    });
                                    setMessage(
                                      "本次操作未放行，未执行业务函数。",
                                    );
                                  })
                                }
                              >
                                不放行
                              </button>
                            </>
                          )}
                          {["queued", "running"].includes(run.status) && (
                            <button
                              className="text-button"
                              onClick={() =>
                                void act(async () => {
                                  await request("runs/control", {
                                    action: "cancel",
                                    run_id: run.id,
                                  });
                                  setMessage(
                                    "已提交停止请求。正在执行的外部操作需要原模块停止按钮和实际终态核对。",
                                  );
                                })
                              }
                            >
                              请求停止
                            </button>
                          )}
                        </div>
                        <RunEvidence run={run} actions={actions} />
                        <details>
                          <summary>查看参数与结果</summary>
                          <div className="run-grid">
                            <div>
                              <h4>参数</h4>
                              {Object.entries(run.inputs || {}).map(
                                ([id, value]) => (
                                  <div key={id}>
                                    <p>{labels[id] || "参数"}</p>
                                    <pre>{show(value)}</pre>
                                  </div>
                                ),
                              )}
                            </div>
                            <div>
                              <h4>结果</h4>
                              {Object.entries(run.outputs || {}).map(
                                ([id, value]) => (
                                  <div key={id}>
                                    <p>{labels[id] || "结果"}</p>
                                    <pre>{show(value)}</pre>
                                  </div>
                                ),
                              )}
                            </div>
                          </div>
                        </details>
                        <div className="run-actions">
                          <button
                            className="secondary"
                            disabled={
                              pending || !taskId || run.status !== "completed"
                            }
                            onClick={() =>
                              void act(async () => {
                                const result = await request("tasks", {
                                  action: "evaluate",
                                  task_id: taskId,
                                  run_id: run.id,
                                });
                                setMessage(
                                  verdictStatus[result.verdict] ||
                                    result.verdict,
                                );
                              })
                            }
                          >
                            运行验收规则
                          </button>
                          <button
                            className="text-button"
                            disabled={pending || !taskId}
                            onClick={() =>
                              void act(async () =>
                                setComparison(
                                  await request("tasks", {
                                    action: "compare",
                                    task_id: taskId,
                                    run_id: run.id,
                                  }),
                                ),
                              )
                            }
                          >
                            与任务基线对比
                          </button>
                          <button
                            className="text-button"
                            disabled={
                              pending || !taskId || !run.acceptance?.passed
                            }
                            onClick={() =>
                              void act(async () => {
                                await request("tasks", {
                                  action: "baseline",
                                  task_id: taskId,
                                  run_id: run.id,
                                });
                                setMessage("已设为任务基线。");
                              })
                            }
                          >
                            设为所选任务基线
                          </button>
                          <button
                            className="text-button"
                            onClick={() => void act(exportEvidence)}
                          >
                            导出运行证据
                          </button>
                        </div>
                        {comparison && (
                          <div className="run-comparison">
                            <h4>与基线的差异</h4>
                            <p className="run-help">{comparison.notice}</p>
                            <p>
                              耗时：基线{" "}
                              {comparison.duration_seconds[0]?.toFixed(2) ??
                                "—"}{" "}
                              秒 / 当前{" "}
                              {comparison.duration_seconds[1]?.toFixed(2) ??
                                "—"}{" "}
                              秒
                            </p>
                            <h4>参数变化</h4>
                            {changes(comparison.inputs)}
                            <h4>结果变化</h4>
                            {changes(comparison.outputs)}
                          </div>
                        )}
                        <div className="run-grid">
                          <details>
                            <summary>人工业务验收</summary>
                            <div className="native-field">
                              <label htmlFor="acceptance-result">
                                实际任务是否通过
                              </label>
                              <select
                                id="acceptance-result"
                                value={String(acceptance)}
                                onChange={(event) =>
                                  setAcceptance(event.target.value === "true")
                                }
                              >
                                <option value="true">通过</option>
                                <option value="false">未通过</option>
                              </select>
                              <label htmlFor="acceptance-reason">
                                核对依据与结论
                              </label>
                              <textarea
                                id="acceptance-reason"
                                maxLength={2000}
                                value={reason}
                                onChange={(event) =>
                                  setReason(event.target.value)
                                }
                              />
                              <button
                                className="secondary"
                                disabled={
                                  pending ||
                                  !["completed", "stopped"].includes(
                                    run.status,
                                  ) ||
                                  !reason.trim()
                                }
                                onClick={() =>
                                  void act(async () => {
                                    await request("tasks", {
                                      action: "review",
                                      run_id: run.id,
                                      passed: acceptance,
                                      reason,
                                    });
                                    setReason("");
                                    setMessage("人工验收结论已保存。");
                                  })
                                }
                              >
                                保存人工验收
                              </button>
                            </div>
                          </details>
                          <details>
                            <summary>从此运行提炼经验</summary>
                            <div className="native-field">
                              <label htmlFor="note-title">经验标题</label>
                              <input
                                id="note-title"
                                maxLength={120}
                                value={title}
                                onChange={(event) =>
                                  setTitle(event.target.value)
                                }
                              />
                              <label htmlFor="note-body">
                                适用条件、做法与结论
                              </label>
                              <textarea
                                id="note-body"
                                maxLength={8000}
                                value={body}
                                onChange={(event) =>
                                  setBody(event.target.value)
                                }
                                placeholder="保留事实依据与适用边界"
                              />
                              <button
                                className="secondary"
                                disabled={
                                  pending || !title.trim() || !body.trim()
                                }
                                onClick={() =>
                                  void act(async () => {
                                    await request("notes", {
                                      run_id: run.id,
                                      title,
                                      body,
                                    });
                                    setTitle("");
                                    setBody("");
                                    setMessage(
                                      "经验草稿已保存，可在“经验”中审阅采用。",
                                    );
                                  })
                                }
                              >
                                保存经验
                              </button>
                            </div>
                          </details>
                        </div>
                        <details>
                          <summary>从此运行保存任务</summary>
                          <div className="native-field">
                            <label htmlFor="task-name">任务名称</label>
                            <input
                              id="task-name"
                              maxLength={120}
                              value={name}
                              onChange={(event) => setName(event.target.value)}
                            />
                            <label htmlFor="task-expectation">验收要点</label>
                            <textarea
                              id="task-expectation"
                              maxLength={2000}
                              value={expectation}
                              onChange={(event) =>
                                setExpectation(event.target.value)
                              }
                              placeholder="例如：命中指定文档，回答包含可核对出处"
                            />
                            <TaskRules
                              rules={newRules}
                              onChange={setNewRules}
                              outputs={Object.keys(run.outputs || {})}
                              labels={labels}
                            />
                            <button
                              className="secondary"
                              disabled={
                                pending || !name.trim() || !expectation.trim()
                              }
                              onClick={() =>
                                void act(async () => {
                                  const task = await request("tasks", {
                                    action: "save",
                                    run_id: run.id,
                                    name,
                                    expectation,
                                    rules: newRules,
                                  });
                                  onExitPreset();setTaskId(task.id);
                                  setName("");
                                  setExpectation("");
                                  setNewRules([]);
                                  setMessage(
                                    "任务已保存。完成人工业务验收后，可将运行设为基线。",
                                  );
                                })
                              }
                            >
                              保存任务
                            </button>
                          </div>
                        </details>
                      </div>
                    )}
                  </>
                )}
                {view === "notes" && (
                  <>
                    <ExperienceConsolidation runs={data.runs} onAction={act} />
                    <p className="run-help">
                      经验先作为草稿保存。审阅采用后，可加入上下文资源；来源运行与历史版本始终保留。
                    </p>
                    {data.notes.length ? (
                      data.notes.map((note) => (
                        <Experience
                          key={note.id}
                          note={note}
                          onSource={() =>
                            void act(() => selectRun(note.source_run))
                          }
                          onAction={act}
                        />
                      ))
                    ) : (
                      <p className="run-help">先从经过核对的运行中提炼经验。</p>
                    )}
                  </>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </details>
  );
}

function Experience({
  note,
  onSource,
  onAction,
}: {
  note: Note;
  onSource: () => void;
  onAction: (work: () => Promise<void>) => Promise<void>;
}) {
  const [body, setBody] = useState(note.body),
    [message, setMessage] = useState(""),
    [historic, setHistoric] = useState<{
      body: string;
      version: number;
    } | null>(null);
  useEffect(() => setBody(note.body), [note.version, note.body]);
  const state = note.status || "draft";
  async function save(status: string) {
    await onAction(async () => {
      await request("notes", {
        action: "update",
        note_id: note.id,
        expected_version: note.version || 1,
        body,
        status,
      });
      setMessage(
        status === "approved" ? "经验已审阅采用。" : "经验版本已保存。",
      );
    });
  }
  return (
    <details className="experience-card">
      <summary>
        {note.title}
        <span>
          {state === "approved"
            ? "已采用"
            : state === "retired"
              ? "已停用"
              : "草稿"}{" "}
          · v{note.version || 1}
        </span>
      </summary>
      <div className="native-field">
        <label htmlFor={"experience-" + note.id}>适用条件、做法与结论</label>
        <textarea
          id={"experience-" + note.id}
          value={body}
          maxLength={8000}
          rows={5}
          onChange={(event) => setBody(event.target.value)}
        />
        <details>
          <summary>经验历史版本</summary>
          <div className="run-actions">
            <button
              className="text-button"
              disabled={(historic?.version || note.version || 1) <= 1}
              onClick={() =>
                void onAction(async () =>
                  setHistoric(
                    await request(
                      `notes/${note.id}?version=${(historic?.version || note.version || 1) - 1}`,
                    ),
                  ),
                )
              }
            >
              读取上一版本
            </button>
            <button className="text-button" onClick={() => setHistoric(null)}>
              返回当前版本
            </button>
          </div>
          {historic && (
            <>
              <p className="run-help">历史 v{historic.version}（只读）</p>
              <pre>{historic.body}</pre>
            </>
          )}
        </details>
        <div className="run-actions">
          <button className="secondary" onClick={() => void save("draft")}>
            保存草稿
          </button>
          <button className="secondary" onClick={() => void save("approved")}>
            审阅采用
          </button>
          <button className="text-button" onClick={() => void save("retired")}>
            停用
          </button>
          <button className="text-button" onClick={onSource}>
            查看来源运行
          </button>
          <button
            className="text-button"
            disabled={state !== "approved"}
            onClick={() =>
              void onAction(async () => {
                await request("context", {
                  action: "from_note",
                  note_id: note.id,
                });
                setMessage(
                  "已加入上下文资源库，可在“上下文工程”的供应配置中组合使用。",
                );
              })
            }
          >
            加入上下文资源
          </button>
        </div>
        {message && (
          <p role="status" className="run-message">
            {message}
          </p>
        )}
      </div>
    </details>
  );
}
