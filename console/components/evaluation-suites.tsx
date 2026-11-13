"use client";
import { useEffect, useState } from "react";
import {
  SavedTask,
  downloadJSON,
  downloadCSV,
  request,
  runStatus,
  verdictStatus,
} from "../lib/workbench-api";
type Suite = {
  id: string;
  name: string;
  task_ids: string[];
  baseline: string | null;
  version: number;
  variants: { name: string; values: Record<string, unknown> }[];
  repeats: number;
};
type Batch = {
  id: string;
  suite_id: string;
  mode: string;
  complete: boolean;
  terminal?: boolean;
  submission_error?: string;
  expected_count?: number;
  recommendation?: { variant: string; basis: string; notice: string } | null;
  rows: {
    task_name: string;
    variant: string;
    trial: number;
    run_id: string;
    status: string;
    verdict: string;
    duration: number | null;
  }[];
  groups: {
    variant: string;
    count: number;
    passed: number;
    pass_rate: number;
    p95_seconds: number | null;
    cost: number | null;
    currency: string | null;
    input_tokens: number | null;
    output_tokens: number | null;
  }[];
};
type Schedule = {
  id: string;
  suite_id: string;
  suite_name: string;
  interval_minutes: number;
  enabled: boolean;
  next_at: number;
  last_batch?: string;
  error?: string;
};
type SuiteComparison = {
  groups: {
    variant: string;
    pass_rate_delta: number | null;
    p95_delta: number | null;
  }[];
  notice: string;
};
export default function EvaluationSuites({
  module,
  suites,
  batches,
  schedules,
  labels,
  onRefresh,
  onRun,
}: {
  module: string;
  suites: Suite[];
  batches: { id: string; suite_id: string; created_at: number }[];
  schedules: Schedule[];
  labels: Record<string, string>;
  onRefresh: () => Promise<void>;
  onRun: (id: string) => void;
}) {
  const [tasks, setTasks] = useState<SavedTask[]>([]),
    [taskIds, setTaskIds] = useState<string[]>([]),
    [name, setName] = useState(""),
    [suiteId, setSuiteId] = useState(""),
    [batchId, setBatchId] = useState(""),
    [batch, setBatch] = useState<Batch | null>(null),
    [repeats, setRepeats] = useState(1),
    [interval, setIntervalMinutes] = useState(60),
    [variants, setVariants] = useState([
      { name: "当前配置", values: {} as Record<string, unknown> },
    ]);
  const [editingSuite, setEditingSuite] = useState<Suite | null>(null);
  const [audit, setAudit] = useState<{
    checks: string[];
    notice: string;
    task_count: number;
  } | null>(null);
  const [invalid, setInvalid] = useState<Record<string, boolean>>({});
  const [message, setMessage] = useState(""),
    [error, setError] = useState(""),
    [pending, setPending] = useState(false),
    [comparison, setComparison] = useState<SuiteComparison | null>(null);
  useEffect(() => {
    void request("evaluations")
      .then((data) => setTasks(data.tasks))
      .catch((e) => setError(e.message));
  }, [module, suites.length]);
  useEffect(() => {
    setSuiteId("");
    setBatchId("");
    setBatch(null);
    setTaskIds([]);
    setComparison(null);
  }, [module]);
  useEffect(() => {
    if (!batchId) return;
    let cancelled = false;
    const load = () =>
      request("evaluations/" + batchId)
        .then((data) => {
          if (!cancelled) setBatch(data);
        })
        .catch((e) => {
          if (!cancelled) setError(e.message);
        });
    void load();
    const timer = setInterval(() => void load(), 2500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [batchId]);
  async function act(work: () => Promise<void>) {
    setPending(true);
    setError("");
    try {
      await work();
      await onRefresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  const chosen = tasks.filter((task) => taskIds.includes(task.id));
  const inputs = Array.from(
    new Set(chosen.flatMap((task) => Object.keys(task.inputs))),
  );
  useEffect(() => {
    const known = new Set(
      tasks
        .filter((task) => taskIds.includes(task.id))
        .flatMap((task) => Object.keys(task.inputs)),
    );
    setVariants((old) =>
      old.map((item) => ({
        ...item,
        values: Object.fromEntries(
          Object.entries(item.values).filter(([id]) => known.has(id)),
        ),
      })),
    );
  }, [taskIds.join(",")]);
  async function start(mode: string) {
    await act(async () => {
      const result = await request("evaluations", {
        action: "start",
        suite_id: suiteId,
        mode,
        request_id: crypto.randomUUID().replaceAll("-", ""),
      });
      setBatchId(result.id);
      setBatch(result);
      setComparison(null);
      setMessage(
        mode === "probe"
          ? "探针已启动：首个任务 × 各变体，各运行一次。"
          : "完整评测已启动。任务在服务端执行。",
      );
    });
  }
  return (
    <div
      data-verify-unit="EvaluationSuites"
      data-verify-state={pending ? "pending" : "ready"}
    >
      <p className="run-help">
        固定任务集 × 参数变体 ×
        重复次数。先用快速探针检查，再运行完整评测。密钥继续由服务端配置，批量任务不会保存密钥值或临时文件。涉及内部会话状态的操作需先准备对应业务环境。
      </p>
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
      <div className="run-grid">
        <div className="native-field">
          <label htmlFor="evaluation-suite">任务集</label>
          <select
            id="evaluation-suite"
            value={suiteId}
            onChange={(e) => {
              setSuiteId(e.target.value);
              setBatchId("");
              setBatch(null);
              setComparison(null);
            }}
          >
            <option value="">选择任务集</option>
            {suites.map((suite) => (
              <option key={suite.id} value={suite.id}>
                {suite.name}
              </option>
            ))}
          </select>
          <div className="run-actions">
            <button
              className="text-button"
              disabled={pending || !suiteId}
              onClick={() => {
                const suite = suites.find((item) => item.id === suiteId);
                if (suite) {
                  setEditingSuite(suite);
                  setName(suite.name);
                  setTaskIds(suite.task_ids);
                  setVariants(
                    suite.variants.map((item) => ({
                      ...item,
                      values: { ...item.values },
                    })),
                  );
                  setRepeats(suite.repeats);
                  setInvalid({});
                }
              }}
            >
              编辑所选任务集
            </button>
            <button
              className="secondary"
              disabled={pending || !suiteId}
              onClick={() =>
                void act(async () =>
                  setAudit(
                    await request("evaluations", {
                      action: "audit",
                      suite_id: suiteId,
                    }),
                  ),
                )
              }
            >
              检查任务集
            </button>
            <button
              className="secondary"
              disabled={pending || !suiteId}
              onClick={() => void start("probe")}
            >
              运行快速探针
            </button>
            <button
              className="primary"
              disabled={pending || !suiteId}
              onClick={() => void start("full")}
            >
              运行完整评测
            </button>
          </div>
        </div>
        <div className="native-field">
          <label htmlFor="evaluation-batch">评测记录</label>
          <select
            id="evaluation-batch"
            value={batchId}
            onChange={(e) => {
              setBatchId(e.target.value);
              setComparison(null);
            }}
          >
            <option value="">选择一次评测</option>
            {batches
              .filter((item) => !suiteId || item.suite_id === suiteId)
              .map((item) => (
                <option key={item.id} value={item.id}>
                  {new Date(item.created_at * 1000).toLocaleString("zh-CN")}
                </option>
              ))}
          </select>
        </div>
      </div>
      {audit && (
        <div className="run-detail">
          <h4>任务集检查 · {audit.task_count} 个任务</h4>
          <ul>
            {audit.checks.map((check, index) => (
              <li key={index}>{check}</li>
            ))}
          </ul>
          <p className="run-help">{audit.notice}</p>
        </div>
      )}
      {batch && (
        <div className="run-detail">
          <h4>
            {batch.mode === "probe" ? "快速探针" : "完整评测"} ·{" "}
            {batch.submission_error ? "提交失败" : batch.complete ? "已结束" : batch.terminal ? "已停止或不完整" : "执行中"}
          </h4>
          {batch.submission_error && <p role="alert" className="inline-error">{batch.submission_error}。本批次不作为完整评测或基线。</p>}
          {batch.recommendation && (
            <p className="run-help">
              当前规则下的候选：{batch.recommendation.variant}。
              {batch.recommendation.basis}
              {batch.recommendation.notice}
            </p>
          )}
          <div className="native-table-scroll">
            <table>
              <thead>
                <tr>
                  <th>变体</th>
                  <th>规则通过 / 总数</th>
                  <th>通过率</th>
                  <th>P95 耗时</th>
                  <th>输入 / 输出 token</th>
                  <th>成本</th>
                </tr>
              </thead>
              <tbody>
                {batch.groups.map((group) => (
                  <tr key={group.variant}>
                    <td>{group.variant}</td>
                    <td>
                      {group.passed} / {group.count}
                    </td>
                    <td>{(group.pass_rate * 100).toFixed(1)}%</td>
                    <td>{group.p95_seconds?.toFixed(2) ?? "—"} 秒</td>
                    <td>
                      {group.input_tokens == null
                        ? "未采集"
                        : `${group.input_tokens} / ${group.output_tokens}`}
                    </td>
                    <td>
                      {group.cost == null
                        ? "未配置或未采集"
                        : `${group.cost.toFixed(6)} ${group.currency}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="run-actions">
            <button
              className="secondary"
              onClick={() => downloadJSON("评测-" + batch.id + ".json", batch)}
            >
              导出评测 JSON
            </button>
            <button
              className="text-button"
              onClick={() =>
                downloadCSV("评测-" + batch.id + ".csv", [
                  [
                    "任务",
                    "变体",
                    "次数",
                    "执行状态",
                    "规则结果",
                    "耗时秒",
                    "运行标识",
                  ],
                  ...batch.rows.map((row) => [
                    row.task_name,
                    row.variant,
                    row.trial,
                    runStatus[row.status],
                    verdictStatus[row.verdict],
                    row.duration,
                    row.run_id,
                  ]),
                ])
              }
            >
              导出评测 CSV
            </button>
            <button
              className="text-button"
              disabled={!batch.complete || batch.mode !== "full"}
              onClick={() =>
                void act(async () => {
                  await request("evaluations", {
                    action: "baseline",
                    suite_id: batch.suite_id,
                    batch_id: batch.id,
                  });
                  setMessage("已设置任务集基线。");
                })
              }
            >
              设为任务集基线
            </button>
            <button
              className="text-button"
              onClick={() =>
                void act(async () =>
                  setComparison(
                    await request("evaluations", {
                      action: "compare",
                      suite_id: batch.suite_id,
                      batch_id: batch.id,
                    }),
                  ),
                )
              }
            >
              比较任务集基线
            </button>
            {!batch.complete && !batch.terminal && (
              <button
                className="text-button"
                onClick={() =>
                  void act(async () =>
                    setBatch(
                      await request("evaluations", {
                        action: "cancel",
                        batch_id: batch.id,
                      }),
                    ),
                  )
                }
              >
                停止本次评测
              </button>
            )}
          </div>
          {comparison && (
            <div className="run-comparison">
              <h4>任务集基线差异</h4>
              <p className="run-help">{comparison.notice}</p>
              <div className="native-table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>变体</th>
                      <th>通过率变化</th>
                      <th>P95 耗时变化</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.groups.map((group) => (
                      <tr key={group.variant}>
                        <td>{group.variant}</td>
                        <td>
                          {group.pass_rate_delta == null
                            ? "新增变体"
                            : `${(group.pass_rate_delta * 100).toFixed(1)} 个百分点`}
                        </td>
                        <td>
                          {group.p95_delta == null
                            ? "未比较"
                            : `${group.p95_delta.toFixed(2)} 秒`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          <div className="native-table-scroll">
            <table>
              <thead>
                <tr>
                  <th>任务</th>
                  <th>变体 / 次数</th>
                  <th>执行</th>
                  <th>规则结果</th>
                  <th>证据</th>
                </tr>
              </thead>
              <tbody>
                {batch.rows.map((row) => (
                  <tr key={row.run_id}>
                    <td>{row.task_name}</td>
                    <td>
                      {row.variant} / {row.trial}
                    </td>
                    <td>{runStatus[row.status]}</td>
                    <td>{verdictStatus[row.verdict]}</td>
                    <td>
                      <button
                        className="text-button"
                        onClick={() => onRun(row.run_id)}
                      >
                        查看运行
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      <details className="run-editor" open={editingSuite != null}>
        <summary>{editingSuite ? "编辑任务集新版本" : "创建任务集"}</summary>
        <div className="native-field">
          <label htmlFor="suite-name">任务集名称</label>
          <input
            id="suite-name"
            maxLength={120}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <fieldset>
            <legend>选择固定任务（可组合不同模块）</legend>
            {tasks.length ? (
              tasks.map((task) => (
                <label className="native-check" key={task.id}>
                  <input
                    type="checkbox"
                    checked={taskIds.includes(task.id)}
                    onChange={(e) =>
                      setTaskIds((old) =>
                        e.target.checked
                          ? [...old, task.id]
                          : old.filter((id) => id !== task.id),
                      )
                    }
                  />
                  {task.name}
                </label>
              ))
            ) : (
              <p className="run-help">先从一次运行保存固定任务。</p>
            )}
          </fieldset>
          {variants.map((variant, index) => (
            <fieldset key={`${editingSuite?.id || "new"}-${index}`}>
              <legend>参数变体 {index + 1}</legend>
              <label htmlFor={`variant-name-${index}`}>变体名称</label>
              <input
                id={`variant-name-${index}`}
                value={variant.name}
                onChange={(e) =>
                  setVariants((old) =>
                    old.map((item, i) =>
                      i === index ? { ...item, name: e.target.value } : item,
                    ),
                  )
                }
              />
              {inputs.map((id) => (
                <div className="variant-input" key={id}>
                  <label>
                    <input
                      type="checkbox"
                      checked={id in variant.values}
                      onChange={(e) =>
                        setVariants((old) =>
                          old.map((item, i) => {
                            if (i !== index) return item;
                            const values = { ...item.values };
                            if (e.target.checked)
                              values[id] =
                                chosen.find((task) => id in task.inputs)
                                  ?.inputs[id] ?? "";
                            else delete values[id];
                            return { ...item, values };
                          }),
                        )
                      }
                    />
                    覆盖 {labels[id] || "参数"}
                  </label>
                  {id in variant.values && (
                    <VariantValue
                      label={`变体${index + 1} ${labels[id] || "参数"}`}
                      initial={variant.values[id]}
                      original={
                        chosen.find((task) => id in task.inputs)?.inputs[id]
                      }
                      onChange={(value, valid) => {
                        setInvalid((old) => ({
                          ...old,
                          [`${index}:${id}`]: !valid,
                        }));
                        if (valid)
                          setVariants((old) =>
                            old.map((item, i) =>
                              i === index
                                ? {
                                    ...item,
                                    values: { ...item.values, [id]: value },
                                  }
                                : item,
                            ),
                          );
                      }}
                    />
                  )}
                </div>
              ))}
            </fieldset>
          ))}
          <button
            className="text-button"
            disabled={variants.length >= 10}
            onClick={() =>
              setVariants((old) => [
                ...old,
                { name: `变体 ${old.length + 1}`, values: {} },
              ])
            }
          >
            添加参数变体
          </button>
          <label htmlFor="suite-repeats">每组重复次数</label>
          <input
            id="suite-repeats"
            type="number"
            min={1}
            max={5}
            value={repeats}
            onChange={(e) => setRepeats(Number(e.target.value))}
          />
          <button
            className="secondary"
            disabled={
              pending ||
              !name.trim() ||
              !taskIds.length ||
              Object.entries(invalid).some(
                ([key, bad]) =>
                  bad &&
                  key.split(":")[1] in
                    (variants[Number(key.split(":")[0])]?.values || {}),
              )
            }
            onClick={() =>
              void act(async () => {
                const suite = await request("evaluations", {
                  action: "save",
                  module,
                  name,
                  suite_id: editingSuite?.id,
                  expected_version: editingSuite?.version,
                  task_ids: taskIds,
                  variants,
                  repeats,
                });
                setSuiteId(suite.id);
                setEditingSuite(null);
                setMessage("任务集已保存，可先运行快速探针。");
              })
            }
          >
            保存任务集{editingSuite ? "新版本" : ""}
          </button>
        </div>
      </details>
      <details className="run-editor">
        <summary>周期验证</summary>
        <p className="run-help">
          启用后由执行服务按周期运行任务集。服务离线期间的周期会合并，不重复补跑；任务版本失效时自动暂停。
        </p>
        <div className="native-field">
          <label htmlFor="schedule-minutes">周期（分钟）</label>
          <input
            id="schedule-minutes"
            type="number"
            min={1}
            max={10080}
            value={interval}
            onChange={(e) => setIntervalMinutes(Number(e.target.value))}
          />
          <button
            className="secondary"
            disabled={!suiteId || pending}
            onClick={() =>
              void act(async () => {
                await request("schedules", {
                  suite_id: suiteId,
                  interval_minutes: interval,
                  enabled: true,
                });
                setMessage("周期验证已启用。");
              })
            }
          >
            为所选任务集启用周期验证
          </button>
        </div>
        {schedules.map((item) => (
          <div className="schedule-row" key={item.id}>
            <strong>{item.suite_name}</strong>
            <span>
              {item.enabled
                ? `每 ${item.interval_minutes} 分钟，下次 ${new Date(item.next_at * 1000).toLocaleString("zh-CN")}`
                : "已暂停"}
            </span>
            {item.error && <p role="alert">{item.error}</p>}
            {item.last_batch && !item.error && (
              <button
                className="text-button"
                onClick={() => setBatchId(item.last_batch!)}
              >
                查看上次评测
              </button>
            )}
            <button
              className="text-button"
              onClick={() =>
                void act(async () => {
                  await request("schedules", {
                    suite_id: item.suite_id,
                    schedule_id: item.id,
                    interval_minutes: item.interval_minutes,
                    enabled: !item.enabled,
                  });
                })
              }
            >
              {item.enabled ? "暂停" : "恢复"}
            </button>
          </div>
        ))}
      </details>
    </div>
  );
}

function VariantValue({
  label,
  initial,
  original,
  onChange,
}: {
  label: string;
  initial: unknown;
  original: unknown;
  onChange: (value: unknown, valid: boolean) => void;
}) {
  const [raw, setRaw] = useState(
      typeof initial === "string" ? initial : JSON.stringify(initial),
    ),
    [error, setError] = useState("");
  useEffect(() => {
    setRaw(typeof initial === "string" ? initial : JSON.stringify(initial));
    setError("");
  }, [JSON.stringify(initial)]);
  function edit(raw: string) {
    setRaw(raw);
    try {
      const value =
        typeof original === "string"
          ? raw
          : typeof original === "number"
            ? Number(raw)
            : JSON.parse(raw);
      if (
        typeof original === "number" &&
        (!raw.trim() || !Number.isFinite(value))
      )
        throw Error();
      setError("");
      onChange(value, true);
    } catch {
      setError("参数格式无效，尚未应用此变更。");
      onChange(undefined, false);
    }
  }
  return (
    <div>
      {typeof original === "boolean" ? (
        <select
          aria-label={label}
          value={raw}
          onChange={(event) => edit(event.target.value)}
        >
          <option value="true">是（true）</option>
          <option value="false">否（false）</option>
        </select>
      ) : (
        <input
          aria-label={label}
          value={raw ?? "null"}
          onChange={(event) => edit(event.target.value)}
        />
      )}{" "}
      {error && (
        <p role="alert" className="inline-error">
          {error}
        </p>
      )}
    </div>
  );
}
