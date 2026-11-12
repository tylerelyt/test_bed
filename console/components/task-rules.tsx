"use client";
import { Rule } from "../lib/workbench-api";
import { useState } from "react";
const names: Record<Rule["op"], string> = {
  contains: "包含文本",
  not_contains: "排除文本",
  equals: "精确相等",
  json_keys: "JSON 必含字段",
  number_between: "数值范围",
  model: "模型辅助评分",
};
export default function TaskRules({
  rules,
  onChange,
  outputs,
  labels,
}: {
  rules: Rule[];
  onChange: (rules: Rule[]) => void;
  outputs: string[];
  labels: Record<string, string>;
}) {
  const edit = (index: number, update: Partial<Rule>) =>
    onChange(
      rules.map((rule, i) => (i === index ? { ...rule, ...update } : rule)),
    );
  return (
    <div className="task-rules">
      <p className="run-help">
        规则检查实际输出。没有规则的任务会标记为“尚未评分”；人工业务验收单独记录。
      </p>
      {rules.map((rule, index) => (
        <fieldset key={index} className="rule-row">
          <legend>规则 {index + 1}</legend>
          <div className="run-grid">
            <div className="native-field">
              <label htmlFor={`rule-op-${index}`}>检查方式</label>
              <select
                id={`rule-op-${index}`}
                value={rule.op}
                onChange={(e) =>
                  edit(index, {
                    op: e.target.value as Rule["op"],
                    expected: e.target.value === "json_keys" ? [] : "",
                    minimum: 0,
                    maximum: 1,
                  })
                }
              >
                {Object.entries(names).map(([key, name]) => (
                  <option key={key} value={key}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <div className="native-field">
              <label htmlFor={`rule-component-${index}`}>检查结果</label>
              <select
                id={`rule-component-${index}`}
                value={rule.component || ""}
                onChange={(e) => edit(index, { component: e.target.value })}
              >
                <option value="">全部输出</option>
                {outputs.map((id) => (
                  <option key={id} value={id}>
                    {labels[id] || id}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="native-field">
            <label htmlFor={`rule-pointer-${index}`}>
              JSON 内部路径（可选）
            </label>
            <input
              id={`rule-pointer-${index}`}
              value={rule.pointer || ""}
              placeholder="例如 /answer 或 /data/0/0"
              onChange={(e) => edit(index, { pointer: e.target.value })}
            />
          </div>
          {rule.op === "number_between" ? (
            <div className="run-grid">
              <div className="native-field">
                <label htmlFor={`rule-min-${index}`}>最小值</label>
                <input
                  id={`rule-min-${index}`}
                  type="number"
                  value={rule.minimum ?? 0}
                  onChange={(e) =>
                    edit(index, { minimum: Number(e.target.value) })
                  }
                />
              </div>
              <div className="native-field">
                <label htmlFor={`rule-max-${index}`}>最大值</label>
                <input
                  id={`rule-max-${index}`}
                  type="number"
                  value={rule.maximum ?? 1}
                  onChange={(e) =>
                    edit(index, { maximum: Number(e.target.value) })
                  }
                />
              </div>
            </div>
          ) : rule.op === "equals" ? (
            <EqualsValue
              index={index}
              initial={rule.expected}
              onChange={(expected) => edit(index, { expected })}
            />
          ) : rule.op === "model" ? (
            <div className="native-field">
              <label htmlFor={`rule-criteria-${index}`}>模型评分标准</label>
              <textarea
                id={`rule-criteria-${index}`}
                maxLength={2000}
                value={rule.criteria || ""}
                onChange={(e) => edit(index, { criteria: e.target.value })}
              />
              <p className="run-help">
                执行评分时会调用服务端配置的模型，意见作为辅助证据。
              </p>
            </div>
          ) : (
            <div className="native-field">
              <label htmlFor={`rule-expected-${index}`}>
                {rule.op === "json_keys" ? "字段名（每行一个）" : "检查文本"}
              </label>
              <textarea
                id={`rule-expected-${index}`}
                value={
                  rule.op === "json_keys"
                    ? Array.isArray(rule.expected)
                      ? rule.expected.join("\n")
                      : ""
                    : String(rule.expected ?? "")
                }
                onChange={(e) => {
                  if (rule.op === "json_keys")
                    edit(index, {
                      expected: e.target.value.split("\n").filter(Boolean),
                    });
                  else edit(index, { expected: e.target.value });
                }}
              />
            </div>
          )}
          <button
            type="button"
            className="text-button"
            onClick={() => onChange(rules.filter((_, i) => i !== index))}
          >
            移除此规则
          </button>
        </fieldset>
      ))}
      <button
        type="button"
        className="secondary"
        disabled={rules.length >= 30}
        onClick={() =>
          onChange([
            ...rules,
            { op: "contains", component: outputs[0] || "", expected: "" },
          ])
        }
      >
        添加验收规则
      </button>
    </div>
  );
}

function EqualsValue({
  index,
  initial,
  onChange,
}: {
  index: number;
  initial: unknown;
  onChange: (value: unknown) => void;
}) {
  const [mode, setMode] = useState(
    typeof initial === "number"
      ? "number"
      : typeof initial === "boolean"
        ? "boolean"
        : typeof initial === "object"
          ? "json"
          : "text",
  );
  const [raw, setRaw] = useState(
      typeof initial === "string" ? initial : JSON.stringify(initial ?? null),
    ),
    [error, setError] = useState("");
  function edit(value: string) {
    setRaw(value);
    try {
      const expected =
        mode === "json"
          ? JSON.parse(value)
          : mode === "number"
            ? Number(value)
            : mode === "boolean"
              ? value === "true"
              : value;
      if (mode === "number" && (!value.trim() || !Number.isFinite(expected)))
        throw Error();
      setError("");
      onChange(expected);
    } catch {
      setError("请填写有效的预期值，当前规则尚未保存。");
      onChange(undefined);
    }
  }
  return (
    <div className="native-field">
      <label htmlFor={`equals-kind-${index}`}>预期值类型</label>
      <select
        id={`equals-kind-${index}`}
        value={mode}
        onChange={(event) => {
          const next = event.target.value;
          setMode(next);
          setError("");
          const value =
            next === "boolean"
              ? true
              : next === "number"
                ? 0
                : next === "json"
                  ? {}
                  : "";
          setRaw(typeof value === "string" ? value : JSON.stringify(value));
          onChange(value);
        }}
      >
        <option value="text">文本</option>
        <option value="number">数字</option>
        <option value="boolean">布尔值</option>
        <option value="json">JSON</option>
      </select>
      <label htmlFor={`equals-value-${index}`}>预期值</label>
      {mode === "boolean" ? (
        <select
          id={`equals-value-${index}`}
          value={raw}
          onChange={(event) => edit(event.target.value)}
        >
          <option value="true">true</option>
          <option value="false">false</option>
        </select>
      ) : (
        <textarea
          id={`equals-value-${index}`}
          value={raw}
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
