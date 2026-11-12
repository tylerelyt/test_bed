"use client";
import { useEffect, useState } from "react";
import { ContextBinding, request } from "../lib/workbench-api";
type Bundle = { id: string; name: string; version: number };
export default function TaskContext({
  binding,
  onChange,
  inputs,
  labels,
}: {
  binding: ContextBinding | null;
  onChange: (binding: ContextBinding | null) => void;
  inputs: string[];
  labels: Record<string, string>;
}) {
  const [bundles, setBundles] = useState<Bundle[]>([]),
    [error, setError] = useState("");
  useEffect(() => {
    void request("context")
      .then((data) => setBundles(data.bundles))
      .catch((e) => setError(e.message));
  }, []);
  const selected = bundles.find((bundle) => bundle.id === binding?.bundle_id);
  return (
    <details>
      <summary>附加企业上下文（可选）</summary>
      <div className="native-field">
        <p className="run-help">
          选定配置内容会加入指定文本输入，会影响检索或推理。资源版本和实际输入会关联到运行记录，建议先用探针验证。配置在原“上下文工程”的“供应配置”中维护。
        </p>
        {error && (
          <p role="alert" className="inline-error">
            {error}
          </p>
        )}
        <label htmlFor="task-context-bundle">使用的供应配置</label>
        <select
          id="task-context-bundle"
          value={binding?.bundle_id || ""}
          onChange={(event) => {
            const bundle = bundles.find(
              (item) => item.id === event.target.value,
            );
            onChange(
              bundle
                ? {
                    bundle_id: bundle.id,
                    version: bundle.version,
                    component_id: inputs[0] || "",
                  }
                : null,
            );
          }}
        >
          <option value="">沿用原任务输入</option>
          {bundles.map((bundle) => (
            <option key={bundle.id} value={bundle.id}>
              {bundle.name} · 最新 v{bundle.version}
            </option>
          ))}
        </select>
        {binding && (
          <>
            <label htmlFor="task-context-input">加入哪个文本输入</label>
            <select
              id="task-context-input"
              value={binding.component_id}
              onChange={(event) =>
                onChange({ ...binding, component_id: event.target.value })
              }
            >
              <option value="">请选择问题或提示词输入</option>
              {inputs.map((id) => (
                <option key={id} value={id}>
                  {labels[id] || "文本输入"}
                </option>
              ))}
            </select>
            <p className="run-help">
              当前固定配置 v{binding.version}。
              {selected &&
                selected.version !== binding.version &&
                "该配置已有新版本，现有任务仍使用原版本。"}
            </p>
            {selected && selected.version !== binding.version && (
              <button
                className="text-button"
                onClick={() =>
                  onChange({ ...binding, version: selected.version })
                }
              >
                采用配置最新版本
              </button>
            )}
          </>
        )}
      </div>
    </details>
  );
}
