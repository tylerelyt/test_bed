"use client";
import { useState } from "react";
import { NativeRun, request, runStatus } from "../lib/workbench-api";
import { displayLabel } from "../lib/presentation";
export default function ExperienceConsolidation({
  runs,
  onAction,
}: {
  runs: NativeRun[];
  onAction: (work: () => Promise<void>) => Promise<void>;
}) {
  const [selected, setSelected] = useState<string[]>([]),
    [title, setTitle] = useState(""),
    [model, setModel] = useState(false),
    [message, setMessage] = useState("");
  return (
    <details className="run-editor">
      <summary>从多次运行整理经验</summary>
      <div className="native-field">
        <p className="run-help">
          选择 2—10
          次已结束运行。默认按实际记录生成事实摘要，也可使用服务端配置的模型辅助整理。结果始终作为草稿，审阅后才可加入上下文资源。
        </p>
        <label htmlFor="consolidation-title">整理后的经验标题</label>
        <input
          id="consolidation-title"
          value={title}
          maxLength={120}
          onChange={(event) => setTitle(event.target.value)}
        />
        <fieldset>
          <legend>选择来源运行</legend>
          {runs
            .filter(
              (run) =>
                !["queued", "running", "awaiting_review"].includes(run.status),
            )
            .slice(0, 20)
            .map((run) => (
              <label className="native-check" key={run.id}>
                <input
                  type="checkbox"
                  checked={selected.includes(run.id)}
                  disabled={!selected.includes(run.id) && selected.length >= 10}
                  onChange={(event) =>
                    setSelected((old) =>
                      event.target.checked
                        ? [...old, run.id]
                        : old.filter((id) => id !== run.id),
                    )
                  }
                />
                {new Date(run.started_at * 1000).toLocaleString("zh-CN")} ·{" "}
                {displayLabel(run.action)} · {runStatus[run.status]}
              </label>
            ))}
        </fieldset>
        <label className="native-check">
          <input
            type="checkbox"
            checked={model}
            onChange={(event) => setModel(event.target.checked)}
          />
          使用已配置的模型辅助整理（将发起模型调用）
        </label>
        <button
          className="secondary"
          disabled={selected.length < 2 || !title.trim()}
          onClick={() =>
            void onAction(async () => {
              await request("notes", {
                action: "consolidate",
                title,
                run_ids: selected,
                use_model: model,
              });
              setSelected([]);
              setTitle("");
              setMessage("经验草稿已生成，请核对证据后审阅采用。");
            })
          }
        >
          生成经验草稿
        </button>
        {message && (
          <p role="status" className="run-message">
            {message}
          </p>
        )}
      </div>
    </details>
  );
}
