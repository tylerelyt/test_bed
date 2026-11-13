"use client";
import { useEffect, useState } from "react";
import { platformRequest as api } from "../lib/platform-api";
export type OperationSupplyValue = {
  model_id?: string;
  context_binding?: {
    bundle_id: string;
    version: number;
    component_id: string;
  };
};
export default function OperationSupply({
  fields,
  modelEnabled,
  value,
  onChange,
}: {
  fields: { id: string; name: string }[];
  modelEnabled: boolean;
  value: OperationSupplyValue;
  onChange: (value: OperationSupplyValue) => void;
}) {
  const [models, setModels] = useState<{ id: string; name: string }[]>([]),
    [bundles, setBundles] = useState<
      { id: string; name: string; version: number }[]
    >([]),
    [error, setError] = useState("");
  useEffect(() => {
    void Promise.all([api("overview"), api("context")])
      .then(([a, b]) => {
        setModels(a.models);
        setBundles(b.bundles);
      })
      .catch((e) => setError(e.message));
  }, []);
  return (
    <details className="run-editor">
      <summary>本次操作使用的供应配置</summary>
      <p className="run-help">
        选择后固定到本次运行记录。检索索引、CTR
        和本地视觉模型仍使用各自的数据与模型；此处模型服务用于兼容 OpenAI
        消息协议的业务调用。
      </p>
      {error && <p role="alert">{error}</p>}
      {modelEnabled && (
        <label className="native-field">
          业务模型服务
          <select
            aria-label="业务模型服务"
            value={value.model_id || ""}
            onChange={(e) =>
              onChange({ ...value, model_id: e.target.value || undefined })
            }
          >
            <option value="">使用模块原有模型配置</option>
            {models.map((x) => (
              <option key={x.id} value={x.id}>
                {x.name}
              </option>
            ))}
          </select>
        </label>
      )}
      {!!fields.length && (
        <>
          <label className="native-field">
            本次上下文配置
            <select
              aria-label="本次上下文配置"
              value={value.context_binding?.bundle_id || ""}
              onChange={(e) => {
                const b = bundles.find((x) => x.id === e.target.value);
                onChange({
                  ...value,
                  context_binding: b
                    ? {
                        bundle_id: b.id,
                        version: b.version,
                        component_id: fields[0].id,
                      }
                    : undefined,
                });
              }}
            >
              <option value="">不附加供应上下文</option>
              {bundles.map((x) => (
                <option key={x.id} value={x.id}>
                  {x.name} · v{x.version}
                </option>
              ))}
            </select>
          </label>
          {value.context_binding && (
            <label className="native-field">
              附加到输入字段
              <select
                aria-label="附加到输入字段"
                value={value.context_binding.component_id}
                onChange={(e) =>
                  onChange({
                    ...value,
                    context_binding: {
                      ...value.context_binding!,
                      component_id: e.target.value,
                    },
                  })
                }
              >
                {fields.map((x) => (
                  <option key={x.id} value={x.id}>
                    {x.name}
                  </option>
                ))}
              </select>
            </label>
          )}
        </>
      )}
    </details>
  );
}
