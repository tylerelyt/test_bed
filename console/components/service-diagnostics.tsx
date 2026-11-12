"use client";
import { useState } from "react";
import { request } from "../lib/workbench-api";
type Report = {
  services: { name: string; reachable: boolean; check: string }[];
  model_key_configured: boolean;
  judge_configured: boolean;
  context_bundles: number;
  note: string;
  scheduler?: { last_tick: number | null; error: string | null };
};
export default function ServiceDiagnostics() {
  const [report, setReport] = useState<Report | null>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  return (
    <details className="service-diagnostics">
      <summary>连接与配置检查</summary>
      <p className="run-help">
        检查当前服务连接和配置，不加载模型或发起模型调用。
      </p>
      <button
        className="secondary"
        disabled={busy}
        onClick={() => {
          setBusy(true);
          setError("");
          void request("diagnostics")
            .then(setReport)
            .catch((e) => setError(e.message))
            .finally(() => setBusy(false));
        }}
      >
        {busy ? "正在检查…" : "检查服务"}
      </button>
      {error && (
        <p role="alert" className="inline-error">
          {error}
        </p>
      )}
      {report && (
        <>
          <div className="native-table-scroll">
            <table>
              <thead>
                <tr>
                  <th>服务</th>
                  <th>当前检查</th>
                </tr>
              </thead>
              <tbody>
                {report.services.map((service) => (
                  <tr key={service.name}>
                    <td>{service.name}</td>
                    <td>{service.reachable ? "端口可达" : "端口未连接"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {report.scheduler?.error && (
            <p role="alert" className="inline-error">
              周期调度异常：{report.scheduler.error}
            </p>
          )}
          <p className="run-help">
            模型密钥：{report.model_key_configured ? "已配置" : "未配置"} ·
            评分模型：{report.judge_configured ? "已配置" : "未配置"} ·
            供应配置：{report.context_bundles}{" "}
            份。端口可达不代表模型和业务健康。
          </p>
        </>
      )}
    </details>
  );
}
