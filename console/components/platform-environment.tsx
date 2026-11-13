"use client";
import { useEffect, useState } from "react";
import { platformRequest as api } from "../lib/platform-api";
type Environment = {
  storage: string;
  context_storage: string;
  worker_running: boolean;
  training_python: string;
  inference_python: string;
  llamafactory: string | null;
  vllm: string | null;
  pypdf: string | null;
  notice: string;
  harnesses: { installed: boolean; kind: string; version: string }[];
};
export default function PlatformEnvironment() {
  const [data, setData] = useState<Environment | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    void api("environment")
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);
  return (
    <section
      className="agent-card"
      data-verify-unit="PlatformEnvironment"
      data-verify-state={data ? "ready" : error ? "unavailable" : "loading"}
    >
      <h2>本机运行环境</h2>
      {error && <p role="alert">{error}</p>}
      {!data && !error && <p>正在检查环境…</p>}
      {data && (
        <>
          <dl className="environment-list">
            <dt>业务数据库</dt>
            <dd>{data.storage}</dd>
            <dt>上下文存储</dt>
            <dd>{data.context_storage}（与业务记录共用）</dd>
            <dt>后台执行进程</dt>
            <dd>
              {data.worker_running
                ? "运行中"
                : "未检测到独立进程；请检查启动日志"}
            </dd>
            <dt>训练环境</dt>
            <dd>
              {data.training_python}
              <br />
              LLaMA-Factory：{data.llamafactory || "未安装"}
            </dd>
            <dt>推理环境</dt>
            <dd>
              {data.inference_python}
              <br />
              vLLM：{data.vllm || "未安装"}
            </dd>
            <dt>PDF 解析</dt>
            <dd>
              {data.pypdf
                ? "pypdf " + data.pypdf
                : "未安装，请安装 requirements-platform.txt"}
            </dd>
            <dt>Harness</dt>
            <dd>
              {data.harnesses.map((x) => (
                <p key={x.kind}>
                  {x.kind} {x.version}：{x.installed ? "已安装" : "未安装"}
                </p>
              ))}
            </dd>
          </dl>
          <p className="run-help">{data.notice}</p>
          <p>
            训练和推理可分别设置 WORKBENCH_TRAIN_PYTHON、WORKBENCH_INFER_PYTHON
            指向独立环境，启动时生效。
          </p>
          <p>
            岗位 Agent
            独立排队执行；供应作业最多同时执行两个。重启后未完成的作业标记为中断，需要核对后重新提交。
          </p>
        </>
      )}
    </section>
  );
}
