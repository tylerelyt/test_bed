"use client";
import { useEffect, useState } from "react";
import { platformRequest as api, requestId } from "../lib/platform-api";
import SupplyJobs from "./supply-jobs";
type Dataset = { id: string; name: string; count: number; digest: string };
type Artifact = {
  id: string;
  name: string;
  format: string;
  path: string;
  source_job: string | null;
};
export default function ModelSupply() {
  const [datasets, setDatasets] = useState<Dataset[]>([]),
    [artifacts, setArtifacts] = useState<Artifact[]>([]),
    [error, setError] = useState(""),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false);
  const [name, setName] = useState(""),
    [file, setFile] = useState<File | null>(null),
    [artifactName, setArtifactName] = useState(""),
    [path, setPath] = useState(""),
    [format, setFormat] = useState("weights"),
    [base, setBase] = useState("");
  const [datasetId, setDatasetId] = useState(""),
    [artifactId, setArtifactId] = useState(""),
    [jobName, setJobName] = useState(""),
    [kind, setKind] = useState("train"),
    [epochs, setEpochs] = useState(1),
    [rate, setRate] = useState(0.0001),
    [template, setTemplate] = useState("default"),
    [port, setPort] = useState(8000),
    [parser, setParser] = useState("");
  async function refresh() {
    const data = await api("model-supply");
    setDatasets(data.datasets);
    setArtifacts(data.artifacts);
  }
  useEffect(() => {
    void refresh().catch((e) => setError(e.message));
    const timer = setInterval(() => void refresh().catch(() => {}), 4000);
    return () => clearInterval(timer);
  }, []);
  async function act(e: React.FormEvent, work: () => Promise<void>) {
    e.preventDefault();
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await work();
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section aria-label="模型供应流程">
      <h2>模型数据、产物与作业</h2>
      <p className="run-help">
        外部服务可以直接登记。自有模型按“数据集 → LLaMA-Factory 训练 → LoRA 产物
        → 合并导出 → vLLM 部署 → 模型服务”的顺序供应给岗位 Agent。
      </p>
      {error && (
        <p role="alert" className="inline-error">
          {error}
        </p>
      )}
      {message && <p role="status">{message}</p>}
      <div className="agent-grid">
        <form
          className="agent-card"
          onSubmit={(e) =>
            void act(e, async () => {
              if (!file) throw Error("请选择数据集");
              if (file.size > 10_000_000) throw Error("数据集不能超过 10 MB");
              const item = await api("datasets", {
                name,
                rows: JSON.parse(await file.text()),
              });
              setDatasetId(item.id);
              setMessage(
                "数据集已校验并固定保存；后续修改请上传为新的数据集版本。",
              );
            })
          }
        >
          <h3>保存训练数据集</h3>
          <p className="run-help">
            JSON 数组，每条包含 instruction、input（可选）和
            output。与上下文资料分别维护。
          </p>
          <label>
            数据集名称
            <input
              value={name}
              required
              maxLength={120}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <label>
            训练数据文件
            <input
              type="file"
              accept=".json"
              required
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </label>
          <button className="primary" disabled={busy}>
            校验并保存数据集
          </button>
          {datasets.map((x) => (
            <p key={x.id}>
              {x.name} · {x.count} 条 ·{" "}
              <small>摘要 {x.digest.slice(0, 12)}</small>
            </p>
          ))}
        </form>
        <form
          className="agent-card"
          onSubmit={(e) =>
            void act(e, async () => {
              const item = await api("artifacts", {
                name: artifactName,
                path,
                format,
                base_model: base,
              });
              setArtifactId(item.id);
              setMessage("本机模型产物已登记；训练和部署会引用该产物。");
            })
          }
        >
          <h3>登记已有模型产物</h3>
          <p className="run-help">
            填写后端主机上的实际目录；系统核对配置和权重文件，不自动下载模型。
          </p>
          <label>
            产物名称
            <input
              required
              value={artifactName}
              onChange={(e) => setArtifactName(e.target.value)}
            />
          </label>
          <label>
            后端模型目录
            <input
              required
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="/srv/models/my-model"
            />
          </label>
          <label>
            产物格式
            <select value={format} onChange={(e) => setFormat(e.target.value)}>
              <option value="weights">完整模型权重</option>
              <option value="adapter">LoRA 适配器</option>
            </select>
          </label>
          <label>
            基础模型目录
            <input
              value={base}
              required={format === "adapter"}
              onChange={(e) => setBase(e.target.value)}
              placeholder="完整模型可留空；LoRA 必填基础模型路径"
            />
          </label>
          <button className="primary" disabled={busy}>
            登记模型产物
          </button>
          {artifacts.map((x) => (
            <div className="agent-row" key={x.id}>
              <div>
                <strong>{x.name}</strong>
                <p>
                  {x.format === "adapter" ? "LoRA 适配器" : "完整权重"} ·{" "}
                  {x.source_job ? "平台作业生成" : "手动登记"}
                </p>
                <small>{x.path}</small>
              </div>
            </div>
          ))}
        </form>
      </div>
      <form
        className="agent-card"
        onSubmit={(e) =>
          void act(e, async () => {
            await api("jobs", {
              kind,
              name: jobName,
              request_id: requestId(),
              artifact_id: artifactId,
              dataset_id: datasetId,
              epochs,
              learning_rate: rate,
              template,
              port,
              tool_parser: parser,
            });
            setMessage(
              "模型作业已提交；日志与结果在下方查看。部署实际就绪后才会登记模型服务。",
            );
          })
        }
      >
        <h3>提交模型作业</h3>
        <div className="agent-grid">
          <label>
            作业名称
            <input
              required
              value={jobName}
              onChange={(e) => setJobName(e.target.value)}
            />
          </label>
          <label>
            模型作业类型
            <select
              value={kind}
              onChange={(e) => {
                setKind(e.target.value);
                setArtifactId("");
              }}
            >
              <option value="train">LLaMA-Factory LoRA 训练</option>
              <option value="merge">LLaMA-Factory 合并导出</option>
              <option value="deploy">vLLM 模型部署</option>
            </select>
          </label>
          <label>
            输入模型产物
            <select
              required
              value={artifactId}
              onChange={(e) => setArtifactId(e.target.value)}
            >
              <option value="">请选择</option>
              {artifacts
                .filter(
                  (x) =>
                    x.format === (kind === "merge" ? "adapter" : "weights"),
                )
                .map((x) => (
                  <option key={x.id} value={x.id}>
                    {x.name}
                  </option>
                ))}
            </select>
          </label>
          {kind === "train" && (
            <>
              <label>
                训练数据集
                <select
                  required
                  value={datasetId}
                  onChange={(e) => setDatasetId(e.target.value)}
                >
                  <option value="">请选择</option>
                  {datasets.map((x) => (
                    <option key={x.id} value={x.id}>
                      {x.name} · {x.count} 条
                    </option>
                  ))}
                </select>
              </label>
              <label>
                训练轮次
                <input
                  type="number"
                  min="0.01"
                  max="100"
                  step="0.01"
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                />
              </label>
              <label>
                学习率
                <input
                  type="number"
                  step="0.000001"
                  min="0.0000001"
                  max="0.01"
                  value={rate}
                  onChange={(e) => setRate(Number(e.target.value))}
                />
              </label>
              <label>
                模型对话模板
                <input
                  value={template}
                  required
                  onChange={(e) => setTemplate(e.target.value)}
                  placeholder="例如 qwen 或 llama3"
                />
              </label>
            </>
          )}
          {kind === "deploy" && (
            <>
              <label>
                推理服务端口
                <input
                  type="number"
                  min="1024"
                  max="65535"
                  value={port}
                  onChange={(e) => setPort(Number(e.target.value))}
                />
              </label>
              <label>
                vLLM 工具调用解析器
                <input
                  value={parser}
                  onChange={(e) => setParser(e.target.value)}
                  placeholder="按模型选择，例如 hermes；留空不启用自动工具调用"
                />
              </label>
              <p className="run-help">
                仅监听本机；部署期间持续占用一个作业槽。给 Pi
                使用前，须确认模型支持工具调用并完成岗位试运行。
              </p>
            </>
          )}
        </div>
        <button className="primary" disabled={busy}>
          提交模型作业
        </button>
      </form>
      <SupplyJobs />
    </section>
  );
}
