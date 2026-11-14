"use client";
import { useEffect, useState } from "react";
import { platformRequest as api, requestId } from "../lib/platform-api";
import { WorkspaceLink, useWorkspace } from "../lib/workspace-navigation";
type Dataset = { id: string; name: string; count: number; digest: string };
type Artifact = {
  id: string;
  name: string;
  format: string;
  path: string;
  source_job: string | null;
};
export default function ModelSupply({ view = "datasets" }: { view?: "datasets" | "artifacts" }) {
  const { navigate, route } = useWorkspace();
  const [editor, setEditor] = useState<"dataset" | "artifact" | "job" | null>(null), [query, setQuery] = useState("");
  const [loaded, setLoaded] = useState(false);
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
    [stage, setStage] = useState("sft"),
    [artifactId, setArtifactId] = useState(""),
    [jobName, setJobName] = useState(""),
    [kind, setKind] = useState("train"),
    [epochs, setEpochs] = useState(1),
    [cutoff, setCutoff] = useState(512),
    [rate, setRate] = useState(0.0001),
    [template, setTemplate] = useState("default"),
    [port, setPort] = useState(8000),
    [parser, setParser] = useState("");
  async function refresh() {
    const data = await api("model-supply");
    setDatasets(data.datasets);
    setArtifacts(data.artifacts);
    setLoaded(true);
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
  const selectedResource = route.record ? (view === "datasets" ? datasets : artifacts).find((item) => item.id === route.record) : undefined;
  function operation(kind: string, artifact: string, dataset = "") { setKind(kind);setArtifactId(artifact);setDatasetId(dataset);setJobName("");setEditor("job"); }
  return (
    <section aria-label="模型供应流程" data-verify-unit="ModelSupply" data-verify-state={loaded ? "ready" : "loading"}>
      <div className="collection-heading"><div><h2>{editor === "job" ? "提交模型作业" : selectedResource?.name || (view === "datasets" ? "数据集列表" : "模型产物列表")}</h2><p>{view === "datasets" ? "维护训练所用的数据快照。" : "管理已有权重和平台作业生成的模型。"}</p></div>
        {editor || route.record ? <button className="text-button" disabled={busy} onClick={() => { if (editor) setEditor(null);else navigate(-3, view); }}>返回{view === "datasets" ? "数据集列表" : "模型产物"}</button> : <div className="agent-actions"><WorkspaceLink area={-4}>查看作业</WorkspaceLink><button className="primary" onClick={() => { setName("");setFile(null);setArtifactName("");setPath("");setBase("");setEditor(view === "datasets" ? "dataset" : "artifact");setMessage(""); }}>{view === "datasets" ? "导入训练数据集" : "登记模型产物"}</button></div>}
      </div>
      {error && (
        <p role="alert" className="inline-error">
          {error}
        </p>
      )}
      {message && <p role="status">{message}</p>}
      {!editor && route.record && <div className="agent-card">{selectedResource ? <><dl className="environment-list"><dt>名称</dt><dd>{selectedResource.name}</dd>{"count" in selectedResource ? <><dt>记录数</dt><dd>{selectedResource.count}</dd><dt>内容摘要</dt><dd><code>{selectedResource.digest}</code></dd></> : <><dt>格式</dt><dd>{selectedResource.format === "adapter" ? "LoRA 适配器" : "完整权重"}</dd><dt>模型目录</dt><dd>{selectedResource.path}</dd><dt>来源</dt><dd>{selectedResource.source_job ? <WorkspaceLink area={-4} record={selectedResource.source_job} className="resource-link">查看来源作业</WorkspaceLink> : "手动登记"}</dd></>}</dl><div className="agent-actions">{"count" in selectedResource ? <button className="primary" onClick={() => operation("train", "", selectedResource.id)}>提交训练</button> : selectedResource.format === "adapter" ? <button className="primary" onClick={() => operation("merge", selectedResource.id)}>合并模型</button> : <><button className="primary" onClick={() => operation("train", selectedResource.id)}>提交训练</button><button className="secondary" onClick={() => operation("deploy", selectedResource.id)}>部署模型</button></>}</div></> : <p>当前列表中未找到该资源，请返回列表重新选择。</p>}</div>}
      {!editor && !route.record && <>
        {(view === "datasets" ? datasets.length : artifacts.length) > 0 && <input className="collection-search" aria-label="搜索模型资源" placeholder="按名称搜索" value={query} onChange={(e) => setQuery(e.target.value)} />}
        {view === "datasets" ? (!datasets.length ? <div className="collection-empty"><h3>尚无训练数据集</h3><p>导入一份 CPT、SFT 或 DPO 数据，校验后可用于训练。</p></div> : <div className="resource-table-scroll"><table className="resource-table"><thead><tr><th>数据集</th><th>记录数</th><th>内容摘要</th><th>操作</th></tr></thead><tbody>{datasets.filter((item) => item.name.includes(query)).map((item) => <tr key={item.id}><td><WorkspaceLink area={-3} view="datasets" record={item.id} className="resource-link">{item.name}</WorkspaceLink></td><td>{item.count} 条</td><td><code>{item.digest.slice(0, 12)}</code></td><td><button className="resource-link" onClick={() => { setDatasetId(item.id);setKind("train");setArtifactId("");setEditor("job"); }}>提交训练</button></td></tr>)}</tbody></table></div>)
        : (!artifacts.length ? <div className="collection-empty"><h3>尚无模型产物</h3><p>登记本机已有权重，或在训练完成后查看生成的产物。</p></div> : <div className="resource-table-scroll"><table className="resource-table"><thead><tr><th>产物名称</th><th>格式</th><th>来源</th><th>操作</th></tr></thead><tbody>{artifacts.filter((item) => item.name.includes(query)).map((item) => <tr key={item.id}><td><WorkspaceLink area={-3} view="artifacts" record={item.id} className="resource-link">{item.name}</WorkspaceLink><small className="table-secondary">{item.path}</small></td><td>{item.format === "adapter" ? "LoRA 适配器" : "完整权重"}</td><td>{item.source_job ? "平台作业" : "手动登记"}</td><td><div className="agent-actions">{(item.format === "adapter" ? ["merge"] : ["train", "deploy"]).map((operation) => <button className="resource-link" key={operation} onClick={() => { setArtifactId(item.id);setKind(operation);setEditor("job"); }}>{({train:"训练",merge:"合并",deploy:"部署"})[operation]}</button>)}</div></td></tr>)}</tbody></table></div>)}
      </>}
      <div className="editor-page">
        {editor === "dataset" &&
        <form
          className="agent-card"
          onSubmit={(e) =>
            void act(e, async () => {
              if (!file) throw Error("请选择数据集");
              if (file.size > 10_000_000) throw Error("数据集不能超过 10 MB");
              const item = await api("datasets", {
                name,
                rows: JSON.parse(await file.text()),
                stage,
              });
              setDatasetId(item.id);
              setEditor(null);
              setMessage(
                "数据集已校验并固定保存；后续修改请上传为新的数据集版本。",
              );
            })
          }
        >
          <h3>保存训练数据集</h3>
          <p className="run-help">
            CPT 使用 text；SFT 使用 instruction、input（可选）、output；DPO 使用
            instruction、chosen、rejected。数据文件为 JSON
            数组，训练阶段须与数据格式一致。
          </p>
          <label>
            训练阶段
            <select
              aria-label="训练阶段"
              value={stage}
              onChange={(e) => setStage(e.target.value)}
            >
              <option value="pt">CPT 语料训练</option>
              <option value="sft">SFT 指令训练</option>
              <option value="dpo">DPO 偏好训练</option>
            </select>
          </label>
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
        </form>
        }
        {editor === "artifact" &&
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
              setEditor(null);
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
        </form>
        }
      </div>
      {editor === "job" &&
      <form
        className="agent-card editor-page"
        onSubmit={(e) =>
          void act(e, async () => {
            const job = await api("model-supply/" + kind, {
              kind,
              name: jobName,
              request_id: requestId(),
              artifact_id: artifactId,
              cutoff_len: cutoff,
              stage,
              dataset_id: datasetId,
              epochs,
              learning_rate: rate,
              template,
              port,
              tool_parser: parser,
            });
            setMessage(
              "模型作业已提交。",
            );
            navigate(-4, "jobs", job.id);
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
              <option value="train">LLaMA-Factory CPT / SFT / DPO 训练</option>
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
                训练任务阶段
                <select
                  aria-label="训练任务阶段"
                  value={stage}
                  onChange={(e) => setStage(e.target.value)}
                >
                  <option value="pt">CPT</option>
                  <option value="sft">SFT</option>
                  <option value="dpo">DPO</option>
                </select>
              </label>
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
                训练截断长度
                <input
                  aria-label="训练截断长度"
                  type="number"
                  min={32}
                  max={8192}
                  value={cutoff}
                  onChange={(e) => setCutoff(Number(e.target.value))}
                />
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
      }
    </section>
  );
}
