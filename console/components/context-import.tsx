"use client";
import { useState } from "react";
import { platformRequest as api, requestId } from "../lib/platform-api";
import SupplyJobs from "./supply-jobs";
export default function ContextImport({ showJobs = true, onSubmitted }: { showJobs?: boolean; onSubmitted?: (id: string) => void }) {
  const [file, setFile] = useState<File | null>(null),
    [name, setName] = useState(""),
    [busy, setBusy] = useState(false),
    [message, setMessage] = useState(""),
    [error, setError] = useState("");
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      if (file.size > 5_000_000) throw Error("文件不能超过 5 MB");
      const bytes = new Uint8Array(await file.arrayBuffer());
      let raw = "";
      for (let i = 0; i < bytes.length; i += 8192)
        raw += String.fromCharCode(...bytes.subarray(i, i + 8192));
      const job = await api("context-supply/import", {
        kind: "import",
        name: name || file.name,
        request_id: requestId(),
        filename: file.name,
        content_base64: btoa(raw),
      });
      onSubmitted?.(job.id);
      setMessage("导入作业已提交。完成后，刷新下方资源库即可选用新资料。");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <form className="agent-card" onSubmit={submit}>
        <h2>导入企业资料</h2>
        <p className="run-help">
          上传后保留原文件与摘要，提取文本形成一个可供岗位引用的资源版本。支持
          UTF-8 文本、Markdown、CSV、JSON、PDF 和 DOCX；扫描件需先做 OCR。
        </p>
        <label>
          资料文件
          <input
            type="file"
            accept=".txt,.md,.csv,.json,.pdf,.docx"
            required
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
        </label>
        <label>
          导入资源名称
          <input
            value={name}
            maxLength={120}
            placeholder="留空使用文件名"
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <button className="primary" disabled={busy || !file}>
          提交资料导入
        </button>
        {error && (
          <p role="alert" className="inline-error">
            {error}
          </p>
        )}
        {message && <p role="status">{message}</p>}
      </form>
      {showJobs && <SupplyJobs kind="import" />}
    </>
  );
}
