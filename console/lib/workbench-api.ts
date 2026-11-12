export type Rule = {
  op:
    | "contains"
    | "not_contains"
    | "equals"
    | "json_keys"
    | "number_between"
    | "model";
  component?: string;
  pointer?: string;
  expected?: unknown;
  minimum?: number;
  maximum?: number;
  criteria?: string;
  label?: string;
};
export type ContextBinding = {
  bundle_id: string;
  version: number;
  component_id: string;
};
export type SavedTask = {
  module?:string;action?:string;
  id: string;
  name: string;
  expectation: string;
  fn_id: number;
  schema: string;
  baseline: string | null;
  inputs: Record<string, unknown>;
  output_ids?: number[];
  version?: number;
  rules?: Rule[];
  requires_review?: boolean;
  secret_env?: Record<string, string>;
  context_binding?: ContextBinding | null;
  context_input_ids?: string[];
  case_kind?: string;
};
export type NativeRun = {
  id: string;
  session: string;
  action: string;
  status: string;
  started_at: number;
  execution_started_at?: number;
  finished_at: number | null;
  config_version: string;
  schema: string;
  fn_id: number;
  error?: string;
  inputs?: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  context?: {
    id: string;
    bundle_id: string;
    bundle_version: number;
    used_chars: number;
    max_chars: number;
    content_sha256: string;
    resources: { id: string; name: string; version: number }[];
  };
  code?: { git_commit: string | null; source_sha256: string };
  evaluation?: {
    verdict: string;
    task_version?: number;
    checks: {
      rule: number;
      label: string;
      verdict: string;
      reason?: string;
      score?: number;
    }[];
  };
  acceptance?: { passed: boolean; reason: string };
  artifacts?: {
    id: string;
    name: string;
    size: number;
    sha256: string;
    url: string;
  }[];
  usage?: {
    model: string;
    input_tokens: number;
    output_tokens: number;
    cost: number | null;
    currency: string | null;
  }[];
};
export const runStatus: Record<string, string> = {
  queued: "排队中",
  running: "执行中",
  awaiting_review: "待人工放行",
  completed: "执行完成",
  failed: "执行失败",
  interrupted: "服务重启，执行中断",
  cancelled: "已取消（未执行）",
  stopped: "执行返回（曾请求停止）",
  rejected: "未放行",
};
export const verdictStatus: Record<string, string> = {
  passed: "规则通过",
  failed: "规则未通过",
  blocked: "无法判定",
  ungraded: "尚未评分",
};
export async function request(path: string, body?: unknown) {
  const response = await fetch(
    "/api/native/" + path,
    body === undefined
      ? { cache: "no-store" }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const result = await response.json();
  if (!response.ok) throw Object.assign(Error(result.error || "操作失败"),{status:response.status});
  return result;
}
export function downloadJSON(name: string, value: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function downloadCSV(name: string, rows: unknown[][]) {
  const cell = (value: unknown) => {
    let text = value == null ? "" : String(value);
    if (/^[=+\-@\t\r]/.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  };
  const url = URL.createObjectURL(
    new Blob(
      ["\ufeff" + rows.map((row) => row.map(cell).join(",")).join("\r\n")],
      { type: "text/csv;charset=utf-8" },
    ),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
