export async function platformRequest(path: string, body?: unknown) {
  const response = await fetch(
    "/api/platform/" + path,
    body === undefined
      ? { cache: "no-store" }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const result = await response.json();
  if (!response.ok) throw Error(result.error || "操作未完成");
  return result;
}
export function requestId() {
  return crypto.randomUUID().replaceAll("-", "");
}
