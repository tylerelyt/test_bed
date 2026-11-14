import { test, expect } from "@playwright/test";
test("检索归上下文供应，运行历史不混入作业管理", async ({ page }) => {
  await page.goto("/");
  await page.locator("aside").getByRole("link", { name: "上下文供应", exact: true }).click();
  await page.locator("aside .nav-tools > summary").click();
  await expect(
    page.locator("aside").getByRole("link", { name: "在线召回排序", exact: true }),
  ).toBeVisible();
  await page.locator("aside").getByRole("link", { name: "在线召回排序", exact: true }).click();
  await expect(
    page.locator('[data-verify-unit="NativeWorkbench"]'),
  ).toHaveAttribute("data-verify-state", "ready");
  const started = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/native/runs") && r.request().method() === "POST",
  );
  await page.getByRole("button", { name: "执行检索", exact: true }).click();
  const id = (await (await started).json()).run_id;
  await expect(page.locator(".native-toolbar")).toContainText("执行完成");
  await page.locator("aside").getByRole("link", { name: "作业管理", exact: true }).click();
  const jobs = await (await page.request.get("/api/platform/jobs")).json();
  expect(jobs.every((x: any) => ["import", "train", "merge", "deploy"].includes(x.kind))).toBe(true);
  expect(jobs.find((x: any) => x.id === "business:" + id)).toBeUndefined();
  await expect(page.locator(".resource-table tr").filter({ hasText: "执行检索" })).toHaveCount(0);
  expect((await page.request.get("/api/platform/jobs/business:" + id)).status()).toBe(400);
  expect((await page.request.post("/api/platform/jobs/business:" + id + "/cancel", {data:{}})).status()).toBe(400);
  const history = await (await page.request.get("/api/native/runs/" + id)).json();
  expect(history.status).toBe("completed");
  await expect(
    page.getByRole("button", { name: "发布输出为上下文资源" }),
  ).toHaveCount(0);
  const rejected = await page.request.post(
    "/api/platform/business-runs/" + id + "/publish",
    { data: { name: "不允许发布" } },
  );
  expect(rejected.status()).toBe(410);
  await page.locator("aside").getByRole("link", { name: "上下文供应", exact: true }).click();
  await page.locator("aside .nav-tools > summary").click();
  await page.locator("aside").getByRole("link", {name:"在线召回排序",exact:true}).click();
  await page.locator(".run-history > summary").click();
  await expect(page.locator('#recorded-run option[value="' + id + '"]')).toBeAttached();
});
