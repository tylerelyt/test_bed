import { test, expect } from "@playwright/test";
test("原业务运行进入统一作业并发布为岗位上下文", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "在线召回排序", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "在线召回排序", exact: true }).click();
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
  await page.getByRole("button", { name: "作业管理", exact: true }).click();
  const jobs = await (await page.request.get("/api/platform/jobs")).json();
  expect(jobs.find((x: any) => x.id === "business:" + id)).toMatchObject({
    kind: "business",
    status: "completed",
  });
  const row = page
    .locator(".agent-row")
    .filter({ hasText: "执行检索" })
    .first();
  await row.getByRole("button", { name: "查看详情与日志" }).click();
  await expect(page.locator(".context-preview")).toContainText(id);
  await page.getByRole("button", { name: "发布输出为上下文资源" }).click();
  await expect(page.getByRole("status")).toContainText("已发布到上下文资源库");
  await page.getByRole("button", { name: "上下文供应", exact: true }).click();
  await expect(
    page
      .locator("#context-resource option")
      .filter({ hasText: "执行检索 · 运行结果" }),
  ).toHaveCount(1);
});
