import { test, expect } from "@playwright/test";

test("岗位 Agent 的预配、原生 Pi 试运行、采用和刷新", async ({ page }) => {
  const suffix = Date.now().toString();
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "岗位 Agent", exact: true, level: 1 }),
  ).toBeVisible();
  await expect(page.locator(".position-agents")).toBeVisible();
  await expect(
    page.locator('[data-verify-unit="NativeWorkbench"]'),
  ).toHaveCount(0);
  await page.getByLabel("岗位名称", { exact: true }).fill("采购审核" + suffix);
  await page
    .getByLabel("岗位职责", { exact: true })
    .fill("按照企业采购规定审核金额，引用资料来源");
  await page.getByRole("button", { name: "保存岗位", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("岗位已创建");
  await page
    .getByLabel("Agent 名称", { exact: true })
    .fill("采购助理" + suffix);
  await page.getByRole("button", { name: "保存 Agent", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("岗位 Agent 已创建");
  const resource = await (
    await page.request.post("/api/native/context", {
      data: {
        action: "resource",
        name: "采购制度" + suffix,
        kind: "knowledge",
        content: "采购超过10000元必须取得部门经理批准。",
        source: "采购制度2026",
      },
    })
  ).json();
  const bundle = await (
    await page.request.post("/api/native/context", {
      data: {
        action: "bundle",
        name: "采购上下文" + suffix,
        resources: [{ id: resource.id, required: true }],
      },
    })
  ).json();
  expect(bundle.id).toBeTruthy();
  await page.getByRole("button", { name: "模型供应", exact: true }).click();
  await page
    .getByLabel("服务名称", { exact: true })
    .fill("本地协议测试" + suffix);
  await page
    .getByLabel("服务地址", { exact: true })
    .fill("http://127.0.0.1:8779/v1");
  await page.getByLabel("模型标识", { exact: true }).fill("test-fixture");
  await page
    .getByLabel("服务密钥（可选）", { exact: true })
    .fill("browser-fixture-model-secret");
  await page.getByRole("button", { name: "保存模型服务", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("模型服务配置已保存");
  await expect(
    page.getByLabel("服务密钥（可选）", { exact: true }),
  ).toHaveValue("");
  await page.getByRole("button", { name: "岗位 Agent", exact: true }).click();
  await page.getByRole("tab", { name: "发布与运行", exact: true }).click();
  await page
    .getByLabel("预配的上下文", { exact: true })
    .selectOption(bundle.id);
  await page.getByRole("button", { name: "生成候选版本", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("候选版本已生成");
  await page
    .getByLabel("岗位任务", { exact: true })
    .fill("12000元采购应按什么规定处理？");
  await page.getByRole("button", { name: "运行任务", exact: true }).click();
  await expect(page.locator("[data-agent-run-state]")).toHaveAttribute(
    "data-agent-run-state",
    "completed",
    { timeout: 45000 },
  );
  await expect(page.locator(".agent-result")).toContainText("10000");
  await expect(page.locator(".agent-result")).toContainText("采购制度2026");
  await expect(page.locator(".agent-result")).toContainText(
    "已调用岗位上下文读取工具",
  );
  await expect(page.locator(".agent-result")).toContainText("原生会话已保存");
  expect(await page.content()).not.toContain("browser-fixture-model-secret");
  const downloaded = page.waitForEvent("download");
  await page.getByRole("link", { name: "下载任务结果", exact: true }).click();
  expect((await downloaded).suggestedFilename()).toBe("岗位任务结果.txt");
  await page
    .getByRole("button", { name: "核对结果后采用此版本", exact: true })
    .click();
  await expect(page.getByRole("status")).toContainText("已采用该 Agent 版本");
  await page.reload();
  await expect(
    page.locator(".agent-row").filter({ hasText: "采购助理" + suffix }),
  ).toContainText("已有采用版本");
  await page.setViewportSize({ width: 390, height: 844 });
  for (const name of ["岗位与 Agent", "发布与运行", "桌面 Agent"]) {
    await page.getByRole("tab", { name, exact: true }).click();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    ).toBe(true);
  }
});
