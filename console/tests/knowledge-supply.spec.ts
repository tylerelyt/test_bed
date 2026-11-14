import { test, expect } from "@playwright/test";
test("从原文档服务到固定索引和岗位引用", async ({ page }) => {
  const suffix = Date.now().toString();
  await page.goto("/");
  await page.locator("aside").getByRole("link", { name: "上下文供应", exact: true }).click();
  await page.getByRole("link", {name:"接入业务资料",exact:true}).click();
  await page
    .getByRole("button", { name: "读取原文档目录", exact: true })
    .click();
  await page
    .getByLabel("原文档", { exact: true })
    .selectOption("fixture-policy");
  await page
    .getByLabel("资料快照名称", { exact: true })
    .fill("原服务制度" + suffix);
  await page.getByRole("button", { name: "保存文档快照", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("已保存原文资料快照");
  await page.locator("aside").getByRole("link", {name:"检索索引",exact:true}).click();
  await page.getByRole("button",{name:"新建索引",exact:true}).click();
  await page.getByLabel("索引名称", { exact: true }).fill("财务索引" + suffix);
  await page
    .getByRole("checkbox", {
      name: "原服务制度" + suffix + " · v1",
      exact: true,
    })
    .check();
  await page.getByRole("button", { name: "构建资料索引", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("索引已构建");
  await page.getByLabel("检索问题", { exact: true }).fill("报销审批");
  await page.getByRole("button", { name: "检查检索结果", exact: true }).click();
  await expect(page.locator(".evidence-results")).toContainText("5000");
  await expect(page.locator(".evidence-results")).toContainText(
    "legacy-index/document/fixture-policy",
  );
  await page
    .getByRole("button", { name: "生成岗位检索引用", exact: true })
    .click();
  await expect(page.getByRole("status")).toContainText("已生成检索资源");
  const context = await (
    await page.request.get("/api/platform/context")
  ).json();
  const ref = context.resources.find(
    (x: any) => x.name === "财务索引" + suffix + " · 岗位检索",
  );
  expect(ref.kind).toBe("retrieval");
  const bundle = await (
    await page.request.post("/api/platform/context", {
      data: {
        action: "bundle",
        name: "财务配置" + suffix,
        resources: [{ id: ref.id, required: true }],
      },
    })
  ).json();
  expect(bundle.resources[0].id).toBe(ref.id);
  await page.setViewportSize({ width: 1280, height: 900 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
});
