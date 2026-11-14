import { test, expect } from "@playwright/test";

test("主导航、文件导入、固定训练数据与真实环境状态", async ({ page }) => {
  const suffix = Date.now().toString();
  await page.goto("/");
  await expect(page.getByText("业务处理与验证", {exact:true})).toHaveCount(0);
  await page.locator("aside").getByRole("link", { name: "上下文供应", exact: true }).click();
  await page.getByRole("button", {name:"导入文件",exact:true}).click();
  await page.getByLabel("资料文件", { exact: true }).setInputFiles({
    name: "审批规定.md",
    mimeType: "text/markdown",
    buffer: Buffer.from("财务审批制度：超过5000元需经理审批。"),
  });
  await page
    .getByLabel("导入资源名称", { exact: true })
    .fill("导入制度" + suffix);
  await page.getByRole("button", { name: "提交资料导入", exact: true }).click();
  const row = page.locator("[data-supply-job-state]");
  await expect(row).toHaveAttribute("data-supply-job-state", "completed", {
    timeout: 20000,
  });
  await page.getByRole("link",{name:"查看生成的资料",exact:true}).click();
  await expect(page.locator(".context-preview")).toContainText("5000");
  await page.locator("aside").getByRole("link", { name: "模型供应", exact: true }).click();
  await page.locator("aside").getByRole("link", {name:"训练数据集",exact:true}).click();
  await page.getByRole("button",{name:"导入训练数据集",exact:true}).click();
  await page
    .getByLabel("数据集名称", { exact: true })
    .fill("训练数据" + suffix);
  // A real payload above the old 2 MB proxy limit.
  const rows = Array.from({ length: 220 }, () => ({
    instruction: "审核这笔报销",
    input: "",
    output: "依据企业制度核对。".repeat(500),
  }));
  await page.getByLabel("训练数据文件", { exact: true }).setInputFiles({
    name: "samples.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(rows)),
  });
  await page
    .getByRole("button", { name: "校验并保存数据集", exact: true })
    .click();
  await expect(page.getByRole("status")).toContainText(
    "数据集已校验并固定保存",
  );
  await expect(
    page.locator(".resource-table tr").filter({ hasText: "训练数据" + suffix }),
  ).toContainText("220 条");
  await page.locator("aside").getByRole("link", { name: "运行环境", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "本机运行环境", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".environment-list")).toContainText("sqlite");
  await expect(page.locator(".environment-list")).toContainText("pypdf");
  await page.locator("aside").getByRole("link", { name: "作业管理", exact: true }).click();
  await expect(
    page.locator(".resource-table tr").filter({ hasText: "导入制度" + suffix }),
  ).toContainText("已完成");
  for (const width of [1440, 1280]) {
    await page.setViewportSize({ width, height: 900 });
    for (const name of ["上下文供应", "模型供应", "作业管理", "运行环境"]) {
      await page.locator("aside").getByRole("link", { name, exact: true }).click();
      await expect
        .poll(() =>
          page.evaluate(() => (window as any).__verify.current().verdict),
        )
        .toBe("PASS");
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 1,
        ),
      ).toBe(true);
    }
  }
});
