import { test, expect } from "@playwright/test";

test("供应能力归属与多模态功能拆分", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("nav [data-platform-index]")).toHaveCount(5);
  await expect(page.getByText("业务处理与验证", {exact:true})).toHaveCount(0);
  await expect(page.getByRole("tab", {name:"模型服务",exact:true})).toHaveCount(0);
  await page.getByRole("button", {name:"上下文供应",exact:true}).click();
  await expect(page.getByRole("group", {name:"上下文供应能力"})).toContainText("离线索引构建");
  await expect(page.getByRole("button", {name:"训练与评测",exact:true})).toHaveCount(0);
  await page.getByRole("button", {name:"图片资料",exact:true}).click();
  for (const name of ["图片上传","图搜图","文搜图","图片管理"])
    await expect(page.getByRole("tab",{name,exact:true})).toBeVisible();
  await expect(page.getByRole("tab",{name:"GUI-Agent",exact:true})).toHaveCount(0);
  await expect(page.getByRole("tab",{name:"图像生成",exact:true})).toHaveCount(0);
  const imageStarted = page.waitForResponse((response) => response.url().endsWith("/api/native/runs") && response.request().method() === "POST");
  await page.getByRole("button", {name:"查看图片上传状态",exact:true}).click();
  const imageRun = (await (await imageStarted).json()).run_id;
  await expect(page.locator(".native-toolbar")).toContainText("执行完成");
  await page.getByRole("button", {name:"模型供应",exact:true}).click();
  await expect(page.getByRole("button", {name:"在线召回排序",exact:true})).toHaveCount(0);
  await expect(page.getByRole("button", {name:"训练与评测",exact:true})).toBeVisible();
  await page.getByRole("button", {name:"图像生成",exact:true}).click();
  await expect(page.getByRole("tab",{name:"图像生成",exact:true})).toBeVisible();
  await expect(page.getByRole("tab",{name:"图片上传",exact:true})).toHaveCount(0);
  await page.getByRole("button",{name:"查看图像生成状态",exact:true}).click();
  await expect(page.locator(".native-toolbar")).toContainText("执行完成");
  await page.locator(".run-history > summary").click();
  await expect(page.locator("#recorded-run option")).toHaveCount(2);
  await expect(page.locator('#recorded-run option[value="' + imageRun + '"]')).toHaveCount(0);
  await page.getByRole("button", {name:"岗位 Agent",exact:true}).click();
  await page.getByRole("tab", {name:"桌面 Agent",exact:true}).click();
  await expect(page.getByRole("tab",{name:"GUI-Agent",exact:true})).toBeVisible();
  await expect(page.getByRole("tab",{name:"图片上传",exact:true})).toHaveCount(0);
  await page.getByRole("button", {name:"运行环境",exact:true}).click();
  await page.getByRole("button", {name:"系统监控",exact:true}).click();
  await expect(page.locator('[data-verify-unit="NativeWorkbench"]')).toHaveAttribute("data-verify-module","7");
});

test("刷新旧入口位置后进入对应供应能力", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => sessionStorage.setItem("workbench-selected-module", "6"));
  await page.reload();
  await expect(page.getByRole("button",{name:"模型供应",exact:true})).toHaveAttribute("aria-pressed","true");
  await expect(page.getByRole("button",{name:"训练与评测",exact:true})).toHaveAttribute("aria-pressed","true");
  await expect(page.locator('[data-verify-unit="NativeWorkbench"]')).toHaveAttribute("data-verify-module","6");
});
