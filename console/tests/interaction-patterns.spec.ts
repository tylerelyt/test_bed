import { test, expect } from "@playwright/test";

test("列表先展示，新建时只打开一个表单，取消后回到列表", async ({page}) => {
  await page.goto("/?area=models&view=services");
  await expect(page.getByRole("heading",{name:"模型服务",exact:true,level:1})).toBeVisible();
  await expect(page.locator("form")).toHaveCount(0);
  await page.getByRole("button",{name:"接入模型服务",exact:true}).click();
  await expect(page.locator("form")).toHaveCount(1);
  await page.getByLabel("服务名称",{exact:true}).fill("尚未提交的服务");
  await page.getByRole("button",{name:"取消",exact:true}).click();
  await expect(page.locator("form")).toHaveCount(0);
  for(const name of ["训练数据集","模型产物"]){
    await page.locator("aside").getByRole("link",{name,exact:true}).click();
    await expect(page.locator("form")).toHaveCount(0);
  }
  await page.locator("aside").getByRole("link",{name:"上下文供应",exact:true}).click();
  await expect(page.locator("form")).toHaveCount(0);
  await expect(page.getByRole("button",{name:"新建资源",exact:true})).toBeVisible();
  await page.locator("aside").getByRole("link",{name:"供应配置",exact:true}).click();
  await expect(page.getByRole("button",{name:"新建供应配置",exact:true})).toBeVisible();
});

test("地址、前进后退和资源详情保持一致", async ({page}) => {
  await page.goto("/?area=context&view=resources");
  const name = "交互检查资料" + Date.now();
  await page.getByRole("button",{name:"新建资源",exact:true}).click();
  await page.getByLabel("资源名称",{exact:true}).fill(name);
  await page.getByLabel("内容",{exact:true}).fill("超过5000元需经理审批。");
  await page.getByRole("button",{name:"保存资源",exact:true}).click();
  await expect(page).toHaveURL(/record=/);
  await expect(page.locator(".context-preview")).toContainText("5000");
  const detail = page.url();
  await page.locator("aside").getByRole("link",{name:"模型供应",exact:true}).click();
  await expect(page).toHaveURL(/area=models/);
  await page.goBack();
  await expect(page).toHaveURL(detail);
  await expect(page.locator(".context-preview")).toContainText("5000");
  await page.reload();
  await expect(page.locator(".context-preview")).toContainText("5000");
  await page.goForward();
  await expect(page).toHaveURL(/area=models/);
});

test("已有岗位可继续新增 Agent，且必须明确选择所属岗位", async ({page}) => {
  const suffix = Date.now().toString();
  const position = await (await page.request.post('/api/platform/positions',{data:{name:'已有岗位'+suffix,duties:'复用既有岗位职责'}})).json();
  await page.goto('/?area=agents&view=agents');
  for(const name of ['第一位Agent'+suffix,'第二位Agent'+suffix]){
    await page.getByRole('button',{name:'新增 Agent',exact:true}).click();
    await expect(page.getByLabel('所属岗位',{exact:true})).toHaveValue('');
    await expect(page.getByRole('button',{name:'保存 Agent',exact:true})).toBeDisabled();
    await page.getByLabel('所属岗位',{exact:true}).selectOption(position.id);
    await page.getByLabel('Agent 名称',{exact:true}).fill(name);
    await page.getByRole('button',{name:'保存 Agent',exact:true}).click();
    await expect(page.getByRole('status')).toContainText('岗位 Agent 已创建');
    await page.locator('aside').getByRole('link',{name:'岗位与 Agent',exact:true}).click();
    await expect(page.locator('.resource-table tr').filter({hasText:name})).toContainText('已有岗位'+suffix);
  }
});
