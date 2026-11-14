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
    await expect(page.getByRole('status')).toHaveCount(0);
    await expect(page.locator('.resource-table tr').filter({hasText:name})).toContainText('已有岗位'+suffix);
  }
});

test("层级归属与 Agent 补充上下文后的返回路径", async ({page}) => {
  const suffix = Date.now().toString();
  const post = async (path: string, data: unknown) => {
    const response = await page.request.post('/api/platform/'+path,{data});
    expect(response.ok()).toBe(true);
    return response.json();
  };
  const position = await post('positions',{name:'财务岗位'+suffix,duties:'按企业制度检查报销'});
  const agent = await post('agents',{name:'财务助理'+suffix,position_id:position.id});
  const model = await post('models',{name:'验证服务'+suffix,model:'test-fixture',base_url:'http://127.0.0.1:8779/v1'});
  const resource = await post('context',{action:'resource',name:'报销规范'+suffix,kind:'knowledge',content:'报销超过5000元需要审批。',source:'企业制度'});
  await page.goto('/?area=agents&view=agents&record='+agent.id);
  await expect(page.getByRole('heading',{level:1})).toHaveText(agent.name);
  await expect(page.locator('aside').getByRole('link',{name:'发布与运行',exact:true})).toHaveCount(0);
  await page.getByLabel('使用的模型服务',{exact:true}).selectOption(model.id);
  await page.getByRole('link',{name:'创建或查看上下文配置',exact:true}).click();
  await expect(page).toHaveURL(/returnTo=/);
  const group = page.locator('aside [data-navigation-area="-2"]');
  await expect(group.getByRole('link',{name:'供应配置',exact:true})).toHaveAttribute('aria-current','page');
  await expect(page.locator('aside [aria-current="page"]')).toHaveCount(1);
  await expect(page.locator('aside').getByRole('link',{name:'接入业务资料',exact:true})).toHaveCount(0);
  await page.getByRole('button',{name:'新建供应配置',exact:true}).click();
  await page.getByLabel('配置名称',{exact:true}).fill('财务上下文'+suffix);
  await page.getByRole('checkbox',{name:'报销规范'+suffix,exact:false}).check();
  await page.getByRole('button',{name:'保存供应配置',exact:true}).click();
  await expect(page.getByRole('status')).toContainText('供应配置已保存');
  await expect(page.locator('.bundle-members')).toContainText('报销规范'+suffix);
  await post('context',{action:'resource',resource_id:resource.id,expected_version:1,name:'更新后的报销规范'+suffix,kind:'knowledge',content:'新的资料版本。',source:'企业制度'});
  await page.reload();
  await expect(page.locator('.bundle-members')).toContainText('报销规范'+suffix);
  await expect(page.locator('.bundle-members')).not.toContainText('更新后的');
  await page.getByRole('button',{name:'返回「'+agent.name+'」配置',exact:true}).click();
  await expect(page.getByRole('heading',{level:1})).toHaveText(agent.name);
  await expect(page.getByLabel('使用的模型服务',{exact:true})).toHaveValue(model.id);
  await expect(page.getByLabel('预配的上下文',{exact:true})).toHaveValue('');
  await page.getByLabel('预配的上下文',{exact:true}).selectOption({label:'财务上下文'+suffix+' · v1'});
  await page.getByRole('button',{name:'生成候选版本',exact:true}).click();
  await expect(page.getByRole('tab',{name:'试运行',exact:true})).toHaveAttribute('aria-selected','true');
  await expect(page).toHaveURL(/record=/);
  const trialUrl = page.url();
  await page.getByRole('tab',{name:'配置与版本',exact:true}).click();
  await expect(page.locator('.resource-table')).toContainText('财务上下文'+suffix);
  await page.goBack();
  await expect(page).toHaveURL(trialUrl);
  await expect(page.getByLabel('Agent 版本',{exact:true})).toHaveValue(/.+/);
  await expect(page.locator('.breadcrumbs')).toContainText(agent.name);
  await page.goto('/?area=agents&view=runs&record='+agent.id);
  await expect(page).toHaveURL(/view=agents/);
  await expect(page.getByRole('heading',{level:1})).toHaveText(agent.name);
});
