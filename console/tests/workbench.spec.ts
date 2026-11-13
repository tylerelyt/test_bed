import { test, expect, Page } from "@playwright/test";

async function open(page: Page) {
  await page.goto("/");
  await page.locator(".legacy-navigation > summary").click();
  await page.getByRole("button", {name:"离线索引构建",exact:true}).click();
  await expect(
    page.locator('[data-verify-unit="NativeWorkbench"]'),
  ).toHaveAttribute("data-verify-state", "ready");
}
async function history(page: Page) {
  const panel = page.locator(".run-history");
  if ((await panel.getAttribute("open")) === null)
    await panel.locator(":scope > summary").click();
  return panel;
}
async function search(page: Page) {
  await page.getByRole("button", { name: "在线召回排序", exact: true }).click();
  const response = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/native/runs") && r.request().method() === "POST",
  );
  await page.getByRole("button", { name: "执行检索", exact: true }).click();
  const result = await (await response).json();
  await expect(page.locator(".native-toolbar")).toContainText("执行完成");
  return result.run_id as string;
}
async function selectRun(page: Page, id: string) {
  await history(page);
  await page.getByRole("tab", { name: "运行证据", exact: true }).click();
  await expect(
    page.locator('#recorded-run option[value="' + id + '"]').first(),
  ).toBeAttached();
  await page.locator("#recorded-run").selectOption(id);
  await expect(page.locator(".run-detail[data-verify-state]")).toHaveAttribute(
    "data-verify-state",
    "completed",
  );
}
async function saveTask(page: Page, name: string) {
  await page.getByText("从此运行保存任务", { exact: true }).click();
  await page.getByLabel("任务名称", { exact: true }).fill(name);
  await page
    .getByLabel("验收要点", { exact: true })
    .fill("回答包含机器学习，并保留文档依据");
  await page.getByRole("button", { name: "添加验收规则", exact: true }).click();
  await page
    .getByLabel("JSON 内部路径（可选）", { exact: true })
    .fill("/answer");
  await page.getByLabel("检查文本", { exact: true }).fill("机器学习");
  const response = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/native/tasks") &&
      r.request().postDataJSON()?.action === "save",
  );
  await page.getByRole("button", { name: "保存任务", exact: true }).click();
  return (await (await response).json()).id as string;
}

test("直接进入工作台，服务端持有凭证且浏览器不接触密钥", async ({page})=>{
  const browserCredentials: string[] = [];
  page.on('request', request=>{
    const authorization=request.headers()['authorization'];
    if(authorization)browserCredentials.push(authorization);
  });
  await open(page);
  await expect(page.getByLabel('访问凭证',{exact:true})).toHaveCount(0);
  await expect(page.getByRole('button',{name:'进入工作台',exact:true})).toHaveCount(0);
  await expect(page.getByRole('button',{name:'退出',exact:true})).toHaveCount(0);
  const response=await page.request.get('/api/native/access');
  expect(await response.json()).toMatchObject({authenticated:true,identity:{id:'fixture-operator'}});
  expect(await page.request.get('/api/access').then(r=>r.status())).toBe(404);
  expect((await page.context().cookies()).some(cookie=>cookie.name==='workbench_access')).toBe(false);
  const clientState=await page.evaluate(()=>document.documentElement.outerHTML+JSON.stringify(localStorage)+JSON.stringify(sessionStorage));
  expect(clientState).not.toContain('fixture-operator-token-not-for-real-use');
  expect(browserCredentials).toEqual([]);
});

test("控制台同源保护及服务端身份不受浏览器凭证覆盖", async ({request})=>{
  const response=await request.get('/api/native/access',{headers:{
    Authorization:'Bearer untrusted-browser-identity',Cookie:'workbench_access=untrusted-cookie',
  }});
  expect(await response.json()).toMatchObject({authenticated:true,identity:{id:'fixture-operator'}});
  expect((await request.get('http://127.0.0.1:8768/api/native/context')).status()).toBe(401);
  expect((await request.post('/api/native/layout',{headers:{Origin:'https://untrusted.example'},data:{session:'blocked'}})).status()).toBe(403);
  expect((await request.post('/api/native/layout',{headers:{Origin:'invalid origin'},data:{session:'blocked'}})).status()).toBe(403);
  expect((await request.post('/api/native/layout',{headers:{'Content-Type':'text/plain'},data:'{}'})).status()).toBe(415);
  expect((await request.post('/api/native/upload',{data:''})).status()).toBe(400);
});

test("八模块机器可读约定与移动端", async ({ page }) => {
  await open(page);
  const names = [
    "离线索引构建",
    "在线召回排序",
    "RAG",
    "OAG 图谱推理",
    "上下文工程",
    "多模态系统",
    "数据回收训练",
    "系统监控",
  ];
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 950 });
    for (const name of names) {
      await page.getByRole("button", { name, exact: true }).click();
      const result = await page.evaluate(() =>
        (window as any).__verify.current(),
      );
      expect(result.verdict, JSON.stringify(result)).toBe("PASS");
    }
  }
  const probes = await page.evaluate(() => {
    const node = document.querySelector<HTMLElement>(
      '[data-verify-unit="NativeWorkbench"]',
    )!;
    node.dataset.verifyBusy = "-1";
    const failed = (window as any).__verify.current();
    node.dataset.verifyBusy = "0";
    node.dataset.verifyState = "unavailable";
    const blocked = (window as any).__verify.current();
    return { failed: failed.verdict, blocked: blocked.verdict };
  });
  expect(probes).toEqual({ failed: "FAIL", blocked: "BLOCKED" });
});

test("实际运行、规则、人工基线、重复评测与周期暂停", async ({ page }) => {
  await open(page);
  const id = await search(page);
  await selectRun(page, id);
  const name = "固定检索-" + Date.now();
  const taskId = await saveTask(page, name);
  await expect(page.locator("#saved-task")).toHaveValue(taskId);
  await page.getByRole("button", { name: "运行验收规则", exact: true }).click();
  await expect(page.locator(".evaluation-verdict")).toContainText("规则通过");
  await page.getByText("人工业务验收", { exact: true }).click();
  await page
    .getByLabel("核对依据与结论", { exact: true })
    .fill("已核对 doc1 与实际回答");
  await page.getByRole("button", { name: "保存人工验收", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "设为所选任务基线", exact: true }),
  ).toBeEnabled();
  await page
    .getByRole("button", { name: "设为所选任务基线", exact: true })
    .click();
  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: "report.txt", exact: true }).click();
  expect((await download).suggestedFilename()).toBe("report.txt");
  await page.getByRole("tab", { name: "批量评测", exact: true }).click();
  await page.getByText("创建任务集", { exact: true }).click();
  await page
    .getByLabel("任务集名称", { exact: true })
    .fill("检索对比-" + Date.now());
  await page.getByRole("checkbox", { name, exact: true }).check();
  await page.getByRole("button", { name: "添加参数变体", exact: true }).click();
  const variant = page.getByRole("group", { name: "参数变体 2", exact: true });
  await variant
    .getByRole("checkbox", { name: "覆盖 排序算法", exact: true })
    .check();
  await page.getByLabel("变体2 排序算法", { exact: true }).fill("ctr");
  await page.getByLabel("每组重复次数", { exact: true }).fill("2");
  await page.getByRole("button", { name: "保存任务集", exact: true }).click();
  await expect(page.locator("#evaluation-suite")).not.toHaveValue("");
  await page.getByRole("button", { name: "运行快速探针", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "快速探针 · 已结束", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "运行完整评测", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "完整评测 · 已结束", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("cell", { name: "规则通过", exact: true }),
  ).toHaveCount(4);
  await expect(
    page.getByRole("cell", { name: "20 / 10", exact: true }),
  ).toHaveCount(2);
  await page
    .getByRole("button", { name: "设为任务集基线", exact: true })
    .click();
  await page
    .getByRole("button", { name: "比较任务集基线", exact: true })
    .click();
  await expect(page.locator(".run-comparison")).toContainText("0.0 个百分点");
  await page
    .getByRole("button", { name: "编辑所选任务集", exact: true })
    .click();
  const baseVariant = page.getByRole("group", {
    name: "参数变体 1",
    exact: true,
  });
  await baseVariant
    .getByRole("checkbox", { name: "覆盖 实验查询", exact: true })
    .check();
  await page.getByLabel("变体1 实验查询", { exact: true }).fill("自然语言");
  await page
    .getByRole("button", { name: "保存任务集新版本", exact: true })
    .click();
  await page.getByRole("button", { name: "运行完整评测", exact: true }).click();
  await expect(
    page.getByRole("cell", { name: "规则未通过", exact: true }),
  ).toHaveCount(2);
  await page
    .getByRole("button", { name: "比较任务集基线", exact: true })
    .click();
  await expect(page.locator(".run-comparison")).toContainText(
    "-100.0 个百分点",
  );
  await page.getByText("周期验证", { exact: true }).click();
  await page
    .getByRole("button", { name: "为所选任务集启用周期验证", exact: true })
    .click();
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  await expect(page.locator(".schedule-row")).toContainText("已暂停");
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test("经验审阅、上下文组合与预算装配", async ({ page }) => {
  await open(page);
  const id = await search(page);
  await selectRun(page, id);
  const taskId = await saveTask(page, "上下文验证任务-" + Date.now());
  await page.getByText("从此运行提炼经验", { exact: true }).click();
  const title = "可复用检索经验-" + Date.now();
  await page.getByLabel("经验标题", { exact: true }).fill(title);
  await page
    .getByLabel("适用条件、做法与结论", { exact: true })
    .fill("适用于当前文档集：先核对查询命中，再核对原文与回答。");
  await page.getByRole("button", { name: "保存经验", exact: true }).click();
  await page.getByRole("tab", { name: /^经验/ }).click();
  const note = page.locator(".experience-card").filter({ hasText: title });
  await note.locator(":scope > summary").click();
  await note.getByRole("button", { name: "审阅采用", exact: true }).click();
  await expect(note.locator(":scope > summary")).toContainText("已采用");
  await note
    .getByRole("button", { name: "加入上下文资源", exact: true })
    .click();
  await expect(note).toContainText("已加入上下文资源库");
  await page.getByRole("button", { name: "上下文工程", exact: true }).click();
  await page.getByRole("tab", { name: "供应配置", exact: true }).click();
  await page.getByText("维护上下文资源", { exact: true }).click();
  const sourceName = "核对说明-" + Date.now();
  await page.getByLabel("资源名称", { exact: true }).fill(sourceName);
  await page
    .getByLabel("资源类型", { exact: true })
    .selectOption("instructions");
  await page
    .getByLabel("内容", { exact: true })
    .fill("回答必须附上可以核对的文档依据。");
  await page.getByRole("button", { name: "保存资源", exact: true }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "资源版本已保存" }),
  ).toBeVisible();
  await page.getByText("组合供应配置", { exact: true }).click();
  const bundleName = "企业上下文-" + Date.now();
  await page.getByLabel("配置名称", { exact: true }).fill(bundleName);
  await page
    .getByRole("checkbox", { name: new RegExp("^" + sourceName) })
    .check();
  await page.getByRole("checkbox", { name: "必选", exact: true }).check();
  await page.getByRole("checkbox", { name: new RegExp("^" + title) }).check();
  await page.getByRole("button", { name: "保存供应配置", exact: true }).click();
  await expect(page.locator("#context-bundle")).not.toHaveValue("");
  await page.getByRole("button", { name: "检查装配结果", exact: true }).click();
  await expect(page.locator(".context-preview")).toContainText("实际装配");
  await expect(page.locator(".context-preview details")).toHaveCount(2);
  const bundleId = await page.locator("#context-bundle").inputValue();
  await page.getByRole("button", { name: "在线召回排序", exact: true }).click();
  await page.locator("#saved-task").selectOption(taskId);
  await page.getByRole("tab", { name: "固定任务", exact: true }).click();
  await page.getByText("附加企业上下文（可选）", { exact: true }).click();
  await page
    .getByLabel("使用的供应配置", { exact: true })
    .selectOption(bundleId);
  await page
    .getByRole("button", { name: "保存任务新版本", exact: true })
    .click();
  await page.getByRole("button", { name: "恢复任务参数", exact: true }).click();
  const contextualRun = await search(page);
  await selectRun(page, contextualRun);
  await expect(
    page.getByText("本次附加上下文 · 配置 v1", { exact: true }),
  ).toBeVisible();
  const record = await (
    await page.request.get("/api/native/runs/" + contextualRun)
  ).json();
  expect(record.context.bundle_id).toBe(bundleId);
  expect(JSON.stringify(record.inputs)).toContain(
    "回答必须附上可以核对的文档依据",
  );
  await expect(page.locator(".preset-badge")).toContainText("v2");
  await page.getByRole("button", { name: "退出任务预设", exact: true }).click();
  await expect(page.locator(".preset-badge")).toHaveCount(0);
  const ordinaryRun = await search(page);
  const ordinary = await (await page.request.get("/api/native/runs/" + ordinaryRun)).json();
  expect(ordinary.task_id).toBeNull();
  expect(ordinary.context_binding).toBeNull();
  expect(JSON.stringify(ordinary.inputs)).not.toContain("[企业上下文]");
  await page.reload();
  await page.getByRole("button", { name: "在线召回排序", exact: true }).click();
  await expect(page.getByLabel("实验查询", { exact: true })).toHaveValue(
    "机器学习",
  );
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test("多次运行整理草稿，并保留采用前的历史版本", async ({ page }) => {
  await open(page);
  await search(page);
  await search(page);
  await history(page);
  await page.getByRole("tab", { name: /^经验/ }).click();
  await page.getByText("从多次运行整理经验", { exact: true }).click();
  const title = "综合经验-" + Date.now();
  await page.getByLabel("整理后的经验标题", { exact: true }).fill(title);
  const choices = page
    .getByRole("group", { name: "选择来源运行", exact: true })
    .getByRole("checkbox");
  await choices.nth(0).check();
  await choices.nth(1).check();
  await page.getByRole("button", { name: "生成经验草稿", exact: true }).click();
  const note = page.locator(".experience-card").filter({ hasText: title });
  await note.locator(":scope > summary").click();
  await expect(note.getByRole("textbox")).toHaveValue(/运行证据/);
  await note.getByRole("button", { name: "审阅采用", exact: true }).click();
  await expect(note.locator(":scope > summary")).toContainText("已采用");
  await note.getByText("经验历史版本", { exact: true }).click();
  await note.getByRole("button", { name: "读取上一版本", exact: true }).click();
  await expect(note).toContainText("历史 v1（只读）");
});

test("人工放行前不执行，放行后执行同一任务", async ({ page }) => {
  await open(page);
  const id = await search(page);
  await selectRun(page, id);
  const taskId = await saveTask(page, "放行任务-" + Date.now());
  await page.getByRole("tab", { name: "固定任务", exact: true }).click();
  await page
    .getByRole("checkbox", { name: "执行前需要人工放行", exact: true })
    .check();
  await page
    .getByRole("button", { name: "保存任务新版本", exact: true })
    .click();
  let releaseRestore!: () => void;
  const restoring = new Promise<void>(resolve => { releaseRestore = resolve; });
  await page.route("**/api/native/tasks", async route => {
    if (route.request().postDataJSON()?.action !== "restore") return route.continue();
    const response = await route.fetch();
    await restoring;
    await route.fulfill({ response });
  });
  await page.getByRole("button", { name: "恢复任务参数", exact: true }).click();
  await expect(page.getByRole("button", { name: "执行检索", exact: true })).toBeDisabled();
  releaseRestore();
  await expect(page.locator(".preset-badge")).toContainText("v2");
  await expect(page.getByRole("button", { name: "执行检索", exact: true })).toBeEnabled();
  await expect(page.locator("#saved-task")).toHaveValue(taskId);
  const response = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/native/runs") && r.request().method() === "POST",
  );
  await page.getByRole("button", { name: "执行检索", exact: true }).click();
  const submitted = await response;
  expect(submitted.request().postDataJSON().task_id).toBe(taskId);
  expect(submitted.request().postDataJSON().task_version).toBe(2);
  const queued = await submitted.json();
  await page.getByRole("tab", { name: "运行证据", exact: true }).click();
  await expect(
    page.locator('#recorded-run option[value="' + queued.run_id + '"]').first(),
  ).toBeAttached();
  await page.locator("#recorded-run").selectOption(queued.run_id);
  await expect(page.locator(".run-detail[data-verify-state]")).toHaveAttribute(
    "data-verify-state",
    "awaiting_review",
  );
  const record = await (
    await page.request.get("/api/native/runs/" + queued.run_id)
  ).json();
  expect(record.outputs).toEqual({});
  await page.getByRole("button", { name: "放行本次操作", exact: true }).click();
  await expect(page.locator(".run-detail[data-verify-state]")).toHaveAttribute(
    "data-verify-state",
    "completed",
  );
});

test("流式刷新只执行一次，失败后界面可继续使用", async ({ page }) => {
  await open(page);
  await page.getByRole("button", { name: "RAG", exact: true }).click();
  await page.getByRole("button", { name: "触发失败", exact: true }).click();
  await expect(page.locator(".native-workbench > .inline-error")).toContainText(
    "可恢复的验收错误",
  );
  await page.getByRole("button", { name: "执行长任务", exact: true }).click();
  await expect(page.getByLabel("回答", { exact: true })).toHaveValue(/步骤/);
  await page.reload();
  await page.getByRole("button", { name: "RAG", exact: true }).click();
  await expect(page.getByLabel("回答", { exact: true })).toHaveValue(
    "结果：刷新验证，执行次数 1",
  );
  await expect(page.locator(".native-toolbar")).toContainText("执行完成");
});


test("执行环境变化后保留历史，避免回放覆盖新会话", async ({ page }) => {
  await open(page);
  const id = await search(page);
  await page.evaluate((runId) => {
    sessionStorage.setItem("workbench-active-runs", JSON.stringify([runId]));
  }, id);
  await page.route("**/api/native/runs/" + id, async route => {
    const response = await route.fetch();
    const body = await response.json();
    await route.fulfill({ response, json: { ...body, current_runtime_id: "another-runtime" } });
  });
  let replayed = false;
  page.on("request", request => {
    if(request.url().includes("/runs/" + id + "/events")) replayed = true;
  });
  await page.reload();
  await expect(page.locator(".native-toolbar")).toContainText("执行环境已变化");
  expect(replayed).toBe(false);
  expect(await page.evaluate(() => sessionStorage.getItem("workbench-active-runs"))).toBe("[]");
});
