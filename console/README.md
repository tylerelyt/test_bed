# 工作台运行与开发

[项目安装入口](../README.md#快速开始) · [模块指南](../docs/README.md)

本页维护运行机制、开发方式和排障；初次安装统一使用项目根 README 的步骤。

## 运行机制

默认入口为岗位 Agent，使用 FastAPI 业务接口管理岗位、模型服务、发布版本和运行；独立的 `platform_worker` 进程启动原生 Pi，详见[岗位 Agent 指南](../docs/position-agents/README.md)。

原八模块的浏览器通过 Next.js 的同源接口连接 Python 执行服务。React 渲染原布局和控件，Python 运行原函数，并复用原参数转换、会话状态、动态更新与流式返回。原函数产生的 HTML 结果在隔离文档中显示。

参照 research-desk，控制台由服务端持有连接凭证，打开页面直接进入原模块。Next.js 默认读取仓库 `.workbench/bootstrap-access.txt`；可通过服务端 `WORKBENCH_TOKEN_FILE`（绝对路径）或 `WORKBENCH_CONSOLE_TOKEN` 指定，后者优先。浏览器不输入或保存凭证，也不能通过 Cookie 或 Authorization 请求头切换服务端身份。

Gradio Python 包仍是兼容依赖，但没有单独的 Gradio Web 服务。旧 `quick_start.sh` 用于历史兼容，不是当前默认入口。

## 统一启动脚本

从仓库根目录执行：

```bash
./start_console.sh
```

脚本先检查依赖与图服务，复用或启动 3001 MCP 和 5001 图像服务，再启动 8766 Python 服务与 3100 Next.js。退出时只停止脚本本次启动的进程。

- 有 `console/.next/BUILD_ID` 时，运行生产构建。
- 没有构建时，运行开发服务。
- `--background` 用独立会话启动，日志与 PID 保存在 `logs/console/`。
- 后台模式不包含开机自启或自动重启；图服务仍由原配置维护。

## 前端开发

先运行 Python 执行服务及所需图/MCP/图像服务，确认 3100 端口没有其他控制台进程，再从根目录运行：

```bash
npm --prefix console run dev
```

生产构建：

```bash
npm --prefix console run build
npm --prefix console start
```

修改代码后，运行中的生产进程需要重新构建并重启。重复执行后台启动命令只会检查已有启动器，不会自动更新旧进程。

## 分别启动服务

在原完整 Python 环境和图服务就绪后，可分别运行：

```bash
python -m src.search_engine.mcp.dynamic_mcp_server
python -m flask --app image_generation_service run --host 127.0.0.1 --port 5001
python -m src.search_engine.platform_api
```

另需运行 `python -m src.search_engine.platform_worker` 处理岗位 Agent 队列；以上进程需要分别保持运行。然后启动前端；不要同时重复启动统一脚本占用相同端口。

Next.js 的服务端 `WORKBENCH_API_URL` 默认指向 `http://127.0.0.1:8766`，可放在 `console/.env.local`。模型密钥继续按原业务环境提供，不进入浏览器公共变量。

## 会话与文件

会话标识保存在当前浏览器的 sessionStorage 中。服务重启后应重新载入页面，不能假定旧会话状态仍有效。

按钮与提交操作由服务端任务执行，运行记录写入 `.workbench/native-runs-v1.sqlite3`。同一模块串行执行，业务并发默认 3，停止操作独立；固定任务可选择人工放行。刷新页面会重新连接活动任务，服务重启保留记录并标记中断。任务、评测、上下文供应及经验的使用范围见[完整指南](../docs/workbench-tasks/README.md)。

可用 `WORKBENCH_STATE_DIR` 设置记录目录；Python 与 Next.js 共用该配置时使用绝对路径。备份时应先停止执行服务，再复制数据库；不要同时启动多个执行服务共享同一目录。控制台使用其服务端配置的项目身份，历史进一步按会话隔离；原业务数据和计算资源仍是共享环境，未提供分布式任务调度。Agent 的独立接入范围见[服务端凭证与 Agent 授权](../docs/workbench-access/README.md)。

上传文件通过浏览器、Next.js 与 Python 流式传输，使用临时引用；已完成操作的输出文件会按大小限制归档，保留大小、SHA-256 和来源，重启后仍可下载。原模型目录和训练输出目录继续由业务模块管理。原数据位置见[数据说明](../data/README.md)。

## 可选评分与成本配置

评分服务使用常见的消息接口：请求含 `model`、`messages` 和 `temperature`，响应从 `choices[0].message.content` 读取 JSON。配置完整评分地址、模型和密钥，密钥只放在执行服务环境中：

```bash
export WORKBENCH_JUDGE_URL='https://your-provider.example/v1/chat/completions'
export WORKBENCH_JUDGE_MODEL='your-model'
export WORKBENCH_JUDGE_API_KEY='<服务端密钥>'
```

模型评分结构会验证通过结论、0—1 分数与依据。没有配置或结构错误时显示“无法判定”，不会给出通过结论。经验整理也可以使用该服务，调用须由操作员在界面明确选择。

成本配置示例，数字仅演示格式，不代表任何模型实际价格：

```bash
export WORKBENCH_MODEL_PRICES='{"your-model":{"input_per_million":1,"output_per_million":2,"currency":"CNY","source":"已核对的定价来源与日期"}}'
```

配置键必须对应接口实际返回的模型标识。记录接口返回的输入、输出 token，再按配置估算；缺少用量、币种或来源时保留为未知。缓存、阶梯计费等特殊价格需要按实际计费另行核对，估算不等于账单。

## 页面运行约定与浏览器检查

页面通过 `data-verify-*` 暴露模块、状态、注册表与执行计数。`window.__verify.manifest()` 列出约定，`window.__verify.current()` 从当前 DOM 读取检查结果，区分 `PASS`、`FAIL` 和无法观察的 `BLOCKED`。

```bash
npm --prefix console run build
npm --prefix console exec -- playwright install chromium
npm --prefix console run test:e2e
```

测试使用独立的 8768/3102 端口和确定性回调，不加载模型、不控制桌面、不调用外部模型服务；Python 需有 Gradio、FastMCP 等基础依赖。可用 `WORKBENCH_TEST_PYTHON` 指定解释器，用 `WORKBENCH_CHROME` 指定本机 Chrome。CI 也运行这些真实浏览器交互，失败时保存轨迹和截图；完整原模块另用原回调与本机浏览器验收。

## 验证与排障

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
npm --prefix console run build
```

完整原环境就绪后，`tools/verify_console_migration.py` 可以对原回调进行以读取和输入校验为主的检查；执行会改变本次会话状态，检查范围见[工具说明](../tools/README.md)。

| 现象 | 先检查什么 |
| --- | --- |
| 无法打开页面 | 3100 进程、启动日志、是否在运行服务的电脑访问 |
| 页面打开但模块未就绪 | 8766 日志、图服务连接与首次初始化 |
| 模型调用失败 | 所选推理方式、端点、密钥与模型加载状态 |
| 图像生成功能不可用 | 5001 服务、加载模型结果、权重与设备 |
| 桌面任务没有效果 | 实际控制环境、截图、解析动作与任务终态 |
| 修改样式后仍旧 | 当前是开发服务还是旧生产构建，是否重新启动 |

统一主题覆盖原控件、表格、子页、隔离 HTML 和嵌套图谱。图谱资源随前端构建提供；模型效果、训练完成与真实桌面任务成功需要各自的业务证据。
