# 工作台运行与开发

[项目安装入口](../README.md#快速开始) · [模块指南](../docs/README.md)

本页维护运行机制、开发方式和排障；初次安装统一使用项目根 README 的步骤。

## 运行机制

浏览器通过 Next.js 的同源接口连接 Python 执行服务。React 渲染原布局和控件，Python 运行原函数，并复用原参数转换、会话状态、动态更新与流式返回。原函数产生的 HTML 结果在隔离文档中显示。

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
python -m src.search_engine.workbench_api
```

以上是三个独立进程，需要分别保持运行。然后启动前端；不要同时重复启动统一脚本占用相同端口。

Next.js 的服务端 `WORKBENCH_API_URL` 默认指向 `http://127.0.0.1:8766`，可放在 `console/.env.local`。模型密钥继续按原业务环境提供，不进入浏览器公共变量。

## 会话与文件

会话标识保存在当前浏览器的 sessionStorage 中。服务重启后应重新载入页面，不能假定旧会话状态仍有效。

文件通过浏览器、Next.js 与 Python 流式传输，并使用临时文件引用预览或下载。原模型目录和训练输出目录继续由业务模块管理，不通过演示数据库替换。原数据位置见[数据说明](../data/README.md)。

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
