# 岗位 Agent：首版使用指南

[返回项目](../../README.md) · [架构方案](../architecture/README.md)

当前提供一条可运行的单机流程：创建岗位与 Agent，预配上下文和模型，通过原生 Pi 执行岗位任务，核对结果后采用该版本。其他 Harness 尚未接入。

## 环境

使用 Python 3.11 和 Node.js 22.19 或更新版本，从仓库根目录执行：

```bash
python -m pip install -r requirements-platform.txt
npm --prefix console ci
npm --prefix runtimes/pi ci --ignore-scripts
npm --prefix console run build
```

Pi 版本固定为 `1.5.2`，依赖锁保存在 `runtimes/pi/package-lock.json`；不依赖全局 Pi 配置。安装 PostgreSQL 17 后执行 `python tools/start_platform_db.py`，数据仅写入项目 `.workbench/postgres`，私有连接配置保存在 `.workbench/database-url.txt`。

macOS 默认查找 `/opt/homebrew/opt/postgresql@17/bin`，其他安装位置通过 `WORKBENCH_POSTGRES_BIN` 指定；端口默认 55432，可通过 `WORKBENCH_POSTGRES_PORT` 设置。脚本不注册系统开机服务。

```bash
./start_console.sh --background
```

统一脚本会启动 FastAPI、独立的岗位执行进程与 Next.js，并保留原模块所需的服务。未配置数据库时仅开发/CI回退 SQLite，不把它当成 PostgreSQL 已经就绪。

## 使用步骤

1. 打开控制台，在“岗位与 Agent”创建岗位，写明职责，再设置该岗位的 Agent。
2. 在“上下文配置”创建资料、规则等资源，并组合成供应配置；这里复用现有上下文服务。
3. 在“模型服务”登记已有的 OpenAI Chat Completions 兼容服务，例如运行中的 vLLM 或 Ollama。填写实际服务地址和模型标识，按需要提供服务密钥。保存配置不表示已经验证业务效果。
4. 在“发布与运行”选择 Agent、模型服务和上下文配置，生成候选版本。职责、资料快照和模型配置在此时固定；后续修改不会原地改变旧候选版本。
5. 输入岗位任务并运行。执行器会启动真实 Pi 进程，由 Pi 调用模型和已注册的 `read_enterprise_context` 工具读取预配置资料。
6. 核对返回内容与来源，下载结果，决定是否采用该版本。只能采用已经完成试运行的对应版本，采用不等于系统自动判定业务质量合格。

## 执行与保存

FastAPI 管理配置与运行记录，`platform_worker` 是一个独立本机进程，顺序领取队列中的任务。Pi 是该进程启动的独立 Harness；没有自建 Agent 执行循环，也没有 LangGraph。

每次运行在 `.workbench/agent-workspaces/<运行标识>/` 下生成独立 HOME、Pi 配置、岗位说明、上下文快照和原生会话目录。发布包保存适配代码、上下文工具及依赖锁的摘要，运行前校验一致性；适配器改动后需要重新生成候选版本。

模型密钥保存在 `.workbench/model-credentials/` 的私有文件中，接口不回显密钥。Pi 通过受控环境变量获取该服务密钥，生成的配置文件只包含变量引用；不继承后台的其他服务凭证。

首版只开放读取本次预配上下文的工具，关闭 Pi 的内置 shell、任意文件读写、MCP 自动发现、插件与全局配置自动发现。没有开放通用业务写操作；这不是完整的操作系统沙箱，后续开放执行工具时须另行验证环境隔离。

刷新页面可以重新选择已保存的运行记录和结果。本次新运行不复用上一次 Pi 会话；原生会话已保存供核对，但继续同一个会话的交互尚未接入。停止请求会终止本次 Pi 进程；服务异常后不自动重跑外部动作。

## 已验证范围与待接入能力

自动化验收使用真实 Pi 和本地协议测试模型，覆盖两轮模型调用、真实上下文工具执行、结果与原生会话落盘、请求去重、停止和版本采用。测试模型是固定协议响应，不具备业务智能，不能将验收通过当成模型质量或岗位胜任证明。

当前环境没有配置真实业务模型密钥，也没有运行中的本地推理模型；使用者需登记自己的模型服务后进行业务验收。

后续接入范围：其他 Harness、更多模型协议与推理参数、自动资料同步和索引构建、训练/微调到模型登记部署的完整链路、继续原生会话。现有训练与检索功能仍可通过原模块使用，本轮没有声称已将它们全部迁入新平台的数据模型。

## 接口与进程

- `src/search_engine/agent_platform/catalog.py`：岗位、模型服务与固定发布包。
- `src/search_engine/agent_platform/pi_runtime.py`：Pi 原生配置与进程接入。
- `src/search_engine/agent_platform/runs.py`：请求去重、队列状态和运行记录。
- `src/search_engine/platform_api.py`：FastAPI 管理 API及旧模块传输适配。
- `src/search_engine/platform_worker.py`：唯一的本机执行进程，使用本机文件锁避免重复启动。

业务接口使用 `/api/platform/positions`、`agents`、`models`、`releases` 和 `runs`；原 `/api/native/*` 接口通过兼容适配器保留。开发和 CI 可用 SQLite，实际单机部署优先使用 PostgreSQL；上下文资源暂沿用原服务。
