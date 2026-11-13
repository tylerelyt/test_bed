# Test Bed：企业上下文工程工作台

面向 FDE（前线部署工程师）的企业上下文供应后台，用于准备、配置和验证 Agent 所需的文档、索引、知识图谱、提示词、工具与反馈数据。

平台承担企业上下文的日常配置与供应；测试床是其中验证配置和业务变更的能力。项目沿用 Test Bed 名称。

工作台通过 Next.js 界面调用原 Python 业务函数，把检索、图谱推理、上下文装配、多模态和训练实验放在同一入口。围绕具体任务检查三件事：**材料从哪里来、模型实际获得了什么、结果能否由依据和执行记录验证。**

[快速开始](#快速开始) · [八个模块](#八个模块) · [上下文供应与任务验证](docs/workbench-tasks/README.md) · [使用指南](docs/README.md) · [运行与开发](console/README.md)

## 八个模块

| 工作台入口 | 主要工作 | 使用说明 |
| --- | --- | --- |
| 离线索引构建 | 查看文档与倒排索引，浏览图谱、审阅三元组 | [索引与检索](docs/search-recommendation/README.md) |
| 在线召回排序 | TF-IDF 召回、CTR 重排、结果详情与点击反馈 | [检索与排序](docs/search-recommendation/README.md) |
| RAG | 用文档检索结果组织回答，核对上下文与实际提示词 | [文档问答](docs/rag/README.md) |
| OAG 图谱推理 | 通过实体关系、图谱游走和逐步事实组织回答 | [图谱推理](docs/rag/README.md#图谱推理) |
| 上下文工程 | MCP 模板选择、占位符替换、推理与历史更新 | [上下文与 MCP](docs/context-engineering/README.md) |
| 多模态系统 | 图片检索、图像生成与桌面任务 | [多模态](docs/multimodal/README.md) · [桌面智能体](docs/gui-agent/README.md) |
| 数据回收训练 | CPT/SFT/DPO、偏好采集、CTR 评估与表示学习实验 | [训练与反馈](docs/llmops/README.md) |
| 系统监控 | 查看数据、索引、质量、性能与模型状态 | [验证工具](tools/README.md) |

## 快速开始

### 1. 安装环境

需要 Python 3.11、Node.js 22+、npm，以及可连接的 JanusGraph/Gremlin 图服务。完整依赖包含多个机器学习框架；模型权重在实际加载时按需获取。

```bash
git clone https://github.com/tylerelyt/test_bed.git
cd test_bed
python3.11 -m venv .venv-full
source .venv-full/bin/activate
python -m pip install -r requirements-console.txt
npm --prefix console ci
```

### 2. 准备图服务

已有图服务时，设置实际地址：

```bash
export JANUSGRAPH_URL='ws://localhost:8182/gremlin'
```

本地没有图服务时，可使用 Docker 开发配置：

```bash
docker compose -f compose.console.yml up -d gremlin
```

此配置使用兼容协议的 TinkerGraph，图数据的持久化没有保证。需要长期保存数据时，连接自己的持久化 JanusGraph。

### 3. 启动工作台

```bash
./start_console.sh
```

打开 **[http://127.0.0.1:3100](http://127.0.0.1:3100)**，直接进入工作台并初始化原模块与数据。参照 research-desk 的单操作员控制台，连接后端的凭证由 Next.js 服务端持有，默认读取自动生成的本机私有文件，无需在浏览器输入。详见[服务端凭证与 Agent 授权](docs/workbench-access/README.md)。前台运行时按 `Ctrl+C` 结束本次启动的进程。

需要关闭终端后继续运行时：

```bash
npm --prefix console run build
./start_console.sh --background
```

后台日志：`logs/console/launcher.log`；启动器 PID：`logs/console/launcher.pid`。后台运行不包含开机自启，图服务仍须先就绪。

## 第一次验证什么

1. 在“离线索引构建”核对文档列表与索引统计。
2. 用文档中有答案的问题，在“在线召回排序”检查命中、详情与点击反馈。
3. 配置模型后，分别在 RAG、OAG 中核对文档依据与关系事实。
4. 在“上下文工程”检查模板、装配后的输入和历史更新。
5. 按任务需要再进入图片、桌面或训练模块，保留输入、参数与实际结果。

不使用模型 API，也可以先验证文档、索引、排序和部分状态查询。在线问答需要对应密钥；本地推理、生成与训练需要相应模型及计算环境。

## 当前能力与边界

- **页面与业务执行**：八个模块已迁到 Next.js，沿用原数据和函数。Gradio Python 包用于原组件参数与状态兼容，新工作台不需要另启 Gradio 站点。
- **身份与项目范围**：独立的操作员和 Agent 凭证；上下文、运行记录及历史按项目授权，Agent 仅能读取获准配置。原共享业务模块需要单独授权，见[权限说明](docs/workbench-access/README.md)。
- **上下文供应**：版本化组合资料、操作说明、工具定义、角色职责、经验和笔记，按预算装配，通过原 MCP 服务供 Agent 按需读取。固定任务也可选用具体配置版本，保留实际输入与资源来源。
- **任务验证**：固定任务、自动规则、可选模型评分与人工验收分开记录；支持参数变体、重复试验、快速探针、任务集检查和基线对比。详见[完整使用指南](docs/workbench-tasks/README.md)。
- **服务端运行**：有界队列、独立停止、可选人工放行、持久化事件和文件产物；刷新可续接，服务重启后标记中断，不自动重跑业务操作。
- **经验与周期验证**：从多次运行整理草稿，审阅后加入上下文；周期验证固定任务集版本，升级后暂停并等待核对。成本仅基于实际模型用量与明确配置的价格估算。
- **指令数据生成**：当前生成器返回模拟样本，用于验证数据格式与流程；尚未接通真实模型扩写。
- **A/B 偏好采集**：当前用同一个已加载模型的两个温度生成回答；不是两个生产模型版本的流量实验。
- **训练与迭代**：提供数据、配置、训练、推理和偏好入口；自动发布、灰度分流和回滚没有组成完整编排。
- **状态与效果**：页面或服务可达，不代表模型质量、任务成功或全部外部服务健康。桌面任务还需核对实际终态和画面。

## 服务与配置

| 服务 | 默认端口 | 启动方式 |
| --- | --- | --- |
| Next.js 控制台 | 3100 | 统一脚本 |
| Python 执行服务 | 8766 | 统一脚本 |
| MCP 服务 | 3001 | 统一脚本启动或复用 |
| 图像生成服务 | 5001 | 统一脚本启动或复用；模型按需加载 |
| Gremlin 图服务 | 8182 | 预先准备，或使用 Docker 开发配置 |
| 独立 CTR API（可选） | 8501 | 另行启动，见[模型服务](docs/model-serving/README.md) |

| 配置项 | 作用 |
| --- | --- |
| `JANUSGRAPH_URL` / `GREMLIN_SERVER_URL` | 图服务地址，前者优先 |
| `WORKBENCH_PYTHON` | 指定已有业务环境的 Python；默认优先使用 `.venv-full/bin/python` |
| `WORKBENCH_API_URL` | Next.js 服务端连接的 Python 地址，默认 `http://127.0.0.1:8766` |
| `WORKBENCH_TOKEN_FILE` / `WORKBENCH_CONSOLE_TOKEN` | 可选的控制台服务端凭证文件或环境变量；默认读取 `.workbench/bootstrap-access.txt`，环境变量优先；均不发送至浏览器 |
| `WORKBENCH_ACCESS_FILE` | 私有身份配置；默认自动初始化 `.workbench/access.json`，无有效凭证不能访问业务接口 |
| `WORKBENCH_STATE_DIR` | 运行记录、任务与经验的保存目录，默认仓库下 `.workbench/` |
| `WORKBENCH_CONCURRENCY` | 业务任务并发上限，默认 3，可设为 1—8；同模块串行，停止独立 |
| `WORKBENCH_ARTIFACT_MAX_MB` | 单个输出文件的归档上限，默认 64 MB |
| `WORKBENCH_JUDGE_URL` / `WORKBENCH_JUDGE_MODEL` / `WORKBENCH_JUDGE_API_KEY` | 可选模型评分与经验整理服务，只在服务端配置 |
| `WORKBENCH_MODEL_PRICES` | 按模型标识配置单价、币种和来源；未配置不推测成本 |
| `DASHSCOPE_API_KEY` | 当前在线问答等 DashScope 调用所需密钥 |
| `OPENAI_API_KEY` | 使用相应配置的视觉模型或其他模块时提供，端点按模块设置 |

密钥放在服务端环境，不使用 `NEXT_PUBLIC_*`。OSWorld、本机桌面控制和图像生成各有独立的环境要求，详见相应模块说明。

## 数据、文档与开发

原数据继续保存在 `data/`、`models/`、`checkpoints/` 及原 MCP 历史位置，详见[数据说明](data/README.md)。

当前操作统一维护在模块 README；专题讲义解释机制并链接真实代码。原代码和图解保留在折叠参考区，旧示例指标不作为验收结果。文档直接在仓库维护，不再使用 Pages 发布流程。

- [全部模块与专题](docs/README.md)
- [运行、开发和故障定位](console/README.md)
- [验证工具与检查范围](tools/README.md)
- [学习路径与资料](docs/learning-resources/README.md)

在已激活的 Python 环境中运行基础验证：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
npm --prefix console run build
```

这些检查覆盖迁移层与构建；模型效果、真实训练和桌面任务需要另外验收。

## 常见问题

**页面打不开。** 确认前后台进程仍在运行并检查后台日志。本机地址 `127.0.0.1:3100` 只对运行服务的电脑有效。

**首次打开一直初始化。** 检查图服务可达性与 Python 日志。首次加载需要初始化原业务模块及图数据。

**提示 Ollama 未连接。** 先核对所选推理方式。Ollama 状态保留在兼容检查中，不等于当前 DashScope 或已加载的本地模型不可用。

**提示缺少模型或密钥。** 按模块配置对应依赖；输入模型名称不等于权重已经加载。

**修改前端后没有变化。** 统一脚本优先运行已有生产构建。更新构建并重启，或按[开发说明](console/README.md#前端开发)运行开发服务。

问题与建议通过 [Issues](https://github.com/tylerelyt/test_bed/issues) 提交。许可证为 [MIT](LICENSE)。
