# 服务端凭证与 Agent 授权

[返回项目](../../README.md) · [上下文供应与任务验证](../workbench-tasks/README.md)

控制台参照 [research-desk](https://github.com/anthropics/cwc-workshops/tree/main/research-desk) 的单操作员模式：服务端持有凭证，浏览器直接进入工作台。参考项目的凭证用于 Anthropic API；本项目将这一服务端持有方式用于连接原 Python 后端，不要求用户在浏览器输入访问凭证。

MCP 的 Agent 接入继续使用独立身份和配置授权。控制台的访问方式与 Agent 的 API 鉴权分别说明如下。

## 控制台启动

第一次启动 Python 后端会在 `.workbench/` 生成本机身份，私有凭证保存在 `bootstrap-access.txt`，身份配置保存在 `access.json`。Next.js 自动从该文件读取凭证并附加到后端请求，页面直接初始化原八模块。浏览器不会收到凭证，也不通过登录 Cookie、浏览器存储或 Authorization 请求头选择后端身份。

默认身份使用 `local` 项目，原上下文、运行和产物仍归该项目。旧 `data/conversation_history.jsonl` 保留，但不会再自动注入新会话。

自定义部署可以在 Next.js 服务端设置 `WORKBENCH_TOKEN_FILE` 指向私有凭证文件，使用绝对路径；也可以通过 `WORKBENCH_CONSOLE_TOKEN` 提供凭证，环境变量优先。没有这两项时读取 `WORKBENCH_STATE_DIR` 下的 `bootstrap-access.txt`，默认目录是仓库 `.workbench/`。这些配置不能使用 `NEXT_PUBLIC_` 前缀。

控制台属于共享操作入口，没有浏览器用户登录和身份切换。与参考项目一样，状态变更请求检查同源来源；普通操作要求 JSON，文件上传另走带长度声明的流式接口。远程部署由外部网关控制页面访问，Python 后端与 MCP 仍独立校验调用凭证。

## 分配身份

从仓库根目录使用已安装依赖的 Python 执行，命令仅输出凭证文件路径：

```bash
# 为项目操作员分配上下文维护权限
python tools/manage_workbench_access.py issue --id fde-alice --project project-a --role operator

# Agent 只读取指定供应配置；配置标识可从配置导出的 JSON 中取得
python tools/manage_workbench_access.py issue --id research-agent --project project-a --role agent --bundle CONFIG_ID

# 撤销身份
python tools/manage_workbench_access.py disable --id research-agent
```

新凭证保存在 `.workbench/credentials/<身份标识>.txt`。对同一标识再次 `issue` 会轮换凭证，旧凭证失效。外部 API 使用者和各 Agent 分配独立凭证，不共用控制台的服务端凭证。

| 身份 | 默认范围 |
| --- | --- |
| API 操作员 | 维护所属项目的资料、配置、笔记、经验、任务与运行记录 |
| Agent | 通过 MCP 读取明确分配的供应配置及其资源版本；只写当前仍授权的笔记 |
| 获准访问原业务的操作员 | 在上述权限之外执行原八模块；须明确增加 `--business-access` |

**原索引、图谱、模型、训练目录及桌面环境仍属于本实例的共享业务环境。** `--business-access` 只应授予获准访问这些共享资源的可信操作员。当前项目隔离覆盖上下文配置、运行台账、产物访问和会话历史，没有为原业务数据或计算资源建立企业多租户隔离。

控制台的服务端身份需要具有原业务权限。只有上下文维护权限的 API 操作员不能调用原业务执行接口；Agent 仅通过 MCP 接入，不能借用控制台代理切换为自己的身份。

## Agent 接入

API 和 MCP 使用 `Authorization: Bearer <访问凭证>`。Agent 的凭证及配置授权均在服务端校验；目录只返回其获准的资源，直接拼接其他资源 URI 不能绕过范围和版本限制。

```python
from pathlib import Path
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

token = Path(".workbench/credentials/research-agent.txt").read_text().strip()
client = Client(StreamableHttpTransport(
    "http://127.0.0.1:3001/mcp",
    headers={"Authorization": "Bearer " + token},
))
```

配置的内容版本可以固定；写权限同时检查该配置的最新授权。将最新配置中的笔记改为只读或移除，旧配置版本也不能继续写入。Agent 读取普通资料时，只能读取当前获准配置固定的资源版本。

身份配置每次请求重新核对。撤权后，新请求被拒绝，排队任务与后续周期执行会重新验证执行身份。已经开始的业务操作仍需通过停止机制处理，并核对实际终态。控制台凭证失效时由部署维护者更新服务端配置，页面不转入登录流程。

## 会话与来源

对话历史保存在 `.workbench/conversation-history-v1.sqlite3`，按项目、身份和会话分别读取、追加和清理。“清空历史”仅影响当前范围。跨会话共享资料通过明确授权的上下文资源和笔记实现。

原业务通过内部 MCP 调用时，服务端传递短期、签名且限定于 MCP 的委托凭证，不把服务端凭证写入任务记录。外部操作员使用对话历史资源时，可以用 `X-Workbench-Session` 指定自己的应用会话；该标识始终受已认证身份和项目约束。

控制台的人工放行与验收记录保存其服务端身份，不宣称识别浏览器背后的独立用户。笔记审计保存 MCP 认证身份和调用方提供的会话来源；调用方填写的来源标签不视为独立认证的人名。

## 配置与备份

可用 `WORKBENCH_ACCESS_FILE` 指定私有身份配置文件；显式配置的文件缺失或格式无效时会关闭访问，不退回匿名模式。默认文件结构如下，`token_sha256` 由管理工具生成：

```json
{
  "version": 1,
  "principals": [
    {
      "id": "research-agent",
      "project": "project-a",
      "role": "agent",
      "token_sha256": "64位十六进制摘要",
      "bundles": ["CONFIG_ID"],
      "enabled": true,
      "business_access": false
    }
  ]
}
```

备份时保护身份配置、凭证文件与签名密钥的系统文件权限；不要把这些文件提交 Git。撤销已分发凭证使用 `disable` 或轮换身份，不依赖删除本机凭证副本。
