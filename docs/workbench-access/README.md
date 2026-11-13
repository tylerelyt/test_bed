# 身份、项目权限与会话隔离

[返回项目](../../README.md) · [上下文供应与任务验证](../workbench-tasks/README.md)

控制台和 MCP 使用同一份身份配置。每个凭证对应一个身份、一个项目和一个角色；服务端核对凭证后决定作用范围，客户端不能通过传入项目名或用户名改变权限。

## 首次进入

第一次启动会在 `.workbench/` 生成本机操作员身份，凭证保存在 `bootstrap-access.txt`，配置保存在 `access.json`。文件仅对当前系统用户开放。打开控制台后，在登录框输入凭证；凭证通过 HttpOnly、SameSite=Strict Cookie 发送，JavaScript 不读取该 Cookie，也不把凭证放进浏览器存储。

默认本机身份可以访问原八模块及 `local` 项目。已有上下文、运行与产物仍归 `local`，不会自动迁入其他项目。旧 `data/conversation_history.jsonl` 保留，但不会再自动注入新会话。

远程部署应使用 HTTPS 和受控反向代理。当前提供独立凭证身份，可通过本机管理工具分配和撤销；企业 SSO 尚未接入。

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

新凭证保存在 `.workbench/credentials/<身份标识>.txt`。对同一标识再次 `issue` 会轮换凭证，旧凭证失效。应为每个人和每个 Agent 单独分配，不共用本机管理员凭证。

| 身份 | 默认范围 |
| --- | --- |
| 操作员 | 维护所属项目的资料、配置、笔记、经验、任务与运行记录 |
| Agent | 通过 MCP 读取明确分配的供应配置及其资源版本；只写当前仍授权的笔记 |
| 获准访问原业务的操作员 | 在上述权限之外执行原八模块；须明确增加 `--business-access` |

**原索引、图谱、模型、训练目录及桌面环境仍属于本实例的共享业务环境。** `--business-access` 只应授予获准访问这些共享资源的可信操作员。当前项目隔离覆盖上下文配置、运行台账、产物访问和会话历史，没有为原业务数据或计算资源建立企业多租户隔离。

只有上下文维护权限的操作员仍可在现有“上下文工程”模块使用供应配置，不初始化或执行原共享业务模块。Agent 凭证不能登录操作员控制台，也不能调用原业务执行接口。

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

身份配置每次请求重新核对。撤权后，新请求被拒绝，排队任务与后续周期执行会重新验证执行身份。已经开始的业务操作仍需通过停止机制处理，并核对实际终态。

## 会话与来源

对话历史保存在 `.workbench/conversation-history-v1.sqlite3`，按项目、身份和会话分别读取、追加和清理。“清空历史”仅影响当前范围。跨会话共享资料通过明确授权的上下文资源和笔记实现。

原业务通过内部 MCP 调用时，服务端传递短期、签名且限定于 MCP 的委托凭证，不把操作员凭证写入任务记录。外部操作员使用对话历史资源时，可以用 `X-Workbench-Session` 指定自己的应用会话；该标识始终受已认证身份和项目约束。

人工放行与验收记录保存操作员身份。笔记审计保存认证身份和调用方提供的会话来源；调用方填写的来源标签不视为独立认证的人名。

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
