# 桌面智能体（GUI Agent）

[项目说明](../../README.md) · [文档导航](../README.md)

工作台入口：**岗位 Agent → 桌面 Agent → GUI-Agent。**

## 输入与输出

| 项目 | 内容 |
| --- | --- |
| 输入 | 任务、截图、历史、模型配置与控制环境 |
| 过程 | 观察 → 模型响应 → 动作解析 → 执行 → 再观察 |
| 输出 | 原始响应、动作、每步截图、环境历史与最终任务状态 |

## 操作与结果核对

1. 先核对环境类型和截图来源，确认控制的是预期桌面。
2. 配置模型后测试一次截图分析，区分真实模型响应与模拟 WAIT。
3. 用明确的小任务验证动作和画面变化，再组合多步任务。
4. 检查 results.final_status、轨迹和最终画面；需要时使用停止入口。

## 当前能力与边界

调用返回 success 不等于任务成功；还需查看任务终态。环境隔离取决于实际控制器和连接，不能仅凭页面模式名称承诺宿主机完全不受影响。

## 按问题阅读

- [架构](lectures/architecture/README.md)
- [环境准备](lectures/environment-setup/README.md)
- [模型配置](lectures/vlm-integration/README.md)
- [截图与动作](lectures/vlm-reasoning/README.md)
- [任务终态](lectures/task-execution/README.md)
- [任务设计](lectures/automation/README.md)
- [OSWorld 适配](lectures/osworld-integration/README.md)
- [故障排查](lectures/troubleshooting/README.md)

## 实现入口

- [src/search_engine/gui_agent_service.py](../../src/search_engine/gui_agent_service.py)
- [src/search_engine/image_tab/image_tab.py](../../src/search_engine/image_tab/image_tab.py)
- [src/search_engine/gui_agent_tab/gui_agent_tab.py](../../src/search_engine/gui_agent_tab/gui_agent_tab.py)

[原有代码与图解参考](implementation/README.md)保留示例；当前操作步骤统一以上述说明为准。
