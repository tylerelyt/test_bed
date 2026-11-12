# 桌面智能体模块实现导航

[当前使用指南](../../../docs/gui-agent/README.md) · [任务状态说明](../../../docs/gui-agent/lectures/task-execution/README.md)

当前工作台的桌面操作入口位于“多模态系统 / GUI-Agent”。操作和环境要求以当前使用指南为准。

## 实现入口

- [原 GUI 页签](gui_agent_tab.py)
- [图片与桌面交互页面](../image_tab/image_tab.py)
- [环境、模型代理与任务服务](../gui_agent_service.py)

`SimpleDesktopEnv` 提供观察和动作执行，`SimplePromptAgent` 构造视觉输入并解析动作，`GUIAgentService` 组织任务循环和结果。

排查时同时保留截图、原始模型响应、解析动作和执行反馈。模型未配置时可能返回模拟 WAIT；外层接口返回 success 时，还需核对 `results.final_status` 与最终画面，不能据此直接判定任务成功。
