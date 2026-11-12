# 桌面任务故障定位

[项目说明](../../../../README.md) · [当前模块说明](../../README.md)

按观察、模型、解析、执行和终态五层排查，通常比反复更换提示词更容易定位问题。先检查截图和环境，再检查模型请求，最后检查动作与目标状态。

当前服务保留环境历史、模型轨迹、错误信息和任务终态。它们分别回答“看到了什么”“建议了什么”“执行了什么”“结果如何”。

## 怎样验证

1. 无截图：检查目标环境、连接、屏幕录制权限。
2. 无有效响应：核对端点、模型能力、密钥与请求错误。
3. 动作无效果：核对坐标空间、控制权限、窗口焦点与执行反馈。

## 使用边界

任务 status 与业务完成状态须分别检查；修复环境或权限后，应从可控的小任务重新验证。

## 实现入口

- [src/search_engine/gui_agent_service.py](../../../../src/search_engine/gui_agent_service.py)
- [src/search_engine/image_tab/image_tab.py](../../../../src/search_engine/image_tab/image_tab.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```bash
# Check Docker
docker ps -a | grep osworld

# View logs
docker logs osworld-vm

# Restart container
docker restart osworld-vm
```

### 示例 2

```bash
curl -X POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
  -d '{"model": "qwen3-vl-plus", "messages": [{"role": "user", "content": "test"}]}'
```

</details>
