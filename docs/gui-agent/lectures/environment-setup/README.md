# 桌面环境的准备与识别

[项目说明](../../../../README.md) · [当前模块说明](../../README.md)

在发出动作前，先明确控制目标是本机桌面还是虚拟机/容器中的桌面。环境状态、截图尺寸和动作坐标必须来自同一个目标。

macOS 本机控制需要相应辅助功能和屏幕录制权限；虚拟机方案还依赖实际启动的控制服务、镜像和连接配置。当前页面中的环境配置应与截图结果交叉核对。

## 怎样验证

1. 只获取状态和截图，核对截图是否来自预期环境。
2. 检查屏幕尺寸、权限与连接错误，再测试一个可观察的简单动作。
3. 记录环境类型、镜像或系统版本，便于复现坐标与窗口变化。

## 使用边界

选择“虚拟机模式”的文案不自动证明隔离已经生效。应依据连接与截图确认实际环境，不能承诺所有桌面操作均不会影响宿主机。

## 实现入口

- [src/search_engine/gui_agent_service.py](../../../../src/search_engine/gui_agent_service.py)
- [src/search_engine/image_tab/image_tab.py](../../../../src/search_engine/image_tab/image_tab.py)
- [src/search_engine/gui_agent_tab/gui_agent_tab.py](../../../../src/search_engine/gui_agent_tab/gui_agent_tab.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```bash
docker run -d \
  --name osworld-vm \
  -p 55000:5000 \
  -p 5901:5900 \
  xlangai/osworld:latest
```

### 示例 2

```
系统设置 → 隐私与安全性 → 辅助功能
- 添加终端应用（Terminal）和 Python

系统设置 → 隐私与安全性 → 屏幕录制
- 添加终端应用（Terminal）和 Python
```

### 示例 3

```bash
# Test screenshot
python -c "import pyautogui; pyautogui.screenshot()"

# Check Ollama (if using local models)
curl http://localhost:11434/api/tags
```

</details>
