# GUI-Agent - 桌面自动化代理

基于 [OSWorld](https://github.com/xlang-ai/OSWorld) 架构实现的多模态桌面自动化代理。

## 功能概述

GUI-Agent 是一个能够像人类一样操作电脑桌面的智能体，具备四种核心能力：

- **👀 观察**：自动捕获屏幕截图
- **🧠 思考**：基于视觉语言模型（如 GPT-4o）理解任务并决策
- **🖱️ 行动**：通过 PyAutoGUI 执行鼠标、键盘操作
- **🔄 循环**：持续执行"观察-决策-行动"闭环，直到任务完成

## 核心架构

### 1. DesktopEnv - 桌面环境

提供真实或虚拟桌面操作环境：

```python
env = SimpleDesktopEnv(
    provider_name="local",          # 环境类型：local/docker/vmware/aws
    os_type="macOS",                # 操作系统
    action_space="pyautogui",       # 动作空间
    screen_size=(1920, 1080)        # 屏幕分辨率
)
```

**支持的 Provider：**
- `local`: 本地环境（直接控制当前系统）
- `docker`: Docker 容器环境
- `vmware`: VMware 虚拟机
- `aws`: AWS 云端环境

### 2. PromptAgent - 智能代理

结合 VL 模型进行任务推理：

```python
agent = SimplePromptAgent(
    model="gpt-4o",                 # 视觉语言模型
    action_space="pyautogui",
    observation_type="screenshot",
    max_trajectory_length=3          # 历史轨迹长度
)
```

**支持的模型：**
- GPT-4o
- GPT-4 Vision Preview
- GPT-4 Turbo
- Claude 3 Opus（需配置相应 API）

### 3. PyAutoGUI - 动作执行

执行实际的用户界面交互：

```python
# 鼠标操作
pyautogui.moveTo(x, y)
pyautogui.click()
pyautogui.doubleClick()
pyautogui.rightClick()

# 键盘操作
pyautogui.typewrite('text')
pyautogui.press('enter')
pyautogui.hotkey('command', 'c')  # macOS 用 command，Windows 用 ctrl
```

## 工作流程

### 步骤 0：环境配置

初始化环境和代理：

```python
from src.search_engine.gui_agent_service import gui_agent_service

# 初始化
result = gui_agent_service.initialize(
    provider_name="local",
    os_type="macOS",
    model="gpt-4o",
    api_key=your_api_key,  # 可选，留空使用环境变量
    base_url=custom_url    # 可选，自定义 API 端点
)
```

### 步骤 1：重置环境

加载任务配置并获取初始观察：

```python
task_config = {
    'instruction': '打开浏览器并搜索 OSWorld 项目',
    'evaluator': None,
    'config': {}
}

obs = env.reset(task_config=task_config)
# obs = {
#     'screenshot': b'...',              # PNG 格式的截图
#     'screenshot_path': 'path/to/file',
#     'instruction': '...',
#     'timestamp': '...'
# }
```

### 步骤 2-6：执行循环

持续执行"观察-思考-行动"循环：

```python
done = False
while not done:
    # 步骤 2: 获取截图（env.reset 或 env.step 自动捕获）
    
    # 步骤 3: VL 模型推理
    response, actions = agent.predict(instruction, obs)
    # response: 模型的思考过程
    # actions: ['pyautogui.moveTo(100, 200)', 'pyautogui.click()']
    
    # 步骤 4-6: 解析并执行动作
    for action in actions:
        obs, reward, done, info = env.step(action, pause=1.0)
        if done:
            break
```

## 使用示例

### 在 UI 中使用

1. 进入"第五部分：多模态系统"
2. 切换到"🤖 GUI-Agent"子标签
3. 在"环境配置"中初始化（可选，使用默认配置）
4. 输入任务指令，例如：
   - "移动鼠标到屏幕中心并点击"
   - "打开 Spotlight 搜索（Command+Space）"
   - "截图并保存（Command+Shift+3）"
5. 点击"▶️ 执行任务"
6. 查看执行过程的截图和步骤记录

### 在代码中使用

```python
from src.search_engine.gui_agent_service import gui_agent_service

# 初始化
gui_agent_service.initialize(
    provider_name="local",
    os_type="macOS",
    model="gpt-4o"
)

# 执行任务
result = gui_agent_service.run_task(
    instruction="打开浏览器并搜索 Python",
    max_steps=15,
    sleep_after_execution=1.5
)

# 查看结果
print(f"状态: {result['results']['final_status']}")
print(f"步数: {result['results']['total_steps']}")

# 查看截图
for step in result['results']['steps']:
    for action_result in step['action_results']:
        screenshot_path = action_result['screenshot_path']
        print(f"截图: {screenshot_path}")
```

## 系统要求

### 必需依赖

```bash
pip install pyautogui Pillow openai
```

- `pyautogui`: 桌面自动化控制
- `Pillow`: 图像处理和截图
- `openai`: OpenAI API 客户端（如果使用 OpenAI 模型）

### 环境变量

可选配置（在"环境配置"中输入或设置环境变量）：

```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选，自定义端点
```

### 权限要求

macOS 系统需要授予以下权限：
- **辅助功能访问**：用于控制鼠标和键盘
- **屏幕录制**：用于截取屏幕

设置路径：
系统偏好设置 → 安全性与隐私 → 隐私 → 辅助功能 / 屏幕录制

## 安全考虑

### 动作白名单

系统只允许执行以下安全动作：

```python
# 允许的 PyAutoGUI 命令
pyautogui.moveTo(x, y)
pyautogui.click()
pyautogui.doubleClick()
pyautogui.rightClick()
pyautogui.typewrite('text')
pyautogui.press('key')
pyautogui.hotkey('modifier', 'key')

# 控制符
DONE   # 任务完成
FAIL   # 任务失败
WAIT   # 等待
```

### 坐标校验

所有鼠标坐标会被校验：
- `0 < x < screen_width`
- `0 < y < screen_height`

越界坐标会被自动丢弃。

### 动作限制

- 每轮最多执行 5 个动作
- 最大执行步数可配置（默认 15）
- 黑名单过滤危险命令（如系统命令、文件操作等）

## 参考资料

- [OSWorld GitHub](https://github.com/xlang-ai/OSWorld)
- [OSWorld 论文](https://arxiv.org/abs/2404.07972)
- [PyAutoGUI 文档](https://pyautogui.readthedocs.io/)

## 注意事项

1. **本地环境谨慎使用**：local provider 会直接控制当前系统，建议在虚拟机或测试环境中使用
2. **API 成本**：每次执行会调用 VL 模型 API，注意控制步数以降低成本
3. **执行速度**：建议设置适当的等待时间（1-2秒），给界面充分响应时间
4. **错误处理**：任务可能因环境变化、模型理解偏差等原因失败，需要人工介入
5. **截图隐私**：所有截图会保存在 `data/gui_agent/screenshots/` 目录

## 故障排除

### PyAutoGUI 权限错误

**问题**：`pyautogui.click()` 无效或报错

**解决**：
1. 检查 macOS 辅助功能权限
2. 重启终端/IDE
3. 尝试手动执行 `pyautogui.displayMousePosition()` 测试

### 模型无法识别屏幕内容

**问题**：模型输出的坐标不准确

**解决**：
1. 检查截图是否正常保存
2. 尝试更高分辨率的屏幕
3. 使用更强大的模型（如 GPT-4o）
4. 在 prompt 中提供更详细的任务描述

### 任务总是达到最大步数

**问题**：任务未完成就达到 max_steps

**解决**：
1. 增加 max_steps 参数
2. 简化任务描述
3. 检查 sleep_after_execution 是否足够
4. 查看执行历史，确认每步是否正常

## 未来计划

- [ ] 支持 Docker 和 VMware Provider
- [ ] 实现任务评估器（Evaluator）
- [ ] 增加无障碍树（Accessibility Tree）观察
- [ ] 支持终端输出捕获
- [ ] 实现任务录制与回放
- [ ] 添加更多示例任务

