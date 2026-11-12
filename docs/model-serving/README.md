# 模型服务

[项目说明](../../README.md) · [文档导航](../README.md)

工作台入口：**工作台 Python 执行服务，以及可选的独立 CTR API。**

## 输入与输出

| 项目 | 内容 |
| --- | --- |
| 工作台调用 | Next.js → 8766 Python 执行服务 → 原业务函数 |
| 独立 CTR API | 单独运行 start_model_serving.py，默认 8501 |
| 图像生成 | 独立 5001 服务，由工作台客户端调用 |

## 操作与结果核对

1. 先明确需要哪一种服务，再检查相应进程与端口。
2. 统一启动脚本处理工作台、MCP 与图像服务，不会自动启动 8501 CTR API。
3. 模型列表、加载状态和预测结果分别核对。
4. 保留请求、模型名、响应与日志，区分传输失败和业务失败。

## 当前能力与边界

不把服务可达性写成模型质量、负载能力或生产可用性的结论。独立 CTR API 的真实路由使用 /predict，不是 :predict。

## 按问题阅读

- [接口与实现](lectures/README.md)

## 实现入口

- [start_console.sh](../../start_console.sh)
- [src/search_engine/workbench_api.py](../../src/search_engine/workbench_api.py)
- [src/search_engine/model_service.py](../../src/search_engine/model_service.py)
- [start_model_serving.py](../../start_model_serving.py)

[原有代码与图解参考](implementation/README.md)保留示例；当前操作步骤统一以上述说明为准。
