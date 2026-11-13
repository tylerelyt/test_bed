# 多模态系统

[项目说明](../../README.md) · [文档导航](../README.md)

工作台入口：**上下文供应 → 图片资料**提供图片上传、图搜图、文搜图和图片管理；图像生成在**模型供应 → 图像生成**。

## 输入与输出

| 项目 | 内容 |
| --- | --- |
| 图片输入 | 原图、文本描述、标签与检索条件 |
| 图片输出 | 图片 ID、索引统计、相似结果与原图 |
| 生成输入与输出 | 提示词及生成参数 → 图像、生成信息与历史 |

## 操作与结果核对

1. 上传已知图片后，核对元数据、索引和原图。
2. 分别用原图和文本描述执行检索，比较命中结果。
3. 图像生成先确认 5001 服务，再加载模型并提交参数。
4. 对照生成图片与返回信息，记录实际模型和参数。

## 当前能力与边界

图片编码与图像生成都按需加载模型，服务启动不等于权重已就绪。GUI Agent 使用独立的环境和模型配置，见桌面智能体指南。

## 按问题阅读

- [图片检索](lectures/image-search/README.md)
- [图像生成](lectures/image-generation/README.md)
- [桌面智能体](../gui-agent/README.md)

## 实现入口

- [src/search_engine/image_service.py](../../src/search_engine/image_service.py)
- [src/search_engine/diffusion_service.py](../../src/search_engine/diffusion_service.py)
- [image_generation_service.py](../../image_generation_service.py)

[原有代码与图解参考](implementation/README.md)保留示例；当前操作步骤统一以上述说明为准。
