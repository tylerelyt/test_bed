---
layout: default
title: 首页
nav_order: 1
description: "企业级 AI 系统工程测试床文档"
permalink: /
---

# AI 工程测试床
{: .fs-9 }

面向企业级场景的 AI 系统工程测试床，覆盖搜索推荐、RAG 与上下文工程、多模态检索、训练与反馈闭环等能力。
{: .fs-6 .fw-300 }

[快速开始](#-快速开始){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[GitHub 仓库](https://github.com/tylerelyt/test_bed){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## 平台定位
{: .text-delta }

- 提供统一实验环境，用于验证 AI 算法、系统架构与工程流程
- 支持从离线数据构建到在线检索、反馈回流、训练迭代的完整闭环
- 默认强调本地可运行、可观测、可扩展

---

## 功能概览
{: .text-delta }

- 🔍 **搜索推荐**：倒排索引、CTR 重排、A/B 测试
- 💬 **RAG 与上下文工程**：检索增强、多步推理、MCP 编排
- 🖼️ **多模态系统**：CLIP 图像检索与跨模态查询
- 🤖 **训练闭环**：样本采集、模型训练与反馈迭代
- 🛡️ **系统监控**：运行状态与关键指标可视化

---

## 🚀 快速开始
{: .text-delta }

### 安装

```bash
git clone https://github.com/tylerelyt/test_bed.git
cd test_bed
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 启动

```bash
./quick_start.sh
```

{: .note }
> 启动后访问 [http://localhost:7861](http://localhost:7861)。

---

## 📚 文档导航
{: .text-delta }

<div class="code-example" markdown="1">

### 核心模块

[搜索推荐系统]({{ site.baseurl }}/docs/search-recommendation){: .btn .btn-outline }

[LLMOps 训练管道]({{ site.baseurl }}/docs/llmops){: .btn .btn-outline }

[RAG 与上下文工程]({{ site.baseurl }}/docs/rag){: .btn .btn-outline }

[多模态系统]({{ site.baseurl }}/docs/multimodal){: .btn .btn-outline }

[GUI 自动化代理]({{ site.baseurl }}/docs/gui-agent){: .btn .btn-outline }

[模型服务]({{ site.baseurl }}/docs/model-serving){: .btn .btn-outline }

</div>

---

## 🛠️ 技术栈
{: .text-delta }

| 类别 | 技术 |
|:---------|:------------|
| **经典 ML** | scikit-learn（逻辑回归）、TensorFlow（Wide & Deep） |
| **大语言模型** | LLaMA-Factory、LoRA、Ollama、OpenAI API |
| **训练技术** | CPT、SFT、DPO |
| **计算机视觉** | OpenAI CLIP（ViT-B/32）、Transformers |
| **Web 框架** | Gradio、Flask |
| **模型服务** | 独立进程 + RESTful 接口 |

---

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](https://github.com/tylerelyt/test_bed/blob/main/LICENSE)。

## 🤝 参与贡献

欢迎提交 Issue 和 Pull Request。提交前请阅读 [贡献指南](https://github.com/tylerelyt/test_bed/blob/main/CONTRIBUTING.md)。

## 📞 联系方式

- **项目主页**: [https://github.com/tylerelyt/test_bed](https://github.com/tylerelyt/test_bed)
- **问题追踪**: [https://github.com/tylerelyt/test_bed/issues](https://github.com/tylerelyt/test_bed/issues)

