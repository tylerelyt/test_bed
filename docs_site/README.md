# 文档网站

这是使用 [just-the-docs](https://github.com/just-the-docs/just-the-docs) Jekyll 主题构建的 GitHub Pages 文档网站。

## 本地开发

### 前置要求

- Ruby 3.1+
- Bundler

### 安装依赖

```bash
cd docs_site
bundle install
```

### 本地运行

```bash
bundle exec jekyll serve
```

然后在浏览器中访问 [http://localhost:4000](http://localhost:4000)

## 部署

网站会自动通过 GitHub Actions 部署到 GitHub Pages。当你推送更改到 `main` 分支的 `docs_site/` 目录时，会自动触发构建和部署。

## 配置

主要配置文件：
- `_config.yml`: Jekyll 和主题配置
- `Gemfile`: Ruby 依赖管理

## 文档结构

- `index.md`: 首页
- `docs/`: 文档页面目录
  - `docs/search-recommendation/`: 搜索推荐系统模块
    - CTR 预测模型
    - 模型评估
    - 可解释性分析
    - 公平性分析
    - AutoML 优化
  - `docs/llmops.md`: LLMOps 训练管道
  - `docs/rag.md`: RAG 与上下文工程
  - `docs/multimodal/`: 多模态 AI 模块
    - 图像搜索
    - 图像生成
  - `docs/gui-agent.md`: GUI 自动化代理
  - `docs/model-serving.md`: 模型服务

