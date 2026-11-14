export type SupplyDomain = "context" | "models" | "environment";
export type SupplyCapability = {
  id: string;
  title: string;
  moduleIndex?: number;
  tabLabels?: string[];
  description?: string;
};

export const supplyCapabilities: Record<SupplyDomain, SupplyCapability[]> = {
  context: [
    { id: "resources", title: "资料库" },
    { id: "configuration", title: "供应配置" },
    { id: "indexes", title: "检索索引" },
    { id: "sources", title: "业务资料来源" },
    { id: "documents", title: "离线索引构建", moduleIndex: 0 },
    { id: "retrieval", title: "在线召回排序", moduleIndex: 1 },
    { id: "rag", title: "RAG", moduleIndex: 2 },
    { id: "graph", title: "OAG 图谱推理", moduleIndex: 3 },
    { id: "assembly", title: "上下文工程", moduleIndex: 4 },
    {
      id: "images",
      title: "图片资料",
      moduleIndex: 5,
      tabLabels: ["图片上传", "图搜图", "文搜图", "图片管理"],
      description: "管理图片资料、描述与标签，使用图搜图和文搜图检索企业图片库。",
    },
  ],
  models: [
    { id: "services", title: "模型服务" },
    { id: "datasets", title: "训练数据集" },
    { id: "artifacts", title: "模型产物" },
    { id: "training", title: "训练与评测", moduleIndex: 6 },
    {
      id: "image-model",
      title: "图像生成",
      moduleIndex: 5,
      tabLabels: ["图像生成"],
      description: "选择图像模型，检查生成参数、结果与历史记录。",
    },
  ],
  environment: [
    { id: "environment", title: "环境检查" },
    { id: "monitoring", title: "系统监控", moduleIndex: 7 },
  ],
};

export const previousModuleLocation: Record<number, { domain: SupplyDomain; capability: string }> = {
  0: { domain: "context", capability: "documents" },
  1: { domain: "context", capability: "retrieval" },
  2: { domain: "context", capability: "rag" },
  3: { domain: "context", capability: "graph" },
  4: { domain: "context", capability: "assembly" },
  5: { domain: "context", capability: "images" },
  6: { domain: "models", capability: "training" },
  7: { domain: "environment", capability: "monitoring" },
};
