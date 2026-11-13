"use client";
import { useEffect, useState } from "react";
import { installVerification } from "../lib/verification";
import {
  Database,
  Search,
  Workflow,
  Network,
  Blocks,
  Image,
  ChartNoAxesCombined,
  Activity,
  PanelLeftClose,
  BookOpen,
  ArrowRight,
} from "lucide-react";
import NativeWorkbench from "../components/native-workbench";
import PositionAgents from "../components/position-agents";
import ContextSupply from "../components/context-supply";
import SupplyJobs from "../components/supply-jobs";
import PlatformEnvironment from "../components/platform-environment";
const platformModules = [
  {
    id: -1,
    title: "岗位 Agent",
    description: "预配上下文与模型，发布并运行岗位 Agent",
    icon: Blocks,
  },
  {
    id: -2,
    title: "上下文供应",
    description: "导入企业资料、维护资源版本与岗位供应配置",
    icon: Database,
  },
  {
    id: -3,
    title: "模型供应",
    description: "管理数据、训练、模型产物、部署与外部模型服务",
    icon: ChartNoAxesCombined,
  },
  {
    id: -4,
    title: "作业管理",
    description: "查看资料与模型作业的状态、日志和结果",
    icon: Activity,
  },
  {
    id: -5,
    title: "运行环境",
    description: "检查数据库、执行进程、Harness 与训练推理环境",
    icon: Network,
  },
];
const modules = [
  {
    title: "离线索引构建",
    description: "文档、倒排索引与知识图谱",
    icon: Database,
  },
  { title: "在线召回排序", description: "召回、排序与点击反馈", icon: Search },
  { title: "RAG", description: "检索增强与回答依据", icon: Workflow },
  { title: "OAG 图谱推理", description: "实体、关系与多步游走", icon: Network },
  { title: "上下文工程", description: "MCP 工具与上下文装配", icon: Blocks },
  {
    title: "多模态系统",
    description: "图片检索、生成与 GUI Agent",
    icon: Image,
  },
  {
    title: "数据回收训练",
    description: "模型训练、评估与优化",
    icon: ChartNoAxesCombined,
  },
  { title: "系统监控", description: "运行状态与样本指标", icon: Activity },
];
export default function Page() {
  const [selected, setSelected] = useState(-1),
    [rail, setRail] = useState(true);
  useEffect(installVerification, []);
  useEffect(() => {
    const saved = sessionStorage.getItem("workbench-selected-module");
    if (saved !== null && Number(saved) >= -5 && Number(saved) < 8)
      setSelected(Number(saved));
  }, []);
  function selectModule(index: number) {
    setSelected(index);
    sessionStorage.setItem("workbench-selected-module", String(index));
  }
  const currentModule =
    selected < 0
      ? platformModules.find((x) => x.id === selected)!
      : modules[selected];
  return (
    <div className={`shell ${rail ? "" : "compact"}`}>
      <aside className="sidebar">
        <a className="brand" href="/">
          <span className="brand-icon">
            <Workflow size={22} />
          </span>
          <span>
            test bed<small>岗位 Agent 供应平台</small>
          </span>
        </a>
        <div className="workspace-label">
          企业 Agent 后台 <span>本地</span>
        </div>
        <nav>
          {platformModules.map(({ id, title, icon: Icon }) => (
            <button
              key={id}
              data-platform-index={id}
              aria-pressed={selected === id}
              className={selected === id ? "active" : ""}
              onClick={() => selectModule(id)}
            >
              <Icon size={18} />
              <span>{title}</span>
            </button>
          ))}
          <details className="legacy-navigation" open>
            <summary>业务处理与验证</summary>
            <div className="legacy-links">
            {modules.map(({ title, icon: Icon }, i) => (
              <button
                key={title}
                data-module-index={i}
                aria-pressed={i === selected}
                className={i === selected ? "active" : ""}
                onClick={() => selectModule(i)}
              >
                <Icon size={18} />
                <span>{title}</span>
              </button>
            ))}
            </div>
          </details>
        </nav>
        <div className="sidebar-note">
          <span className="eyebrow">面向业务任务</span>
          <p>
            供应与验证
            <br />
            岗位 Agent 所需的上下文与模型。
          </p>
          <a
            href="https://github.com/tylerelyt/test_bed/blob/main/README.md"
            target="_blank"
            rel="noreferrer"
          >
            <BookOpen size={15} />
            项目文档
            <ArrowRight size={14} />
          </a>
        </div>
      </aside>
      <main>
        <header>
          <div className="breadcrumbs">
            <button
              className="icon-button collapse"
              aria-label="切换侧栏"
              onClick={() => setRail(!rail)}
            >
              <PanelLeftClose size={18} />
            </button>
            <span>FDE 工作台</span>
            <span>/</span>
            <strong>{currentModule.title}</strong>
          </div>
        </header>
        <div
          className="page-content"
          data-platform-module={selected < 0 ? selected : undefined}
        >
          <div className="page-heading">
            <div>
              <div className="eyebrow">岗位 Agent 供应平台</div>
              <h1>{currentModule.title}</h1>
              <p>{currentModule.description}</p>
            </div>
          </div>
          {selected === -1 ? (
            <PositionAgents />
          ) : selected === -2 ? (
            <div className="agent-card">
              <ContextSupply />
            </div>
          ) : selected === -3 ? (
            <PositionAgents key="model-supply" initialView="models" hideTabs />
          ) : selected === -4 ? (
            <SupplyJobs />
          ) : selected === -5 ? (
            <PlatformEnvironment />
          ) : (
            <NativeWorkbench moduleIndex={selected} />
          )}
        </div>
        <footer>
          TEST BED <span>岗位 Agent · 上下文 · 模型</span>
          <a
            href="https://github.com/tylerelyt/test_bed"
            target="_blank"
            rel="noreferrer"
          >
            GitHub ↗
          </a>
        </footer>
      </main>
    </div>
  );
}
