"use client";
import { useEffect, useState } from "react";
import { installVerification } from "../lib/verification";
import {
  Database,
  Workflow,
  Network,
  Blocks,
  ChartNoAxesCombined,
  Activity,
  PanelLeftClose,
  BookOpen,
  ArrowRight,
} from "lucide-react";
import PositionAgents from "../components/position-agents";
import SupplyJobs from "../components/supply-jobs";
import SupplyWorkspace from "../components/supply-workspace";
import { WorkspaceNavigation, WorkspaceLink, pagesFor, useWorkspace } from "../lib/workspace-navigation";
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
export default function Page() {
  return <WorkspaceNavigation><WorkbenchPage /></WorkspaceNavigation>;
}
function WorkbenchPage() {
  const { route, navigate } = useWorkspace();
  const selected = route.area;
  const [rail, setRail] = useState(true);
  useEffect(installVerification, []);
  const currentModule = platformModules.find((x) => x.id === selected)!;
  const pages = pagesFor(selected);
  const currentPage = pages.find((page) => page.id === route.view);
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
            <WorkspaceLink
              key={id}
              area={id}
              aria-label={title}
              title={title}
              data-platform-index={id}
              aria-current={selected === id ? "page" : undefined}
              className={selected === id ? "active" : ""}
            >
              <Icon size={18} />
              <span>{title}</span>
            </WorkspaceLink>
          ))}
        </nav>
        {pages.length > 0 && <div className="workspace-subnav" role="group" aria-label={currentModule.title + "能力"}>
          {pages.map((page) => <WorkspaceLink key={page.id} area={selected} view={page.id}
            aria-current={route.view === page.id ? "page" : undefined}>{page.title}</WorkspaceLink>)}
        </div>}
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
            <span className="breadcrumb-root">FDE 工作台</span>
            <span className="breadcrumb-root">/</span>
            <WorkspaceLink area={selected}>{currentModule.title}</WorkspaceLink>
            {currentPage && <><span>/</span><strong>{currentPage.title}</strong></>}
            {!rail && pages.length > 0 && <select aria-label="切换功能页面" value={route.view} onChange={(event) => navigate(selected, event.target.value)}>{pages.map((page) => <option key={page.id} value={page.id}>{page.title}</option>)}</select>}
          </div>
        </header>
        <div
          className="page-content"
          data-platform-module={selected}
        >
          <div className="page-heading">
            <div>
              <h1>{currentPage?.title || currentModule.title}</h1>
              <p>{currentModule.description}</p>
            </div>
          </div>
          {selected === -1 ? (
            <PositionAgents initialView={route.view} hideTabs />
          ) : selected === -2 ? (
            <SupplyWorkspace key="context" domain="context" selectedView={route.view} />
          ) : selected === -3 ? (
            <SupplyWorkspace key="models" domain="models" selectedView={route.view} />
          ) : selected === -4 ? (
            <SupplyJobs />
          ) : selected === -5 ? (
            <SupplyWorkspace key="environment" domain="environment" selectedView={route.view} />
          ) : null}
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
