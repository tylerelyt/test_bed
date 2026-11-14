"use client";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { previousModuleLocation, supplyCapabilities } from "./supply-capabilities";

export type WorkspaceRoute = { area: number; view: string; record?: string };
const areas: Record<number, string> = { [-1]: "agents", [-2]: "context", [-3]: "models", [-4]: "jobs", [-5]: "environment" };
export const agentPages = [{ id: "agents", title: "岗位与 Agent" }, { id: "runs", title: "发布与运行" }, { id: "desktop", title: "桌面 Agent" }];
export function pagesFor(area: number) {
  return area === -1 ? agentPages : area === -2 ? supplyCapabilities.context : area === -3 ? supplyCapabilities.models : area === -5 ? supplyCapabilities.environment : [];
}
export function workspaceHref(area: number, view?: string, record?: string) {
  const params = new URLSearchParams({ area: areas[area] || "agents" });
  const resolved = view || pagesFor(area)[0]?.id;
  if (resolved) params.set("view", resolved);
  if (record) params.set("record", record);
  return "/?" + params.toString();
}
function readRoute(): WorkspaceRoute {
  const params = new URLSearchParams(location.search);
  let area = Number(Object.entries(areas).find(([, name]) => name === params.get("area"))?.[0]);
  if (!areas[area]) {
    const saved = sessionStorage.getItem("workbench-selected-module");
    const old = saved === null ? undefined : previousModuleLocation[Number(saved)];
    area = old ? old.domain === "context" ? -2 : old.domain === "models" ? -3 : -5 : areas[Number(saved)] ? Number(saved) : -1;
    if (old) sessionStorage.setItem("workbench-capability-" + areas[area], old.capability);
  }
  const pages = pagesFor(area);
  const savedView = sessionStorage.getItem("workbench-capability-" + areas[area]);
  const requested = params.get("view") || savedView;
  return { area, view: pages.find((page) => page.id === requested)?.id || pages[0]?.id || "jobs", record: params.get("record") || undefined };
}
const Navigation = createContext<{ route: WorkspaceRoute; navigate: (area: number, view?: string, record?: string) => void }>({ route: { area: -1, view: "agents" }, navigate: () => {} });
export const useWorkspace = () => useContext(Navigation);
export function WorkspaceNavigation({ children }: { children: React.ReactNode }) {
  const [route, setRoute] = useState<WorkspaceRoute>({ area: -1, view: "agents" });
  useEffect(() => {
    const initial = readRoute();setRoute(initial);
    history.replaceState(null, "", workspaceHref(initial.area, initial.view, initial.record));
    const pop = () => setRoute(readRoute());
    window.addEventListener("popstate", pop);
    return () => window.removeEventListener("popstate", pop);
  }, []);
  const navigate = useCallback((area: number, view?: string, record?: string) => {
    const pages = pagesFor(area);
    const next = { area, view: view || pages[0]?.id || "jobs", record };
    history.pushState(null, "", workspaceHref(area, next.view, record));
    sessionStorage.setItem("workbench-selected-module", String(area));
    sessionStorage.setItem("workbench-capability-" + areas[area], next.view);
    setRoute(next);
    window.scrollTo({ top: 0 });
  }, []);
  return <Navigation.Provider value={{ route, navigate }}>{children}</Navigation.Provider>;
}
export function WorkspaceLink({ area, view, record, children, ...props }: { area: number; view?: string; record?: string; children: React.ReactNode } & Omit<React.AnchorHTMLAttributes<HTMLAnchorElement>, "href">) {
  const { navigate } = useWorkspace();
  return <a {...props} href={workspaceHref(area, view, record)} onClick={(event) => {
    props.onClick?.(event);
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();navigate(area, view, record);
  }}>{children}</a>;
}
