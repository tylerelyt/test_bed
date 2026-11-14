"use client";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import {
  previousModuleLocation,
  supplyCapabilities,
  SupplyCapability,
} from "./supply-capabilities";

export type RouteOptions = {
  tab?: string;
  release?: string;
  run?: string;
  returnTo?: string;
  returnLabel?: string;
};
export type WorkspaceRoute = {
  area: number;
  view: string;
  record?: string;
} & RouteOptions;
const areas: Record<number, string> = {
  [-1]: "agents",
  [-2]: "context",
  [-3]: "models",
  [-4]: "jobs",
  [-5]: "environment",
};
export const agentPages: SupplyCapability[] = [
  { id: "agents", title: "岗位与 Agent" },
  { id: "desktop", title: "桌面 Agent", navigation: "tool" },
];
export function pagesFor(area: number) {
  return area === -1
    ? agentPages
    : area === -2
      ? supplyCapabilities.context
      : area === -3
        ? supplyCapabilities.models
        : area === -5
          ? supplyCapabilities.environment
          : [];
}
export function workspaceHref(
  area: number,
  view?: string,
  record?: string,
  options: RouteOptions = {},
) {
  const params = new URLSearchParams({ area: areas[area] || "agents" });
  const resolved = view || pagesFor(area)[0]?.id;
  if (resolved) params.set("view", resolved);
  if (record) params.set("record", record);
  for (const key of [
    "tab",
    "release",
    "run",
    "returnTo",
    "returnLabel",
  ] as const)
    if (options[key]) params.set(key, options[key]!);
  return "/?" + params.toString();
}
function readRoute(search = location.search, restore = true): WorkspaceRoute {
  const params = new URLSearchParams(search);
  let area = Number(
    Object.entries(areas).find(([, name]) => name === params.get("area"))?.[0],
  );
  if (!areas[area]) {
    const saved = restore
      ? sessionStorage.getItem("workbench-selected-module")
      : null;
    const old =
      saved === null ? undefined : previousModuleLocation[Number(saved)];
    area = old
      ? old.domain === "context"
        ? -2
        : old.domain === "models"
          ? -3
          : -5
      : areas[Number(saved)]
        ? Number(saved)
        : -1;
    if (old)
      sessionStorage.setItem(
        "workbench-capability-" + areas[area],
        old.capability,
      );
  }
  const pages = pagesFor(area);
  const requested =
    params.get("view") ||
    (restore
      ? sessionStorage.getItem("workbench-capability-" + areas[area])
      : null);
  const extras: RouteOptions = {};
  for (const key of [
    "tab",
    "release",
    "run",
    "returnTo",
    "returnLabel",
  ] as const)
    if (params.get(key)) extras[key] = params.get(key)!;
  // Previous releases linked to a global run page. The same URL now opens its Agent.
  return {
    area,
    view:
      pages.find((page) => page.id === requested)?.id || pages[0]?.id || "jobs",
    record: params.get("record") || undefined,
    ...extras,
  };
}
type NavigationValue = {
  route: WorkspaceRoute;
  navigate: (
    area: number,
    view?: string,
    record?: string,
    options?: RouteOptions,
  ) => void;
  resume: () => void;
  detailName: string;
  setDetailName: (name: string) => void;
};
const Navigation = createContext<NavigationValue>({
  route: { area: -1, view: "agents" },
  navigate: () => {},
  resume: () => {},
  detailName: "",
  setDetailName: () => {},
});
export const useWorkspace = () => useContext(Navigation);
export function useDetailName(name?: string) {
  const { setDetailName } = useWorkspace();
  useEffect(() => {
    setDetailName(name || "");
    return () => setDetailName("");
  }, [name, setDetailName]);
}
export function WorkspaceNavigation({
  children,
}: {
  children: React.ReactNode;
}) {
  const [route, setRoute] = useState<WorkspaceRoute>({
    area: -1,
    view: "agents",
  });
  const [detailName, setDetailName] = useState("");
  useEffect(() => {
    const initial = readRoute();
    setRoute(initial);
    history.replaceState(
      null,
      "",
      workspaceHref(initial.area, initial.view, initial.record, initial),
    );
    const pop = () => setRoute(readRoute());
    window.addEventListener("popstate", pop);
    return () => window.removeEventListener("popstate", pop);
  }, []);
  const open = useCallback((next: WorkspaceRoute) => {
    history.pushState(
      null,
      "",
      workspaceHref(next.area, next.view, next.record, next),
    );
    sessionStorage.setItem("workbench-selected-module", String(next.area));
    sessionStorage.setItem(
      "workbench-capability-" + areas[next.area],
      next.view,
    );
    setRoute(next);
    window.scrollTo({ top: 0 });
  }, []);
  const navigate = useCallback(
    (
      area: number,
      view?: string,
      record?: string,
      options: RouteOptions = {},
    ) => {
      open({
        area,
        view: view || pagesFor(area)[0]?.id || "jobs",
        record,
        returnTo: route.returnTo,
        returnLabel: route.returnLabel,
        ...options,
      });
    },
    [open, route.returnTo, route.returnLabel],
  );
  const resume = useCallback(() => {
    if (!route.returnTo?.startsWith("/?")) return;
    open(readRoute(route.returnTo.slice(1), false));
  }, [route.returnTo, open]);
  return (
    <Navigation.Provider
      value={{ route, navigate, resume, detailName, setDetailName }}
    >
      {children}
    </Navigation.Provider>
  );
}
export function WorkspaceLink({
  area,
  view,
  record,
  options,
  related,
  resetContext,
  children,
  ...props
}: {
  area: number;
  view?: string;
  record?: string;
  options?: RouteOptions;
  related?: string;
  resetContext?: boolean;
  children: React.ReactNode;
} & Omit<React.AnchorHTMLAttributes<HTMLAnchorElement>, "href">) {
  const { route, navigate } = useWorkspace();
  const next = {
    returnTo: route.returnTo,
    returnLabel: route.returnLabel,
    ...options,
    ...(related
      ? {
          returnTo: workspaceHref(route.area, route.view, route.record, route),
          returnLabel: related,
        }
      : {}),
    ...(resetContext ? { returnTo: "", returnLabel: "" } : {}),
  };
  return (
    <a
      {...props}
      href={workspaceHref(area, view, record, next)}
      onClick={(event) => {
        props.onClick?.(event);
        if (
          event.defaultPrevented ||
          event.button !== 0 ||
          event.metaKey ||
          event.ctrlKey ||
          event.shiftKey ||
          event.altKey
        )
          return;
        event.preventDefault();
        navigate(area, view, record, next);
      }}
    >
      {children}
    </a>
  );
}
