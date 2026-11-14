import { supplyCapabilities } from "./supply-capabilities";

export function installVerification() {
  const visible = (element: Element) => element instanceof HTMLElement && element.getClientRects().length > 0;
  const current = () => {
    const platform = document.querySelector<HTMLElement>("[data-platform-module]");
    const navigation = Array.from(document.querySelectorAll<HTMLElement>("nav [data-platform-index]"));
    const activeDomains = navigation.filter((item) => item.classList.contains("domain-active"));
    const currentPages = Array.from(document.querySelectorAll<HTMLElement>('aside nav [aria-current="page"]'));
    const owner = currentPages[0]?.closest<HTMLElement>("[data-navigation-area]");
    const controls = platform ? Array.from(platform.querySelectorAll("input,textarea,select,button")).filter(visible) : [];
    const unnamed = controls.filter((element) => element.tagName === "BUTTON"
      ? !element.textContent?.trim() && !element.getAttribute("aria-label")
      : !element.getAttribute("aria-label") && !Array.from((element as HTMLInputElement).labels || []).some((label) => label.textContent?.trim()));
    const units = platform ? Array.from(platform.querySelectorAll<HTMLElement>("[data-verify-state]")) : [];
    const ready = units.length > 0 && units.every((unit) => !["loading", "unavailable"].includes(unit.dataset.verifyState || ""));
    const native = platform?.querySelector<HTMLElement>('[data-verify-unit="NativeWorkbench"]');
    const checks = [
      { id: "platform-navigation", ok: navigation.length === 5 && activeDomains.length === 1 && activeDomains[0].dataset.platformIndex === platform?.dataset.platformModule && currentPages.length === 1 && owner?.dataset.navigationArea === platform?.dataset.platformModule, detail: "五个主入口，当前页面唯一且位于所属栏目下" },
      { id: "named-controls", ok: unnamed.length === 0, detail: `${unnamed.length} 个未命名可见控件` },
      { id: "viewport", ok: document.documentElement.scrollWidth <= innerWidth + 1, detail: "页面无横向溢出" },
    ];
    if (native) checks.push(
      { id: "native-contract", ok: Number(native.dataset.verifyComponents || 0) > 0 && Number(native.dataset.verifyEvents || 0) > 0, detail: "业务组件与操作注册表存在" },
      { id: "nonnegative-busy", ok: Number(native.dataset.verifyBusy || 0) >= 0, detail: "执行计数不为负数" },
    );
    return { format_version: 1, module: Number(platform?.dataset.platformModule), state: ready ? "ready" : "loading",
      verdict: !ready ? "BLOCKED" : checks.every((check) => check.ok) ? "PASS" : "FAIL", checks };
  };
  const manifest = () => ({
    format_version: 3,
    units: ["NativeWorkbench", "RunHistory", "RunEvidence", "EvaluationSuites", "ContextSupply", "ContextIndexes", "ModelServices", "ModelSupply", "PositionAgents", "SupplyJobs", "PlatformEnvironment"],
    capabilities: supplyCapabilities,
    platform_modules: Array.from(document.querySelectorAll<HTMLElement>("nav [data-platform-index]")).map((item) => ({ index: Number(item.dataset.platformIndex), name: item.textContent?.trim() })),
    invariants: ["platform-navigation", "named-controls", "viewport", "native-contract", "nonnegative-busy"],
  });
  (window as Window & { __verify?: { manifest: typeof manifest; current: typeof current } }).__verify = { manifest, current };
  return () => { delete (window as Window & { __verify?: unknown }).__verify; };
}
