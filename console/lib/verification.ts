export function installVerification() {
  const visible = (element: Element) =>
    element instanceof HTMLElement && element.getClientRects().length > 0;
  const current = () => {
    const workbench = document.querySelector<HTMLElement>(
      '[data-verify-unit="NativeWorkbench"]',
    );
    const modules = Array.from(
      document.querySelectorAll<HTMLElement>("nav [data-module-index]"),
    );
    const controls = workbench
      ? Array.from(
          workbench.querySelectorAll("input,textarea,select,button"),
        ).filter(visible)
      : [];
    const unnamed = controls.filter((element) => {
      if (element.tagName === "BUTTON")
        return (
          !element.textContent?.trim() && !element.getAttribute("aria-label")
        );
      const input = element as HTMLInputElement;
      return (
        !element.getAttribute("aria-label") &&
        !Array.from(input.labels || []).some((label) =>
          label.textContent?.trim(),
        )
      );
    });
    const checks = [
      {
        id: "native-contract",
        ok:
          Number(workbench?.dataset.verifyComponents || 0) > 0 &&
          Number(workbench?.dataset.verifyEvents || 0) > 0,
        detail: "原组件与操作注册表存在",
      },
      {
        id: "module-count",
        ok: modules.length === 8,
        detail: `${modules.length} 个一级模块`,
      },
      {
        id: "selected-module",
        ok:
          modules.filter(
            (element) => element.getAttribute("aria-pressed") === "true",
          ).length === 1,
        detail: "恰有一个当前模块",
      },
      {
        id: "named-controls",
        ok: unnamed.length === 0,
        detail: `${unnamed.length} 个未命名可见控件`,
      },
      {
        id: "viewport",
        ok: document.documentElement.scrollWidth <= innerWidth + 1,
        detail: "页面无横向溢出",
      },
      {
        id: "nonnegative-busy",
        ok: Number(workbench?.dataset.verifyBusy || 0) >= 0,
        detail: "执行计数不为负数",
      },
    ];
    const blocked =
      !workbench ||
      workbench.dataset.verifyState === "loading" ||
      workbench.dataset.verifyState === "unavailable";
    return {
      format_version: 1,
      module: Number(workbench?.dataset.verifyModule),
      state: workbench?.dataset.verifyState,
      verdict: blocked
        ? "BLOCKED"
        : checks.every((check) => check.ok)
          ? "PASS"
          : "FAIL",
      checks,
    };
  };
  const manifest = () => ({
    format_version: 1,
    units: [
      "NativeWorkbench",
      "RunHistory",
      "RunEvidence",
      "EvaluationSuites",
      "ContextSupply",
    ],
    modules: Array.from(
      document.querySelectorAll<HTMLElement>("nav [data-module-index]"),
    ).map((element) => ({
      index: Number(element.dataset.moduleIndex),
      name: element.textContent?.trim(),
    })),
    invariants: [
      "module-count",
      "selected-module",
      "named-controls",
      "viewport",
      "nonnegative-busy",
    ],
  });
  const verify = { manifest, current };
  (window as Window & { __verify?: typeof verify }).__verify = verify;
  return () => {
    delete (window as Window & { __verify?: typeof verify }).__verify;
  };
}
