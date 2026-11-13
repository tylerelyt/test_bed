"use client";
import { useEffect, useState } from "react";
import { supplyCapabilities, SupplyDomain } from "../lib/supply-capabilities";
import ContextSupply from "./context-supply";
import NativeWorkbench from "./native-workbench";
import PositionAgents from "./position-agents";
import PlatformEnvironment from "./platform-environment";

const names = { context: "上下文供应能力", models: "模型供应能力", environment: "运行环境能力" };
export default function SupplyWorkspace({ domain }: { domain: SupplyDomain }) {
  const items = supplyCapabilities[domain];
  const [selected, setSelected] = useState(items[0].id);
  useEffect(() => {
    const saved = sessionStorage.getItem("workbench-capability-" + domain);
    if (items.some((item) => item.id === saved)) setSelected(saved!);
  }, [domain]);
  const capability = items.find((item) => item.id === selected) || items[0];
  return (
    <div className="supply-workspace" data-capability-domain={domain}>
      <div className="history-tabs capability-navigation" role="group" aria-label={names[domain]}>
        {items.map((item) => (
          <button key={item.id} aria-pressed={capability.id === item.id}
            data-capability={item.id} onClick={() => {
              setSelected(item.id);
              sessionStorage.setItem("workbench-capability-" + domain, item.id);
            }}>{item.title}</button>
        ))}
      </div>
      {capability.moduleIndex !== undefined ? (
        <NativeWorkbench key={domain}
          moduleIndex={capability.moduleIndex} tabLabels={capability.tabLabels} description={capability.description} />
      ) : domain === "context" ? (
        <div className="agent-card"><ContextSupply /></div>
      ) : domain === "models" ? (
        <PositionAgents initialView="models" hideTabs />
      ) : <PlatformEnvironment />}
    </div>
  );
}
