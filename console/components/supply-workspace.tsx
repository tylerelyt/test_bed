"use client";
import { supplyCapabilities, SupplyDomain } from "../lib/supply-capabilities";
import ContextSupply from "./context-supply";
import NativeWorkbench from "./native-workbench";
import ModelServices from "./model-services";
import ModelSupply from "./model-supply";
import ContextIndexes from "./context-indexes";
import PlatformEnvironment from "./platform-environment";

export default function SupplyWorkspace({ domain, selectedView }: { domain: SupplyDomain; selectedView: string }) {
  const items = supplyCapabilities[domain];
  const capability = items.find((item) => item.id === selectedView) || items[0];
  return (
    <div className="supply-workspace" data-capability-domain={domain}>
      {capability.moduleIndex !== undefined ? (
        <NativeWorkbench key={domain}
          moduleIndex={capability.moduleIndex} tabLabels={capability.tabLabels} description={capability.description} />
      ) : domain === "context" ? (
        capability.id === "indexes" || capability.id === "sources"
          ? <ContextIndexes initialMode={capability.id === "sources" ? "sources" : "search"} key={capability.id} />
          : <ContextSupply key={capability.id} view={capability.id === "resources" ? "resources" : "bundles"} />
      ) : domain === "models" ? (
        capability.id === "services" ? <ModelServices /> : <ModelSupply view={capability.id === "datasets" ? "datasets" : "artifacts"} key={capability.id} />
      ) : <PlatformEnvironment />}
    </div>
  );
}
