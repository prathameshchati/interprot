import { useEffect, useState } from "react";
import MolstarMulti from "./components/MolstarMulti";
import SeqViewer, { SingleSeq } from "./components/SeqViewer";
import CustomSeqPlayground from "./components/CustomSeqPlayground";
import { CuratedFeature, SAE_CONFIGS } from "./SAEConfigs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sidebar,
  SidebarContent,
  SidebarProvider,
  SidebarHeader,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import HomeNavigator from "@/components/HomeNavigator";
import "./App.css";
import { Toggle } from "./components/ui/toggle";

const NUM_SEQS_TO_DISPLAY = 9;

interface FeatureListProps {
  config: {
    curated?: CuratedFeature[];
    numHiddenDims: number;
  };
  feature: number;
  setFeature: (feature: number) => void;
}

function FeatureList({ config, feature, setFeature }: FeatureListProps) {
  const { setOpenMobile } = useSidebar();

  const handleFeatureChange = (feature: number) => {
    setFeature(feature);
    setOpenMobile(false);
  };

  return (
    <ul className="space-y-2 font-medium">
      {config.curated?.map((c) => (
        <Toggle
          key={`feature-${c.dim}`}
          style={{ width: "100%", paddingLeft: 20, textAlign: "left" }}
          className="justify-start"
          pressed={feature === c.dim}
          onPressedChange={() => handleFeatureChange(c.dim)}
        >
          {c.name}
        </Toggle>
      ))}
      {Array.from({ length: config.numHiddenDims }, (_, i) => i).map((i) => (
        <Toggle
          key={`feature-${i}`}
          style={{ width: "100%", paddingLeft: 20 }}
          className="justify-start"
          pressed={feature === i}
          onPressedChange={() => handleFeatureChange(i)}
        >
          {i}
        </Toggle>
      ))}
    </ul>
  );
}

function SAEVisualizer() {
  const [selectedModel, setSelectedModel] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get("model") || "SAE4096-L24";
  });
  const config = SAE_CONFIGS[selectedModel];
  const dimToCuratedMap = new Map(config.curated?.map((i) => [i.dim, i]));

  const [feature, setFeature] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return parseInt(params.get("feature") || config.defaultDim.toString(), 10);
  });

  useEffect(() => {
    const updateUrl = () => {
      const newUrl = new URL(window.location.href);
      newUrl.searchParams.set("model", selectedModel);
      newUrl.searchParams.set("feature", feature.toString());
      window.history.pushState({}, "", newUrl);
    };
    updateUrl();
  }, [config, feature, selectedModel]);

  const [featureData, setFeatureData] = useState<SingleSeq[]>([]);

  useEffect(() => {
    const fileURL = `${config.baseUrl}${feature}.json`;
    fetch(fileURL)
      .then((response) => response.json())
      .then((data) => {
        setFeatureData(data.slice(0, NUM_SEQS_TO_DISPLAY));
      });
  }, [config, feature]);

  return (
    <SidebarProvider>
      <Sidebar>
        <SidebarHeader>
          <div className="m-4">
            <HomeNavigator />
          </div>
          <Select value={selectedModel} onValueChange={(value) => setSelectedModel(value)}>
            <SelectTrigger>
              <SelectValue placeholder="Select SAE Model" />
            </SelectTrigger>
            <SelectContent>
              {Object.keys(SAE_CONFIGS).map((model) => (
                <SelectItem key={model} value={model}>
                  {model}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div className="text-sm mt-2 text-left" style={{ paddingLeft: 10, paddingRight: 10 }}>
            <p>
              This SAE was trained on layer {config.plmLayer} of{" "}
              <a href="https://huggingface.co/facebook/esm2_t33_650M_UR50D" className="underline">
                ESM2-650M
              </a>{" "}
              and has {config.numHiddenDims} hidden dimensions. Click on a feature below to
              visualize its activation pattern.
            </p>
          </div>
          <Separator />
        </SidebarHeader>
        <SidebarContent>
          <FeatureList config={config} feature={feature} setFeature={setFeature} />
        </SidebarContent>
      </Sidebar>
      <main className="text-left max-w-full overflow-x-auto">
        {/* HACK to make the divider extend the entire width of the screen */}
        <div
          className="fixed top-0 w-full bg-background border-b border-border z-50 py-4 px-8 md:hidden"
          style={{ marginLeft: -28, width: "calc(100% + 56px)" }}
        >
          <div className="flex items-center justify-between">
            <SidebarTrigger />
          </div>
        </div>
        <h1 className="text-3xl font-semibold md:mt-0 mt-16">Feature {feature}</h1>
        {dimToCuratedMap.has(feature) && (
          <p className="mt-2">{dimToCuratedMap.get(feature)?.desc}</p>
        )}
        {config?.supportsCustomSequence && <CustomSeqPlayground feature={feature} />}
        <div className="p-4 mt-5 border-2 border-gray-200 border-dashed rounded-lg">
          <div className="overflow-x-auto">
            {featureData.map((seq) => (
              <SeqViewer seq={seq} key={`seq-${seq.alphafold_id}`} />
            ))}
          </div>
        </div>

        <MolstarMulti proteins={featureData} />
      </main>
    </SidebarProvider>
  );
}

export default SAEVisualizer;
