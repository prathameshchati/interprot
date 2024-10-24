import { useEffect, useState } from "react";
import MolstarViewer from "./components/MolstarViewer";
import SeqViewer, { SingleSeq } from "./components/SeqViewer";
import CustomViewer from "./components/CustomViewer";
import { SAE_CONFIGS } from "./SAEConfigs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Sidebar, SidebarContent, SidebarProvider, SidebarHeader } from "@/components/ui/sidebar";

import "./App.css";
import { Toggle } from "./components/ui/toggle";

const NUM_SEQS_TO_DISPLAY = 9;

function App() {
  const [selectedModel, setSelectedModel] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get("model") || "4096-dim SAE on ESM2-650M Layer 24";
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
          <h2 className="text-xl font-semibold" style={{ margin: 5 }}>
            SAE Feature
          </h2>
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
        </SidebarHeader>
        <SidebarContent>
          <ul className="space-y-2 font-medium">
            {config.curated?.map((i) => (
              <Toggle
                key={`feature-${i.dim}`}
                style={{ width: "100%", paddingLeft: 20 }}
                className="justify-start"
                pressed={feature === i.dim}
                onPressedChange={() => setFeature(i.dim)}
              >
                {i.name}
              </Toggle>
            ))}
            {Array.from({ length: config.numHiddenDims }, (_, i) => i).map((i) => (
              <Toggle
                key={`feature-${i}`}
                style={{ width: "100%", paddingLeft: 20 }}
                className="justify-start"
                pressed={feature === i}
                onPressedChange={() => setFeature(i)}
              >
                {i}
              </Toggle>
            ))}
          </ul>
        </SidebarContent>
      </Sidebar>
      <main className="text-left max-w-full overflow-x-auto">
        <h1 className="text-3xl font-bold">Feature {feature}</h1>
        {dimToCuratedMap.has(feature) && <p>{dimToCuratedMap.get(feature)?.desc}</p>}
        {config?.supportsCustomSequence && <CustomViewer feature={feature} />}
        <div className="p-4 mt-5 border-2 border-gray-200 border-dashed rounded-lg">
          <div className="overflow-x-auto">
            {featureData.map((seq) => (
              <SeqViewer seq={seq} key={`seq-${seq.alphafold_id}`} />
            ))}
          </div>
        </div>
        <div className="container mx-auto p-4">
          <div className="grid grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
            {featureData.map((seq) => (
              <div className="bg-white rounded-lg flex items-center justify-center">
                <MolstarViewer
                  key={`molstar-${seq.alphafold_id}`}
                  alphafold_id={seq.alphafold_id}
                  activation_list={seq.tokens_acts_list}
                />
              </div>
            ))}
          </div>
        </div>
      </main>
    </SidebarProvider>
  );
}

export default App;
