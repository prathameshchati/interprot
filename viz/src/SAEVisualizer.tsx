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
  SidebarGroup,
  SidebarGroupLabel,
} from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import HomeNavigator from "@/components/HomeNavigator";
import "./App.css";
import { Toggle } from "./components/ui/toggle";
import { useParams, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Dices } from "lucide-react";

const NUM_SEQS_TO_DISPLAY = 9;
const SHOW_MODEL_SELECTOR = false;

interface FeatureListProps {
  config: {
    curated?: CuratedFeature[];
    plmLayer: number;
    numHiddenDims: number;
    supportsCustomSequence?: boolean;
  };
  model: string;
  setModel: (model: string) => void;
  feature: number;
  setFeature: (feature: number) => void;
}

function FeatureSidebar({ config, model, setModel, feature, setFeature }: FeatureListProps) {
  const { setOpenMobile } = useSidebar();

  const handleFeatureChange = (feature: number) => {
    setFeature(feature);
    setOpenMobile(false);
  };

  const groupedFeatures = config.curated?.reduce((acc, feature) => {
    const group = feature.group || "not classified";
    if (!acc[group]) acc[group] = [];
    acc[group].push(feature);
    return acc;
  }, {} as Record<string, CuratedFeature[]>);

  return (
    <Sidebar>
      <SidebarHeader>
        <div className="m-3">
          <HomeNavigator />
        </div>
        {SHOW_MODEL_SELECTOR && (
          <Select value={model} onValueChange={(value) => setModel(value)}>
            <SelectTrigger className="mb-3">
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
        )}
        <div className="text-sm text-left px-3 mb-2">
          <p>
            This{" "}
            <a href="https://huggingface.co/liambai/InterProt-ESM2-SAEs" className="underline">
              SAE
            </a>{" "}
            was trained on layer {config.plmLayer} of{" "}
            <a href="https://huggingface.co/facebook/esm2_t33_650M_UR50D" className="underline">
              ESM2-650M
            </a>{" "}
            and has {config.numHiddenDims} hidden dimensions. Click on a feature below to visualize
            its activation pattern.
          </p>
        </div>
        <Button
          variant="outline"
          className="mb-3 mx-3"
          onClick={() => {
            handleFeatureChange(Math.floor(Math.random() * config.numHiddenDims));
          }}
        >
          <Dices className="w-4 h-4 mr-2" /> Random Feature
        </Button>
        <Separator />
      </SidebarHeader>
      <SidebarContent>
        <ul className="space-y-2 font-medium">
          {groupedFeatures &&
            Object.entries(groupedFeatures).map(([group, features]) => (
              <SidebarGroup key={group}>
                <SidebarGroupLabel>{group}</SidebarGroupLabel>
                {features.map((c) => (
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
              </SidebarGroup>
            ))}
          <SidebarGroup>
            <SidebarGroupLabel>all features</SidebarGroupLabel>
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
          </SidebarGroup>
        </ul>
      </SidebarContent>
    </Sidebar>
  );
}

function SAEVisualizer() {
  const { model, feature } = useParams();
  const navigate = useNavigate();

  const [selectedModel, setSelectedModel] = useState(() => {
    return model && SAE_CONFIGS[model] ? model : "SAE4096-L24";
  });
  const config = SAE_CONFIGS[selectedModel] || SAE_CONFIGS["SAE4096-L24"];
  const dimToCuratedMap = new Map(config?.curated?.map((i) => [i.dim, i]) || []);

  const [selectedFeature, setSelectedFeature] = useState(() => {
    return parseInt(feature || config.defaultDim?.toString(), 10);
  });

  useEffect(() => {
    navigate(`/sae-viz/${selectedModel}/${selectedFeature}`);
  }, [selectedModel, selectedFeature, navigate]);

  const [featureData, setFeatureData] = useState<SingleSeq[]>([]);

  useEffect(() => {
    const fileURL = `${config.baseUrl}${selectedFeature}.json`;
    fetch(fileURL)
      .then((response) => response.json())
      .then((data) => {
        setFeatureData(data.slice(0, NUM_SEQS_TO_DISPLAY));
      });
  }, [config, selectedFeature]);

  return (
    <SidebarProvider>
      <FeatureSidebar
        config={config}
        model={selectedModel}
        setModel={setSelectedModel}
        feature={selectedFeature}
        setFeature={setSelectedFeature}
      />
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
        <h1 className="text-3xl font-semibold md:mt-0 mt-16">Feature {selectedFeature}</h1>
        {dimToCuratedMap.has(selectedFeature) && (
          <p className="mt-3">{dimToCuratedMap.get(selectedFeature)?.desc}</p>
        )}
        {config?.supportsCustomSequence && <CustomSeqPlayground feature={selectedFeature} />}
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
