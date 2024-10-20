import { useEffect, useState } from "react";
import MolstarViewer from "./components/MolstarViewer";
import SeqViewer, { SingleSeq } from "./components/SeqViewer";
import { SAE_CONFIGS } from "./SAEConfigs";

import "./App.css";

function App() {
  const [selectedModel, setSelectedModel] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get("model") || "4096-dim SAE on ESM2-650M Layer 24";
  });
  const config = SAE_CONFIGS[selectedModel];
  const dimToCuratedMap = new Map(config.curated?.map((i) => [i.dim, i]));

  const [feature, setFeature] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return parseInt(params.get("feature") || "2293", 10);
  });
  const [isSidebarOpen, setSidebarOpen] = useState(false);

  const toggleSidebar = () => {
    setSidebarOpen(!isSidebarOpen);
  };

  useEffect(() => {
    const updateUrl = () => {
      const newUrl = new URL(window.location.href);
      newUrl.searchParams.set("model", selectedModel);
      newUrl.searchParams.set("feature", feature.toString());
      window.history.pushState({}, "", newUrl);
    };

    updateUrl();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "ArrowDown") {
        setFeature((prev) => Math.min(prev + 1, config.numHiddenDims - 1));
      } else if (event.key === "ArrowUp") {
        setFeature((prev) => Math.max(prev - 1, 0));
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [config, feature, selectedModel]);

  const [featureData, setFeatureData] = useState<SingleSeq[]>([]);

  useEffect(() => {
    const fileURL = `${config.baseUrl}${feature}.json`;
    fetch(fileURL)
      .then((response) => response.json())
      .then((data) => {
        setFeatureData(data);
      });
  }, [config, feature]);

  return (
    <div>
      <button
        onClick={toggleSidebar}
        aria-controls="default-sidebar"
        type="button"
        className="inline-flex items-center p-2 mt-2 ms-3 text-sm text-gray-500 rounded-lg sm:hidden hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-200"
      >
        <span className="sr-only">Open sidebar</span>
        <svg
          className="w-6 h-6"
          aria-hidden="true"
          fill="currentColor"
          viewBox="0 0 20 20"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            clip-rule="evenodd"
            fill-rule="evenodd"
            d="M2 4.75A.75.75 0 012.75 4h14.5a.75.75 0 010 1.5H2.75A.75.75 0 012 4.75zm0 10.5a.75.75 0 01.75-.75h7.5a.75.75 0 010 1.5h-7.5a.75.75 0 01-.75-.75zM2 10a.75.75 0 01.75-.75h14.5a.75.75 0 010 1.5H2.75A.75.75 0 012 10z"
          ></path>
        </svg>
      </button>

      <aside
        id="default-sidebar"
        className={`fixed top-0 left-0 z-40 w-100 h-screen transition-transform transform ${
          isSidebarOpen ? "translate-x-0" : "-translate-x-full"
        } sm:translate-x-0`}
        aria-label="Sidebar"
      >
        <div className="h-full px-3 py-4 overflow-y-auto bg-gray-50">
          <div className="mb-4">
            <label
              htmlFor="model-select"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Select SAE Model
            </label>
            <select
              id="model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5"
            >
              {Object.keys(SAE_CONFIGS).map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>
          <ul className="space-y-2 font-medium">
            {config.curated?.map((i) => (
              <li key={`feature-${i.dim}`}>
                <a
                  onClick={() => setFeature(i.dim)}
                  className={`flex items-center p-2 text-gray-900 rounded-lg hover:bg-gray-100 cursor-pointer group ${
                    feature === i.dim ? "font-bold" : ""
                  }`}
                >
                  <span className="ms-3">{i.name}</span>
                </a>
              </li>
            ))}
            {Array.from({ length: config.numHiddenDims }, (_, i) => i).map(
              (i) => (
                <li key={`feature-${i}`}>
                  <a
                    onClick={() => setFeature(i)}
                    className={`flex items-center p-2 text-gray-900 rounded-lg hover:bg-gray-100 cursor-pointer group ${
                      feature === i ? "font-bold" : ""
                    }`}
                  >
                    <span className="ms-3">{i}</span>
                  </a>
                </li>
              )
            )}
          </ul>
        </div>
      </aside>
      <div className="sm:ml-64 text-left">
        <h1 className="text-3xl font-bold">Feature: {feature}</h1>
        {dimToCuratedMap.has(feature) && (
          <p>{dimToCuratedMap.get(feature)?.desc}</p>
        )}
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
      </div>
    </div>
  );
}

export default App;
