import React, { useEffect, useState } from "react";
import MolstarViewer from "./components/MolstarViewer";
import SeqViewer, { SingleSeq } from "./components/SeqViewer";

import "./App.css";

// TODO: Filter this down to a curated, interesting set of dims
const hiddenDims = Array.from({ length: 4096 }, (_, index) => index);

const CONFIG: { baseUrl: string; hiddenDims: number[] } = {
  baseUrl:
    "https://raw.githubusercontent.com/liambai/plm-interp-viz-data/refs/heads/main/esm2_plm1280_l24_sae4096_100Kseqs/",
  hiddenDims: hiddenDims,
};

function App() {
  const [feature, setFeature] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return parseInt(params.get("feature") || "0", 10);
  });

  useEffect(() => {
    const updateUrl = () => {
      const newUrl = new URL(window.location.href);
      newUrl.searchParams.set("feature", feature.toString());
      window.history.pushState({}, "", newUrl);
    };

    updateUrl();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "ArrowDown") {
        setFeature((prev) => Math.min(prev + 1, CONFIG.hiddenDims.length - 1));
      } else if (event.key === "ArrowUp") {
        setFeature((prev) => Math.max(prev - 1, 0));
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [feature]);

  const [featureData, setFeatureData] = useState<SingleSeq[]>([]);

  useEffect(() => {
    const fileURL = `${CONFIG.baseUrl}${feature}.json`;
    fetch(fileURL)
      .then((response) => response.json())
      .then((data) => {
        setFeatureData(data);
      });
  }, [feature]);
  return (
    <div>
      <aside
        className="fixed top-0 left-0 z-40 w-64 h-screen transition-transform -translate-x-full sm:translate-x-0"
        aria-label="Sidebar"
      >
        <div className="h-full px-3 py-4 overflow-y-auto bg-gray-50 dark:bg-gray-800">
          <ul className="space-y-2 font-medium">
            {CONFIG.hiddenDims.map((i) => (
              <li>
                <a
                  onClick={() => setFeature(i)}
                  className={`flex items-center p-2 text-gray-900 rounded-lg dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700 group ${
                    feature === i ? "font-bold" : ""
                  }`}
                >
                  <span className="ms-3">{i}</span>
                </a>
              </li>
            ))}
          </ul>
        </div>
      </aside>
      <div className="sm:ml-64">
        <h1 className="text-3xl font-bold">{feature}</h1>
        <div className="p-4 mt-5 border-2 border-gray-200 border-dashed rounded-lg dark:border-gray-700">
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
