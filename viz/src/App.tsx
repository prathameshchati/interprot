import { useEffect, useState } from "react";
import MolstarViewer from "./components/MolstarViewer";
import SeqViewer, { SingleSeq } from "./components/SeqViewer";

import "./App.css";

// TODO: Filter this down to a curated, interesting set of dims
const hiddenDims = Array.from({ length: 4096 }, (_, index) => index);

const CONFIG: { baseUrl: string; hiddenDims: number[], curated?: {name: string, dim: number, desc: string}[] } = {
  baseUrl:
    "https://raw.githubusercontent.com/liambai/plm-interp-viz-data/refs/heads/main/esm2_plm1280_l24_sae4096_100Kseqs/",
  hiddenDims: hiddenDims,
  curated: [
    {name: "free alpha helices", dim: 2293, desc: "Activates on every fourth amino acid in free alpha helices"},
    {name: "long alpha helices", dim: 1008, desc: "Activates on most amino acids in long alpha helices"},
    {name: "alpha helix turn", dim: 56, desc: "Activates on the turn between two alpha helices in ABC transporter proteins"},
    {name: "single beta sheet", dim: 1299, desc: "Activates on a single beta sheet"},
    {name: "beta sheet: first aa", dim: 782, desc: "Activates on the first amino acid in beta sheets"},
    {name: "leucine rich repeats", dim: 3425, desc: "Activates on the amino acid before the start of a beta sheet in a leucine rich repeat"},
    {name: "start M", dim: 600, desc: "Activates on the M amino acid at the start of a sequence"},
    {name: "second residue", dim: 3728, desc: "Mostly activates on the second amino acid in a sequence"},
  ]
};

const dimToCuratedMap = new Map(CONFIG.curated?.map((i) => [i.dim, i]));

function App() {
  const [feature, setFeature] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return parseInt(params.get("feature") || "2293", 10);
  });
  const [isSidebarOpen, setSidebarOpen] = useState(false);

  // Step 2: Function to toggle the sidebar's visibility
  const toggleSidebar = () => {
    setSidebarOpen(!isSidebarOpen);
  };

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
        className={`fixed top-0 left-0 z-40 w-64 h-screen transition-transform transform ${
          isSidebarOpen ? "translate-x-0" : "-translate-x-full"
        } sm:translate-x-0`}
        aria-label="Sidebar"
      >
        <div className="h-full px-3 py-4 overflow-y-auto bg-gray-50">
          <ul className="space-y-2 font-medium">
            {
              CONFIG.curated?.map((i) => (
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
              ))
            }
            {CONFIG.hiddenDims.map((i) => (
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
            ))}
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
