import MolstarViewer from "./components/MolstarViewer";
import SeqViewer from "./components/SeqViewer";

import "./App.css";
import React, { useState } from "react";

import { data } from "./data";

function App() {
  const [feature, setFeature] = useState(0);

  return (
    <div>
      <aside
        className="fixed top-0 left-0 z-40 w-64 h-screen transition-transform -translate-x-full sm:translate-x-0"
        aria-label="Sidebar"
      >
        <div className="h-full px-3 py-4 overflow-y-auto bg-gray-50 dark:bg-gray-800">
          <ul className="space-y-2 font-medium">
            {data.map((seqInfo, i) => (
              <li>
                <a
                  href="#"
                  onClick={() => setFeature(i)}
                  className={`flex items-center p-2 text-gray-900 rounded-lg dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700 group ${
                    feature === i ? "font-bold" : ""
                  }`}
                >
                  <span className="ms-3">{seqInfo.dimenstion}</span>
                </a>
              </li>
            ))}
          </ul>
        </div>
      </aside>
      <div className="sm:ml-64">
        <h1 className="text-3xl font-bold">{data[feature].dimenstion}</h1>

        <div className="container mx-auto p-4">
          <div className="grid grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
            {data[feature].examples.map((seq) => (
              <div
                className="bg-white rounded-lg flex items-center justify-center"
              >
                <MolstarViewer
                  key={`molstar-${seq.alphafold_id}`}
                  alphafold_id={seq.alphafold_id}
                  activation_list={seq.tokens_acts_list}
                />
              </div>
            ))}
          </div>
        </div>
        <div className="p-4 mt-5 border-2 border-gray-200 border-dashed rounded-lg dark:border-gray-700">
          {data[feature].examples.map((seq) => (
            <SeqViewer key={`seq-${seq.alphafold_id}`} seq={seq} />
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
