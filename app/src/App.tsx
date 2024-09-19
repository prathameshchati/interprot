import MolstarViewer from "./components/MolstarViewer";
import SeqViewer from "./components/SeqViewer";

import "./App.css";
import React from "react";

function App() {
  return (
    <>
      <div className="flex">
        <div className="w-1/4 bg-gray-200">
          {/* Sidebar content */}
        </div>
        <div className="w-3/4">
          <div className="card">
            <SeqViewer
              seq={{
                tokens_acts_list: [1, 2, 3, 4, 5, 6],
                tokens_id_list: [1, 2, 3, 4, 5, 6],
              }}
            />
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
